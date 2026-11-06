/** Copyright 2026 RWTH Aachen University. All rights reserved.
 *
 *  Licensed under the RWTH ASR License (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "StateManagedOnnxLabelScorer.hh"

#include <algorithm>

#include <Core/XmlStream.hh>

#include <Nn/Module.hh>

#include "OnnxEncoderStates.hh"
#include "ScoreAccessor.hh"

/*
 * Scoring context for ONNX models with state managed by a StateManager.
 * The state stores only the slice produced for the most recent token.
 */
namespace Nn {

StateManagedOnnxScoringContext::StateManagedOnnxScoringContext(HistoryState&& initialState)
        : labelSeq(),
          parent(),
          state(std::make_shared<HistoryState>(std::move(initialState))) {
}

StateManagedOnnxScoringContext::StateManagedOnnxScoringContext(std::vector<LabelIndex>&&                       labelSeq,
                                                               Core::Ref<StateManagedOnnxScoringContext const> parent)
        : labelSeq(std::move(labelSeq)),
          parent(parent),
          state() {
}

bool StateManagedOnnxScoringContext::isEqual(ScoringContextRef const& other) const {
    auto* otherPtr = dynamic_cast<StateManagedOnnxScoringContext const*>(other.get());

    if (otherPtr == nullptr) {
        return false;
    }

    return labelSeqEqual(labelSeq, otherPtr->labelSeq);
}

size_t StateManagedOnnxScoringContext::hash() const {
    return labelSeqHash(labelSeq);
}

static const std::vector<Onnx::IOSpecification> ioSpec = {
        Onnx::IOSpecification{
                "token",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{-1, 1}, {-1, -1}}},
        Onnx::IOSpecification{
                "token-length",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{-1}}},
        Onnx::IOSpecification{
                "prefix-length",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{-1}}},
        Onnx::IOSpecification{
                "encoder-states",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{1, -1, -2}, {-1, -1, -2}}},
        Onnx::IOSpecification{
                "encoder-states-size",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{1}, {-1}}},
        Onnx::IOSpecification{
                "encoder-states-index",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{-1}}},
        Onnx::IOSpecification{
                "scores",
                Onnx::IODirection::OUTPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, 1, -2}, {-1, -1, -2}}}};

/*
 * ==================================
 * == StateManagedOnnxLabelScorer ===
 * ==================================
 */

const Core::ParameterIntVector StateManagedOnnxLabelScorer::paramStartLabels(
        "start-labels",
        "Initial context consists of these tokens.",
        "",
        0,
        Core::Type<s32>::max,
        1);

const Core::ParameterBool StateManagedOnnxLabelScorer::paramBlankUpdatesHistory(
        "blank-updates-history",
        "Whether previously emitted blank labels should be used to update the recurrent state.",
        false);

const Core::ParameterBool StateManagedOnnxLabelScorer::paramSilenceUpdatesHistory(
        "silence-updates-history",
        "Whether previously emitted silence labels should be used to update the recurrent state.",
        false);

const Core::ParameterBool StateManagedOnnxLabelScorer::paramLoopUpdatesHistory(
        "loop-updates-history",
        "Whether loop transitions should update the recurrent state.",
        false);

const Core::ParameterInt StateManagedOnnxLabelScorer::paramMaxBatchSize(
        "max-batch-size",
        "Max number of scoring contexts that can be fed into the ONNX model at once.",
        Core::Type<int>::max);

const Core::ParameterInt StateManagedOnnxLabelScorer::paramMaxCachedScores(
        "max-cached-score-vectors",
        "Maximum size of cache that maps scoring contexts to scores and state slices. This prevents memory overflow in case of very long audio segments."
        " In multi-segment usage this applies to each segment separately.",
        10000);

const Core::ParameterBool StateManagedOnnxLabelScorer::paramStackEncoderStates(
        "stack-encoder-states",
        "Whether contexts of different segments are forwarded together by stacking the encoder states of their segments to shape"
        " [B, T_max, E] (zero-padded). Requires a model with dynamic batch dimension of the encoder states that masks the padding"
        " based on the encoder states size. Otherwise contexts are forwarded per segment.",
        false);

StateManagedOnnxLabelScorer::StateManagedOnnxLabelScorer(Core::Configuration const& config, ModelCache& modelCache)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::LM),
          blankUpdatesHistory_(paramBlankUpdatesHistory(config)),
          silenceUpdatesHistory_(paramSilenceUpdatesHistory(config)),
          loopUpdatesHistory_(paramLoopUpdatesHistory(config)),
          maxBatchSize_(paramMaxBatchSize(config)),
          maxCachedScores_(paramMaxCachedScores(config)),
          stackEncoderStates_(paramStackEncoderStates(config)),
          stateManager_(Module::instance().createStateManager(
                  select("state-manager"),
                  Onnx::Session::executionProviderType(Core::Configuration(Core::Configuration(config, "onnx-model"), "session")))),
          stateVectorFactory_(Module::instance().createCompressedVectorFactory(select("state-compression"))),
          segmentData_() {
    Core::Configuration modelConfig(config, "onnx-model");
    auto                key = modelConfig.getSelection();
    onnxModel_              = modelCache.getOrCreate<Onnx::Model>(key, modelConfig, ioSpec);
    tokenName_              = onnxModel_->mapping.getOnnxName("token");
    tokenLengthName_        = onnxModel_->mapping.getOnnxName("token-length");
    prefixLengthName_       = onnxModel_->mapping.getOnnxName("prefix-length");
    scoresName_             = onnxModel_->mapping.getOnnxName("scores");
    encoderStatesName_      = onnxModel_->mapping.getOnnxName("encoder-states");
    encoderStatesSizeName_  = onnxModel_->mapping.getOnnxName("encoder-states-size");
    encoderStatesIndexName_ = onnxModel_->mapping.getOnnxName("encoder-states-index");

    stateVariables_ = onnxModel_->session.getStateVariablesMetadata();

    auto startLabels = paramStartLabels(config);
    startLabels_.insert(startLabels_.begin(), startLabels.begin(), startLabels.end());

    if (stackEncoderStates_ and encoderStatesName_ != "") {
        if (encoderStatesSizeName_ == "") {
            error() << "stack-encoder-states requires the encoder-states-size input to mask the padding";
        }
        auto problem = checkStackableEncoderStatesInput(onnxModel_->session, encoderStatesName_);
        if (not problem.empty()) {
            error() << "stack-encoder-states is not possible with this model: " << problem;
        }
    }
}

StateManagedOnnxLabelScorer::SegmentData::SegmentData(size_t maxCachedScores)
        : encoderStatesValue(),
          encoderStatesSizeValue(),
          scoreCache(maxCachedScores),
          stateCache(maxCachedScores) {}

StateManagedOnnxLabelScorer::SegmentData& StateManagedOnnxLabelScorer::segmentData(SegmentId segment) {
    return segmentData_.try_emplace(segment, maxCachedScores_).first->second;
}

bool StateManagedOnnxLabelScorer::usesEncoderStates() const {
    return encoderStatesName_ != "" or encoderStatesSizeName_ != "";
}

void StateManagedOnnxLabelScorer::addSegment(SegmentId segment) {
    Precursor::addSegment(segment);
    // Create the segment data here, so that it is never inserted while contexts of other segments are processed
    segmentData_.erase(segment);
    segmentData(segment);
}

void StateManagedOnnxLabelScorer::removeSegment(SegmentId segment) {
    Precursor::removeSegment(segment);
    segmentData_.erase(segment);
}

void StateManagedOnnxLabelScorer::addSegmentInput(SegmentId segment, DataView const& input) {
    Precursor::addSegmentInput(segment, input);
    auto& data = segmentData(segment);
    data.encoderStatesValue.clear();
    data.encoderStatesSizeValue = Onnx::Value();
}

ScoringContextRef StateManagedOnnxLabelScorer::getInitialSegmentScoringContext(SegmentId segment) {
    auto rootState = stateManager_->initialState(stateVariables_, *stateVectorFactory_);
    if (rootState.empty()) {
        error("Initial recurrent state is empty.");
    }

    auto context     = Core::ref(new StateManagedOnnxScoringContext(std::move(rootState)));
    context->segment = segment;

    for (auto const label : startLabels_) {
        std::vector<LabelIndex> labelSeq(context->labelSeq);
        labelSeq.push_back(label);
        context          = Core::ref(new StateManagedOnnxScoringContext(std::move(labelSeq), context));
        context->segment = segment;
        cacheStatesAndScores({context});
    }
    return context;
}

ScoringContextRef StateManagedOnnxLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) {
    bool updateState = false;
    switch (transitionType) {
        case TransitionType::BLANK_LOOP:
            updateState = blankUpdatesHistory_ and loopUpdatesHistory_;
            break;
        case TransitionType::SILENCE_LOOP:
            updateState = silenceUpdatesHistory_ and loopUpdatesHistory_;
            break;
        case TransitionType::LABEL_TO_BLANK:
        case TransitionType::INITIAL_BLANK:
            updateState = blankUpdatesHistory_;
            break;
        case TransitionType::LABEL_TO_SILENCE:
        case TransitionType::INITIAL_SILENCE:
            updateState = silenceUpdatesHistory_;
            break;
        case TransitionType::LABEL_LOOP:
            updateState = loopUpdatesHistory_;
            break;
        case TransitionType::BLANK_TO_LABEL:
        case TransitionType::SILENCE_TO_LABEL:
        case TransitionType::LABEL_TO_LABEL:
        case TransitionType::INITIAL_LABEL:
        case TransitionType::SENTENCE_END:
            updateState = true;
            break;
        default:
            error() << "Unknown transition type " << transitionType;
    }
    if (not updateState) {
        return scoringContext;
    }

    StateManagedOnnxScoringContextRef context(dynamic_cast<StateManagedOnnxScoringContext const*>(scoringContext.get()));
    verify(context);

    std::vector<LabelIndex> labelSeq(context->labelSeq);
    labelSeq.push_back(nextToken);
    auto newContext     = Core::ref(new StateManagedOnnxScoringContext(std::move(labelSeq), context));
    newContext->segment = context->segment;
    return newContext;
}

void StateManagedOnnxLabelScorer::resetInternal() {
    Precursor::resetInternal();
    segmentData_.clear();
}

void StateManagedOnnxLabelScorer::resetStatisticsInternal() {
    Precursor::resetStatisticsInternal();
    onnxSessionTime_.reset();
    stateMarshallingTime_.reset();
    contextPreparationTime_.reset();
}

std::optional<ScoreAccessorRef> StateManagedOnnxLabelScorer::computeScoreAccessor(ScoringContextRef scoringContext) {
    return computeScoreAccessors({scoringContext})[0];
}

std::vector<std::optional<ScoreAccessorRef>> StateManagedOnnxLabelScorer::computeScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) {
    if (scoringContexts.empty()) {
        return {};
    }

    std::vector<std::optional<ScoreAccessorRef>> scoreAccessors(scoringContexts.size(), std::nullopt);

    // Only allow scoring once all encoder states of a segment have been passed
    auto segmentReady = [this](SegmentId segment) {
        return not usesEncoderStates() or (not expectMoreFeatures(segment) and bufferSize(segment) != 0);
    };

    /*
     * Identify unique scoring contexts that still need session runs
     */
    contextPreparationTime_.start();
    std::unordered_set<StateManagedOnnxScoringContextRef, ScoringContextHash, ScoringContextEq> uniqueUncachedScoringContexts;
    for (auto const& scoringContext : scoringContexts) {
        StateManagedOnnxScoringContextRef context(dynamic_cast<StateManagedOnnxScoringContext const*>(scoringContext.get()));
        verify(context);
        if (not segmentReady(context->segment)) {
            continue;
        }
        auto& data = segmentData(context->segment);
        if (not context->state and data.stateCache.contains(context)) {
            context->state = *data.stateCache.get(context);
        }
        // A context with a state was already forwarded (e.g. the start labels in `getInitialSegmentScoringContext`) and must not be
        // forwarded again: its parent may already be released, as for state managers that don't require all parent states
        if (not context->state or not data.scoreCache.contains(context)) {
            uniqueUncachedScoringContexts.emplace(context);
        }
    }

    /*
     * Group contexts that can be forwarded together: all contexts if the model doesn't need
     * encoder states or they are stacked, otherwise the contexts of each segment
     */
    bool                                                        groupBySegment = usesEncoderStates() and not stackEncoderStates_;
    std::vector<std::vector<StateManagedOnnxScoringContextRef>> contextGroups;
    std::unordered_map<SegmentId, size_t>                       segmentToGroupIndex;
    for (auto const& scoringContext : uniqueUncachedScoringContexts) {
        SegmentId groupKey  = groupBySegment ? scoringContext->segment : defaultSegmentId;
        auto [it, inserted] = segmentToGroupIndex.emplace(groupKey, contextGroups.size());
        if (inserted) {
            contextGroups.emplace_back();
        }
        contextGroups[it->second].push_back(scoringContext);
    }
    contextPreparationTime_.stop();

    countUncomputedAccessors(scoringContexts.size() - uniqueUncachedScoringContexts.size());

    /*
     * Fill state and score caches for all uncached scoring contexts
     */
    for (auto const& contextGroup : contextGroups) {
        std::vector<StateManagedOnnxScoringContextRef> scoringContextBatch;
        scoringContextBatch.reserve(std::min(contextGroup.size(), maxBatchSize_));
        for (auto const& scoringContext : contextGroup) {
            scoringContextBatch.push_back(scoringContext);
            if (scoringContextBatch.size() == maxBatchSize_) {
                cacheStatesAndScores(scoringContextBatch);
                scoringContextBatch.clear();
            }
        }
        cacheStatesAndScores(scoringContextBatch);
    }

    /*
     * Assign scores from cache to result vector
     */
    contextPreparationTime_.start();
    for (size_t i = 0ul; i < scoringContexts.size(); ++i) {
        StateManagedOnnxScoringContextRef context(dynamic_cast<StateManagedOnnxScoringContext const*>(scoringContexts[i].get()));
        if (not segmentReady(context->segment)) {
            continue;
        }
        auto scores = segmentData(context->segment).scoreCache.get(context);
        verify(scores);
        scoreAccessors[i] = Core::ref(new VectorScoreAccessor(scores->get(), context->labelSeq.empty() ? 0ul : context->labelSeq.size() - 1ul));
    }
    contextPreparationTime_.stop();

    return scoreAccessors;
}

size_t StateManagedOnnxLabelScorer::getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const {
    return 0ul;
}

void StateManagedOnnxLabelScorer::setupEncoderStatesValue(SegmentId segment) {
    auto& data = segmentData(segment);
    if (not data.encoderStatesValue.empty()) {
        return;
    }

    u32  T                    = bufferSize(segment);
    auto inputFeatureDataView = getInput(segment, 0);
    verify(inputFeatureDataView);

    auto encoderStates = Onnx::Value::createEmpty<f32>({1l, static_cast<int64_t>(T), static_cast<int64_t>(inputFeatureDataView->size())});
    for (size_t t = 0ul; t < T; ++t) {
        inputFeatureDataView = getInput(segment, t);
        verify(inputFeatureDataView);
        std::copy(inputFeatureDataView->data(), inputFeatureDataView->data() + inputFeatureDataView->size(), encoderStates.data<f32>(0, t));
    }
    data.encoderStatesValue.set(std::move(encoderStates));
}

void StateManagedOnnxLabelScorer::setupEncoderStatesSizeValue(SegmentId segment) {
    auto& data = segmentData(segment);
    if (not data.encoderStatesSizeValue.empty()) {
        return;
    }
    data.encoderStatesSizeValue = Onnx::Value::create(std::vector<s32>{static_cast<s32>(bufferSize(segment))});
}

void StateManagedOnnxLabelScorer::cacheStatesAndScores(std::vector<StateManagedOnnxScoringContextRef> const& scoringContextBatch) {
    if (scoringContextBatch.empty()) {
        return;
    }

    // Can't score before any encoder features are buffered; defer (mirrors the guard in computeScoreAccessors()).
    if (usesEncoderStates() and std::any_of(scoringContextBatch.begin(), scoringContextBatch.end(), [this](auto const& context) { return bufferSize(context->segment) == 0ul; })) {
        return;
    }

    std::vector<size_t>              prefixLengths;
    std::vector<HistoryState const*> prefixStates;
    prefixLengths.reserve(scoringContextBatch.size());
    prefixStates.reserve(scoringContextBatch.size());

    // Fill in prefix states
    for (auto const& context : scoringContextBatch) {
        auto parent = context->parent;

        if (stateManager_->requiresAllParentStates()) {
            size_t prefixLength = parent->labelSeq.size();
            prefixLengths.push_back(prefixLength);
            prefixStates.resize(prefixStates.size() + prefixLength);

            size_t offset = prefixStates.size() - prefixLength;
            for (size_t i = 0ul; i < prefixLength; ++i) {
                prefixStates[offset + prefixLength - i - 1] = parent->state.get();
                parent                                      = parent->parent;
            }
        }
        else {
            prefixLengths.push_back(parent->labelSeq.size());
            prefixStates.push_back(parent->state.get());
        }
    }

    std::vector<std::pair<std::string, Onnx::Value>> inputs;
    std::vector<std::string>                         targets;
    stateMarshallingTime_.start();
    stateManager_->mergeStates(stateVariables_, prefixLengths, prefixStates, inputs, targets);
    stateMarshallingTime_.stop();

    Math::FastMatrix<s32> tokens(scoringContextBatch.size(), 1);
    for (size_t i = 0ul; i < scoringContextBatch.size(); ++i) {
        tokens.at(i, 0) = static_cast<s32>(scoringContextBatch[i]->labelSeq.back());
    }
    inputs.emplace_back(tokenName_, Onnx::Value::create(tokens));
    inputs.emplace_back(tokenLengthName_, Onnx::Value::create(std::vector<s32>(scoringContextBatch.size(), 1)));  // All suffix lengths are 1

    if (prefixLengthName_ != "") {
        std::vector<s32> stateLengths(prefixLengths.begin(), prefixLengths.end());
        inputs.emplace_back(prefixLengthName_, Onnx::Value::create(stateLengths));
    }
    if (usesEncoderStates()) {
        std::vector<SegmentId> segments;
        segments.reserve(scoringContextBatch.size());
        for (auto const& context : scoringContextBatch) {
            segments.push_back(context->segment);
        }
        addEncoderStatesInputs(
                {.states = encoderStatesName_, .size = encoderStatesSizeName_, .index = encoderStatesIndexName_},
                onnxModel_->session,
                segments,
                [this](SegmentId segment) -> Onnx::PersistentInput& {
                    setupEncoderStatesValue(segment);
                    return segmentData(segment).encoderStatesValue;
                },
                [this](SegmentId segment) -> Onnx::Value const& {
                    setupEncoderStatesSizeValue(segment);
                    return segmentData(segment).encoderStatesSizeValue;
                },
                [this](SegmentId segment) { return bufferSize(segment); },
                inputs);
    }

    targets.emplace(targets.begin(), scoresName_);

    std::vector<Onnx::Value> outputs;
    onnxSessionTime_.start();
    onnxModel_->session.run(std::move(inputs), targets, outputs);
    onnxSessionTime_.stop();

    for (size_t b = 0ul; b < scoringContextBatch.size(); ++b) {
        auto scores = std::make_shared<std::vector<Score>>();
        outputs.front().get(b, 0, *scores);
        segmentData(scoringContextBatch[b]->segment).scoreCache.put(scoringContextBatch[b], scores);
    }

    std::vector<Onnx::Value> stateOutputs(std::make_move_iterator(outputs.begin() + 1), std::make_move_iterator(outputs.end()));
    std::vector<size_t>      suffixLengths(scoringContextBatch.size(), 1ul);
    stateMarshallingTime_.start();
    auto splitStates = stateManager_->splitStates(stateVariables_, suffixLengths, stateOutputs, *stateVectorFactory_);
    stateMarshallingTime_.stop();
    verify_eq(splitStates.size(), scoringContextBatch.size());
    for (size_t i = 0ul; i < scoringContextBatch.size(); ++i) {
        scoringContextBatch[i]->state = std::make_shared<HistoryState>(std::move(splitStates[i]));
        segmentData(scoringContextBatch[i]->segment).stateCache.put(scoringContextBatch[i], scoringContextBatch[i]->state);
        if (not stateManager_->requiresAllParentStates()) {
            scoringContextBatch[i]->parent.reset();
        }
    }
}

void StateManagedOnnxLabelScorer::logScoringBreakdown() const {
    statisticsChannel_ << Core::XmlFull("onnx-session-time", onnxSessionTime_.elapsedMilliseconds());
    statisticsChannel_ << Core::XmlFull("state-marshalling-time", stateMarshallingTime_.elapsedMilliseconds());
    statisticsChannel_ << Core::XmlFull("context-preparation-time", contextPreparationTime_.elapsedMilliseconds());
}

}  // namespace Nn
