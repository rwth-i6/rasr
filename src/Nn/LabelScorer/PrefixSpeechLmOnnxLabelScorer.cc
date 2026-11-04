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

#include "PrefixSpeechLmOnnxLabelScorer.hh"

#include <Core/XmlStream.hh>

#include <Nn/Module.hh>
#include <algorithm>
#include <unordered_set>

#include "ScoreAccessor.hh"

namespace Nn {

/*
 * ====================================
 * === PrefixSpeechLmScoringContext ===
 * ====================================
 */
PrefixSpeechLmScoringContext::PrefixSpeechLmScoringContext()
        : labelSeq(),
          historyLength(0ul),
          parent(),
          state() {
}

PrefixSpeechLmScoringContext::PrefixSpeechLmScoringContext(std::vector<LabelIndex>&&                     labelSeq,
                                                           size_t                                        historyLength,
                                                           Core::Ref<PrefixSpeechLmScoringContext const> parent,
                                                           std::shared_ptr<HistoryState>                 state)
        : labelSeq(std::move(labelSeq)),
          historyLength(historyLength),
          parent(parent),
          state(std::move(state)) {
}

size_t PrefixSpeechLmScoringContext::hash() const {
    return labelSeqHash(labelSeq);
}

bool PrefixSpeechLmScoringContext::isEqual(ScoringContextRef const& other) const {
    auto* otherPtr = dynamic_cast<const PrefixSpeechLmScoringContext*>(other.get());
    if (otherPtr == nullptr) {
        return false;
    }

    return labelSeqEqual(labelSeq, otherPtr->labelSeq);
}

/*
 * =====================================
 * === PrefixSpeechLmOnnxLabelScorer ===
 * =====================================
 */

static const std::vector<Onnx::IOSpecification> initializerIoSpec = {
        Onnx::IOSpecification{
                "initial-prompt",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{-1, -1}, {1, -1}}},
        Onnx::IOSpecification{
                "initial-prompt-length",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{-1}, {1}}},
        Onnx::IOSpecification{
                "encoder-states",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -1, -2}, {1, -1, -2}}},
        Onnx::IOSpecification{
                "encoder-states-size",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{-1}, {1}}},
        Onnx::IOSpecification{
                "suffix-prompt",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{-1, -1}, {1, -1}}},
        Onnx::IOSpecification{
                "suffix-prompt-length",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{-1}, {1}}},
        Onnx::IOSpecification{
                "scores",
                Onnx::IODirection::OUTPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -2}, {1, -2}}}};

static const std::vector<Onnx::IOSpecification> stepIoSpec = {
        Onnx::IOSpecification{
                "token",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{-1, -1}, {-1, 1}}},
        Onnx::IOSpecification{
                "token-length",
                Onnx::IODirection::INPUT,
                true,
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
                "scores",
                Onnx::IODirection::OUTPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -1, -2}, {-1, 1, -2}}}};

const Core::ParameterIntVector PrefixSpeechLmOnnxLabelScorer::paramInitialPromptLabels(
        "initial-prompt-labels",
        "Token ids for the prompt fed into the LM before the input features.",
        "",
        0,
        Core::Type<s32>::max,
        0);

const Core::ParameterIntVector PrefixSpeechLmOnnxLabelScorer::paramSuffixPromptLabels(
        "suffix-prompt-labels",
        "Token ids for the prompt fed into the LM after the input features.",
        "",
        0,
        Core::Type<s32>::max,
        0);

const Core::ParameterBool PrefixSpeechLmOnnxLabelScorer::paramBlankUpdatesHistory(
        "blank-updates-history",
        "Whether previously emitted blank labels should be used to update the recurrent state.",
        false);

const Core::ParameterBool PrefixSpeechLmOnnxLabelScorer::paramSilenceUpdatesHistory(
        "silence-updates-history",
        "Whether previously emitted silence labels should be used to update the recurrent state.",
        false);

const Core::ParameterBool PrefixSpeechLmOnnxLabelScorer::paramLoopUpdatesHistory(
        "loop-updates-history",
        "Whether loop transitions should update the recurrent state.",
        false);

const Core::ParameterInt PrefixSpeechLmOnnxLabelScorer::paramMaxBatchSize(
        "max-batch-size",
        "Max number of scoring contexts that can be fed into the step ONNX model at once.",
        Core::Type<int>::max);

const Core::ParameterInt PrefixSpeechLmOnnxLabelScorer::paramMaxCachedScores(
        "max-cached-score-vectors",
        "Maximum size of cache that maps scoring contexts to scores and state slices. In multi-segment usage this applies to each segment separately.",
        10000);

PrefixSpeechLmOnnxLabelScorer::PrefixSpeechLmOnnxLabelScorer(Core::Configuration const& config, ModelCache& modelCache)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::LM),
          blankUpdatesHistory_(paramBlankUpdatesHistory(config)),
          silenceUpdatesHistory_(paramSilenceUpdatesHistory(config)),
          loopUpdatesHistory_(paramLoopUpdatesHistory(config)),
          maxBatchSize_(paramMaxBatchSize(config)),
          maxCachedScores_(paramMaxCachedScores(config)),
          stateManager_(Module::instance().createStateManager(select("state-manager"))),
          stateVectorFactory_(Module::instance().createCompressedVectorFactory(select("state-compression"))),
          segmentData_() {
    Core::Configuration initializerModelConfig(config, "initializer-model");
    Core::Configuration stepModelConfig(config, "step-model");

    auto initializerKey = initializerModelConfig.getSelection();
    auto stepKey        = stepModelConfig.getSelection();

    initializerOnnxModel_ = modelCache.getOrCreate<Onnx::Model>(initializerKey, initializerModelConfig, initializerIoSpec);
    stepOnnxModel_        = modelCache.getOrCreate<Onnx::Model>(stepKey, stepModelConfig, stepIoSpec);

    stateVariables_ = stepOnnxModel_->session.getStateVariablesMetadata();

    initializerInitialPromptName_       = initializerOnnxModel_->mapping.getOnnxName("initial-prompt");
    initializerInitialPromptLengthName_ = initializerOnnxModel_->mapping.getOnnxName("initial-prompt-length");
    initializerEncoderStatesName_       = initializerOnnxModel_->mapping.getOnnxName("encoder-states");
    initializerEncoderStatesSizeName_   = initializerOnnxModel_->mapping.getOnnxName("encoder-states-size");
    initializerSuffixPromptName_        = initializerOnnxModel_->mapping.getOnnxName("suffix-prompt");
    initializerSuffixPromptLengthName_  = initializerOnnxModel_->mapping.getOnnxName("suffix-prompt-length");
    initializerScoresName_              = initializerOnnxModel_->mapping.getOnnxName("scores");
    stepTokenName_                      = stepOnnxModel_->mapping.getOnnxName("token");
    stepTokenLengthName_                = stepOnnxModel_->mapping.getOnnxName("token-length");
    stepPrefixLengthName_               = stepOnnxModel_->mapping.getOnnxName("prefix-length");
    stepScoresName_                     = stepOnnxModel_->mapping.getOnnxName("scores");

    auto initialPromptLabels = paramInitialPromptLabels(config);
    auto suffixPromptLabels  = paramSuffixPromptLabels(config);
    initialPromptLabels_.insert(initialPromptLabels_.begin(), initialPromptLabels.begin(), initialPromptLabels.end());
    suffixPromptLabels_.insert(suffixPromptLabels_.begin(), suffixPromptLabels.begin(), suffixPromptLabels.end());
}

PrefixSpeechLmOnnxLabelScorer::SegmentData::SegmentData(size_t maxCachedScores)
        : encoderStatesValue(),
          encoderStatesSizeValue(),
          initialContext(),
          scoreCache(maxCachedScores),
          stateCache(maxCachedScores) {}

PrefixSpeechLmOnnxLabelScorer::SegmentData& PrefixSpeechLmOnnxLabelScorer::segmentData(SegmentId segment) {
    return segmentData_.try_emplace(segment, maxCachedScores_).first->second;
}

void PrefixSpeechLmOnnxLabelScorer::resetInternal() {
    Precursor::resetInternal();
    segmentData_.clear();
}

void PrefixSpeechLmOnnxLabelScorer::resetStatisticsInternal() {
    Precursor::resetStatisticsInternal();
    onnxSessionTime_.reset();
    stateMarshallingTime_.reset();
    contextPreparationTime_.reset();
}

void PrefixSpeechLmOnnxLabelScorer::addSegment(SegmentId segment) {
    Precursor::addSegment(segment);
    // Create the segment data here, so that it is never inserted while contexts of other segments are processed
    segmentData_.erase(segment);
    segmentData(segment);
}

void PrefixSpeechLmOnnxLabelScorer::removeSegment(SegmentId segment) {
    Precursor::removeSegment(segment);
    segmentData_.erase(segment);
}

void PrefixSpeechLmOnnxLabelScorer::addSegmentInput(SegmentId segment, DataView const& input) {
    Precursor::addSegmentInput(segment, input);
    auto& data                  = segmentData(segment);
    data.encoderStatesValue     = Onnx::Value();
    data.encoderStatesSizeValue = Onnx::Value();
    data.scoreCache.clear();
    data.stateCache.clear();
}

void PrefixSpeechLmOnnxLabelScorer::setInitialPrompt(std::vector<s32> const& prompt, SegmentId segment) {
    auto& data         = segmentData(segment);
    data.initialPrompt = prompt;
    // The prefill may have been computed with the previous prompt
    data.scoreCache.clear();
    data.stateCache.clear();
}

void PrefixSpeechLmOnnxLabelScorer::setSuffixPrompt(std::vector<s32> const& prompt, SegmentId segment) {
    auto& data        = segmentData(segment);
    data.suffixPrompt = prompt;
    // The prefill may have been computed with the previous prompt
    data.scoreCache.clear();
    data.stateCache.clear();
}

ScoringContextRef PrefixSpeechLmOnnxLabelScorer::getInitialSegmentScoringContext(SegmentId segment) {
    auto& data = segmentData(segment);
    if (not data.initialContext) {
        auto context        = Core::ref(new PrefixSpeechLmScoringContext());
        context->segment    = segment;
        data.initialContext = context;
    }
    return data.initialContext;
}

ScoringContextRef PrefixSpeechLmOnnxLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) {
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

    PrefixSpeechLmScoringContextRef context(dynamic_cast<PrefixSpeechLmScoringContext const*>(scoringContext.get()));
    verify(context);

    std::vector<LabelIndex> labelSeq(context->labelSeq);
    labelSeq.push_back(nextToken);
    auto newContext     = Core::ref(new PrefixSpeechLmScoringContext(std::move(labelSeq), context->historyLength + 1ul, context));
    newContext->segment = context->segment;
    return newContext;
}

std::vector<std::optional<ScoreAccessorRef>> PrefixSpeechLmOnnxLabelScorer::computeScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) {
    if (scoringContexts.empty()) {
        return {};
    }

    std::vector<std::optional<ScoreAccessorRef>> scoreAccessors(scoringContexts.size(), std::nullopt);

    // Only allow scoring once all encoder states of a segment have been passed
    auto segmentReady = [this](SegmentId segment) {
        return not expectMoreFeatures(segment) and bufferSize(segment) != 0ul;
    };

    // Run the initializer of all ready segments; it scores their empty emitted-label contexts
    std::unordered_set<SegmentId> initializedSegments;
    for (auto const& scoringContext : scoringContexts) {
        if (segmentReady(scoringContext->segment) and initializedSegments.insert(scoringContext->segment).second) {
            setupInitialStates(scoringContext->segment);
        }
    }

    /*
     * Identify unique scoring contexts that still need session runs
     */
    contextPreparationTime_.start();
    std::unordered_set<PrefixSpeechLmScoringContextRef, ScoringContextHash, ScoringContextEq> uniqueUncachedScoringContexts;
    for (auto const& scoringContext : scoringContexts) {
        PrefixSpeechLmScoringContextRef context(dynamic_cast<PrefixSpeechLmScoringContext const*>(scoringContext.get()));
        verify(context);
        if (not segmentReady(context->segment)) {
            continue;
        }

        auto& data = segmentData(context->segment);
        if (not context->state and data.stateCache.contains(context)) {
            context->state = *data.stateCache.get(context);
        }

        if (data.scoreCache.contains(context)) {
            continue;
        }

        verify(not context->labelSeq.empty());  // The empty emitted-label context is scored by setupInitialStates().
        uniqueUncachedScoringContexts.emplace(context);
    }
    contextPreparationTime_.stop();

    countUncomputedAccessors(scoringContexts.size() - uniqueUncachedScoringContexts.size());

    /*
     * Fill state and score caches for all uncached scoring contexts
     */
    std::vector<PrefixSpeechLmScoringContextRef> scoringContextBatch;
    scoringContextBatch.reserve(std::min(uniqueUncachedScoringContexts.size(), maxBatchSize_));
    for (auto const& scoringContext : uniqueUncachedScoringContexts) {
        scoringContextBatch.push_back(scoringContext);
        if (scoringContextBatch.size() == maxBatchSize_) {
            cacheStatesAndScores(scoringContextBatch);
            scoringContextBatch.clear();
        }
    }
    cacheStatesAndScores(scoringContextBatch);

    /*
     * Assign scores from cache to result vector
     */
    contextPreparationTime_.start();
    for (size_t i = 0ul; i < scoringContexts.size(); ++i) {
        PrefixSpeechLmScoringContextRef context(dynamic_cast<PrefixSpeechLmScoringContext const*>(scoringContexts[i].get()));
        if (not segmentReady(context->segment)) {
            continue;
        }
        auto scores = segmentData(context->segment).scoreCache.get(context);
        verify(scores);
        scoreAccessors[i] = Core::ref(new VectorScoreAccessor(scores->get(), context->labelSeq.size()));
    }
    contextPreparationTime_.stop();

    return scoreAccessors;
}

std::optional<ScoreAccessorRef> PrefixSpeechLmOnnxLabelScorer::computeScoreAccessor(ScoringContextRef scoringContext) {
    return computeScoreAccessors({scoringContext})[0];
}

size_t PrefixSpeechLmOnnxLabelScorer::getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const {
    return 0ul;
}

void PrefixSpeechLmOnnxLabelScorer::setupEncoderStatesValue(SegmentId segment) {
    auto& data = segmentData(segment);
    if (not data.encoderStatesValue.empty()) {
        return;
    }

    u32  T                    = bufferSize(segment);
    auto inputFeatureDataView = getInput(segment, 0);
    verify(inputFeatureDataView);

    data.encoderStatesValue = Onnx::Value::createEmpty<f32>({1l, static_cast<int64_t>(T), static_cast<int64_t>(inputFeatureDataView->size())});
    for (size_t t = 0ul; t < T; ++t) {
        inputFeatureDataView = getInput(segment, t);
        verify(inputFeatureDataView);
        std::copy(inputFeatureDataView->data(), inputFeatureDataView->data() + inputFeatureDataView->size(), data.encoderStatesValue.data<f32>(0, t));
    }
}

void PrefixSpeechLmOnnxLabelScorer::setupEncoderStatesSizeValue(SegmentId segment) {
    auto& data = segmentData(segment);
    if (not data.encoderStatesSizeValue.empty()) {
        return;
    }
    data.encoderStatesSizeValue = Onnx::Value::create(std::vector<s32>{static_cast<s32>(bufferSize(segment))});
}

void PrefixSpeechLmOnnxLabelScorer::setupInitialStates(SegmentId segment) {
    auto& data           = segmentData(segment);
    auto  initialContext = Core::ref(dynamic_cast<PrefixSpeechLmScoringContext const*>(getInitialSegmentScoringContext(segment).get()));
    if (data.scoreCache.contains(initialContext)) {
        return;
    }

    auto const& initialPrompt = data.initialPrompt ? *data.initialPrompt : initialPromptLabels_;
    auto const& suffixPrompt  = data.suffixPrompt ? *data.suffixPrompt : suffixPromptLabels_;

    size_t promptLength = initialPrompt.size() + bufferSize(segment) + suffixPrompt.size();
    if (promptLength == 0ul) {
        error("Prefix speech LLM prefill has empty prompt and no encoder states.");
    }

    std::vector<std::pair<std::string, Onnx::Value>> inputs;

    if (initializerInitialPromptName_ != "") {
        inputs.emplace_back(
                initializerInitialPromptName_,
                Onnx::Value::create(
                        initialPrompt.data(),
                        std::vector<s64>{1l, static_cast<s64>(initialPrompt.size())}));
        if (initializerInitialPromptLengthName_ != "") {
            inputs.emplace_back(
                    initializerInitialPromptLengthName_,
                    Onnx::Value::create(
                            std::vector<s32>{static_cast<s32>(initialPrompt.size())}));
        }
    }

    setupEncoderStatesValue(segment);
    inputs.emplace_back(initializerEncoderStatesName_, data.encoderStatesValue);
    if (initializerEncoderStatesSizeName_ != "") {
        setupEncoderStatesSizeValue(segment);
        inputs.emplace_back(initializerEncoderStatesSizeName_, data.encoderStatesSizeValue);
    }

    if (initializerSuffixPromptName_ != "") {
        inputs.emplace_back(
                initializerSuffixPromptName_,
                Onnx::Value::create(
                        suffixPrompt.data(),
                        std::vector<s64>{1l, static_cast<s64>(suffixPrompt.size())}));
        if (initializerSuffixPromptLengthName_ != "") {
            inputs.emplace_back(
                    initializerSuffixPromptLengthName_,
                    Onnx::Value::create(
                            std::vector<s32>{static_cast<s32>(suffixPrompt.size())}));
        }
    }

    std::vector<std::string> targets;
    targets.reserve(stateVariables_.size() + 1ul);
    targets.emplace_back(initializerScoresName_);
    for (auto const& stateVariable : stateVariables_) {
        targets.emplace_back(stateVariable.output_state_key);
    }

    std::vector<Onnx::Value> outputs;
    onnxSessionTime_.start();
    initializerOnnxModel_->session.run(std::move(inputs), targets, outputs);
    onnxSessionTime_.stop();

    verify(not stateVariables_.empty());
    verify_ge(outputs.size(), 2ul);
    size_t stateLength = 0ul;
    // Find time dim and take its size as prompt length
    for (size_t d = 1ul; d < stateVariables_.front().shape.size(); ++d) {
        if (stateVariables_.front().shape[d] < 0l) {
            stateLength = outputs[1].dimSize(d);
            break;
        }
    }
    verify_gt(stateLength, 0ul);

    auto scores = std::make_shared<std::vector<Score>>();
    outputs.front().get(0, *scores);

    std::vector<Onnx::Value> stateOutputs(std::make_move_iterator(outputs.begin() + 1), std::make_move_iterator(outputs.end()));
    std::vector<size_t>      suffixLengths{stateLength};
    stateMarshallingTime_.start();
    auto promptStates = stateManager_->splitStates(stateVariables_, suffixLengths, stateOutputs, *stateVectorFactory_);
    stateMarshallingTime_.stop();

    PrefixSpeechLmScoringContextRef parent;
    if (stateManager_->requiresAllParentStates()) {
        verify_eq(promptStates.size(), stateLength);
        for (size_t i = 0ul; i + 1ul < promptStates.size(); ++i) {
            auto state       = std::make_shared<HistoryState>(std::move(promptStates[i]));
            auto context     = Core::ref(new PrefixSpeechLmScoringContext(std::vector<LabelIndex>(), i + 1ul, parent, state));
            context->segment = segment;
            parent           = context;
        }
    }
    else {
        verify_eq(promptStates.size(), 1ul);
    }

    initialContext->historyLength = stateLength;
    initialContext->parent        = parent;
    initialContext->state         = std::make_shared<HistoryState>(std::move(promptStates.back()));

    data.scoreCache.put(initialContext, scores);
    data.stateCache.put(initialContext, initialContext->state);
}

void PrefixSpeechLmOnnxLabelScorer::cacheStatesAndScores(std::vector<PrefixSpeechLmScoringContextRef> const& scoringContextBatch) {
    if (scoringContextBatch.empty()) {
        return;
    }

    std::vector<size_t>              prefixLengths;
    std::vector<HistoryState const*> prefixStates;
    prefixLengths.reserve(scoringContextBatch.size());
    prefixStates.reserve(scoringContextBatch.size());

    for (auto const& context : scoringContextBatch) {
        verify(not context->labelSeq.empty());

        auto parent = context->parent;
        verify(parent);

        if (stateManager_->requiresAllParentStates()) {
            size_t prefixLength = parent->historyLength;
            prefixLengths.push_back(prefixLength);
            prefixStates.resize(prefixStates.size() + prefixLength);

            size_t offset = prefixStates.size() - prefixLength;
            for (size_t i = 0ul; i < prefixLength; ++i) {
                verify(parent);
                verify(parent->state);
                prefixStates[offset + prefixLength - i - 1] = parent->state.get();
                parent                                      = parent->parent;
            }
        }
        else {
            prefixLengths.push_back(parent->historyLength);
            verify(parent->state);
            prefixStates.push_back(parent->state.get());
        }
    }

    std::vector<std::pair<std::string, Onnx::Value>> inputs;
    std::vector<std::string>                         targets;
    stateMarshallingTime_.start();
    stateManager_->mergeStates(stateVariables_, prefixLengths, prefixStates, inputs, targets);
    stateMarshallingTime_.stop();

    Onnx::Value tokens = Onnx::Value::createEmpty<s32>({static_cast<s64>(scoringContextBatch.size()), 1l});
    for (size_t i = 0ul; i < scoringContextBatch.size(); ++i) {
        verify(not scoringContextBatch[i]->labelSeq.empty());
        tokens.data<s32>(i, 0)[0] = static_cast<s32>(scoringContextBatch[i]->labelSeq.back());
    }
    inputs.emplace_back(stepTokenName_, std::move(tokens));

    if (stepTokenLengthName_ != "") {
        inputs.emplace_back(stepTokenLengthName_, Onnx::Value::create(std::vector<s32>(scoringContextBatch.size(), 1)));
    }
    if (stepPrefixLengthName_ != "") {
        std::vector<s32> stateLengths(prefixLengths.begin(), prefixLengths.end());
        inputs.emplace_back(stepPrefixLengthName_, Onnx::Value::create(stateLengths));
    }

    targets.emplace(targets.begin(), stepScoresName_);

    std::vector<Onnx::Value> outputs;
    onnxSessionTime_.start();
    stepOnnxModel_->session.run(std::move(inputs), targets, outputs);
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

void PrefixSpeechLmOnnxLabelScorer::logScoringBreakdown() const {
    statisticsChannel_ << Core::XmlFull("onnx-session-time", onnxSessionTime_.elapsedMilliseconds());
    statisticsChannel_ << Core::XmlFull("state-marshalling-time", stateMarshallingTime_.elapsedMilliseconds());
    statisticsChannel_ << Core::XmlFull("context-preparation-time", contextPreparationTime_.elapsedMilliseconds());
}

}  // namespace Nn
