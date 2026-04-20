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

#include <Nn/Module.hh>

#include "ScoreAccessor.hh"

namespace Nn {

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
                "scores",
                Onnx::IODirection::OUTPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -2}, {-1, 1, -2}, {-1, -1, -2}}}};

/*
 * ==================================
 * == StateManagedOnnxLabelScorer ===
 * ==================================
 */

const Core::ParameterInt StateManagedOnnxLabelScorer::paramStartLabelIndex(
        "start-label-index",
        "Initial recurrent update token used to obtain the first score distribution.",
        0);

const Core::ParameterBool StateManagedOnnxLabelScorer::paramBlankUpdatesHistory(
        "blank-updates-history",
        "Whether previously emitted blank labels should be used to update the recurrent state.",
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
        "Maximum size of cache that maps scoring contexts to scores and state slices. This prevents memory overflow in case of very long audio segments.",
        1000);

StateManagedOnnxLabelScorer::StateManagedOnnxLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::LM),
          startLabelIndex_(paramStartLabelIndex(config)),
          blankUpdatesHistory_(paramBlankUpdatesHistory(config)),
          loopUpdatesHistory_(paramLoopUpdatesHistory(config)),
          maxBatchSize_(paramMaxBatchSize(config)),
          onnxModel_(select("onnx-model"), ioSpec),
          stateManager_(Module::instance().createStateManager(select("state-manager"))),
          stateVariables_(onnxModel_.session.getStateVariablesMetadata()),
          stateVectorFactory_(Module::instance().createCompressedVectorFactory(select("state-compression"))),
          tokenName_(onnxModel_.mapping.getOnnxName("token")),
          tokenLengthName_(onnxModel_.mapping.getOnnxName("token-length")),
          prefixLengthName_(onnxModel_.mapping.getOnnxName("prefix-length")),
          scoresName_(onnxModel_.mapping.getOnnxName("scores")),
          encoderStatesName_(onnxModel_.mapping.getOnnxName("encoder-states")),
          encoderStatesSizeName_(onnxModel_.mapping.getOnnxName("encoder-states-size")),
          encoderStatesValue_(),
          encoderStatesSizeValue_(),
          scoreCache_(paramMaxCachedScores(config)),
          stateCache_(scoreCache_.maxSize()) {
}

void StateManagedOnnxLabelScorer::reset() {
    Precursor::reset();
    encoderStatesValue_     = Onnx::Value();
    encoderStatesSizeValue_ = Onnx::Value();
    scoreCache_.clear();
    stateCache_.clear();
}

void StateManagedOnnxLabelScorer::addInput(DataView const& input) {
    Precursor::addInput(input);
    encoderStatesValue_     = Onnx::Value();
    encoderStatesSizeValue_ = Onnx::Value();
}

ScoringContextRef StateManagedOnnxLabelScorer::getInitialScoringContext() {
    auto rootState = stateManager_->initialState(stateVariables_, *stateVectorFactory_);
    if (rootState.empty()) {
        error("Initial recurrent state is empty.");
    }

    StateManagedOnnxScoringContextRef root = Core::ref(new StateManagedOnnxScoringContext(std::move(rootState)));

    std::vector<LabelIndex> labelSeq{static_cast<LabelIndex>(startLabelIndex_)};
    return Core::ref(new StateManagedOnnxScoringContext(std::move(labelSeq), root));
}

ScoringContextRef StateManagedOnnxLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) {
    bool updateState = false;
    switch (transitionType) {
        case TransitionType::BLANK_LOOP:
            updateState = blankUpdatesHistory_ and loopUpdatesHistory_;
            break;
        case TransitionType::LABEL_TO_BLANK:
        case TransitionType::INITIAL_BLANK:
            updateState = blankUpdatesHistory_;
            break;
        case TransitionType::LABEL_LOOP:
            updateState = loopUpdatesHistory_;
            break;
        case TransitionType::BLANK_TO_LABEL:
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
    return Core::ref(new StateManagedOnnxScoringContext(std::move(labelSeq), context));
}

std::vector<std::optional<ScoreAccessorRef>> StateManagedOnnxLabelScorer::getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) {
    if (scoringContexts.empty()) {
        return {};
    }

    std::vector<std::optional<ScoreAccessorRef>> scoreAccessors(scoringContexts.size(), std::nullopt);
    if ((encoderStatesName_ != "" or encoderStatesSizeName_ != "") and (expectMoreFeatures_ or bufferSize() == 0)) {
        return scoreAccessors;
    }

    /*
     * Identify unique scoring contexts that still need session runs
     */
    std::unordered_set<StateManagedOnnxScoringContextRef, ScoringContextHash, ScoringContextEq> uniqueUncachedScoringContexts;
    for (auto const& scoringContext : scoringContexts) {
        StateManagedOnnxScoringContextRef context(dynamic_cast<StateManagedOnnxScoringContext const*>(scoringContext.get()));
        verify(context);
        if (not context->state and stateCache_.contains(context)) {
            context->state = *stateCache_.get(context);
        }
        else {
            uniqueUncachedScoringContexts.emplace(context);
        }
    }

    /*
     * Fill state and score caches for all uncached scoring contexts
     */
    std::vector<StateManagedOnnxScoringContextRef> scoringContextBatch;
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
    for (size_t i = 0ul; i < scoringContexts.size(); ++i) {
        StateManagedOnnxScoringContextRef context(dynamic_cast<StateManagedOnnxScoringContext const*>(scoringContexts[i].get()));
        auto                              scores = scoreCache_.get(context);
        verify(scores);
        scoreAccessors[i] = Core::ref(new VectorScoreAccessor(scores->get(), context->labelSeq.empty() ? 0ul : context->labelSeq.size() - 1ul));
    }

    return scoreAccessors;
}

std::optional<ScoreAccessorRef> StateManagedOnnxLabelScorer::getScoreAccessor(ScoringContextRef scoringContext) {
    return getScoreAccessors({scoringContext})[0];
}

size_t StateManagedOnnxLabelScorer::getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const {
    return 0ul;
}

void StateManagedOnnxLabelScorer::setupEncoderStatesValue() {
    if (not encoderStatesValue_.empty()) {
        return;
    }

    u32  T                    = bufferSize();
    auto inputFeatureDataView = getInput(0);
    verify(inputFeatureDataView);

    encoderStatesValue_ = Onnx::Value::createEmpty<f32>({1l, static_cast<int64_t>(T), static_cast<int64_t>(inputFeatureDataView->size())});
    for (size_t t = 0ul; t < T; ++t) {
        inputFeatureDataView = getInput(t);
        verify(inputFeatureDataView);
        std::copy(inputFeatureDataView->data(), inputFeatureDataView->data() + inputFeatureDataView->size(), encoderStatesValue_.data<f32>(0, t));
    }
}

void StateManagedOnnxLabelScorer::setupEncoderStatesSizeValue() {
    if (not encoderStatesSizeValue_.empty()) {
        return;
    }
    encoderStatesSizeValue_ = Onnx::Value::create(std::vector<s32>{static_cast<s32>(bufferSize())});
}

void StateManagedOnnxLabelScorer::cacheStatesAndScores(std::vector<StateManagedOnnxScoringContextRef> const& scoringContextBatch) {
    if (scoringContextBatch.empty()) {
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
    stateManager_->mergeStates(stateVariables_, prefixLengths, prefixStates, inputs, targets);

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
    if (encoderStatesName_ != "") {
        setupEncoderStatesValue();
        inputs.emplace_back(encoderStatesName_, encoderStatesValue_);
    }
    if (encoderStatesSizeName_ != "") {
        setupEncoderStatesSizeValue();
        inputs.emplace_back(encoderStatesSizeName_, encoderStatesSizeValue_);
    }

    targets.emplace(targets.begin(), scoresName_);

    std::vector<Onnx::Value> outputs;
    onnxModel_.session.run(std::move(inputs), targets, outputs);

    for (size_t b = 0ul; b < scoringContextBatch.size(); ++b) {
        auto scores = std::make_shared<std::vector<Score>>();
        if (outputs.front().numDims() == 3) {
            outputs.front().get(b, 0, *scores);
        }
        else {
            outputs.front().get(b, *scores);
        }
        scoreCache_.put(scoringContextBatch[b], scores);
    }

    std::vector<Onnx::Value> stateOutputs(std::make_move_iterator(outputs.begin() + 1), std::make_move_iterator(outputs.end()));
    std::vector<size_t>      suffixLengths(scoringContextBatch.size(), 1ul);
    auto                     splitStates = stateManager_->splitStates(stateVariables_, suffixLengths, stateOutputs, *stateVectorFactory_);
    verify_eq(splitStates.size(), scoringContextBatch.size());
    for (size_t i = 0ul; i < scoringContextBatch.size(); ++i) {
        scoringContextBatch[i]->state = std::make_shared<HistoryState>(std::move(splitStates[i]));
        stateCache_.put(scoringContextBatch[i], scoringContextBatch[i]->state);
        if (not stateManager_->requiresAllParentStates()) {
            scoringContextBatch[i]->parent.reset();
        }
    }
}

}  // namespace Nn
