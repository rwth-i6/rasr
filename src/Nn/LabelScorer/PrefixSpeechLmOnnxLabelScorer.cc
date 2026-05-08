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

#include <Nn/Module.hh>
#include <algorithm>
#include <unordered_set>

#include "ScoreAccessor.hh"

namespace Nn {

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
                {{-1, -1}}},
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
                {{-1, -1, -2}}}};

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
        "Maximum size of cache that maps scoring contexts to scores and state slices.",
        10000);

PrefixSpeechLmOnnxLabelScorer::PrefixSpeechLmOnnxLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::LM),
          blankUpdatesHistory_(paramBlankUpdatesHistory(config)),
          loopUpdatesHistory_(paramLoopUpdatesHistory(config)),
          maxBatchSize_(paramMaxBatchSize(config)),
          initializerOnnxModel_(select("initializer-model"), initializerIoSpec),
          stepOnnxModel_(select("step-model"), stepIoSpec),
          stateManager_(Module::instance().createStateManager(select("state-manager"))),
          stateVariables_(stepOnnxModel_.session.getStateVariablesMetadata()),
          stateVectorFactory_(Module::instance().createCompressedVectorFactory(select("state-compression"))),
          initializerInitialPromptName_(initializerOnnxModel_.mapping.getOnnxName("initial-prompt")),
          initializerInitialPromptLengthName_(initializerOnnxModel_.mapping.getOnnxName("initial-prompt-length")),
          initializerEncoderStatesName_(initializerOnnxModel_.mapping.getOnnxName("encoder-states")),
          initializerEncoderStatesSizeName_(initializerOnnxModel_.mapping.getOnnxName("encoder-states-size")),
          initializerSuffixPromptName_(initializerOnnxModel_.mapping.getOnnxName("suffix-prompt")),
          initializerSuffixPromptLengthName_(initializerOnnxModel_.mapping.getOnnxName("suffix-prompt-length")),
          initializerScoresName_(initializerOnnxModel_.mapping.getOnnxName("scores")),
          stepTokenName_(stepOnnxModel_.mapping.getOnnxName("token")),
          stepTokenLengthName_(stepOnnxModel_.mapping.getOnnxName("token-length")),
          stepPrefixLengthName_(stepOnnxModel_.mapping.getOnnxName("prefix-length")),
          stepScoresName_(stepOnnxModel_.mapping.getOnnxName("scores")),
          encoderStatesValue_(),
          encoderStatesSizeValue_(),
          initialContext_(),
          scoreCache_(paramMaxCachedScores(config)),
          stateCache_(scoreCache_.maxSize()) {
    auto initialPromptLabels = paramInitialPromptLabels(config);
    auto suffixPromptLabels  = paramSuffixPromptLabels(config);
    initialPromptLabels_.insert(initialPromptLabels_.begin(), initialPromptLabels.begin(), initialPromptLabels.end());
    suffixPromptLabels_.insert(suffixPromptLabels_.begin(), suffixPromptLabels.begin(), suffixPromptLabels.end());
}

void PrefixSpeechLmOnnxLabelScorer::reset() {
    Precursor::reset();
    encoderStatesValue_     = Onnx::Value();
    encoderStatesSizeValue_ = Onnx::Value();
    initialContext_         = PrefixSpeechLmScoringContextRef();
    scoreCache_.clear();
    stateCache_.clear();
}

void PrefixSpeechLmOnnxLabelScorer::addInput(DataView const& input) {
    Precursor::addInput(input);
    encoderStatesValue_     = Onnx::Value();
    encoderStatesSizeValue_ = Onnx::Value();
    scoreCache_.clear();
    stateCache_.clear();
}

ScoringContextRef PrefixSpeechLmOnnxLabelScorer::getInitialScoringContext() {
    if (not initialContext_) {
        initialContext_ = Core::ref(new PrefixSpeechLmScoringContext());
    }
    return initialContext_;
}

ScoringContextRef PrefixSpeechLmOnnxLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) {
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

    PrefixSpeechLmScoringContextRef context(dynamic_cast<PrefixSpeechLmScoringContext const*>(scoringContext.get()));
    verify(context);

    std::vector<LabelIndex> labelSeq(context->labelSeq);
    labelSeq.push_back(nextToken);
    return Core::ref(new PrefixSpeechLmScoringContext(std::move(labelSeq), context->historyLength + 1ul, context));
}

std::vector<std::optional<ScoreAccessorRef>> PrefixSpeechLmOnnxLabelScorer::getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) {
    if (scoringContexts.empty()) {
        return {};
    }

    std::vector<std::optional<ScoreAccessorRef>> scoreAccessors(scoringContexts.size(), std::nullopt);
    if (expectMoreFeatures_ or bufferSize() == 0ul) {
        return scoreAccessors;
    }

    setupInitialStates();

    std::unordered_set<PrefixSpeechLmScoringContextRef, ScoringContextHash, ScoringContextEq> uniqueUncachedScoringContexts;
    for (auto const& scoringContext : scoringContexts) {
        PrefixSpeechLmScoringContextRef context(dynamic_cast<PrefixSpeechLmScoringContext const*>(scoringContext.get()));
        verify(context);

        if (not context->state and stateCache_.contains(context)) {
            context->state = *stateCache_.get(context);
        }

        if (scoreCache_.contains(context)) {
            continue;
        }

        verify(not context->labelSeq.empty());  // The empty emitted-label context is scored by setupInitialStates().
        uniqueUncachedScoringContexts.emplace(context);
    }

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

    for (size_t i = 0ul; i < scoringContexts.size(); ++i) {
        PrefixSpeechLmScoringContextRef context(dynamic_cast<PrefixSpeechLmScoringContext const*>(scoringContexts[i].get()));
        auto                            scores = scoreCache_.get(context);
        verify(scores);
        scoreAccessors[i] = Core::ref(new VectorScoreAccessor(scores->get(), context->labelSeq.size()));
    }

    return scoreAccessors;
}

std::optional<ScoreAccessorRef> PrefixSpeechLmOnnxLabelScorer::getScoreAccessor(ScoringContextRef scoringContext) {
    return getScoreAccessors({scoringContext})[0];
}

size_t PrefixSpeechLmOnnxLabelScorer::getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const {
    return 0ul;
}

void PrefixSpeechLmOnnxLabelScorer::setupEncoderStatesValue() {
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

void PrefixSpeechLmOnnxLabelScorer::setupEncoderStatesSizeValue() {
    if (not encoderStatesSizeValue_.empty()) {
        return;
    }
    encoderStatesSizeValue_ = Onnx::Value::create(std::vector<s32>{static_cast<s32>(bufferSize())});
}

void PrefixSpeechLmOnnxLabelScorer::setupInitialStates() {
    if (not initialContext_) {
        getInitialScoringContext();
    }
    if (scoreCache_.contains(initialContext_)) {
        return;
    }

    size_t promptLength = initialPromptLabels_.size() + bufferSize() + suffixPromptLabels_.size();
    if (promptLength == 0ul) {
        error("Prefix speech LLM prefill has empty prompt and no encoder states.");
    }

    std::vector<std::pair<std::string, Onnx::Value>> inputs;

    if (initializerInitialPromptName_ != "") {
        inputs.emplace_back(
                initializerInitialPromptName_,
                Onnx::Value::create(
                        initialPromptLabels_.data(),
                        std::vector<s64>{1l, static_cast<s64>(initialPromptLabels_.size())}));
        if (initializerInitialPromptLengthName_ != "") {
            inputs.emplace_back(
                    initializerInitialPromptLengthName_,
                    Onnx::Value::create(
                            std::vector<s32>{static_cast<s32>(initialPromptLabels_.size())}));
        }
    }

    setupEncoderStatesValue();
    inputs.emplace_back(initializerEncoderStatesName_, encoderStatesValue_);
    if (initializerEncoderStatesSizeName_ != "") {
        setupEncoderStatesSizeValue();
        inputs.emplace_back(initializerEncoderStatesSizeName_, encoderStatesSizeValue_);
    }

    if (initializerSuffixPromptName_ != "") {
        inputs.emplace_back(
                initializerSuffixPromptName_,
                Onnx::Value::create(
                        suffixPromptLabels_.data(),
                        std::vector<s64>{1l, static_cast<s64>(suffixPromptLabels_.size())}));
        if (initializerSuffixPromptLengthName_ != "") {
            inputs.emplace_back(
                    initializerSuffixPromptLengthName_,
                    Onnx::Value::create(
                            std::vector<s32>{static_cast<s32>(suffixPromptLabels_.size())}));
        }
    }

    std::vector<std::string> targets;
    targets.reserve(stateVariables_.size() + 1ul);
    targets.emplace_back(initializerScoresName_);
    for (auto const& stateVariable : stateVariables_) {
        targets.emplace_back(stateVariable.output_state_key);
    }

    std::vector<Onnx::Value> outputs;
    initializerOnnxModel_.session.run(std::move(inputs), targets, outputs);

    auto scores = std::make_shared<std::vector<Score>>();
    outputs.front().get(0, *scores);

    std::vector<Onnx::Value> stateOutputs(std::make_move_iterator(outputs.begin() + 1), std::make_move_iterator(outputs.end()));
    std::vector<size_t>      suffixLengths{promptLength};
    auto                     promptStates = stateManager_->splitStates(stateVariables_, suffixLengths, stateOutputs, *stateVectorFactory_);

    PrefixSpeechLmScoringContextRef parent;
    if (stateManager_->requiresAllParentStates()) {
        verify_eq(promptStates.size(), promptLength);
        for (size_t i = 0ul; i + 1ul < promptStates.size(); ++i) {
            auto state = std::make_shared<HistoryState>(std::move(promptStates[i]));
            parent     = Core::ref(new PrefixSpeechLmScoringContext(std::vector<LabelIndex>(), i + 1ul, parent, state));
        }
    }
    else {
        verify_eq(promptStates.size(), 1ul);
    }

    initialContext_->historyLength = promptLength;
    initialContext_->parent        = parent;
    initialContext_->state         = std::make_shared<HistoryState>(std::move(promptStates.back()));

    scoreCache_.put(initialContext_, scores);
    stateCache_.put(initialContext_, initialContext_->state);
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
    stateManager_->mergeStates(stateVariables_, prefixLengths, prefixStates, inputs, targets);

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
    stepOnnxModel_.session.run(std::move(inputs), targets, outputs);

    for (size_t b = 0ul; b < scoringContextBatch.size(); ++b) {
        auto scores = std::make_shared<std::vector<Score>>();
        outputs.front().get(b, 0, *scores);
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
