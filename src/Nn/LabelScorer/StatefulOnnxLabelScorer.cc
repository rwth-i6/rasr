/** Copyright 2025 RWTH Aachen University. All rights reserved.
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

#include "StatefulOnnxLabelScorer.hh"

#include <algorithm>
#include <cstddef>
#include <utility>

#include <Core/Assertions.hh>
#include <Core/ReferenceCounting.hh>
#include <Flow/Timestamp.hh>
#include <Math/FastMatrix.hh>
#include <Mm/Module.hh>
#include <Onnx/OnnxStateVariable.hh>
#include <Onnx/Value.hh>
#include <Speech/Types.hh>

#include "ScoreAccessor.hh"
#include "ScoringContext.hh"

namespace Nn {

OnnxHiddenStateScoringContext::OnnxHiddenStateScoringContext()
        : labelSeq(), hiddenState(), requiresFinalize(false) {}

OnnxHiddenStateScoringContext::OnnxHiddenStateScoringContext(std::vector<LabelIndex> const& labelSeq, OnnxHiddenStateRef state, bool requiresFinalize)
        : labelSeq(labelSeq), hiddenState(state), requiresFinalize(requiresFinalize) {}

bool OnnxHiddenStateScoringContext::isEqual(ScoringContextRef const& other) const {
    auto* otherPtr = dynamic_cast<OnnxHiddenStateScoringContext const*>(other.get());
    if (otherPtr == nullptr) {
        return false;
    }

    return labelSeqEqual(labelSeq, otherPtr->labelSeq);
}

size_t OnnxHiddenStateScoringContext::hash() const {
    return labelSeqHash(labelSeq);
}

typedef Core::Ref<OnnxHiddenStateScoringContext const> OnnxHiddenStateScoringContextRef;

/*
 * =============================
 * == StatefulOnnxLabelScorer ==
 * =============================
 */

const Core::ParameterBool StatefulOnnxLabelScorer::paramBlankUpdatesHistory(
        "blank-updates-history",
        "Whether previously emitted blank labels should be used to update the history.",
        false);

const Core::ParameterBool StatefulOnnxLabelScorer::paramSilenceUpdatesHistory(
        "silence-updates-history",
        "Whether previously emitted silence labels should be used to update the history.",
        false);

const Core::ParameterBool StatefulOnnxLabelScorer::paramLoopUpdatesHistory(
        "loop-updates-history",
        "Whether in the case of loop transitions every repeated emission should be used to update the history.",
        false);

const Core::ParameterInt StatefulOnnxLabelScorer::paramMaxBatchSize(
        "max-batch-size",
        "Max number of hidden-states that can be fed into the scorer ONNX model at once.",
        Core::Type<int>::max);

const Core::ParameterInt StatefulOnnxLabelScorer::paramMaxCachedScores(
        "max-cached-score-vectors",
        "Maximum size of cache that maps scoring contexts to scores. This prevents memory overflow in case of very long audio segments.",
        10000);

// The hidden states and the scores output are not part of the IO specs; they are handled by `OnnxHiddenStateModel`.
// The scorer takes nothing but hidden states, so it has no IO spec of its own.
static const std::vector<Onnx::IOSpecification> stateInitializerModelIoSpec = {
        Onnx::IOSpecification{
                "encoder-states",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{1, -1, -2}, {-1, -1, -2}}},  // [1, T, E] or [B, T, E]
        Onnx::IOSpecification{
                "encoder-states-size",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{1}, {-1}}}};  // [1] or [B]

static const std::vector<Onnx::IOSpecification> stateUpdaterModelIoSpec = {
        Onnx::IOSpecification{
                "encoder-states",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{1, -1, -2}, {-1, -1, -2}}},  // [1, T, E] or [B, T, E]
        Onnx::IOSpecification{
                "encoder-states-size",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{1}, {-1}}},  // [1] or [B]
        Onnx::IOSpecification{
                "token",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{1}, {-1}}}};  // [1] or [B]

StatefulOnnxLabelScorer::StatefulOnnxLabelScorer(Core::Configuration const& config, ModelCache& modelCache)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::LM),
          blankUpdatesHistory_(paramBlankUpdatesHistory(config)),
          silenceUpdatesHistory_(paramSilenceUpdatesHistory(config)),
          loopUpdatesHistory_(paramLoopUpdatesHistory(config)),
          maxBatchSize_(paramMaxBatchSize(config)),
          hiddenStateModel_(config, modelCache, stateInitializerModelIoSpec, stateUpdaterModelIoSpec, {}),
          initialHiddenState_(),
          encoderStatesValue_(),
          encoderStatesSizeValue_(),
          scoreCache_(paramMaxCachedScores(config)),
          stateCache_(scoreCache_.maxSize()) {
    initializerEncoderStatesName_     = hiddenStateModel_.stateInitializerOnnxName("encoder-states");
    initializerEncoderStatesSizeName_ = hiddenStateModel_.stateInitializerOnnxName("encoder-states-size");
    updaterEncoderStatesName_         = hiddenStateModel_.stateUpdaterOnnxName("encoder-states");
    updaterEncoderStatesSizeName_     = hiddenStateModel_.stateUpdaterOnnxName("encoder-states-size");
    updaterTokenName_                 = hiddenStateModel_.stateUpdaterOnnxName("token");
}

void StatefulOnnxLabelScorer::reset() {
    Precursor::reset();
    stateCache_.clear();
    scoreCache_.clear();
}

ScoringContextRef StatefulOnnxLabelScorer::getInitialScoringContext() {
    return Core::ref(new OnnxHiddenStateScoringContext());
}

void StatefulOnnxLabelScorer::addInput(DataView const& input) {
    Precursor::addInput(input);

    initialHiddenState_ = OnnxHiddenStateRef();

    if (not encoderStatesValue_.empty()) {  // Any previously computed hidden state values are outdated now so reset them
        encoderStatesValue_     = Onnx::Value();
        encoderStatesSizeValue_ = Onnx::Value();
    }
}

size_t StatefulOnnxLabelScorer::getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const {
    return 0u;
}

ScoringContextRef StatefulOnnxLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) {
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

    // If scoring context is not going to be modified, return the original one
    if (not updateState) {
        return scoringContext;
    }

    OnnxHiddenStateScoringContextRef onnxHiddenStateScoringContext(dynamic_cast<OnnxHiddenStateScoringContext const*>(scoringContext.get()));
    std::vector<LabelIndex>          newLabelSeq(onnxHiddenStateScoringContext->labelSeq);
    newLabelSeq.push_back(nextToken);

    // Re-use previous hidden-state but mark that finalization (i.e. hidden-state update) is required
    auto newScoringContext = Core::ref(new OnnxHiddenStateScoringContext(std::move(newLabelSeq), onnxHiddenStateScoringContext->hiddenState, true));

    auto hiddenState = stateCache_.get(newScoringContext);
    if (hiddenState) {
        newScoringContext->hiddenState      = *hiddenState;
        newScoringContext->requiresFinalize = false;
    }

    return newScoringContext;
}

std::vector<std::optional<ScoreAccessorRef>> StatefulOnnxLabelScorer::getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) {
    if (scoringContexts.empty()) {
        return {};
    }

    std::vector<std::optional<ScoreAccessorRef>> scoreAccessors(scoringContexts.size(), std::nullopt);

    if ((initializerEncoderStatesName_ != "" or initializerEncoderStatesSizeName_ != "" or updaterEncoderStatesName_ != "" or updaterEncoderStatesSizeName_ != "") and (expectMoreFeatures_ or bufferSize() == 0)) {
        // Only allow scoring once all encoder states have been passed
        return scoreAccessors;
    }

    /*
     * Identify unique scoring contexts that still need session runs
     */
    std::unordered_set<OnnxHiddenStateScoringContextRef, ScoringContextHash, ScoringContextEq> uniqueUncachedScoringContexts;

    for (auto const& scoringContext : scoringContexts) {
        // We need to finalize all scoring contexts before using them for scoring again.

        OnnxHiddenStateScoringContextRef onnxHiddenStateScoringContext(dynamic_cast<OnnxHiddenStateScoringContext const*>(scoringContext.get()));
        if (not scoreCache_.contains(onnxHiddenStateScoringContext)) {
            // Group by unique scoring context
            uniqueUncachedScoringContexts.emplace(onnxHiddenStateScoringContext);
        }
    }

    /*
     * Fill state and score caches for all uncached scoring contexts
     */
    std::vector<OnnxHiddenStateScoringContextRef> scoringContextBatch;
    scoringContextBatch.reserve(std::min(uniqueUncachedScoringContexts.size(), maxBatchSize_));
    for (auto scoringContext : uniqueUncachedScoringContexts) {
        scoringContextBatch.push_back(scoringContext);
        if (scoringContextBatch.size() == maxBatchSize_) {  // Batch is full -> forward now
            cacheStates(scoringContextBatch);
            cacheScores(scoringContextBatch);
            scoringContextBatch.clear();
        }
    }

    // Forward remaining scoring contexts
    cacheStates(scoringContextBatch);
    cacheScores(scoringContextBatch);

    /*
     * Assign states from cache to scoring contexts and scores from cache to result vector
     */
    for (size_t contextIndex = 0ul; contextIndex < scoringContexts.size(); ++contextIndex) {
        OnnxHiddenStateScoringContextRef onnxHiddenStateScoringContext(dynamic_cast<OnnxHiddenStateScoringContext const*>(scoringContexts[contextIndex].get()));

        if (onnxHiddenStateScoringContext->requiresFinalize) {
            auto hiddenState = stateCache_.get(onnxHiddenStateScoringContext);
            verify(hiddenState);
            onnxHiddenStateScoringContext->hiddenState      = *hiddenState;
            onnxHiddenStateScoringContext->requiresFinalize = false;
        }

        verify(scoreCache_.contains(onnxHiddenStateScoringContext));
        auto const& scoreVec         = scoreCache_.get(onnxHiddenStateScoringContext)->get();
        auto const  timeframe        = onnxHiddenStateScoringContext->labelSeq.size();
        scoreAccessors[contextIndex] = Core::ref(new VectorScoreAccessor(scoreVec, timeframe));
    }

    return scoreAccessors;
}

std::optional<ScoreAccessorRef> StatefulOnnxLabelScorer::getScoreAccessor(ScoringContextRef scoringContext) {
    return getScoreAccessors({scoringContext})[0];
}

void StatefulOnnxLabelScorer::setupEncoderStatesValue() {
    if (not encoderStatesValue_.empty()) {
        return;
    }

    u32  T                    = bufferSize();
    auto inputFeatureDataView = getInput(0);

    encoderStatesValue_ = Onnx::Value::createEmpty<f32>({1l, static_cast<int64_t>(T), static_cast<int64_t>(inputFeatureDataView->size())});

    for (size_t t = 0ul; t < T; ++t) {
        inputFeatureDataView = getInput(t);
        std::copy(inputFeatureDataView->data(), inputFeatureDataView->data() + inputFeatureDataView->size(), encoderStatesValue_.data<f32>(0, t));
    }
}

void StatefulOnnxLabelScorer::setupEncoderStatesSizeValue() {
    if (not encoderStatesSizeValue_.empty()) {
        return;
    }

    u32 T = bufferSize();

    encoderStatesSizeValue_ = Onnx::Value::create(std::vector<s32>{static_cast<s32>(T)});
}

OnnxHiddenStateRef StatefulOnnxLabelScorer::computeInitialHiddenState() {
    verify(not expectMoreFeatures_ or (initializerEncoderStatesName_ == "" and initializerEncoderStatesSizeName_ == ""));

    if (not initialHiddenState_) {  // initialHiddenState_ is still sentinel value -> compute it
        OnnxHiddenStateModel::SessionInputs sessionInputs;

        if (initializerEncoderStatesName_ != "") {
            setupEncoderStatesValue();
            sessionInputs.emplace_back(initializerEncoderStatesName_, encoderStatesValue_);
        }
        if (initializerEncoderStatesSizeName_ != "") {
            setupEncoderStatesSizeValue();
            sessionInputs.emplace_back(initializerEncoderStatesSizeName_, encoderStatesSizeValue_);
        }

        initialHiddenState_ = hiddenStateModel_.initialHiddenState(std::move(sessionInputs));
    }

    return initialHiddenState_;
}

std::vector<OnnxHiddenStateRef> StatefulOnnxLabelScorer::updatedHiddenStates(std::vector<OnnxHiddenStateRef> const& hiddenStatesBatch, std::vector<s32> nextTokensBatch) {
    OnnxHiddenStateModel::SessionInputs sessionInputs;

    if (updaterEncoderStatesName_ != "") {
        setupEncoderStatesValue();
        sessionInputs.emplace_back(updaterEncoderStatesName_, encoderStatesValue_);
    }
    if (updaterEncoderStatesSizeName_ != "") {
        setupEncoderStatesSizeValue();
        sessionInputs.emplace_back(updaterEncoderStatesSizeName_, encoderStatesSizeValue_);
    }
    if (updaterTokenName_ != "") {
        sessionInputs.emplace_back(updaterTokenName_, Onnx::Value::create(nextTokensBatch));
    }

    return hiddenStateModel_.updatedHiddenStates(hiddenStatesBatch, std::move(sessionInputs));
}

void StatefulOnnxLabelScorer::cacheStates(std::vector<OnnxHiddenStateScoringContextRef> const& scoringContextBatch) {
    std::vector<OnnxHiddenStateScoringContextRef> nonFinalizedContexts;
    std::vector<OnnxHiddenStateRef>               hiddenStates;
    std::vector<s32>                              nextTokens;
    for (auto const& scoringContext : scoringContextBatch) {
        if (not scoringContext->requiresFinalize) {
            continue;
        }
        nonFinalizedContexts.push_back(scoringContext);
        if (scoringContext->hiddenState) {
            hiddenStates.push_back(scoringContext->hiddenState);
        }
        else {
            hiddenStates.push_back(computeInitialHiddenState());
        }
        verify(not scoringContext->labelSeq.empty());
        nextTokens.push_back(scoringContext->labelSeq.back());
    }

    // If no scoring contexts need finalization, nothing has to be done
    if (nonFinalizedContexts.empty()) {
        return;
    }

    auto newHiddenStates = updatedHiddenStates(hiddenStates, nextTokens);
    verify(newHiddenStates.size() == nonFinalizedContexts.size());

    for (size_t i = 0ul; i < nonFinalizedContexts.size(); ++i) {
        stateCache_.put(nonFinalizedContexts[i], newHiddenStates[i]);
    }
}

void StatefulOnnxLabelScorer::cacheScores(std::vector<OnnxHiddenStateScoringContextRef> const& scoringContextBatch) {
    if (scoringContextBatch.empty()) {
        return;
    }

    /*
     * Collect the hidden state of each scoring context in the batch
     */
    std::vector<OnnxHiddenStateRef> hiddenStates;
    hiddenStates.reserve(scoringContextBatch.size());
    for (auto const& scoringContext : scoringContextBatch) {
        if (scoringContext->labelSeq.empty()) {
            hiddenStates.push_back(computeInitialHiddenState());
        }
        else {
            hiddenStates.push_back((*stateCache_.get(scoringContext)).get());
        }
    }

    /*
     * Run scorer and put resulting scores into cache map
     */
    auto scoreVecs = hiddenStateModel_.scores(hiddenStates);
    verify(scoreVecs.size() == scoringContextBatch.size());

    for (size_t b = 0ul; b < scoringContextBatch.size(); ++b) {
        scoreCache_.put(scoringContextBatch[b], scoreVecs[b]);
    }
}

}  // namespace Nn
