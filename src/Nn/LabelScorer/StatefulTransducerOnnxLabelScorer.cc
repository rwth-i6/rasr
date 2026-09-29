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

#include "StatefulTransducerOnnxLabelScorer.hh"

#include <algorithm>
#include <cstddef>
#include <utility>

#include <Core/Assertions.hh>
#include <Core/ReferenceCounting.hh>
#include <Flow/Timestamp.hh>
#include <Math/FastMatrix.hh>
#include <Mm/Module.hh>
#include <Speech/Types.hh>

#include "ScoreAccessor.hh"
#include "ScoringContext.hh"

namespace Nn {

StepOnnxHiddenStateScoringContext::StepOnnxHiddenStateScoringContext()
        : currentStep(0), labelSeq(), hiddenState(), requiresFinalize(false) {}

StepOnnxHiddenStateScoringContext::StepOnnxHiddenStateScoringContext(
        Speech::TimeframeIndex         step,
        std::vector<LabelIndex> const& labelSeq,
        OnnxHiddenStateRef             state,
        bool                           requiresFinalize)
        : currentStep(step),
          labelSeq(labelSeq),
          hiddenState(state),
          requiresFinalize(requiresFinalize) {}

bool StepOnnxHiddenStateScoringContext::isEqual(ScoringContextRef const& other) const {
    auto* otherPtr = dynamic_cast<StepOnnxHiddenStateScoringContext const*>(other.get());
    if (otherPtr == nullptr) {
        return false;
    }

    return currentStep == otherPtr->currentStep and labelSeqEqual(labelSeq, otherPtr->labelSeq);
}

size_t StepOnnxHiddenStateScoringContext::hash() const {
    return Core::combineHashes(currentStep, labelSeqHash(labelSeq));
}

/*
 * =======================================
 * == StatefulTransducerOnnxLabelScorer ==
 * =======================================
 */

const Core::ParameterBool StatefulTransducerOnnxLabelScorer::paramBlankUpdatesHistory(
        "blank-updates-history",
        "Whether previously emitted blank labels should be used to update the history.",
        false);

const Core::ParameterBool StatefulTransducerOnnxLabelScorer::paramSilenceUpdatesHistory(
        "silence-updates-history",
        "Whether previously emitted silence labels should be used to update the history.",
        false);

const Core::ParameterBool StatefulTransducerOnnxLabelScorer::paramLoopUpdatesHistory(
        "loop-updates-history",
        "Whether in the case of loop transitions every repeated emission should be used to update the history.",
        false);

const Core::ParameterBool StatefulTransducerOnnxLabelScorer::paramVerticalLabelTransition(
        "vertical-label-transition",
        "Whether (non-blank) label transitions should be vertical, i.e. not increase the time step.",
        false);

const Core::ParameterInt StatefulTransducerOnnxLabelScorer::paramMaxBatchSize(
        "max-batch-size",
        "Max number of hidden-states that can be fed into the scorer ONNX model at once.",
        Core::Type<int>::max);

const Core::ParameterInt StatefulTransducerOnnxLabelScorer::paramMaxCachedScores(
        "max-cached-score-vectors",
        "Maximum size of cache that maps histories to scores. This prevents memory overflow in case of very long audio segments.",
        10000);

// The hidden states and the scores output are not part of the IO spec; they are handled by `OnnxHiddenStateModel`
static const std::vector<Onnx::IOSpecification> scorerModelIoSpec = {
        Onnx::IOSpecification{
                "input-feature",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -2}, {1, -2}}}};  // [1, E]

static const std::vector<Onnx::IOSpecification> stateUpdaterModelIoSpec = {
        Onnx::IOSpecification{
                "token",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{1}, {-1}}}};  // [1] or [B]

StatefulTransducerOnnxLabelScorer::StatefulTransducerOnnxLabelScorer(Core::Configuration const& config, ModelCache& modelCache)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::TRANSDUCER),
          blankUpdatesHistory_(paramBlankUpdatesHistory(config)),
          silenceUpdatesHistory_(paramSilenceUpdatesHistory(config)),
          loopUpdatesHistory_(paramLoopUpdatesHistory(config)),
          verticalLabelTransition_(paramVerticalLabelTransition(config)),
          maxBatchSize_(paramMaxBatchSize(config)),
          hiddenStateModel_(config, modelCache, {}, stateUpdaterModelIoSpec, scorerModelIoSpec),
          initialScoringContext_(),
          scoreCache_(paramMaxCachedScores(config)),
          stateCache_(paramMaxCachedScores(config)) {
    scorerInputFeatureName_ = hiddenStateModel_.scorerOnnxName("input-feature");
    updaterTokenName_       = hiddenStateModel_.stateUpdaterOnnxName("token");
}

void StatefulTransducerOnnxLabelScorer::reset() {
    Precursor::reset();
    stateCache_.clear();
    scoreCache_.clear();
}

ScoringContextRef StatefulTransducerOnnxLabelScorer::getInitialScoringContext() {
    if (not initialScoringContext_) {
        auto initialHiddenState = hiddenStateModel_.initialHiddenState();
        initialScoringContext_  = Core::ref(new StepOnnxHiddenStateScoringContext(0ul, std::vector<LabelIndex>(), initialHiddenState, false));
    }

    return initialScoringContext_;
}

ScoringContextRef StatefulTransducerOnnxLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) {
    bool   pushToken     = false;
    size_t timeIncrement = 0ul;
    switch (transitionType) {
        case TransitionType::BLANK_LOOP:
            pushToken     = blankUpdatesHistory_ and loopUpdatesHistory_;
            timeIncrement = 1ul;
            break;
        case TransitionType::SILENCE_LOOP:
            pushToken     = silenceUpdatesHistory_ and loopUpdatesHistory_;
            timeIncrement = 1ul;
            break;
        case TransitionType::LABEL_TO_BLANK:
        case TransitionType::INITIAL_BLANK:
            pushToken     = blankUpdatesHistory_;
            timeIncrement = 1ul;
            break;
        case TransitionType::LABEL_TO_SILENCE:
        case TransitionType::INITIAL_SILENCE:
            pushToken     = silenceUpdatesHistory_;
            timeIncrement = 1ul;
            break;
        case TransitionType::LABEL_LOOP:
            pushToken     = loopUpdatesHistory_;
            timeIncrement = not verticalLabelTransition_;
            break;
        case TransitionType::BLANK_TO_LABEL:
        case TransitionType::SILENCE_TO_LABEL:
        case TransitionType::LABEL_TO_LABEL:
        case TransitionType::INITIAL_LABEL:
        case TransitionType::SENTENCE_END:
            pushToken     = true;
            timeIncrement = not verticalLabelTransition_;
            break;
        default:
            error() << "Unknown transition type " << transitionType;
    }

    // If scoringContext is not going to be modified, return the original one
    if (not pushToken and timeIncrement == 0ul) {
        return scoringContext;
    }

    StepOnnxHiddenStateScoringContextRef stepOnnxHiddenStateScoringContext(dynamic_cast<StepOnnxHiddenStateScoringContext const*>(scoringContext.get()));
    std::vector<LabelIndex>              newLabelSeq(stepOnnxHiddenStateScoringContext->labelSeq);
    bool                                 requiresFinalize = false;

    if (pushToken) {
        newLabelSeq.push_back(nextToken);
        requiresFinalize = true;
    }

    // Re-use previous hidden-state but mark that finalization (i.e. hidden-state update) is required
    auto newScoringContext              = Core::ref(new StepOnnxHiddenStateScoringContext(stepOnnxHiddenStateScoringContext->currentStep + timeIncrement, std::move(newLabelSeq), stepOnnxHiddenStateScoringContext->hiddenState, true));
    newScoringContext->requiresFinalize = requiresFinalize;

    auto hiddenState = stateCache_.get(newScoringContext);
    if (hiddenState) {
        newScoringContext->hiddenState      = *hiddenState;
        newScoringContext->requiresFinalize = false;
    }

    return newScoringContext;
}

std::vector<std::optional<ScoreAccessorRef>> StatefulTransducerOnnxLabelScorer::getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) {
    if (scoringContexts.empty()) {
        return {};
    }

    std::vector<std::optional<ScoreAccessorRef>> scoreAccessors(scoringContexts.size(), std::nullopt);

    // Cast scoring contexts to concrete types
    std::vector<StepOnnxHiddenStateScoringContextRef> stepOnnxHiddenStateScoringContexts;
    stepOnnxHiddenStateScoringContexts.reserve(scoringContexts.size());
    for (auto const& scoringContext : scoringContexts) {
        stepOnnxHiddenStateScoringContexts.push_back(Core::ref(dynamic_cast<StepOnnxHiddenStateScoringContext const*>(scoringContext.get())));
    }

    /*
     * Collect all contexts that are based on the same timestep (-> same input feature) and
     * group them together
     */
    std::unordered_map<size_t, std::vector<size_t>> contextsWithTimestep;  // Maps timestep to list of all indices of contexts with that timestep

    for (size_t contextIndex = 0ul; contextIndex < scoringContexts.size(); ++contextIndex) {
        auto const& context = stepOnnxHiddenStateScoringContexts[contextIndex];
        auto        step    = context->currentStep;

        if (not getInput(step)) {
            // If input is not available, this context can't be forwarded
            continue;
        }

        // Create new vector if step value isn't present in map yet
        auto [it, inserted] = contextsWithTimestep.emplace(step, std::vector<size_t>());
        it->second.push_back(contextIndex);
    }

    /*
     * Iterate over distinct timesteps
     */
    for (auto const& [timestep, contextIndices] : contextsWithTimestep) {
        /*
         * Identify unique scoring contexts that still need session runs
         */
        std::unordered_set<StepOnnxHiddenStateScoringContextRef, ScoringContextHash, ScoringContextEq> uniqueUncachedScoringContexts;

        for (auto contextIndex : contextIndices) {
            auto const& scoringContextRef = stepOnnxHiddenStateScoringContexts[contextIndex];
            if (not scoreCache_.contains(scoringContextRef)) {
                // Group by unique scoringContext
                uniqueUncachedScoringContexts.emplace(scoringContextRef);
            }
        }

        if (uniqueUncachedScoringContexts.empty()) {
            continue;
        }

        std::vector<StepOnnxHiddenStateScoringContextRef> scoringContextBatch;
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
    }

    /*
     * Assign states from cache to scoring contexts and scores from cache to result vector
     */
    for (size_t contextIndex = 0ul; contextIndex < scoringContexts.size(); ++contextIndex) {
        auto const& scoringContext = stepOnnxHiddenStateScoringContexts[contextIndex];

        if (not getInput(scoringContext->currentStep)) {
            continue;
        }

        if (scoringContext->requiresFinalize) {
            auto hiddenState = stateCache_.get(scoringContext);
            verify(hiddenState);
            scoringContext->hiddenState      = *hiddenState;
            scoringContext->requiresFinalize = false;
        }

        verify(scoreCache_.contains(scoringContext));
        auto const& scoreVec         = scoreCache_.get(scoringContext)->get();
        scoreAccessors[contextIndex] = Core::ref(new VectorScoreAccessor(scoreVec, scoringContext->currentStep));
    }

    return scoreAccessors;
}

std::optional<ScoreAccessorRef> StatefulTransducerOnnxLabelScorer::getScoreAccessor(ScoringContextRef scoringContext) {
    return getScoreAccessors({scoringContext})[0];
}

size_t StatefulTransducerOnnxLabelScorer::getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const {
    return 0u;
}

std::vector<OnnxHiddenStateRef> StatefulTransducerOnnxLabelScorer::updatedHiddenStates(std::vector<OnnxHiddenStateRef> const& hiddenStatesBatch, std::vector<s32> nextTokensBatch) {
    return hiddenStateModel_.updatedHiddenStates(hiddenStatesBatch, {{updaterTokenName_, Onnx::Value::create(nextTokensBatch)}});
}

void StatefulTransducerOnnxLabelScorer::cacheStates(std::vector<StepOnnxHiddenStateScoringContextRef> const& scoringContextBatch) {
    std::vector<StepOnnxHiddenStateScoringContextRef> nonFinalizedContexts;
    std::vector<OnnxHiddenStateRef>                   hiddenStates;
    std::vector<s32>                                  nextTokens;
    for (auto const& scoringContext : scoringContextBatch) {
        if (not scoringContext->requiresFinalize) {
            stateCache_.put(scoringContext, scoringContext->hiddenState);
            continue;
        }
        nonFinalizedContexts.push_back(scoringContext);
        verify(scoringContext->hiddenState);
        hiddenStates.push_back(scoringContext->hiddenState);
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

void StatefulTransducerOnnxLabelScorer::cacheScores(std::vector<StepOnnxHiddenStateScoringContextRef> const& scoringContextBatch) {
    if (scoringContextBatch.empty()) {
        return;
    }

    /*
     * Collect the hidden state of each scoring context in the batch
     */
    std::vector<OnnxHiddenStateRef> hiddenStates;
    hiddenStates.reserve(scoringContextBatch.size());
    for (auto const& scoringContext : scoringContextBatch) {
        if (scoringContext->requiresFinalize) {
            hiddenStates.push_back((*stateCache_.get(scoringContext)).get());
        }
        else {
            hiddenStates.push_back(scoringContext->hiddenState);
        }
    }

    /*
     * All scoring contexts in the batch are based on the same timestep and thus share the same input feature
     */
    auto                 inputFeatureDataView = getInput(scoringContextBatch.front()->currentStep);
    f32 const*           inputFeatureData     = inputFeatureDataView->data();
    std::vector<int64_t> inputFeatureShape    = {1ul, static_cast<int64_t>(inputFeatureDataView->size())};

    /*
     * Run scorer and put resulting scores into cache map
     */
    auto scoreVecs = hiddenStateModel_.scores(hiddenStates, {{scorerInputFeatureName_, Onnx::Value::create(inputFeatureData, inputFeatureShape)}});
    verify(scoreVecs.size() == scoringContextBatch.size());

    for (size_t b = 0ul; b < scoringContextBatch.size(); ++b) {
        scoreCache_.put(scoringContextBatch[b], scoreVecs[b]);
    }
}

}  // namespace Nn
