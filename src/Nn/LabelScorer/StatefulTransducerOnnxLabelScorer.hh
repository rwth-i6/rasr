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

#ifndef STATEFUL_TRANSDUCER_ONNX_LABEL_SCORER_HH
#define STATEFUL_TRANSDUCER_ONNX_LABEL_SCORER_HH

#include <optional>

#include <Core/Component.hh>
#include <Core/Configuration.hh>
#include <Core/FIFOCache.hh>
#include <Core/ReferenceCounting.hh>
#include <Mm/FeatureScorer.hh>
#include <Speech/Feature.hh>

#include "BufferedLabelScorer.hh"
#include "ModelCache.hh"
#include "OnnxHiddenStateModel.hh"
#include "ScoringContext.hh"

namespace Nn {

/*
 * Scoring context consisting of a hidden state and a step.
 * Assumes that two hidden states are equal if and only if they were created
 * from the same label history.
 */
struct StepOnnxHiddenStateScoringContext : public ScoringContext {
    Speech::TimeframeIndex     currentStep;
    std::vector<LabelIndex>    labelSeq;  // Used for hashing
    mutable OnnxHiddenStateRef hiddenState;
    mutable bool               requiresFinalize;

    StepOnnxHiddenStateScoringContext();
    StepOnnxHiddenStateScoringContext(Speech::TimeframeIndex step, std::vector<LabelIndex> const& labelSeq, OnnxHiddenStateRef state, bool requiresFinalize);

    bool   isEqual(ScoringContextRef const& other) const override;
    size_t hash() const override;
};

typedef Core::Ref<const StepOnnxHiddenStateScoringContext> StepOnnxHiddenStateScoringContextRef;

/*
 * Label Scorer that performs scoring by forwarding hidden states through an ONNX model.
 * This Label Scorer requires three ONNX models:
 *  - A State Initializer which produces the hidden states for the first step
 *  - A State Updater which produces updated hidden states based on the previous hidden states and the next token
 *  - A Scorer which computes scores based on the current input feature and the hidden states
 *
 * The models themselves as well as the mapping between their inputs/outputs and the hidden states are
 * handled by `OnnxHiddenStateModel`; see there for the metadata convention that the models have to follow.
 *
 * A common use case for this Label Scorer would be a Transducer model with unlimited context.
 *
 * Note: This LabelScorer is similar to the `StatefulOnnxLabelScorer`. The difference is that in this one the ScoringContext also
 * contains the current step and the input feature at the current step is fed to the Scorer. Furthermore, the state initializer
 * and updater here only take tokens and no input features.
 */
class StatefulTransducerOnnxLabelScorer : public BufferedLabelScorer {
    using Precursor = BufferedLabelScorer;

    static const Core::ParameterBool paramBlankUpdatesHistory;
    static const Core::ParameterBool paramSilenceUpdatesHistory;
    static const Core::ParameterBool paramLoopUpdatesHistory;
    static const Core::ParameterBool paramVerticalLabelTransition;
    static const Core::ParameterInt  paramMaxBatchSize;
    static const Core::ParameterInt  paramMaxCachedScores;

public:
    StatefulTransducerOnnxLabelScorer(Core::Configuration const& config, ModelCache& modelCache);
    virtual ~StatefulTransducerOnnxLabelScorer() = default;

    void reset() override;

    // If startLabelIndex is set, forward that through the state updater to obtain the start ScoringContext
    ScoringContextRef getInitialScoringContext() override;

    // Append the new token to the label sequence; does not update the hidden-state. This is only done once the scoringContext is used for scoring again.
    ScoringContextRef extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) override;

    // Update hidden state, run scorer and get an accessor for the output score vector
    std::optional<ScoreAccessorRef> getScoreAccessor(ScoringContextRef scoringContext) override;

    // Update hidden states (batched), run scorers (batched) and get accessor for the output score vectors
    std::vector<std::optional<ScoreAccessorRef>> getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) override;

protected:
    size_t getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const override;

private:
    // Forward a batch of scoringContexts through the ONNX scorer model and put the resulting scores into the score cache
    void cacheScores(std::vector<StepOnnxHiddenStateScoringContextRef> const& scoringContextBatch);

    // Computes new hidden state based on previous hidden state and next token with batched state-updater call
    std::vector<OnnxHiddenStateRef> updatedHiddenStates(std::vector<OnnxHiddenStateRef> const& hiddenStatesBatch, std::vector<s32> nextTokensBatch);

    // Compute updated states for all non-finalized scoring contexts and put them into the state cache
    void cacheStates(std::vector<StepOnnxHiddenStateScoringContextRef> const& scoringContextBatch);

    bool   blankUpdatesHistory_;
    bool   silenceUpdatesHistory_;
    bool   loopUpdatesHistory_;
    bool   verticalLabelTransition_;
    size_t maxBatchSize_;

    OnnxHiddenStateModel hiddenStateModel_;

    StepOnnxHiddenStateScoringContextRef initialScoringContext_;

    std::string scorerInputFeatureName_;
    std::string updaterTokenName_;

    Core::FIFOCache<StepOnnxHiddenStateScoringContextRef, std::shared_ptr<std::vector<Score>>, ScoringContextHash, ScoringContextEq> scoreCache_;
    Core::FIFOCache<StepOnnxHiddenStateScoringContextRef, OnnxHiddenStateRef, ScoringContextHash, ScoringContextEq>                  stateCache_;
};

}  // namespace Nn

#endif  // STATEFUL_TRANSDUCER_ONNX_LABEL_SCORER_HH
