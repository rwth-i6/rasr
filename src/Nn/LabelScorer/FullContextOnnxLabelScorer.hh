/** Copyright 2025 RWTH Aachen University. All rights reserved.
 *
 * Licensed under the RWTH ASR License (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FULL_CONTEXT_ONNX_LABEL_SCORER_HH
#define FULL_CONTEXT_ONNX_LABEL_SCORER_HH

#include <Onnx/Model.hh>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "BufferedLabelScorer.hh"
#include "ModelCache.hh"

namespace Nn {

/*
 * Label scorer that forwards the complete label history together with some form of
 * acoustic input through a single ONNX model.
 *
 * This is stateless with respect to the ONNX model: the full history is supplied
 * as input on every scoring call. Unlike FixedContextOnnxLabelScorer, the history
 * is never truncated.
 *
 * The acoustic input can be given to the ONNX model in one of two ways, depending on
 * which inputs are mapped in the model's ONNX I/O spec:
 *  - "input-feature": only the single input feature at the hypothesis' current timestep
 *    is fed in, re-selected on every scoring call. This works incrementally as features
 *    arrive.
 *  - "encoder-states"/"encoder-states-size": the complete input sequence collected so far
 *    is fed in as one tensor, together with its length. Since this requires the whole
 *    sequence, scoring only starts once all features of the segment have been passed
 *    (i.e. after `signalNoMoreFeatures`), and the input buffer is never trimmed.
 * Both can also be mapped at once, in which case the model receives both the current
 * frame and the full sequence.
 */
class FullContextOnnxLabelScorer : public BufferedLabelScorer {
    using Precursor = BufferedLabelScorer;

    static const Core::ParameterInt  paramStartLabelIndex;
    static const Core::ParameterInt  paramInitialHistoryLength;
    static const Core::ParameterBool paramBlankUpdatesHistory;
    static const Core::ParameterBool paramSilenceUpdatesHistory;
    static const Core::ParameterBool paramLoopUpdatesHistory;
    static const Core::ParameterBool paramVerticalLabelTransition;
    static const Core::ParameterInt  paramMaxBatchSize;

public:
    FullContextOnnxLabelScorer(Core::Configuration const& config, ModelCache& modelCache);
    virtual ~FullContextOnnxLabelScorer() = default;

    // Clear feature buffer and cached scores
    void reset() override;

    // Add a single input feature to the buffer
    void addInput(DataView const& input) override;

    // Initial scoring context contains step 0 and initial-history-length copies of start-label-index
    // For the common BOS-prefix case, leave this at 1
    ScoringContextRef getInitialScoringContext() override;

    // May increment the step by 1 (except for vertical transitions) and may append the next token to the
    // history label sequence depending on the transition type and whether loops/blanks update the history
    // or not
    ScoringContextRef extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) override;

    // Clean up input buffer as well as cached score vectors that are no longer needed
    void cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) override;

    // If scores for the given scoring contexts are not yet cached, prepare and run an ONNX session to
    // compute the scores and cache them
    std::vector<std::optional<ScoreAccessorRef>> getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) override;

    // Uses `getScoreAccessors` internally with some wrapping for vector packing/expansion
    std::optional<ScoreAccessorRef> getScoreAccessor(ScoringContextRef scoringContext) override;

protected:
    size_t getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const override;

private:
    // Forward a batch of histories through the ONNX model and put the resulting scores into the score cache
    // Assumes that all histories in the batch share the same history length and if a per-frame input-feature
    // is used, are also based on the same timestep
    void forwardBatch(std::vector<SeqStepScoringContextRef> const& scoringContextBatch);

    // Set up encoderStatesValue_/encoderStatesSizeValue_ from the full input buffer, unless already cached
    void setupEncoderStatesValue();
    void setupEncoderStatesSizeValue();

    size_t startLabelIndex_;
    size_t initialHistoryLength_;
    bool   blankUpdatesHistory_;
    bool   silenceUpdatesHistory_;
    bool   loopUpdatesHistory_;
    bool   verticalLabelTransition_;
    size_t maxBatchSize_;

    std::shared_ptr<Onnx::Model> onnxModel_;

    std::string inputFeatureName_;
    std::string encoderStatesName_;
    std::string encoderStatesSizeName_;
    std::string historyName_;
    std::string historySizeName_;
    std::string scoresName_;

    // Store the onnx values with all encoder states and lengths inside so that it doesn't have to be recomputed every time
    Onnx::Value encoderStatesValue_;
    Onnx::Value encoderStatesSizeValue_;

    std::unordered_map<SeqStepScoringContextRef, std::shared_ptr<std::vector<Score>>, ScoringContextHash, ScoringContextEq> scoreCache_;
};

}  // namespace Nn

#endif  // FULL_CONTEXT_ONNX_LABEL_SCORER_HH
