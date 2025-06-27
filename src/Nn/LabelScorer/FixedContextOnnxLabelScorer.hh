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

#ifndef FIXED_CONTEXT_ONNX_LABEL_SCORER_HH
#define FIXED_CONTEXT_ONNX_LABEL_SCORER_HH

#include <Onnx/Model.hh>

#include "BufferedLabelScorer.hh"

namespace Nn {

/*
 * Label Scorer that performs scoring by forwarding the input feature at the current timestep together
 * with a fixed-size sequence of history tokens through an ONNX model.
 * A common use case would be a neural transducer model with a fixed-size history.
 */
class FixedContextOnnxLabelScorer : public BufferedLabelScorer {
    using Precursor = BufferedLabelScorer;

    static const Core::ParameterInt  paramStartLabelIndex;
    static const Core::ParameterInt  paramHistoryLength;
    static const Core::ParameterBool paramBlankUpdatesHistory;
    static const Core::ParameterBool paramLoopUpdatesHistory;
    static const Core::ParameterBool paramVerticalLabelTransition;
    static const Core::ParameterInt  paramMaxBatchSize;

public:
    FixedContextOnnxLabelScorer(Core::Configuration const& config);
    virtual ~FixedContextOnnxLabelScorer() = default;

    // Clear feature buffer and cached scores
    void reset() override;

    // Initial scoring context contains step 0 and a history vector filled with the start label index
    ScoringContextRef getInitialScoringContext() override;

    // May increment the step by 1 (except for vertical transitions) and may append the next token to the
    // history label sequence depending on the transition type and whether loops/blanks update the history
    // or not
    ScoringContextRef extendedScoringContext(LabelScorer::Request const& request) override;

    // Clean up input buffer as well as cached score vectors that are no longer needed
    void cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) override;

    // If scores for the given scoring contexts are not yet cached, prepare and run an ONNX session to
    // compute the scores and cache them
    // Then, retreive scores from cache
    std::optional<LabelScorer::ScoresWithTimes> computeScoresWithTimes(std::vector<LabelScorer::Request> const& requests) override;

    // Uses `getScoresWithTimes` internally with some wrapping for vector packing/expansion
    std::optional<LabelScorer::ScoreWithTime> computeScoreWithTime(LabelScorer::Request const& request) override;

protected:
    Speech::TimeframeIndex minActiveTimeIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const override;

private:
    // Forward a batch of histories through the ONNX model and put the resulting scores into the score cache
    // Assumes that all histories in the batch are based on the same timestep
    void forwardBatch(std::vector<SeqStepScoringContextRef> const& contextBatch);

    size_t startLabelIndex_;
    size_t historyLength_;
    bool   blankUpdatesHistory_;
    bool   loopUpdatesHistory_;
    bool   verticalLabelTransition_;
    size_t maxBatchSize_;

    Onnx::Model onnxModel_;

    std::string inputFeatureName_;
    std::string historyName_;
    std::string scoresName_;

    std::unordered_map<SeqStepScoringContextRef, std::vector<Score>, ScoringContextHash, ScoringContextEq> scoreCache_;
};

}  // namespace Nn

#endif  // FIXED_CONTEXT_ONNX_LABEL_SCORER_HH
