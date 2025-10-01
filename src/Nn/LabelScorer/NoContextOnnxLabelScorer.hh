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

#ifndef NO_CONTEXT_ONNX_LABEL_SCORER_HH
#define NO_CONTEXT_ONNX_LABEL_SCORER_HH

#include <Onnx/Model.hh>

#include "BufferedLabelScorer.hh"

namespace Nn {

/*
 * A LabelScorer that computes scores by forwarding only the input feature at the current timestep through an ONNX model,
 * without any label history.
 * This is suitable for example for CTC outputs consisting of a linear layer + -log_softmax activation.
 *
 * If the CTC output is the only output, the encoder and output layer can be put together into an "encoder-only" label scorer.
 * However, when the CTC output is one of several outputs based on a shared encoder, the CTC output head must be separated
 * from the encoder. The NoContextOnnxLabelScorer can be used for this purpose.
 */
class NoContextOnnxLabelScorer : public BufferedLabelScorer {
    using Precursor = BufferedLabelScorer;

public:
    NoContextOnnxLabelScorer(Core::Configuration const& config);
    virtual ~NoContextOnnxLabelScorer() = default;

    // Clear feature buffer and cached scores
    void reset() override;

    // Initial scoring context contains step 0
    ScoringContextRef getInitialScoringContext() override;

    // Clean up input buffer as well as cached score vectors that are no longer needed
    void cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) override;

protected:
    size_t getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const override;

    // Increment the step by 1
    ScoringContextRef extendedScoringContextInternal(LabelScorer::Request const& request) override;

    // If scores for the given scoring contexts are not yet cached, prepare and run an ONNX session to
    // compute the scores and cache them
    // Then, retreive scores from cache
    std::optional<LabelScorer::ScoresWithTimes> computeScoresWithTimesInternal(std::vector<LabelScorer::Request> const& requests) override;

    // Uses `getScoresWithTimes` internally with some wrapping for vector packing/expansion
    std::optional<LabelScorer::ScoreWithTime> computeScoreWithTimeInternal(LabelScorer::Request const& request) override;

private:
    void forwardContext(StepScoringContextRef const& context);

    Onnx::Model onnxModel_;

    std::string inputFeatureName_;
    std::string scoresName_;

    std::unordered_map<StepScoringContextRef, std::vector<Score>, ScoringContextHash, ScoringContextEq> scoreCache_;
};

}  // namespace Nn

#endif  // NO_CONTEXT_ONNX_LABEL_SCORER_HH
