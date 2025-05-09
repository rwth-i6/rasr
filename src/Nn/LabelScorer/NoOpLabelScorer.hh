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

#ifndef NO_OP_LABEL_SCORER_HH
#define NO_OP_LABEL_SCORER_HH

#include "BufferedLabelScorer.hh"

namespace Nn {

/*
 * Label Scorer that performs no computation internally. It assumes that the input features are already
 * finished score vectors and just returns the score at the current time step.
 *
 * This is useful for example when the scores are computed externally and transmitted via a pybind interface
 * or when they are computed inside a flow node.
 */
class StepwiseNoOpLabelScorer : public BufferedLabelScorer {
public:
    using Precursor = BufferedLabelScorer;

    StepwiseNoOpLabelScorer(const Core::Configuration& config);

    // Initial scoring context just contains step 0.
    ScoringContextRef getInitialScoringContext() override;

    // Scoring context with step incremented by 1.
    ScoringContextRef extendedScoringContext(LabelScorer::Request const& request) override;

    // Gets the buffered score for the requested token at the requested step
    std::optional<LabelScorer::ScoreWithTime> computeScoreWithTime(LabelScorer::Request const& request) override;

protected:
    Speech::TimeframeIndex minActiveTimeIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const override;
};

}  // namespace Nn

#endif  // NO_OP_LABEL_SCORER_HH
