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

#ifndef TRANSITION_LABEL_SCORER_HH
#define TRANSITION_LABEL_SCORER_HH

#include "LabelScorer.hh"

namespace Nn {

/*
 * Label Scorer that adds predefined transition scores to the scores of an underlying base
 * label scorer based on the transition type of each request.
 * The score for each transition type is set via config parameters.
 */
class TransitionLabelScorer : public LabelScorer {
    static const Core::ParameterFloat paramLabelToLabelScore;
    static const Core::ParameterFloat paramLabelLoopScore;
    static const Core::ParameterFloat paramLabelToBlankScore;
    static const Core::ParameterFloat paramBlankToLabelScore;
    static const Core::ParameterFloat paramBlankLoopScore;
    static const Core::ParameterFloat paramInitialLabelScore;
    static const Core::ParameterFloat paramInitialBlankScore;

public:
    using Precursor = LabelScorer;

    TransitionLabelScorer(Core::Configuration const& config);
    virtual ~TransitionLabelScorer() = default;

    // Reset base scorer
    void reset() override;

    // Forward signal to base scorer
    void signalNoMoreFeatures() override;

    // Initial context of base scorer
    ScoringContextRef getInitialScoringContext() override;

    // Extend context via base scorer
    ScoringContextRef extendedScoringContext(Request const& request) override;

    // Clean up base scorer
    void cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) override;

    // Add input to base scorer
    void addInput(DataView const& input) override;

    // Add inputs to sub-scorer
    void addInputs(DataView const& input, size_t nTimesteps) override;

    // Compute score of base scorer and add transition score based on transition type of the request
    std::optional<ScoreWithTime> computeScoreWithTime(Request const& request) override;

    // Compute scores of base scorer and add transition scores based on transition types of the requests
    std::optional<ScoresWithTimes> computeScoresWithTimes(std::vector<Request> const& requests) override;

private:
    Score labelToLabelScore_;
    Score labelLoopScore_;
    Score labelToBlankScore_;
    Score blankToLabelScore_;
    Score blankLoopScore_;
    Score initialLabelScore_;
    Score initialBlankScore_;

    Core::Ref<LabelScorer> baseLabelScorer_;

    Score getTransitionScore(TransitionType transitionType) const;
};

}  // namespace Nn

#endif  // TRANSITION_LABEL_SCORER_HH
