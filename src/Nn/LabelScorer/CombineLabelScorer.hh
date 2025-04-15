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

#ifndef COMBINE_LABEL_SCORER_HH
#define COMBINE_LABEL_SCORER_HH

#include "LabelScorer.hh"

namespace Nn {

/*
 * Performs log-linear combination of multiple sub-label-scorers, assuming
 * that the sub-scorers have the same label alphabet, i.e.
 * combined_score(request) = sum_i { score_i(request) * scale_i }
 * The assigned timeframe is the maximum over the sub-timeframes, i.e.
 * combined_timeframe(request) = max_i { timeframe_i(request) }
 */
class CombineLabelScorer : public LabelScorer {
    using Precursor = LabelScorer;

public:
    static Core::ParameterInt   paramNumLabelScorers;
    static Core::ParameterFloat paramScale;

    CombineLabelScorer(const Core::Configuration& config);
    virtual ~CombineLabelScorer() = default;

    // Reset all sub-scorers
    void reset();

    // Forward signal to all sub-scorers
    void signalNoMoreFeatures();

    // Combine initial ScoringContexts from all sub-scorers
    ScoringContextRef getInitialScoringContext();

    // Combine extended ScoringContexts from all sub-scorers
    ScoringContextRef extendedScoringContext(Request const& request);

    // Add input to all sub-scorers
    void addInput(std::shared_ptr<const f32[]> const& input, size_t featureSize);

    // Add inputs to all sub-scorers
    virtual void addInputs(std::shared_ptr<const f32[]> const& input, size_t timeSize, size_t featureSize);

    // Compute weighted score of request with all sub-scorers
    std::optional<ScoreWithTime> computeScoreWithTime(Request const& request);

    // Compute weighted scores of requests with all sub-scorers
    std::optional<ScoresWithTimes> computeScoresWithTimes(const std::vector<Request>& requests);

protected:
    struct ScaledLabelScorer {
        Core::Ref<LabelScorer> scorer;
        Score                  scale;
    };

    std::vector<ScaledLabelScorer> scaledScorers_;
};

}  // namespace Nn

#endif  // COMBINE_LABEL_SCORER_HH
