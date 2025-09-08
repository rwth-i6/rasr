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
    void reset() override;

    // Forward signal to all sub-scorers
    void signalNoMoreFeatures() override;

    // Combine initial ScoringContexts from all sub-scorers
    ScoringContextRef getInitialScoringContext() override;

    // Combine extended ScoringContexts from all sub-scorers
    ScoringContextRef extendedScoringContext(Request const& request) override;

    // Cleanup all sub-scorers
    void cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) override;

    // Add input to all sub-scorers
    void addInput(DataView const& input) override;

    // Add inputs to all sub-scorers
    virtual void addInputs(DataView const& input, size_t nTimesteps) override;

    // Compute weighted score of request with all sub-scorers
    std::optional<ScoreWithTime> computeScoreWithTime(Request const& request) override;

    // Compute weighted scores of requests with all sub-scorers
    std::optional<ScoresWithTimes> computeScoresWithTimes(std::vector<Request> const& requests) override;

    // Get number of scorers inside combined scorer
    size_t numSubScorers() const override;

    // Compute weighted score of request with a specific sub-scorer
    std::optional<ScoreWithTime> computeScoreWithTime(Request const& request, size_t scorerIdx) override;

    // Compute weighted scores of requests with a specific sub-scorer
    std::optional<ScoresWithTimes> computeScoresWithTimes(const std::vector<Request>& requests, size_t scorerIdx) override;

#ifdef MODULE_PYTHON
    virtual void registerPythonCallback(std::string const& name, pybind11::function const& callback) override;
#endif

protected:
    struct ScaledLabelScorer {
        Core::Ref<LabelScorer> scorer;
        Score                  scale;
    };

    std::vector<ScaledLabelScorer> scaledScorers_;
};

}  // namespace Nn

#endif  // COMBINE_LABEL_SCORER_HH
