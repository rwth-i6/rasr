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
 * Adds the scores of multiple sub-label-scorers for each request.
 * This assumes that the sub-scorers work on the same token level.
 */
class CombineLabelScorer : public LabelScorer {
    using Precursor = LabelScorer;

public:
    static Core::ParameterInt paramNumLabelScorers;

    CombineLabelScorer(const Core::Configuration& config);
    virtual ~CombineLabelScorer() = default;

    void                           reset();
    void                           signalNoMoreFeatures();
    ScoringContextRef              getInitialScoringContext();
    ScoringContextRef              extendedScoringContext(Request const& request);
    void                           addInput(SharedDataHolder const& input, size_t featureSize);
    void                           addInput(std::vector<f32> const& input);
    void                           addInputs(SharedDataHolder const& input, size_t timeSize, size_t featureSize);
    std::optional<ScoreWithTime>   computeScoreWithTime(Request const& request);
    std::optional<ScoresWithTimes> computeScoresWithTimes(const std::vector<Request>& requests);

protected:
    std::vector<Core::Ref<LabelScorer>> scorers_;
};

}  // namespace Nn

#endif  // COMBINE_LABEL_SCORER_HH
