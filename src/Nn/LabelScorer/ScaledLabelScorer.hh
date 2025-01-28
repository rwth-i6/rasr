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

#ifndef SCALED_LABEL_SCORER_HH
#define SCALED_LABEL_SCORER_HH

#include "LabelScorer.hh"

namespace Nn {

/*
 * Wrapper around a LabelScorer that scales all the scores by some factor.
 */
class ScaledLabelScorer : public LabelScorer {
    using Precursor = LabelScorer;

public:
    static Core::ParameterFloat paramScale;

    ScaledLabelScorer(const Core::Configuration& config, const Core::Ref<LabelScorer>& scorer);

    void                           reset() override;
    void                           signalNoMoreFeatures() override;
    ScoringContextRef              getInitialScoringContext() override;
    ScoringContextRef              extendedScoringContext(Request const& request) override;
    void                           addInput(SharedDataHolder const& input, size_t featureSize) override;
    void                           addInput(std::vector<f32> const& input) override;
    void                           addInputs(SharedDataHolder const& input, size_t timeSize, size_t featureSize) override;
    std::optional<ScoreWithTime>   computeScoreWithTime(Request const& request) override;
    std::optional<ScoresWithTimes> computeScoresWithTimes(std::vector<Request> const& requests) override;

protected:
    Core::Ref<LabelScorer> scorer_;
    Score                  scale_;
};

}  // namespace Nn

#endif  // SCALED_LABEL_SCORER_HH
