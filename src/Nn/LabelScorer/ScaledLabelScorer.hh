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

#include <Core/Configuration.hh>

#include "LabelScorer.hh"

namespace Nn {

/*
 * Wraps a sub label scorer and scaled all the scores by a given factor
 */
class ScaledLabelScorer : public LabelScorer {
public:
    static const Core::ParameterFloat paramScale;

    ScaledLabelScorer(Core::Configuration const& config, Core::Ref<LabelScorer> const& scorer);

    // Reset sub-scorer
    void reset() override;

    // Forward signal to sub-scorer
    void signalNoMoreFeatures() override;

    // Initial ScoringContext from sub-scorer
    ScoringContextRef getInitialScoringContext() override;

    // Cleanup sub-scorer
    void cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) override;

    // Add input to sub-scorer
    void addInput(DataView const& input) override;

    // Add inputs to sub-scorer
    virtual void addInputs(DataView const& input, size_t nTimesteps) override;

protected:
    // Extended ScoringContext from sub-scorer
    ScoringContextRef extendedScoringContextInternal(Request const& request) override;

    // Compute scaled score of request with sub-scorer
    std::optional<ScoreWithTime> computeScoreWithTimeInternal(Request const& request) override;

    // Compute scaled scores of requests with sub-scorer
    std::optional<ScoresWithTimes> computeScoresWithTimesInternal(std::vector<Request> const& requests) override;

private:
    Core::Ref<LabelScorer> scorer_;
    Score                  scale_;
};

}  // namespace Nn

#endif  // SCALED_LABEL_SCORER_HH
