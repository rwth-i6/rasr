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
 * This LabelScorer returns predefined transition scores depending on the transition type of each request.
 * The transition scores are all individually specified as config parameters.
 * It should be used together with a main LabelScorer
 */
class TransitionLabelScorer : public LabelScorer {
public:
    using Precursor = LabelScorer;

    TransitionLabelScorer(Core::Configuration const& config);
    virtual ~TransitionLabelScorer() = default;

    // No op
    void reset() override;

    // No op
    void signalNoMoreFeatures() override;

    // Return dummy-context
    ScoringContextRef getInitialScoringContext() override;

    // Return dummy-context
    ScoringContextRef extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) override;

    // No op
    void addInput(DataView const& input) override;

    // Return transition score based on transition type of the request
    std::optional<ScoreAccessorRef> getScoreAccessor(ScoringContextRef scoringContext) override;

private:
    Core::Ref<FixedTransitionScoreAccessor> transitionScoreAccessor_;
};

}  // namespace Nn

#endif  // TRANSITION_LABEL_SCORER_HH
