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

#include "NoOpLabelScorer.hh"
#include "ScoringContext.hh"

namespace Nn {

StepwiseNoOpLabelScorer::StepwiseNoOpLabelScorer(Core::Configuration const& config)
        : Core::Component(config), Precursor(config) {}

ScoringContextRef StepwiseNoOpLabelScorer::getInitialScoringContext() {
    return Core::ref(new StepScoringContext());
}

ScoringContextRef StepwiseNoOpLabelScorer::extendedScoringContext(LabelScorer::Request const& request) {
    StepScoringContextRef stepHistory(dynamic_cast<const StepScoringContext*>(request.context.get()));
    return Core::ref(new StepScoringContext(stepHistory->currentStep + 1));
}

std::optional<LabelScorer::ScoreWithTime> StepwiseNoOpLabelScorer::computeScoreWithTime(LabelScorer::Request const& request) {
    StepScoringContextRef stepHistory(dynamic_cast<const StepScoringContext*>(request.context.get()));
    auto                  input = getInput(stepHistory->currentStep);
    if (not input) {
        return {};
    }

    return ScoreWithTime{(*input)[request.nextToken], stepHistory->currentStep};
}

size_t StepwiseNoOpLabelScorer::getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const {
    auto minInputIndex = Core::Type<size_t>::max;
    for (auto const& context : activeContexts.internalData()) {
        StepScoringContextRef stepHistory(dynamic_cast<const StepScoringContext*>(context.get()));
        minInputIndex = std::min(minInputIndex, static_cast<size_t>(stepHistory->currentStep));
    }

    return minInputIndex;
}

}  // namespace Nn
