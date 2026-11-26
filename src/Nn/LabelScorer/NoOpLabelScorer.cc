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
        : Core::Component(config),
          Precursor(config, TransitionPresetType::CTC) {}

Core::Status StepwiseNoOpLabelScorer::getInitialScoringContext(ScoringContextRef& initialScoringContext) {
    initialScoringContext = Core::ref(new StepScoringContext());
    return Core::Status();
}

Core::Status StepwiseNoOpLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType, ScoringContextRef& extendedContext) {
    StepScoringContextRef stepScoringContext(dynamic_cast<StepScoringContext const*>(scoringContext.get()));
    extendedContext = Core::ref(new StepScoringContext(stepScoringContext->currentStep + 1));
    return Core::Status();
}

Core::Status StepwiseNoOpLabelScorer::computeScoreAccessor(ScoringContextRef scoringContext, std::optional<ScoreAccessorRef>& scoreAccessor) {
    StepScoringContextRef stepScoringContext(dynamic_cast<StepScoringContext const*>(scoringContext.get()));
    auto                  input = getInput(stepScoringContext->currentStep);
    if (not input) {
        countUncomputedAccessors();
        scoreAccessor = std::nullopt;
        return Core::Status();
    }

    scoreAccessor = Core::ref(new DataViewScoreAccessor(*input, stepScoringContext->currentStep));
    return Core::Status();
}

size_t StepwiseNoOpLabelScorer::getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const {
    auto minInputIndex = Core::Type<size_t>::max;
    for (auto const& context : activeContexts.internalData()) {
        StepScoringContextRef stepHistory(dynamic_cast<StepScoringContext const*>(context.get()));
        minInputIndex = std::min(minInputIndex, static_cast<size_t>(stepHistory->currentStep));
    }

    return minInputIndex;
}

}  // namespace Nn
