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

#include "TransitionLabelScorer.hh"

#include <Nn/Module.hh>

namespace Nn {

TransitionLabelScorer::TransitionLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::ALL),
          transitionScoreAccessor_(new FixedTransitionScoreAccessor()) {
    for (auto const& [stringIdentifier, enumValue] : TransitionTypeArray) {
        auto paramName = std::string(stringIdentifier) + "-score";
        transitionScoreAccessor_->setScore(enumValue, Core::ParameterFloat(paramName.c_str(), "", 0.0)(config));
    }
}

void TransitionLabelScorer::reset() {}

void TransitionLabelScorer::signalNoMoreFeatures() {}

ScoringContextRef TransitionLabelScorer::getInitialScoringContext() {
    return Core::ref(new ScoringContext());
}

void TransitionLabelScorer::addInput(DataView const& input) {}

ScoringContextRef TransitionLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) {
    return Core::ref(new ScoringContext());
}

std::optional<ScoreAccessorRef> TransitionLabelScorer::getScoreAccessor(ScoringContextRef scoringContext) {
    return transitionScoreAccessor_;
}

}  // namespace Nn
