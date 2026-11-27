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

/*
 * ==================================
 * == FixedTransitionScoreAccessor ==
 * ==================================
 */

FixedTransitionScoreAccessor::FixedTransitionScoreAccessor()
        : transitionScores_() {
    for (auto const& [stringIdentifier, enumValue] : TransitionTypeArray) {
        setScore(enumValue, 0.0);
    }
}

void FixedTransitionScoreAccessor::setScore(TransitionType transitionType, Score score) {
    transitionScores_[static_cast<size_t>(transitionType)] = score;
}

Score FixedTransitionScoreAccessor::getScore(TransitionType transitionType, LabelIndex labelIndex) const {
    return transitionScores_.at(transitionType);
}

TimeframeIndex FixedTransitionScoreAccessor::getTime() const {
    return 0;
}

/*
 * ===========================
 * == TransitionLabelScorer ==
 * ===========================
 */

TransitionLabelScorer::TransitionLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::ALL),
          transitionScoreAccessor_(new FixedTransitionScoreAccessor()) {
    for (auto const& [stringIdentifier, enumValue] : TransitionTypeArray) {
        auto paramName = std::string(stringIdentifier) + "-score";
        transitionScoreAccessor_->setScore(enumValue, Core::ParameterFloat(paramName.c_str(), "", 0.0)(config));
    }
}

Core::Status TransitionLabelScorer::signalNoMoreFeatures() {
    return Core::Status();
}

Core::Status TransitionLabelScorer::getInitialScoringContext(ScoringContextRef& initialScoringContext) {
    initialScoringContext = Core::ref(new ScoringContext());
    return Core::Status();
}

Core::Status TransitionLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType, ScoringContextRef& extendedContext) {
    extendedContext = Core::ref(new ScoringContext());
    return Core::Status();
}

Core::Status TransitionLabelScorer::addInput(DataView const& input) {
    return Core::Status();
}

void TransitionLabelScorer::resetInternal() {}

Core::Status TransitionLabelScorer::computeScoreAccessor(ScoringContextRef scoringContext, std::optional<ScoreAccessorRef>& scoreAccessor) {
    scoreAccessor = transitionScoreAccessor_;
    return Core::Status();
}

}  // namespace Nn
