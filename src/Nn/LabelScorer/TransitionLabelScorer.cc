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
          Precursor(config),
          transitionScores_(),
          baseLabelScorer_(Nn::Module::instance().labelScorerFactory().createLabelScorer(select("base-scorer"))) {
    for (size_t idx = 0ul; idx < paramNames.size(); ++idx) {
        transitionScores_[idx] = Core::ParameterFloat(paramNames[idx], "", 0.0)(config);
    }
}

void TransitionLabelScorer::reset() {
    baseLabelScorer_->reset();
}

void TransitionLabelScorer::signalNoMoreFeatures() {
    baseLabelScorer_->signalNoMoreFeatures();
}

ScoringContextRef TransitionLabelScorer::getInitialScoringContext() {
    return baseLabelScorer_->getInitialScoringContext();
}

ScoringContextRef TransitionLabelScorer::extendedScoringContext(LabelScorer::Request const& request) {
    return baseLabelScorer_->extendedScoringContext(request);
}

void TransitionLabelScorer::cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) {
    baseLabelScorer_->cleanupCaches(activeContexts);
}

void TransitionLabelScorer::addInput(DataView const& input) {
    baseLabelScorer_->addInput(input);
}

void TransitionLabelScorer::addInputs(DataView const& input, size_t nTimesteps) {
    baseLabelScorer_->addInputs(input, nTimesteps);
}

std::optional<LabelScorer::ScoreWithTime> TransitionLabelScorer::computeScoreWithTime(LabelScorer::Request const& request) {
    auto result = baseLabelScorer_->computeScoreWithTime(request);
    if (result) {
        result->score += getTransitionScore(request.transitionType);
    }
    return result;
}

std::optional<LabelScorer::ScoresWithTimes> TransitionLabelScorer::computeScoresWithTimes(std::vector<LabelScorer::Request> const& requests) {
    auto results = baseLabelScorer_->computeScoresWithTimes(requests);
    if (results) {
        for (size_t i = 0ul; i < requests.size(); ++i) {
            results->scores[i] += getTransitionScore(requests[i].transitionType);
        }
    }
    return results;
}

LabelScorer::Score TransitionLabelScorer::getTransitionScore(LabelScorer::TransitionType transitionType) const {
    return transitionScores_[transitionTypeToIndex(transitionType)];
}

}  // namespace Nn
