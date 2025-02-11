/** Copyright 2024 RWTH Aachen University. All rights reserved.
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

#include "ScaledLabelScorer.hh"

namespace Nn {

Core::ParameterFloat ScaledLabelScorer::paramScale(
        "scale",
        "Scores of the label scorer are scaled by this factor",
        1.0f);

ScaledLabelScorer::ScaledLabelScorer(Core::Configuration const& config, Core::Ref<LabelScorer> const& scorer)
        : Core::Component(config),
          Precursor(config),
          scorer_(scorer),
          scale_(paramScale(config)) {
    log() << "Create scaled label scorer with scale " << scale_;
}

void ScaledLabelScorer::reset() {
    scorer_->reset();
}

void ScaledLabelScorer::signalNoMoreFeatures() {
    scorer_->signalNoMoreFeatures();
}

ScoringContextRef ScaledLabelScorer::getInitialScoringContext() {
    return scorer_->getInitialScoringContext();
}

ScoringContextRef ScaledLabelScorer::extendedScoringContext(Request const& request) {
    return scorer_->extendedScoringContext(request);
}

void ScaledLabelScorer::addInput(SharedDataHolder const& input, size_t featureSize) {
    scorer_->addInput(input, featureSize);
}

void ScaledLabelScorer::addInputs(SharedDataHolder const& input, size_t timeSize, size_t featureSize) {
    scorer_->addInputs(input, timeSize, featureSize);
}

std::optional<LabelScorer::ScoreWithTime> ScaledLabelScorer::computeScoreWithTime(Request const& request) {
    auto result = scorer_->computeScoreWithTime(request);
    if (scale_ == 1.0f) {
        return result;
    }

    if (result.has_value()) {
        result.value().score *= scale_;
    }

    return result;
}

std::optional<LabelScorer::ScoresWithTimes> ScaledLabelScorer::computeScoresWithTimes(std::vector<Request> const& requests) {
    auto result = scorer_->computeScoresWithTimes(requests);
    if (scale_ == 1.0f) {
        return result;
    }

    if (result.has_value()) {
        std::transform(result.value().scores.begin(),
                       result.value().scores.end(),
                       result.value().scores.begin(),
                       [this](Score s) { return s * scale_; });
    }

    return result;
}

}  // namespace Nn
