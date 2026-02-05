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

#include "ScaledLabelScorer.hh"

namespace Nn {

const Core::ParameterFloat ScaledLabelScorer::paramScale(
        "scale",
        "Scale used to multiply the sub-scorer scores.",
        1.0);

ScaledLabelScorer::ScaledLabelScorer(Core::Configuration const& config, Core::Ref<LabelScorer> const& scorer)
        : Core::Component(config),
          LabelScorer(config, TransitionPresetType::ALL),
          scorer_(scorer),
          scale_(paramScale(config)) {
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

void ScaledLabelScorer::cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) {
    scorer_->cleanupCaches(activeContexts);
}

void ScaledLabelScorer::addInput(DataView const& input) {
    scorer_->addInput(input);
}

void ScaledLabelScorer::addInputs(DataView const& input, size_t nTimesteps) {
    scorer_->addInputs(input, nTimesteps);
}

ScoringContextRef ScaledLabelScorer::extendedScoringContextInternal(Request const& request) {
    return scorer_->extendedScoringContext(request);
}

std::optional<LabelScorer::ScoreWithTime> ScaledLabelScorer::computeScoreWithTimeInternal(Request const& request) {
    auto result = scorer_->computeScoreWithTime(request);
    if (result and scale_ != 1) {
        result->score *= scale_;
    }
    return result;
}

std::optional<LabelScorer::ScoresWithTimes> ScaledLabelScorer::computeScoresWithTimesInternal(std::vector<LabelScorer::Request> const& requests) {
    auto result = scorer_->computeScoresWithTimes(requests);
    if (result and scale_ != 1) {
        for (auto& score : result->scores) {
            score *= scale_;
        }
    }
    return result;
}

}  // namespace Nn
