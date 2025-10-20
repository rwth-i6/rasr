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
          transitionScores_() {
    for (auto const& [stringIdentifier, enumValue] : transitionTypeArray_) {
        auto paramName               = std::string(stringIdentifier) + "-score";
        transitionScores_[enumValue] = Core::ParameterFloat(paramName.c_str(), "", 0.0)(config);
    }
}

void TransitionLabelScorer::reset() {}

void TransitionLabelScorer::signalNoMoreFeatures() {}

ScoringContextRef TransitionLabelScorer::getInitialScoringContext() {
    return Core::ref(new StepScoringContext());
}

void TransitionLabelScorer::addInput(DataView const& input) {}

ScoringContextRef TransitionLabelScorer::extendedScoringContextInternal(LabelScorer::Request const& request) {
    return Core::ref(new StepScoringContext());
}

std::optional<LabelScorer::ScoreWithTime> TransitionLabelScorer::computeScoreWithTimeInternal(LabelScorer::Request const& request) {
    LabelScorer::ScoreWithTime result;
    result.score = transitionScores_[request.transitionType];
    result.timeframe = static_cast<Speech::TimeframeIndex>(0);
    return result;
}

std::optional<LabelScorer::ScoresWithTimes> TransitionLabelScorer::computeScoresWithTimesInternal(std::vector<LabelScorer::Request> const& requests) {
    if (requests.empty()) {
        return ScoresWithTimes{};
    }

    LabelScorer::ScoresWithTimes results;
    for (size_t i = 0ul; i < requests.size(); ++i) {
        results.scores.push_back(transitionScores_[requests[i].transitionType]);
        results.timeframes.push_back(static_cast<Speech::TimeframeIndex>(0));
    }
    return results;
}

}  // namespace Nn
