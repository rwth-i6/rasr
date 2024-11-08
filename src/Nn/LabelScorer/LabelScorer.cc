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

#include "LabelScorer.hh"
#include <Flow/Timestamp.hh>

namespace Nn {

/*
 * =============================
 * === LabelScorer =============
 * =============================
 */

LabelScorer::LabelScorer(const Core::Configuration& config)
        : Core::Component(config) {}

void LabelScorer::addInput(Core::Ref<const Speech::Feature> input) {
    addInput(Flow::dataPtr(new FeatureVector(*input->mainStream(), input->timestamp().startTime(), input->timestamp().endTime())));
}

void LabelScorer::addInputs(const std::vector<FeatureVectorRef>& inputs) {
    for (const auto& input : inputs) {
        addInput(input);
    }
}

void LabelScorer::addInputs(const std::vector<Core::Ref<const Speech::Feature>>& inputs) {
    for (const auto& input : inputs) {
        addInput(input);
    }
}

std::optional<LabelScorer::ScoresWithTimes> LabelScorer::getScoresWithTimes(const std::vector<LabelScorer::Request>& requests) {
    // By default, just loop over the non-batched `getScoreWithTime` and collect the results
    ScoresWithTimes result;

    result.scores.reserve(requests.size());
    result.timesteps.reserve(requests.size());
    for (auto& request : requests) {
        auto singleResult = getScoreWithTime(request);
        if (not singleResult.has_value()) {
            return {};
        }
        result.scores.push_back(singleResult->score);
        result.timesteps.push_back(singleResult->timestep);
    }

    return result;
}

}  // namespace Nn
