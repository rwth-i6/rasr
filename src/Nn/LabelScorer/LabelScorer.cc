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

namespace Nn {

LabelScorer::LabelScorer(Core::Configuration const& config)
        : Core::Component(config) {}

void LabelScorer::addInput(std::vector<f32> const& input) {
    addInput(input, input.size());
}

void LabelScorer::addInputs(SharedDataHolder const& input, size_t timeSize, size_t featureSize) {
    for (size_t t = 0ul; t < timeSize; ++t) {
        // Use aliasing constructor to create sub-`shared_ptr`s that share ownership with the original one but point to different memory locations
        addInput({input, t * featureSize}, featureSize);
    }
}

std::optional<LabelScorer::ScoresWithTimes> LabelScorer::computeScoresWithTimes(std::vector<LabelScorer::Request> const& requests) {
    // By default, just loop over the non-batched `computeScoreWithTime` and collect the results
    ScoresWithTimes result;

    result.scores.reserve(requests.size());
    result.timeframes.reserve(requests.size());
    for (auto& request : requests) {
        auto singleResult = computeScoreWithTime(request);
        if (not singleResult.has_value()) {
            return {};
        }
        result.scores.push_back(singleResult->score);
        result.timeframes.push_back(singleResult->timeframe);
    }

    return result;
}

}  // namespace Nn
