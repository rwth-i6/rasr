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

/*
 * =============================
 * === LabelScorer =============
 * =============================
 */

LabelScorer::LabelScorer(const Core::Configuration& config)
        : Core::Component(config) {}

void LabelScorer::addInput(std::vector<f32> const& input) {
    // The custom deleter ties the lifetime of vector `input` to the lifetime
    // of `dataPtr` by capturing the `inputWrapper` by value.
    // This makes sure that the underlying data isn't invalidated prematurely.
    auto inputWrapper = std::make_shared<std::vector<f32>>(input);
    auto dataPtr      = std::shared_ptr<const f32>(
            inputWrapper->data(),
            [inputWrapper](const f32*) mutable {});
    addInput(dataPtr, input.size());
}

void LabelScorer::addInputs(std::shared_ptr<const f32> const& input, size_t T, size_t F) {
    for (size_t t = 0ul; t < T; ++t) {
        // Use aliasing constructor to create sub-`shared_ptr`s that share ownership with the original one but point to different memory locations
        addInput(std::shared_ptr<const f32>(input, input.get() + t * F), F);
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
