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

void LabelScorer::addInput(const std::vector<f32>& input) {
    addInput(input.data(), input.size());
}

void LabelScorer::addInput(const FeatureVectorRef input) {
    addInput(*input);
}

void LabelScorer::addInput(const Core::Ref<const Speech::Feature> input) {
    addInput(*input->mainStream());
}

void LabelScorer::addInputs(const f32* input, size_t T, size_t F) {
    for (size_t t = 0ul; t < T; ++t) {
        addInput(input + t * F, F);
    }
}

void LabelScorer::addInputs(const std::vector<std::vector<f32>>& inputs) {
    for (const auto& input : inputs) {
        addInput(input);
    }
}

void LabelScorer::addInputs(const std::vector<FeatureVectorRef>& inputs) {
    for (auto input : inputs) {
        addInput(input);
    }
}

void LabelScorer::addInputs(const std::vector<Core::Ref<const Speech::Feature>>& inputs) {
    for (auto input : inputs) {
        addInput(input);
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