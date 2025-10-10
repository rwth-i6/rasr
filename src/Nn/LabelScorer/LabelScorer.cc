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

const Core::ParameterStringVector LabelScorer::paramIgnoredTransitionTypes(
        "ignored-transition-types",
        "Transition types that should be ignored by the label scorer (i.e. get assigned score 0 and do not affect the ScoringContext)",
        ",");

LabelScorer::LabelScorer(const Core::Configuration& config)
        : Core::Component(config),
          ignoredTransitionTypes_() {
    auto ignoredTransitionTypeStrings = paramIgnoredTransitionTypes(config);
    for (auto const& transitionTypeString : ignoredTransitionTypeStrings) {
        auto it = std::find_if(transitionTypeArray_.begin(),
                               transitionTypeArray_.end(),
                               [&](auto const& entry) { return entry.first == transitionTypeString; });
        if (it != transitionTypeArray_.end()) {
            ignoredTransitionTypes_.insert(it->second);
        }
        else {
            error() << "Ignored transition type name '" << transitionTypeString << "' is not a valid identifier";
        }
    }
}

void LabelScorer::addInputs(DataView const& input, size_t nTimesteps) {
    auto featureSize = input.size() / nTimesteps;
    for (size_t t = 0ul; t < nTimesteps; ++t) {
        addInput({input, featureSize, t * featureSize});
    }
}

ScoringContextRef LabelScorer::extendedScoringContext(Request const& request) {
    if (ignoredTransitionTypes_.contains(request.transitionType)) {
        return request.context;
    }
    return extendedScoringContextInternal(request);
}

std::optional<LabelScorer::ScoreWithTime> LabelScorer::computeScoreWithTime(Request const& request) {
    if (ignoredTransitionTypes_.contains(request.transitionType)) {
        return ScoreWithTime{0.0, 0};
    }
    return computeScoreWithTimeInternal(request);
}

std::optional<LabelScorer::ScoresWithTimes> LabelScorer::computeScoresWithTimes(std::vector<LabelScorer::Request> const& requests) {
    // First, collect all requests for which the transition type is not ignored
    std::vector<Request> nonIgnoredRequests;
    nonIgnoredRequests.reserve(requests.size());

    std::vector<size_t> nonIgnoredRequestIndices;
    nonIgnoredRequestIndices.reserve(requests.size());

    for (size_t requestIndex = 0ul; requestIndex < requests.size(); ++requestIndex) {
        auto const& request = requests[requestIndex];
        if (not ignoredTransitionTypes_.contains(request.transitionType)) {
            nonIgnoredRequests.push_back(request);
            nonIgnoredRequestIndices.push_back(requestIndex);
        }
    }

    // Compute scores for non-ignored requests
    auto nonIgnoredResults = computeScoresWithTimesInternal(nonIgnoredRequests);
    if (not nonIgnoredResults) {
        return {};
    }

    // Interleave actual results with 0 scores for requests with ignored transition types
    ScoresWithTimes result{{requests.size(), 0.0}, {requests.size(), 0}};
    for (size_t i = 0ul; i < nonIgnoredRequestIndices.size(); ++i) {
        auto requestResult          = nonIgnoredResults[i];
        auto requestIndex           = nonIgnoredRequestIndices[i];
        result.scores[requestIndex] = requestResult.score;
        result.timeframes.set(requestIndex, requestResult.timeframe);
    }

    return result;
}

std::optional<LabelScorer::ScoresWithTimes> LabelScorer::computeScoresWithTimesInternal(std::vector<LabelScorer::Request> const& requests) {
    if (requests.empty()) {
        return ScoresWithTimes{};
    }

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
