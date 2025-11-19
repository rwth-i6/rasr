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

#include "CombineLabelScorer.hh"
#include <Nn/Module.hh>

namespace Nn {

Core::ParameterInt CombineLabelScorer::paramNumLabelScorers(
        "num-scorers", "Number of label scorers to combine", 1, 1);

Core::ParameterFloat CombineLabelScorer::paramScale(
        "scale", "Scores of a sub-label-scorer are scaled by this factor", 1.0);

CombineLabelScorer::CombineLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::ALL) {
    size_t numLabelScorers = paramNumLabelScorers(config);
    for (size_t i = 0ul; i < numLabelScorers; ++i) {
        Core::Configuration subConfig = select(std::string("scorer-") + std::to_string(i + 1));
        scaledScorers_.push_back({Nn::Module::instance().labelScorerFactory().createLabelScorer(subConfig), static_cast<Score>(paramScale(subConfig))});
    }
}

void CombineLabelScorer::reset() {
    for (auto& scaledScorer : scaledScorers_) {
        scaledScorer.scorer->reset();
    }
}

void CombineLabelScorer::signalNoMoreFeatures() {
    for (auto& scaledScorer : scaledScorers_) {
        scaledScorer.scorer->signalNoMoreFeatures();
    }
}

ScoringContextRef CombineLabelScorer::getInitialScoringContext() {
    std::vector<ScoringContextRef> scoringContexts;
    scoringContexts.reserve(scaledScorers_.size());

    for (const auto& scaledScorer : scaledScorers_) {
        scoringContexts.push_back(scaledScorer.scorer->getInitialScoringContext());
    }
    return Core::ref(new CombineScoringContext(std::move(scoringContexts)));
}

void CombineLabelScorer::cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) {
    std::vector<const CombineScoringContext*> combineContexts;
    combineContexts.reserve(activeContexts.internalSize());
    for (auto const& activeContext : activeContexts.internalData()) {
        combineContexts.push_back(dynamic_cast<const CombineScoringContext*>(activeContext.get()));
    }

    for (size_t scorerIdx = 0ul; scorerIdx < scaledScorers_.size(); ++scorerIdx) {
        auto const&                              scaledScorer = scaledScorers_[scorerIdx];
        Core::CollapsedVector<ScoringContextRef> subScoringContexts;
        for (auto const& combineContext : combineContexts) {
            subScoringContexts.push_back(combineContext->scoringContexts[scorerIdx]);
        }

        scaledScorer.scorer->cleanupCaches(subScoringContexts);
    }
}

void CombineLabelScorer::addInput(DataView const& input) {
    for (auto& scaledScorer : scaledScorers_) {
        scaledScorer.scorer->addInput(input);
    }
}

void CombineLabelScorer::addInputs(DataView const& input, size_t nTimesteps) {
    for (auto& scaledScorer : scaledScorers_) {
        scaledScorer.scorer->addInputs(input, nTimesteps);
    }
}

ScoringContextRef CombineLabelScorer::extendedScoringContextInternal(Request const& request) {
    auto combineContext = dynamic_cast<const CombineScoringContext*>(request.context.get());

    std::vector<ScoringContextRef> extScoringContexts;
    extScoringContexts.reserve(scaledScorers_.size());

    auto scorerIt  = scaledScorers_.begin();
    auto contextIt = combineContext->scoringContexts.begin();

    for (; scorerIt != scaledScorers_.end(); ++scorerIt, ++contextIt) {
        Request subRequest{*contextIt, request.nextToken, request.transitionType};
        extScoringContexts.push_back(scorerIt->scorer->extendedScoringContext(subRequest));
    }
    return Core::ref(new CombineScoringContext(std::move(extScoringContexts)));
}

std::optional<LabelScorer::ScoreWithTime> CombineLabelScorer::computeScoreWithTimeInternal(Request const& request) {
    // Initialize accumulated result with zero-valued score and timestep
    ScoreWithTime accumResult{0.0, 0};

    auto combineContext = dynamic_cast<const CombineScoringContext*>(request.context.get());

    // Iterate over all the scorers and accumulate their results into `accumResult`
    auto scorerIt  = scaledScorers_.begin();
    auto contextIt = combineContext->scoringContexts.begin();
    for (; scorerIt != scaledScorers_.end(); ++scorerIt, ++contextIt) {
        // Prepare sub-request for the current scorer by extracting the appropriate
        // ScoringContext from the combined ScoringContext
        Request subRequest{*contextIt, request.nextToken, request.transitionType};

        // Run current scorer
        auto result = scorerIt->scorer->computeScoreWithTime(subRequest);
        if (!result) {
            return {};
        }

        // Merge results of current scorer into `accumResult`
        // Scores are weighted sum, timeframes are maximum
        accumResult.score += result->score * scorerIt->scale;
        accumResult.timeframe = std::max(accumResult.timeframe, result->timeframe);
    }

    return accumResult;
}

std::optional<LabelScorer::ScoresWithTimes> CombineLabelScorer::computeScoresWithTimesInternal(std::vector<Request> const& requests) {
    if (requests.empty()) {
        return ScoresWithTimes{};
    }

    // Initialize accumulated results with zero-valued scores and timesteps
    ScoresWithTimes accumResult{std::vector<Score>(requests.size(), 0.0), {requests.size(), 0}};

    // Collect CombineScoringContexts from requests
    std::vector<const CombineScoringContext*> combineContexts;
    combineContexts.reserve(requests.size());
    for (const auto& request : requests) {
        combineContexts.push_back(dynamic_cast<const CombineScoringContext*>(request.context.get()));
    }

    // Iterate over all the scorers and accumulate their results into `accumResult`
    for (size_t scorerIdx = 0ul; scorerIdx < scaledScorers_.size(); ++scorerIdx) {
        // Prepare sub-requests for the current scorer by extracting the appropriate
        // ScoringContext from all the CombineScoringContexts
        std::vector<Request> subRequests;
        subRequests.reserve(requests.size());
        auto requestIt = requests.begin();
        auto contextIt = combineContexts.begin();
        for (; requestIt != requests.end(); ++requestIt, ++contextIt) {
            subRequests.push_back(Request{(*contextIt)->scoringContexts[scorerIdx], requestIt->nextToken, requestIt->transitionType});
        }

        // Run current scorer
        auto subResults = scaledScorers_[scorerIdx].scorer->computeScoresWithTimes(subRequests);
        if (!subResults) {
            return {};
        }

        // Merge results of current scorer into `accumResult`
        // Scores are weighted sum, timeframes are maximum
        Core::CollapsedVector<Speech::TimeframeIndex> newTimeframes;
        for (size_t requestIdx = 0ul; requestIdx < requests.size(); ++requestIdx) {
            accumResult.scores[requestIdx] += subResults->scores[requestIdx] * scaledScorers_[scorerIdx].scale;
            newTimeframes.push_back(std::max(accumResult.timeframes[requestIdx], subResults->timeframes[requestIdx]));
        }
        accumResult.timeframes = newTimeframes;
    }

    return accumResult;
}

size_t CombineLabelScorer::numSubScorers() const {
    return scaledScorers_.size();
}

std::optional<CombineLabelScorer::ScoreWithTime> CombineLabelScorer::computeScoreWithTime(Request const& request, size_t scorerIdx) {
    verify(scorerIdx < scaledScorers_.size());

    auto        combineContext = dynamic_cast<const CombineScoringContext*>(request.context.get());
    auto const& subContext     = combineContext->scoringContexts.at(scorerIdx);

    // Iterate over all the scorers and accumulate their results into `accumResult`
    auto const& scaledScorer = scaledScorers_.at(scorerIdx);

    // Prepare sub-request for the current scorer by extracting the appropriate
    // ScoringContext from the combined ScoringContext
    Request subRequest{subContext, request.nextToken, request.transitionType};

    // Run current scorer
    auto result = scaledScorer.scorer->computeScoreWithTime(subRequest);
    if (!result) {
        return {};
    }

    // Merge results of current scorer into `accumResult`
    // Scores are weighted sum, timeframes are maximum
    return ScoreWithTime{result->score * scaledScorer.scale, result->timeframe};
}

std::optional<CombineLabelScorer::ScoresWithTimes> CombineLabelScorer::computeScoresWithTimes(const std::vector<Request>& requests, size_t scorerIdx) {
    // Collect CombineScoringContexts from requests
    std::vector<const CombineScoringContext*> combineContexts;
    combineContexts.reserve(requests.size());
    for (const auto& request : requests) {
        combineContexts.push_back(dynamic_cast<const CombineScoringContext*>(request.context.get()));
    }

    // Prepare sub-requests for the current scorer by extracting the appropriate
    // ScoringContext from all the CombineScoringContexts
    std::vector<Request> subRequests;
    subRequests.reserve(requests.size());
    auto requestIt = requests.begin();
    auto contextIt = combineContexts.begin();
    for (; requestIt != requests.end(); ++requestIt, ++contextIt) {
        subRequests.push_back(Request{(*contextIt)->scoringContexts[scorerIdx], requestIt->nextToken, requestIt->transitionType});
    }

    // Run scorer
    auto subResults = scaledScorers_[scorerIdx].scorer->computeScoresWithTimes(subRequests);
    if (!subResults) {
        return {};
    }

    // Initialize accumulated results with zero-valued scores and timesteps
    ScoresWithTimes result{std::vector<Score>(requests.size(), 0.0), {requests.size(), 0}};

    // Merge results of current scorer into `accumResult`
    // Scores are weighted sum, timeframes are maximum
    Core::CollapsedVector<Speech::TimeframeIndex> timeframes;
    for (size_t requestIdx = 0ul; requestIdx < requests.size(); ++requestIdx) {
        result.scores[requestIdx] = subResults->scores[requestIdx] * scaledScorers_[scorerIdx].scale;
        result.timeframes.push_back(subResults->timeframes[requestIdx]);
    }

    return result;
}

#ifdef MODULE_PYTHON
void CombineLabelScorer::registerPythonCallback(std::string const& name, pybind11::function const& callback) {
    for (auto& scaledScorer : scaledScorers_) {
        scaledScorer.scorer->registerPythonCallback(name, callback);
    }
}
#endif

}  // namespace Nn
