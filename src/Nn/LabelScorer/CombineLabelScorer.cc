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

CombineLabelScorer::CombineLabelScorer(Core::Configuration const& config)
        : Core::Component(config), Precursor(config), scorers_() {
    size_t numLabelScorers = paramNumLabelScorers(config);
    for (size_t i = 0ul; i < numLabelScorers; i++) {
        Core::Configuration subConfig = select(std::string("scorer-") + std::to_string(i + 1));
        scorers_.push_back(Nn::Module::instance().createScaledLabelScorer(subConfig));
    }
}

void CombineLabelScorer::reset() {
    for (auto& scorer : scorers_) {
        scorer->reset();
    }
}

void CombineLabelScorer::signalNoMoreFeatures() {
    for (auto& scorer : scorers_) {
        scorer->signalNoMoreFeatures();
    }
}

ScoringContextRef CombineLabelScorer::getInitialScoringContext() {
    std::vector<ScoringContextRef> scoringContexts;
    scoringContexts.reserve(scorers_.size());

    for (const auto& scorer : scorers_) {
        scoringContexts.push_back(scorer->getInitialScoringContext());
    }
    return Core::ref(new CombineScoringContext(std::move(scoringContexts)));
}

ScoringContextRef CombineLabelScorer::extendedScoringContext(Request const& request) {
    auto combineContext = dynamic_cast<const CombineScoringContext*>(request.context.get());

    std::vector<ScoringContextRef> extScoringContexts;
    extScoringContexts.reserve(scorers_.size());

    auto scorerIt  = scorers_.begin();
    auto contextIt = combineContext->scoringContexts.begin();

    for (; scorerIt != scorers_.end(); ++scorerIt, ++contextIt) {
        Request subRequest{*contextIt, request.nextToken, request.transitionType};
        extScoringContexts.push_back((*scorerIt)->extendedScoringContext(subRequest));
    }
    return Core::ref(new CombineScoringContext(std::move(extScoringContexts)));
}

void CombineLabelScorer::addInput(SharedDataHolder const& input, size_t featureSize) {
    for (auto& scorer : scorers_) {
        scorer->addInput(input, featureSize);
    }
}

void CombineLabelScorer::addInput(std::vector<f32> const& input) {
    for (auto& scorer : scorers_) {
        scorer->addInput(input);
    }
}

void CombineLabelScorer::addInputs(SharedDataHolder const& input, size_t timeSize, size_t featureSize) {
    for (auto& scorer : scorers_) {
        scorer->addInputs(input, timeSize, featureSize);
    }
}

std::optional<LabelScorer::ScoreWithTime> CombineLabelScorer::computeScoreWithTime(Request const& request) {
    ScoreWithTime accumResult{0.0, 0};

    auto combineContext = dynamic_cast<const CombineScoringContext*>(request.context.get());

    auto scorerIt  = scorers_.begin();
    auto contextIt = combineContext->scoringContexts.begin();

    for (; scorerIt != scorers_.end(); ++scorerIt, ++contextIt) {
        Request subRequest{*contextIt, request.nextToken, request.transitionType};
        auto    result = (*scorerIt)->computeScoreWithTime(subRequest);
        if (!result) {
            return {};
        }
        accumResult.score += result->score;
        accumResult.timeframe = std::max(accumResult.timeframe, result->timeframe);
    }

    return accumResult;
}

std::optional<LabelScorer::ScoresWithTimes> CombineLabelScorer::computeScoresWithTimes(std::vector<Request> const& requests) {
    ScoresWithTimes accumResult{std::vector<Score>(requests.size(), 0.0f), {requests.size(), 0}};

    std::vector<const CombineScoringContext*> combineContexts;
    combineContexts.reserve(requests.size());
    for (const auto& request : requests) {
        combineContexts.push_back(dynamic_cast<const CombineScoringContext*>(request.context.get()));
    }

    for (size_t scorerIdx = 0ul; scorerIdx < scorers_.size(); ++scorerIdx) {
        std::vector<Request> subRequests;
        subRequests.reserve(requests.size());
        auto requestIt = requests.begin();
        auto contextIt = combineContexts.begin();
        for (; requestIt != requests.end(); ++requestIt, ++contextIt) {
            subRequests.push_back(Request{(*contextIt)->scoringContexts[scorerIdx], requestIt->nextToken, requestIt->transitionType});
        }

        auto subResults = scorers_[scorerIdx]->computeScoresWithTimes(subRequests);
        if (!subResults) {
            return {};
        }

        Core::CollapsedVector<Speech::TimeframeIndex> newTimesteps;
        for (size_t requestIdx = 0ul; requestIdx < requests.size(); ++requestIdx) {
            accumResult.scores[requestIdx] += subResults->scores[requestIdx];
            newTimesteps.push_back(std::max(accumResult.timeframes[requestIdx], subResults->timeframes[requestIdx]));
        }
        accumResult.timeframes = newTimesteps;
    }

    return accumResult;
}

}  // namespace Nn
