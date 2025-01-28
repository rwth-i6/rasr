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
#include <Mm/Module.hh>
#include <Nn/Module.hh>
#include "Nn/LabelScorer/SharedDataHolder.hh"

namespace Nn {

/*
 * =============================
 * === LabelScorer =============
 * =============================
 */

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

/*
 * =============================
 * === ScaledLabelScorer =======
 * =============================
 */

Core::ParameterFloat ScaledLabelScorer::paramScale(
        "scale",
        "Scores of the label scorer are scaled by this factor",
        1.0f);

ScaledLabelScorer::ScaledLabelScorer(Core::Configuration const& config, Core::Ref<LabelScorer> const& scorer)
        : Core::Component(config),
          Precursor(config),
          scorer_(scorer),
          scale_(paramScale(config)) {}

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

void ScaledLabelScorer::addInput(std::vector<f32> const& input) {
    scorer_->addInput(input);
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
/*
 * =============================
 * == StepwiseNoOpLabelScorer ==
 * =============================
 */

StepwiseNoOpLabelScorer::StepwiseNoOpLabelScorer(Core::Configuration const& config)
        : Core::Component(config), Precursor(config) {}

ScoringContextRef StepwiseNoOpLabelScorer::getInitialScoringContext() {
    return Core::ref(new StepScoringContext());
}

ScoringContextRef StepwiseNoOpLabelScorer::extendedScoringContext(LabelScorer::Request const& request) {
    StepScoringContextRef stepHistory(dynamic_cast<const StepScoringContext*>(request.context.get()));
    return Core::ref(new StepScoringContext(stepHistory->currentStep + 1));
}

std::optional<LabelScorer::ScoreWithTime> StepwiseNoOpLabelScorer::computeScoreWithTime(LabelScorer::Request const& request) {
    StepScoringContextRef stepHistory(dynamic_cast<const StepScoringContext*>(request.context.get()));
    if (inputBuffer_.size() <= stepHistory->currentStep) {
        return {};
    }
    if (request.nextToken >= featureSize_) {
        error() << "Tried to get score for token " << request.nextToken << " but only have " << featureSize_ << " scores available.";
    }

    return ScoreWithTime{inputBuffer_.at(stepHistory->currentStep).get()[request.nextToken], stepHistory->currentStep};
}

/*
 * ==================================
 * = LegacyFeatureScorerLabelScorer =
 * ==================================
 */

LegacyFeatureScorerLabelScorer::LegacyFeatureScorerLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          featureScorer_(Mm::Module::instance().createFeatureScorer(config)),
          scoreCache_() {}

void LegacyFeatureScorerLabelScorer::reset() {
    featureScorer_.reset();
    scoreCache_.clear();
}

void LegacyFeatureScorerLabelScorer::addInput(SharedDataHolder const& input, size_t featureSize) {
    std::vector<f32> featureVector(featureSize);
    std::copy(input.get(), input.get() + featureSize, featureVector.begin());
    addInput(featureVector);
}

void LegacyFeatureScorerLabelScorer::addInput(std::vector<f32> const& input) {
    auto feature = Core::ref(new Mm::Feature(input));

    if (featureScorer_->isBuffered() and not featureScorer_->bufferFilled()) {
        featureScorer_->addFeature(feature);
    }
    else {
        scoreCache_.push_back(featureScorer_->getScorer(feature));
    }
}

void LegacyFeatureScorerLabelScorer::signalNoMoreFeatures() {
    while (!featureScorer_->bufferEmpty()) {
        scoreCache_.push_back(featureScorer_->flush());
    }
}

ScoringContextRef LegacyFeatureScorerLabelScorer::getInitialScoringContext() {
    return Core::ref(new StepScoringContext());
}

ScoringContextRef LegacyFeatureScorerLabelScorer::extendedScoringContext(LabelScorer::Request const& request) {
    StepScoringContextRef stepHistory(dynamic_cast<const StepScoringContext*>(request.context.get()));
    return Core::ref(new StepScoringContext(stepHistory->currentStep + 1));
}

std::optional<LabelScorer::ScoreWithTime> LegacyFeatureScorerLabelScorer::computeScoreWithTime(LabelScorer::Request const& request) {
    StepScoringContextRef stepHistory(dynamic_cast<const StepScoringContext*>(request.context.get()));
    if (scoreCache_.size() <= stepHistory->currentStep) {
        return {};
    }
    // Retrieve score from score cache at the right index
    auto cachedScore = scoreCache_.at(stepHistory->currentStep);
    return ScoreWithTime{cachedScore->score(request.nextToken), stepHistory->currentStep};
}

/*
 * ==================================
 * === CombineLabelScorer ===========
 * ==================================
 */

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
