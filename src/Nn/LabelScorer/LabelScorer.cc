/** Copyright 2020 RWTH Aachen University. All rights reserved.
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

namespace Nn {

/*
 * =============================
 * === LabelScorer =============
 * =============================
 */

LabelScorer::LabelScorer(const Core::Configuration& config)
        : Core::Component(config) {}

std::optional<LabelScorer::ScoresWithTimes> LabelScorer::getScoresWithTimes(const std::vector<LabelScorer::Request>& requests) {
    ScoresWithTimes result;

    result.scores.reserve(requests.size());
    result.timesteps.reserve(requests.size());
    for (auto& request : requests) {
        auto singleResult = getScoreWithTime(request);
        if (not singleResult.has_value()) {
            return {};
        }
        result.scores.push_back(singleResult->score);
        result.timesteps.push_back(singleResult->timeframe);
    }

    return result;
}

void LabelScorer::addInput(std::vector<f32> const& input) {
    // The custom deleter ties the lifetime of vector `input` to the lifetime
    // of `dataPtr` by capturing the `inputWrapper` by value.
    // This makes sure that the underlying data isn't invalidated prematurely.
    auto inputWrapper = std::make_shared<std::vector<f32>>(input);
    auto dataPtr      = std::shared_ptr<const f32[]>(
            inputWrapper->data(),
            [inputWrapper](const f32*) mutable {});
    addInput(dataPtr, input.size());
}

void LabelScorer::addInputs(std::shared_ptr<const f32[]> const& input, size_t T, size_t F) {
    for (size_t t = 0ul; t < T; ++t) {
        // Use aliasing constructor to create sub-`shared_ptr`s that share ownership with the original one but point to different memory locations
        addInput(std::shared_ptr<const f32[]>(input, input.get() + t * F), F);
    }
}

/*
 * =============================
 * === BufferedLabelScorer =====
 * =============================
 */
BufferedLabelScorer::BufferedLabelScorer(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config),
          featureSize_(Core::Type<size_t>::max),
          inputBuffer_(),
          featuresMissing_(true) {
}

void BufferedLabelScorer::reset() {
    inputBuffer_.clear();
    featuresMissing_ = true;
    featureSize_     = Core::Type<size_t>::max;
}

void BufferedLabelScorer::signalNoMoreFeatures() {
    featuresMissing_ = false;
}

void BufferedLabelScorer::addInput(std::shared_ptr<const f32[]> const& input, size_t F) {
    if (featureSize_ == Core::Type<size_t>::max) {
        featureSize_ = F;
    }
    else if (featureSize_ != F) {
        error() << "Label scorer received incompatible feature size " << F << "; was set to " << featureSize_ << " before.";
    }

    inputBuffer_.push_back(input);
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

ScaledLabelScorer::ScaledLabelScorer(const Core::Configuration& config, const Core::Ref<LabelScorer>& scorer)
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

ScoringContextRef ScaledLabelScorer::extendedScoringContext(Request request) {
    return scorer_->extendedScoringContext(request);
}

void ScaledLabelScorer::addInput(std::shared_ptr<const f32[]> const& input, size_t F) {
    scorer_->addInput(input, F);
}

void ScaledLabelScorer::addInput(std::vector<f32> const& input) {
    scorer_->addInput(input);
}

void ScaledLabelScorer::addInputs(std::shared_ptr<const f32[]> const& input, size_t T, size_t F) {
    scorer_->addInputs(input, T, F);
}

std::optional<LabelScorer::ScoreWithTime> ScaledLabelScorer::getScoreWithTime(const Request request) {
    auto result = scorer_->getScoreWithTime(request);
    if (scale_ == 1.0f) {
        return result;
    }

    if (result.has_value()) {
        result.value().score *= scale_;
    }

    return result;
}

std::optional<LabelScorer::ScoresWithTimes> ScaledLabelScorer::getScoresWithTimes(const std::vector<Request>& requests) {
    auto result = scorer_->getScoresWithTimes(requests);
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

StepwiseNoOpLabelScorer::StepwiseNoOpLabelScorer(const Core::Configuration& config)
        : Core::Component(config), Precursor(config) {}

ScoringContextRef StepwiseNoOpLabelScorer::getInitialScoringContext() {
    return Core::ref(new StepScoringContext());
}

ScoringContextRef StepwiseNoOpLabelScorer::extendedScoringContext(LabelScorer::Request request) {
    StepScoringContextRef stepHistory(dynamic_cast<const StepScoringContext*>(request.context.get()));
    return Core::ref(new StepScoringContext(stepHistory->currentStep + 1));
}

std::optional<LabelScorer::ScoreWithTime> StepwiseNoOpLabelScorer::getScoreWithTime(const LabelScorer::Request request) {
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

LegacyFeatureScorerLabelScorer::LegacyFeatureScorerLabelScorer(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config),
          featureScorer_(Mm::Module::instance().createFeatureScorer(config)),
          scoreCache_() {}

void LegacyFeatureScorerLabelScorer::reset() {
    featureScorer_.reset();
    scoreCache_.clear();
}

void LegacyFeatureScorerLabelScorer::addInput(std::shared_ptr<const f32[]> const& input, size_t F) {
    std::vector<f32> featureVector(F);
    std::copy(input.get(), input.get() + F, featureVector.begin());
    addInput(featureVector);
}

void LegacyFeatureScorerLabelScorer::addInput(const std::vector<f32>& input) {
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

ScoringContextRef LegacyFeatureScorerLabelScorer::extendedScoringContext(LabelScorer::Request request) {
    StepScoringContextRef stepHistory(dynamic_cast<const StepScoringContext*>(request.context.get()));
    return Core::ref(new StepScoringContext(stepHistory->currentStep + 1));
}

std::optional<LabelScorer::ScoreWithTime> LegacyFeatureScorerLabelScorer::getScoreWithTime(const LabelScorer::Request request) {
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

CombineLabelScorer::CombineLabelScorer(const Core::Configuration& config)
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

ScoringContextRef CombineLabelScorer::extendedScoringContext(Request request) {
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

void CombineLabelScorer::addInput(std::shared_ptr<const f32[]> const& input, size_t F) {
    for (auto& scorer : scorers_) {
        scorer->addInput(input, F);
    }
}

void CombineLabelScorer::addInput(std::vector<f32> const& input) {
    for (auto& scorer : scorers_) {
        scorer->addInput(input);
    }
}

void CombineLabelScorer::addInputs(std::shared_ptr<const f32[]> const& input, size_t T, size_t F) {
    for (auto& scorer : scorers_) {
        scorer->addInputs(input, T, F);
    }
}

std::optional<LabelScorer::ScoreWithTime> CombineLabelScorer::getScoreWithTime(const Request request) {
    ScoreWithTime accumResult{0.0, 0};

    auto combineContext = dynamic_cast<const CombineScoringContext*>(request.context.get());

    auto scorerIt  = scorers_.begin();
    auto contextIt = combineContext->scoringContexts.begin();

    for (; scorerIt != scorers_.end(); ++scorerIt, ++contextIt) {
        Request subRequest{*contextIt, request.nextToken, request.transitionType};
        auto    result = (*scorerIt)->getScoreWithTime(subRequest);
        if (!result) {
            return {};
        }
        accumResult.score += result->score;
        accumResult.timeframe = std::max(accumResult.timeframe, result->timeframe);
    }

    return accumResult;
}

std::optional<LabelScorer::ScoresWithTimes> CombineLabelScorer::getScoresWithTimes(const std::vector<Request>& requests) {
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

        auto subResults = scorers_[scorerIdx]->getScoresWithTimes(subRequests);
        if (!subResults) {
            return {};
        }

        Core::CollapsedVector<Speech::TimeframeIndex> newTimesteps;
        for (size_t requestIdx = 0ul; requestIdx < requests.size(); ++requestIdx) {
            accumResult.scores[requestIdx] += subResults->scores[requestIdx];
            newTimesteps.push_back(std::max(accumResult.timesteps[requestIdx], subResults->timesteps[requestIdx]));
        }
        accumResult.timesteps = newTimesteps;
    }

    return accumResult;
}

}  // namespace Nn
