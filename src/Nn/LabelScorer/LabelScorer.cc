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
#include <Flow/Timestamp.hh>
#include <Mm/Module.hh>
#include "Mm/Feature.hh"

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

/*
 * =============================
 * === BufferedLabelScorer =====
 * =============================
 */
BufferedLabelScorer::BufferedLabelScorer(const Core::Configuration& config)
        : Core::Component(config),
          LabelScorer(config),
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

void BufferedLabelScorer::addInput(std::shared_ptr<const f32> const& input, size_t F) {
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

void LegacyFeatureScorerLabelScorer::addInput(std::shared_ptr<const f32> const& input, size_t F) {
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

}  // namespace Nn
