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

/*
 * =============================
 * === BufferedLabelScorer =====
 * =============================
 */
BufferedLabelScorer::BufferedLabelScorer(const Core::Configuration& config)
        : Core::Component(config), LabelScorer(config), featureSize_(Core::Type<size_t>::max), inputBuffer_(), inputsAreContiguous_(true), featuresMissing_(true) {
}

void BufferedLabelScorer::reset() {
    inputBuffer_.clear();
    inputsAreContiguous_ = true;
    featuresMissing_     = true;
    featureSize_         = Core::Type<size_t>::max;
}

void BufferedLabelScorer::signalNoMoreFeatures() {
    featuresMissing_ = false;
}

void BufferedLabelScorer::addInput(const f32* input, size_t F) {
    if (featureSize_ == Core::Type<size_t>::max) {
        featureSize_ = F;
    }
    else if (featureSize_ != F) {
        error() << "Label scorer received incompatible feature size " << F << "; was set to " << featureSize_ << " before.";
    }

    if (not inputBuffer_.empty() and input != inputBuffer_.back().get() + F) {
        inputsAreContiguous_ = false;
    }
    auto dataPtr = std::shared_ptr<const f32>(input, [](const f32*) {});
    inputBuffer_.push_back(dataPtr);
}

void BufferedLabelScorer::addInput(const std::vector<f32>& input) {
    if (featureSize_ == Core::Type<size_t>::max) {
        featureSize_ = input.size();
    }
    else if (featureSize_ != input.size()) {
        error() << "Label scorer received incompatible feature size " << input.size() << "; was set to " << featureSize_ << " before.";
    }

    inputsAreContiguous_ = false;

    // `dataPtr` contains the underlying pointer `input.data()`.
    // It has a custom deleter that captures a shared pointer to the vector input by value,
    // thus making sure the vector stays alive as long as dataPtr exists and the
    // underlying data isn't invalidated prematurely.
    auto dataPtr = std::shared_ptr<const float>(
            input.data(),
            [vecPtr = std::make_shared<std::vector<f32>>(input)](const f32*) mutable {});
    inputBuffer_.push_back(dataPtr);
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

void LegacyFeatureScorerLabelScorer::addInput(const f32* input, size_t F) {
    std::vector<f32> featureVector(F);
    std::copy(input, input + F, featureVector.begin());
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
