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

namespace Nn {

/*
 * =============================
 * === LabelScorer =============
 * =============================
 */

LabelScorer::LabelScorer(const Core::Configuration& config)
        : Core::Component(config) {}

void LabelScorer::addInput(Core::Ref<const Speech::Feature> input) {
    addInput(Flow::dataPtr(new FeatureVector(*input->mainStream(), input->timestamp().startTime(), input->timestamp().endTime())));
}

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

/*
 * =============================
 * === BufferedLabelScorer =====
 * =============================
 */
BufferedLabelScorer::BufferedLabelScorer(const Core::Configuration& config)
        : Core::Component(config), LabelScorer(config), inputBuffer_(), featuresMissing_(true), timestamps_() {
}

void BufferedLabelScorer::reset() {
    inputBuffer_.clear();
    featuresMissing_ = true;
    timestamps_.clear();
}

void BufferedLabelScorer::signalNoMoreFeatures() {
    featuresMissing_ = false;
}

const std::vector<Flow::Timestamp>& BufferedLabelScorer::getTimestamps() const {
    return timestamps_;
}

void BufferedLabelScorer::addInput(FeatureVectorRef input) {
    inputBuffer_.push_back(input);
}

/*
 * =============================
 * == StepwiseNoOpLabelScorer ==
 * =============================
 */

StepwiseNoOpLabelScorer::StepwiseNoOpLabelScorer(const Core::Configuration& config)
        : Core::Component(config), Precursor(config) {}

void StepwiseNoOpLabelScorer::addInput(FeatureVectorRef input) {
    Precursor::addInput(input);
    timestamps_.push_back(Flow::Timestamp(*input));
}

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

    return ScoreWithTime{inputBuffer_.at(stepHistory->currentStep)->at(request.nextToken), stepHistory->currentStep};
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

const std::vector<Flow::Timestamp>& LegacyFeatureScorerLabelScorer::getTimestamps() const {
    return timestamps_;
}

void LegacyFeatureScorerLabelScorer::addInput(FeatureVectorRef encoderOutput) {
    auto feature = Core::ref(new Mm::Feature(*encoderOutput));
    timestamps_.push_back(Flow::Timestamp(encoderOutput));
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
