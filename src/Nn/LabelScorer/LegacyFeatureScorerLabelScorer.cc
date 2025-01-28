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

#include "LegacyFeatureScorerLabelScorer.hh"
#include <Mm/Module.hh>

namespace Nn {

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

}  // namespace Nn
