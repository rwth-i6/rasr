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

#include "Decoder.hh"
#include <Core/ReferenceCounting.hh>
#include <Mm/Module.hh>
#include "LabelHistory.hh"
#include "LabelScorer.hh"

namespace Nn {

/*
 * =============================
 * === Decoder =================
 * =============================
 */

Decoder::Decoder(const Core::Configuration& config)
        : Core::Component(config) {
}

void Decoder::reset() {
    encoderOutputBuffer_.clear();
    segmentEnd_ = false;
}

void Decoder::addEncoderOutput(FeatureVectorRef encoderOutput) {
    encoderOutputBuffer_.push_back(encoderOutput);
}

void Decoder::signalNoMoreEncoderOutputs() {
    segmentEnd_ = true;
}

/*
 * =============================
 * === NoOpDecoder =============
 * =============================
 */

NoOpDecoder::NoOpDecoder(const Core::Configuration& config)
        : Core::Component(config), Precursor(config) {}

Core::Ref<LabelHistory> NoOpDecoder::getStartHistory() {
    StepLabelHistory startHistory;
    startHistory.currentStep = 0ul;
    return Core::ref(&startHistory);
}

void NoOpDecoder::extendHistory(LabelScorer::Request& request) {
    auto stepHistory = dynamic_cast<StepLabelHistory*>(request.history.get());
    ++stepHistory->currentStep;
}

std::optional<Score> NoOpDecoder::getScore(const LabelScorer::Request& request) {
    const auto& stepHistory = dynamic_cast<const StepLabelHistory&>(*request.history);
    if (encoderOutputBuffer_.size() <= stepHistory.currentStep) {
        return {};
    }
    return encoderOutputBuffer_.at(stepHistory.currentStep)->at(request.nextToken);
}

/*
 * =============================
 * = LegacyFeatureScorerDecoder
 * =============================
 */

LegacyFeatureScorerDecoder::LegacyFeatureScorerDecoder(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config),
          featureScorer_(Mm::Module::instance().createFeatureScorer(config)),
          scoreCache_() {}

void LegacyFeatureScorerDecoder::reset() {
    Precursor::reset();
    featureScorer_.reset();
    scoreCache_.clear();
}

void LegacyFeatureScorerDecoder::addEncoderOutput(FeatureVectorRef encoderOutput) {
    auto feature = Core::ref(new Mm::Feature(*encoderOutput));
    if (featureScorer_->isBuffered() and not featureScorer_->bufferFilled()) {
        featureScorer_->addFeature(feature);
    }
    else {
        scoreCache_.push_back(featureScorer_->getScorer(feature));
    }
}

void LegacyFeatureScorerDecoder::signalNoMoreEncoderOutputs() {
    Precursor::signalNoMoreEncoderOutputs();
    while (!featureScorer_->bufferEmpty()) {
        scoreCache_.push_back(featureScorer_->flush());
    }
}

Core::Ref<LabelHistory> LegacyFeatureScorerDecoder::getStartHistory() {
    StepLabelHistory startHistory;
    startHistory.currentStep = 0ul;
    return Core::ref(&startHistory);
}

void LegacyFeatureScorerDecoder::extendHistory(LabelScorer::Request& request) {
    auto stepHistory = dynamic_cast<StepLabelHistory*>(request.history.get());
    ++stepHistory->currentStep;
}

std::optional<Score> LegacyFeatureScorerDecoder::getScore(const LabelScorer::Request& request) {
    const auto& stepHistory = dynamic_cast<const StepLabelHistory&>(*request.history);
    if (scoreCache_.size() <= stepHistory.currentStep) {
        return {};
    }
    // Retrieve score from score cache at the right index
    return scoreCache_.at(stepHistory.currentStep)->score(request.nextToken);
}

}  // namespace Nn
