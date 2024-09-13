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
#include "Flow/Timestamp.hh"
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
    Core::Ref<LabelHistory> hist = Core::ref(new StepLabelHistory());
    return hist;
}

void NoOpDecoder::extendHistory(LabelScorer::Request request) {
    auto& stepHistory = dynamic_cast<StepLabelHistory&>(*request.history);
    ++stepHistory.currentStep;
}

std::optional<LabelScorer::ScoreWithTime> NoOpDecoder::getScoreWithTime(const LabelScorer::Request request) {
    const auto& stepHistory = dynamic_cast<const StepLabelHistory&>(*request.history);
    if (encoderOutputBuffer_.size() <= stepHistory.currentStep) {
        return {};
    }
    auto encoderOutput = encoderOutputBuffer_.at(stepHistory.currentStep);
    return LabelScorer::ScoreWithTime{NegLogScore::fromLogProb(encoderOutput->at(request.nextToken)), Flow::Timestamp(*encoderOutput)};
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
        scoreCache_.push_back(std::make_pair(featureScorer_->getScorer(feature), Flow::Timestamp(encoderOutput)));
    }
}

void LegacyFeatureScorerDecoder::signalNoMoreEncoderOutputs() {
    Precursor::signalNoMoreEncoderOutputs();
    while (!featureScorer_->bufferEmpty()) {
        scoreCache_.push_back(std::make_pair(featureScorer_->flush(), scoreCache_.back().second));
    }
}

Core::Ref<LabelHistory> LegacyFeatureScorerDecoder::getStartHistory() {
    return Core::ref(new StepLabelHistory());
}

void LegacyFeatureScorerDecoder::extendHistory(LabelScorer::Request request) {
    auto stepHistory = dynamic_cast<StepLabelHistory*>(request.history.get());
    ++stepHistory->currentStep;
}

std::optional<LabelScorer::ScoreWithTime> LegacyFeatureScorerDecoder::getScoreWithTime(const LabelScorer::Request request) {
    const auto& stepHistory = dynamic_cast<const StepLabelHistory&>(*request.history);
    if (scoreCache_.size() <= stepHistory.currentStep) {
        return {};
    }
    // Retrieve score from score cache at the right index
    auto cachedScore = scoreCache_.at(stepHistory.currentStep);
    return LabelScorer::ScoreWithTime{
            NegLogScore(cachedScore.first->score(request.nextToken)), cachedScore.second};
}

}  // namespace Nn
