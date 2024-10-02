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
#include <Flow/Timestamp.hh>
#include <Mm/Module.hh>
#include <utility>
#include "LabelHistory.hh"
#include "LabelScorer.hh"

namespace Nn {

/*
 * =============================
 * === Decoder =================
 * =============================
 */

Decoder::Decoder(const Core::Configuration& config)
        : Core::Component(config), encoderOutputBuffer_(), segmentEnd_(false), timestamps_() {
}

void Decoder::reset() {
    encoderOutputBuffer_.clear();
    segmentEnd_ = false;
    timestamps_.clear();
}

const std::vector<Flow::Timestamp>& Decoder::getTimestamps() const {
    return timestamps_;
}

void Decoder::addEncoderOutput(FeatureVectorRef encoderOutput) {
    encoderOutputBuffer_.push_back(encoderOutput);
}

void Decoder::signalNoMoreEncoderOutputs() {
    segmentEnd_ = true;
}

std::optional<std::pair<std::vector<Score>, Core::CollapsedVector<Speech::TimeframeIndex>>> Decoder::getScoresWithTime(const std::vector<LabelScorer::Request>& requests) {
    std::vector<Score>                            scores;
    Core::CollapsedVector<Search::TimeframeIndex> timeframes;

    scores.reserve(requests.size());
    timeframes.reserve(requests.size());
    for (auto& request : requests) {
        auto score_time = getScoreWithTime(request);
        if (not score_time.has_value()) {
            return {};
        }
        scores.push_back(score_time->first);
        timeframes.push_back(score_time->second);
    }

    return std::make_pair(scores, timeframes);
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

std::optional<std::pair<Score, Speech::TimeframeIndex>> NoOpDecoder::getScoreWithTime(const LabelScorer::Request request) {
    const auto& stepHistory = dynamic_cast<const StepLabelHistory&>(*request.history);
    if (encoderOutputBuffer_.size() <= stepHistory.currentStep) {
        return {};
    }
    while (stepHistory.currentStep >= timestamps_.size()) {
        timestamps_.push_back(Flow::Timestamp(*encoderOutputBuffer_.at(timestamps_.size())));
    }

    return std::make_pair(encoderOutputBuffer_.at(stepHistory.currentStep)->at(request.nextToken), stepHistory.currentStep);
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
    timestamps_.push_back(Flow::Timestamp(encoderOutput));
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
    return Core::ref(new StepLabelHistory());
}

void LegacyFeatureScorerDecoder::extendHistory(LabelScorer::Request request) {
    auto stepHistory = dynamic_cast<StepLabelHistory*>(request.history.get());
    ++stepHistory->currentStep;
}

std::optional<std::pair<Score, Speech::TimeframeIndex>> LegacyFeatureScorerDecoder::getScoreWithTime(const LabelScorer::Request request) {
    const auto& stepHistory = dynamic_cast<const StepLabelHistory&>(*request.history);
    if (scoreCache_.size() <= stepHistory.currentStep) {
        return {};
    }
    // Retrieve score from score cache at the right index
    auto cachedScore = scoreCache_.at(stepHistory.currentStep);
    return std::make_pair(cachedScore->score(request.nextToken), stepHistory.currentStep);
}

}  // namespace Nn
