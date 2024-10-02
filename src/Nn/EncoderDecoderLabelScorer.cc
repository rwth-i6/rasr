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

#include "EncoderDecoderLabelScorer.hh"

namespace Nn {

EncoderDecoderLabelScorer::EncoderDecoderLabelScorer(const Core::Configuration& config, const Core::Ref<Encoder> encoder, const Core::Ref<Decoder> decoder)
        : Core::Component(config),
          Nn::LabelScorer(config),
          encoder_(encoder),
          decoder_(decoder) {
}

void EncoderDecoderLabelScorer ::reset() {
    encoder_->reset();
    decoder_->reset();
}

Core::Ref<LabelHistory> EncoderDecoderLabelScorer::getStartHistory() {
    return decoder_->getStartHistory();
}

void EncoderDecoderLabelScorer::extendHistory(Request request) {
    decoder_->extendHistory(request);
}

const std::vector<Flow::Timestamp>& EncoderDecoderLabelScorer::getTimestamps() const {
    return decoder_->getTimestamps();
}

void EncoderDecoderLabelScorer::addInput(FeatureVectorRef input) {
    encoder_->addInput(input);
    encode();
}

void EncoderDecoderLabelScorer::addInput(Core::Ref<const Speech::Feature> input) {
    encoder_->addInput(input);
    encode();
}

void EncoderDecoderLabelScorer::signalNoMoreFeatures() {
    encoder_->signalNoMoreFeatures();
    // Call `encode()` before signaling segment end to the decoder since the decoder
    // is supposed to receive all available encoder outputs before this signal
    encode();
    decoder_->signalNoMoreEncoderOutputs();
}

std::optional<std::pair<Score, Speech::TimeframeIndex>> EncoderDecoderLabelScorer::getScoreWithTime(const LabelScorer::Request request) {
    return decoder_->getScoreWithTime(request);
}

std::optional<std::pair<std::vector<Score>, Core::CollapsedVector<Speech::TimeframeIndex>>> EncoderDecoderLabelScorer::getScoresWithTime(const std::vector<LabelScorer::Request>& requests) {
    return decoder_->getScoresWithTime(requests);
}

void EncoderDecoderLabelScorer::encode() {
    std::optional<FeatureVectorRef> encoderOutput;
    while ((encoderOutput = encoder_->getNextOutput())) {
        decoder_->addEncoderOutput(*encoderOutput);
    }
}

}  // namespace Nn
