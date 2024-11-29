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

EncoderDecoderLabelScorer::EncoderDecoderLabelScorer(const Core::Configuration& config, const Core::Ref<Encoder> encoder, const Core::Ref<LabelScorer> decoder)
        : Core::Component(config),
          LabelScorer(config),
          encoder_(encoder),
          decoder_(decoder) {
}

void EncoderDecoderLabelScorer ::reset() {
    encoder_->reset();
    decoder_->reset();
}

ScoringContextRef EncoderDecoderLabelScorer::getInitialScoringContext() {
    return decoder_->getInitialScoringContext();
}

ScoringContextRef EncoderDecoderLabelScorer::extendedScoringContext(Request request) {
    return decoder_->extendedScoringContext(request);
}

void EncoderDecoderLabelScorer::addInput(f32 const* data, size_t F) {
    encoder_->addInput(data, F);
    encode();
}

void EncoderDecoderLabelScorer::addInputs(f32 const* data, size_t T, size_t F) {
    encoder_->addInputs(data, T, F);
    encode();
}

void EncoderDecoderLabelScorer::signalNoMoreFeatures() {
    encoder_->signalNoMoreFeatures();
    // Call `encode()` before signaling segment end to the decoder since the decoder
    // is supposed to receive all available encoder outputs before this signal
    encode();
    decoder_->signalNoMoreFeatures();
}

std::optional<LabelScorer::ScoreWithTime> EncoderDecoderLabelScorer::getScoreWithTime(const LabelScorer::Request request) {
    return decoder_->getScoreWithTime(request);
}

std::optional<LabelScorer::ScoresWithTimes> EncoderDecoderLabelScorer::getScoresWithTimes(const std::vector<LabelScorer::Request>& requests) {
    return decoder_->getScoresWithTimes(requests);
}

void EncoderDecoderLabelScorer::encode() {
    std::optional<f32 const*> encoderOutput;
    while ((encoderOutput = encoder_->getNextOutput())) {
        decoder_->addInput(*encoderOutput, encoder_->getOutputSize());
    }
}

}  // namespace Nn
