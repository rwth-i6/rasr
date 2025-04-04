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

#include "EncoderDecoderLabelScorer.hh"

namespace Nn {

EncoderDecoderLabelScorer::EncoderDecoderLabelScorer(Core::Configuration const& config, Core::Ref<Encoder> const& encoder, Core::Ref<LabelScorer> const& decoder)
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

ScoringContextRef EncoderDecoderLabelScorer::extendedScoringContext(Request const& request) {
    return decoder_->extendedScoringContext(request);
}

void EncoderDecoderLabelScorer::addInput(std::shared_ptr<const f32[]> const& input, size_t featureSize) {
    encoder_->addInput(input, featureSize);
    passEncoderOutputsToDecoder();
}

void EncoderDecoderLabelScorer::addInput(std::vector<f32> const& input) {
    // The custom deleter ties the lifetime of the vector to the lifetime
    // of `dataPtr` by capturing the `inputWrapper` by value.
    // This makes sure that the underlying data isn't invalidated prematurely.
    auto inputWrapper = std::make_shared<std::vector<f32>>(input);
    auto dataPtr      = std::shared_ptr<const f32[]>(
            inputWrapper->data(),
            [inputWrapper](const f32*) mutable {});
    encoder_->addInput(dataPtr, input.size());
    passEncoderOutputsToDecoder();
}

void EncoderDecoderLabelScorer::addInputs(std::shared_ptr<const f32[]> const& input, size_t timeSize, size_t featureSize) {
    encoder_->addInputs(input, timeSize, featureSize);
    passEncoderOutputsToDecoder();
}

void EncoderDecoderLabelScorer::signalNoMoreFeatures() {
    encoder_->signalNoMoreFeatures();
    // Call `passEncoderOutputsToDecoder()` before signaling segment end to the decoder since the decoder
    // is supposed to receive all available encoder outputs before this signal
    passEncoderOutputsToDecoder();
    decoder_->signalNoMoreFeatures();
}

std::optional<LabelScorer::ScoreWithTime> EncoderDecoderLabelScorer::computeScoreWithTime(LabelScorer::Request const& request) {
    return decoder_->computeScoreWithTime(request);
}

std::optional<LabelScorer::ScoresWithTimes> EncoderDecoderLabelScorer::computeScoresWithTimes(std::vector<LabelScorer::Request> const& requests) {
    return decoder_->computeScoresWithTimes(requests);
}

void EncoderDecoderLabelScorer::passEncoderOutputsToDecoder() {
    std::optional<std::shared_ptr<const f32[]>> encoderOutput;
    while ((encoderOutput = encoder_->getNextOutput())) {
        decoder_->addInput(*encoderOutput, encoder_->getOutputSize());
    }
}

}  // namespace Nn
