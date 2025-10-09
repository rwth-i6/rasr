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

void EncoderDecoderLabelScorer::cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) {
    decoder_->cleanupCaches(activeContexts);
}

void EncoderDecoderLabelScorer::addInput(DataView const& input) {
    encoder_->addInput(input);
    passEncoderOutputsToDecoder();
}

void EncoderDecoderLabelScorer::addInputs(DataView const& input, size_t nTimesteps) {
    encoder_->addInputs(input, nTimesteps);
    passEncoderOutputsToDecoder();
}

void EncoderDecoderLabelScorer::signalNoMoreFeatures() {
    encoder_->signalNoMoreFeatures();
    // Call `passEncoderOutputsToDecoder()` before signaling segment end to the decoder since the decoder
    // is supposed to receive all available encoder outputs before this signal
    passEncoderOutputsToDecoder();
    decoder_->signalNoMoreFeatures();
}

ScoringContextRef EncoderDecoderLabelScorer::extendedScoringContextInternal(Request const& request) {
    return decoder_->extendedScoringContext(request);
}

std::optional<LabelScorer::ScoreWithTime> EncoderDecoderLabelScorer::computeScoreWithTimeInternal(LabelScorer::Request const& request) {
    return decoder_->computeScoreWithTime(request);
}

std::optional<LabelScorer::ScoresWithTimes> EncoderDecoderLabelScorer::computeScoresWithTimesInternal(std::vector<LabelScorer::Request> const& requests) {
    if (requests.empty()) {
        return ScoresWithTimes{};
    }

    return decoder_->computeScoresWithTimes(requests);
}

void EncoderDecoderLabelScorer::passEncoderOutputsToDecoder() {
    std::optional<DataView> encoderOutput;
    while ((encoderOutput = encoder_->getNextOutput())) {
        decoder_->addInput(*encoderOutput);
    }
}

}  // namespace Nn
