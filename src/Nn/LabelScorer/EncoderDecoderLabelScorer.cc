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

EncoderDecoderLabelScorer::EncoderDecoderLabelScorer(Core::Configuration const& config, Core::Ref<Encoder> const& encoder, Core::Ref<ScaledLabelScorer> const& decoder)
        : Core::Component(config),
          LabelScorer(config),
          encoder_(encoder),
          decoder_(decoder) {
    enabledTransitions_ = decoder_->enabledTransitions();
}

Core::Ref<ScaledLabelScorer> EncoderDecoderLabelScorer::getDecoderLabelScorer() const {
    return decoder_;
}

void EncoderDecoderLabelScorer::logStatistics() const {
    encoder_->logStatistics();
    decoder_->logStatistics();
}

Core::Status EncoderDecoderLabelScorer::signalNoMoreFeatures() {
    encoder_->signalNoMoreFeatures();
    // Call `passEncoderOutputsToDecoder()` before signaling segment end to the decoder since the decoder
    // is supposed to receive all available encoder outputs before this signal
    auto status = passEncoderOutputsToDecoder();
    if (not status.ok()) {
        return status;
    }
    return decoder_->signalNoMoreFeatures();
}

Core::Status EncoderDecoderLabelScorer::getInitialScoringContext(ScoringContextRef& initialScoringContext) {
    return decoder_->getInitialScoringContext(initialScoringContext);
}

Core::Status EncoderDecoderLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType, ScoringContextRef& extendedContext) {
    return decoder_->extendedScoringContext(scoringContext, nextToken, transitionType, extendedContext);
}

void EncoderDecoderLabelScorer::cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) {
    decoder_->cleanupCaches(activeContexts);
}

Core::Status EncoderDecoderLabelScorer::addInput(DataView const& input) {
    encoder_->addInput(input);
    return passEncoderOutputsToDecoder();
}

Core::Status EncoderDecoderLabelScorer::addInputs(DataView const& input, size_t nTimesteps) {
    encoder_->addInputs(input, nTimesteps);
    return passEncoderOutputsToDecoder();
}

void EncoderDecoderLabelScorer::reset() {
    encoder_->reset();
    decoder_->reset();
}

Core::Status EncoderDecoderLabelScorer::getScoreAccessor(ScoringContextRef scoringContext, std::optional<ScoreAccessorRef>& scoreAccessor) {
    return decoder_->getScoreAccessor(scoringContext, scoreAccessor);
}

Core::Status EncoderDecoderLabelScorer::getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts, std::vector<std::optional<ScoreAccessorRef>>& scoreAccessors) {
    return decoder_->getScoreAccessors(scoringContexts, scoreAccessors);
}

Core::Status EncoderDecoderLabelScorer::passEncoderOutputsToDecoder() {
    std::optional<EncodedSpan> encoderOutput;
    while (true) {
        auto status = encoder_->getNextOutput(encoderOutput);
        if (not status.ok() or not encoderOutput) {
            return status;
        }
        status = decoder_->addInput(encoderOutput->encoding);
        if (not status.ok()) {
            return status;
        }
    }
}

}  // namespace Nn
