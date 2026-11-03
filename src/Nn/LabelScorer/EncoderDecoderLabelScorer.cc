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

EncoderDecoderLabelScorer::EncoderDecoderLabelScorer(Core::Configuration const&          config,
                                                     Core::Ref<Encoder> const&           encoder,
                                                     Core::Ref<ScaledLabelScorer> const& decoder,
                                                     EncoderCreationFunction             createEncoder)
        : Core::Component(config),
          LabelScorer(config),
          encoder_(encoder),
          decoder_(decoder),
          createEncoder_(std::move(createEncoder)),
          segmentEncoders_(),
          unusedEncoders_() {
    enabledTransitions_ = decoder_->enabledTransitions();
}

Core::Ref<ScaledLabelScorer> EncoderDecoderLabelScorer::getDecoderLabelScorer() const {
    return decoder_;
}

void EncoderDecoderLabelScorer::logStatistics() const {
    collectEncoderStatistics();
    encoder_->logStatistics();
    decoder_->logStatistics();
}

void EncoderDecoderLabelScorer::resetStatistics() {
    encoder_->resetStatistics();
    for (auto& [segment, encoder] : segmentEncoders_) {
        encoder->resetStatistics();
    }
    for (auto& encoder : unusedEncoders_) {
        encoder->resetStatistics();
    }
    decoder_->resetStatistics();
}

void EncoderDecoderLabelScorer::collectEncoderStatistics() const {
    // The statistics are moved, so they are not counted twice when they are collected again
    for (auto const& [segment, encoder] : segmentEncoders_) {
        encoder_->addStatistics(*encoder);
        encoder->resetStatistics();
    }
}

void EncoderDecoderLabelScorer::signalNoMoreFeatures() {
    signalNoMoreSegmentFeatures(defaultSegmentId);
}

ScoringContextRef EncoderDecoderLabelScorer::getInitialScoringContext() {
    return decoder_->getInitialScoringContext();
}

ScoringContextRef EncoderDecoderLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) {
    return decoder_->extendedScoringContext(scoringContext, nextToken, transitionType);
}

void EncoderDecoderLabelScorer::cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) {
    decoder_->cleanupCaches(activeContexts);
}

void EncoderDecoderLabelScorer::addInput(DataView const& input) {
    encoder_->addInput(input);
    passEncoderOutputsToDecoder(defaultSegmentId);
}

void EncoderDecoderLabelScorer::addInputs(DataView const& input, size_t nTimesteps) {
    encoder_->addInputs(input, nTimesteps);
    passEncoderOutputsToDecoder(defaultSegmentId);
}

void EncoderDecoderLabelScorer::reset() {
    encoder_->reset();
    decoder_->reset();
    for (auto& [segment, encoder] : segmentEncoders_) {
        encoder->reset();
        unusedEncoders_.push_back(encoder);
    }
    segmentEncoders_.clear();
}

std::optional<ScoreAccessorRef> EncoderDecoderLabelScorer::getScoreAccessor(ScoringContextRef scoringContext) {
    return decoder_->getScoreAccessor(scoringContext);
}

std::vector<std::optional<ScoreAccessorRef>> EncoderDecoderLabelScorer::getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) {
    return decoder_->getScoreAccessors(scoringContexts);
}

bool EncoderDecoderLabelScorer::supportsMultipleSegments() const {
    return decoder_->supportsMultipleSegments() and createEncoder_;
}

void EncoderDecoderLabelScorer::addSegment(SegmentId segment) {
    segmentEncoder(segment)->reset();
    decoder_->addSegment(segment);
}

void EncoderDecoderLabelScorer::removeSegment(SegmentId segment) {
    if (segment == defaultSegmentId) {
        encoder_->reset();
    }
    else {
        auto it = segmentEncoders_.find(segment);
        if (it != segmentEncoders_.end()) {
            encoder_->addStatistics(*it->second);
            it->second->reset();
            unusedEncoders_.push_back(it->second);
            segmentEncoders_.erase(it);
        }
    }
    decoder_->removeSegment(segment);
}

void EncoderDecoderLabelScorer::signalNoMoreSegmentFeatures(SegmentId segment) {
    segmentEncoder(segment)->signalNoMoreFeatures();
    // Call `passEncoderOutputsToDecoder()` before signaling segment end to the decoder since the decoder
    // is supposed to receive all available encoder outputs before this signal
    passEncoderOutputsToDecoder(segment);
    if (segment == defaultSegmentId) {
        decoder_->signalNoMoreFeatures();
    }
    else {
        decoder_->signalNoMoreSegmentFeatures(segment);
    }
}

ScoringContextRef EncoderDecoderLabelScorer::getInitialSegmentScoringContext(SegmentId segment) {
    return decoder_->getInitialSegmentScoringContext(segment);
}

void EncoderDecoderLabelScorer::addSegmentInput(SegmentId segment, DataView const& input) {
    segmentEncoder(segment)->addInput(input);
    passEncoderOutputsToDecoder(segment);
}

void EncoderDecoderLabelScorer::addSegmentInputs(SegmentId segment, DataView const& input, size_t nTimesteps) {
    segmentEncoder(segment)->addInputs(input, nTimesteps);
    passEncoderOutputsToDecoder(segment);
}

Core::Ref<Encoder> const& EncoderDecoderLabelScorer::segmentEncoder(SegmentId segment) {
    if (segment == defaultSegmentId) {
        return encoder_;
    }

    auto it = segmentEncoders_.find(segment);
    if (it != segmentEncoders_.end()) {
        return it->second;
    }

    Core::Ref<Encoder> encoder;
    if (not unusedEncoders_.empty()) {
        encoder = unusedEncoders_.back();
        unusedEncoders_.pop_back();
    }
    else {
        if (not createEncoder_) {
            error() << "Encoder-decoder label scorer can't create an encoder for segment " << segment << " because no encoder creation function was given";
        }
        encoder = createEncoder_();
    }
    return segmentEncoders_.emplace(segment, encoder).first->second;
}

void EncoderDecoderLabelScorer::passEncoderOutputsToDecoder(SegmentId segment) {
    auto const&                encoder = segmentEncoder(segment);
    std::optional<EncodedSpan> encoderOutput;
    while ((encoderOutput = encoder->getNextOutput())) {
        if (segment == defaultSegmentId) {
            decoder_->addInput(encoderOutput->encoding);
        }
        else {
            decoder_->addSegmentInput(segment, encoderOutput->encoding);
        }
    }
}

}  // namespace Nn
