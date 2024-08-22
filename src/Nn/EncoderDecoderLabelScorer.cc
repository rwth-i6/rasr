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

#ifdef MODULE_ONNX
#include "OnnxEncoder.hh"
#endif

namespace Nn {

const Core::Choice EncoderDecoderLabelScorer::choiceEncoderType(
        // Assume encoder inputs are already finished states and just pass them on without transformations
        "no-op", EncoderType::NoOpEncoder,
        // Forward encoder inputs through an onnx network
        "onnx-encoder", EncoderType::OnnxEncoder);

const Core::ParameterChoice EncoderDecoderLabelScorer::paramEncoderType(
        "encoder-type",
        &choiceEncoderType,
        "Choice from a set of encoder types.",
        EncoderType::NoOpEncoder);

const Core::Choice EncoderDecoderLabelScorer::choiceDecoderType(
        // Assume encoder states are already finished scores and just pass them on without transformations
        "no-op", DecoderType::NoOpDecoder,
        // Wrapper around legacy Mm::FeatureScorer
        "legacy-feature-scorer", DecoderType::LegacyFeatureScorerDecoder);

const Core::ParameterChoice EncoderDecoderLabelScorer::paramDecoderType(
        "decoder-type",
        &choiceDecoderType,
        "Choice from a set of decoder types.",
        DecoderType::NoOpDecoder);

EncoderDecoderLabelScorer::EncoderDecoderLabelScorer(const Core::Configuration& config)
        : Core::Component(config),
          Nn::LabelScorer(config) {
    switch (paramEncoderType(config)) {
        case EncoderType::NoOpEncoder:
            encoder_ = Core::ref(new Nn::NoOpEncoder(config));
            break;
#ifdef MODULE_ONNX
        case EncoderType::OnnxEncoder:
            encoder_ = Core::ref(new Nn::OnnxEncoder(config));
            break;
#endif
        default:
            Core::Application::us()->criticalError("unknown encoder type: %d", paramEncoderType(config));
    }

    switch (paramDecoderType(config)) {
        case DecoderType::NoOpDecoder:
            decoder_ = Core::ref(new Nn::NoOpDecoder(config));
            break;
        case DecoderType::LegacyFeatureScorerDecoder:
            decoder_ = Core::ref(new Nn::LegacyFeatureScorerDecoder(config));
            break;
        default:
            Core::Application::us()->criticalError("unknown decoder type: %d", paramDecoderType(config));
    }
}

void EncoderDecoderLabelScorer ::reset() {
    encoder_->reset();
    decoder_->reset();
}

Core::Ref<LabelHistory> EncoderDecoderLabelScorer::getStartHistory() {
    return decoder_->getStartHistory();
}

void EncoderDecoderLabelScorer::extendHistory(Request& request) {
    decoder_->extendHistory(request);
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

std::optional<Score> EncoderDecoderLabelScorer::getScore(const Request& request) {
    return decoder_->getScore(request);
}

void EncoderDecoderLabelScorer::encode() {
    std::optional<FeatureVectorRef> encoderOutput;
    while ((encoderOutput = encoder_->getNextOutput())) {
        decoder_->addEncoderOutput(*encoderOutput);
    }
}

}  // namespace Nn
