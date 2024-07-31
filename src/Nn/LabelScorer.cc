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

#include "LabelScorer.hh"
#include "Core/Choice.hh"
#include "Nn/Encoder.hh"

#ifdef MODULE_ONNX
#include "OnnxEncoder.hh"
#endif

namespace Nn {

/*
 * =============================
 * = EncoderDecoderLabelScorer =
 * =============================
 */

const Core::Choice LabelScorer::choiceType(
        // Assume encoder inputs are already finished scores and just pass them on without transformations
        "no-op", LabelScorerType::NoOpLabelScorer,
        // Onnx encoder with no-op decoder
        "onnx-encoder-only", LabelScorerType::OnnxEncoderLabelScorer,
        // Wrapper around legacy Mm::FeatureScorer for backward compatibility
        "legacy-feature-scorer", LabelScorerType::LegacyFeatureScorerLabelScorer);

const Core::ParameterChoice LabelScorer::paramType(
        "type",
        &choiceType,
        "Choice from a set of label scorer types.",
        LabelScorerType::NoOpLabelScorer);

LabelScorer::LabelScorer(const Core::Configuration& config)
        : Core::Component(config),
          type_(static_cast<LabelScorerType>(paramType(config))) {
    initEncoderDecoder();
}

void LabelScorer::initEncoderDecoder() {
    const auto& encoderConfig = select("encoder");
    const auto& decoderConfig = select("decoder");

    Encoder* encoder = nullptr;
    Decoder* decoder = nullptr;
    switch (type_) {
        case LabelScorerType::NoOpLabelScorer:
            encoder = new NoOpEncoder(encoderConfig);
            decoder = new NoOpDecoder(decoderConfig);
            break;
#ifdef MODULE_ONNX
        case LabelScorerType::OnnxEncoderLabelScorer:
            encoder = new OnnxEncoder(encoderConfig);
            decoder = new NoOpDecoder(decoderConfig);
            break;
#endif
        case LabelScorerType::LegacyFeatureScorerLabelScorer:
            encoder = new NoOpEncoder(encoderConfig);
            decoder = new LegacyFeatureScorerDecoder(decoderConfig);
            break;
        default:
            error() << "Failed to initialize label scorer. Type is not known.";
            break;
    }
    encoder_ = Core::Ref<Encoder>(encoder);
    decoder_ = Core::Ref<Decoder>(decoder);
}

void LabelScorer ::reset() {
    encoder_->reset();
    decoder_->reset();
}

LabelHistory LabelScorer::getStartHistory() {
    return decoder_->getStartHistory();
}

void LabelScorer::extendHistory(LabelHistory& history, LabelIndex label, bool isLoop) {
    decoder_->extendHistory(history, label, isLoop);
}

void LabelScorer::addInput(FeatureVectorRef input) {
    encoder_->addInput(input);
    encode();
}

void LabelScorer::addInput(Core::Ref<const Speech::Feature> input) {
    encoder_->addInput(input);
    encode();
}

void LabelScorer::signalSegmentEnd() {
    encoder_->signalSegmentEnd();
    // Call `encode()` before signaling segment end to the decoder since the decoder
    // is supposed to receive all available encoder outputs before this signal
    encode();
    decoder_->signalSegmentEnd();
}

void LabelScorer::getDecoderStepScores(std::vector<ScoreRequest>& requests) {
    decoder_->getDecoderStepScores(requests);
}

void LabelScorer::encode() {
    std::optional<FeatureVectorRef> encoderOutput;
    // As long as the encoder returns more outputs, add them to the decoder buffer
    while ((encoderOutput = encoder_->getNextOutput())) {
        decoder_->addEncoderOutput(*encoderOutput);
    }
}

}  // namespace Nn
