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

#ifndef ENCODER_DECODER_LABEL_SCORER_HH
#define ENCODER_DECODER_LABEL_SCORER_HH

#include <Core/Component.hh>
#include "Decoder.hh"
#include "Encoder.hh"
#include "LabelScorer.hh"

namespace Nn {

// Glue class that couples encoder and decoder
// Purpose is creation of right encoder/decoder combination through choice parameters and
// automatic information flow between encoder and decoder
class EncoderDecoderLabelScorer : public LabelScorer {
private:
    static const Core::Choice          choiceEncoderType;
    static const Core::ParameterChoice paramEncoderType;

    static const Core::Choice          choiceDecoderType;
    static const Core::ParameterChoice paramDecoderType;

public:
    EncoderDecoderLabelScorer(const Core::Configuration& config);
    virtual ~EncoderDecoderLabelScorer() = default;

    enum EncoderType {
        NoOpEncoder,
        OnnxEncoder
    };

    enum DecoderType {
        NoOpDecoder,
        LegacyFeatureScorerDecoder
    };

    // Clear buffers and reset segment end flag in both encoder and decoder
    void reset();

    // Signal that no more features are expected for the current segment.
    // When segment end is signaled, encoder can run regardless of whether the buffer has been filled.
    // Thus, all encoder states are computed and forwarded to the decoder.
    // Relevant for e.g. Attention models that require all encoder states of a segment before decoding can begin
    void signalNoMoreFeatures();

    // Get start history from decoder
    Core::Ref<LabelHistory> getStartHistory();

    // Extend history for decoder
    void extendHistory(Request& request);

    // Add a single input feature to the encoder
    void addInput(FeatureVectorRef input);
    void addInput(Core::Ref<const Speech::Feature> input);

    // Runs requests through decoder given available encoder states
    std::optional<Score> getScore(const Request& request);

protected:
    Core::Ref<Encoder> encoder_;
    Core::Ref<Decoder> decoder_;

private:
    // Run encoder as long as it can return outputs and add all results to the decoder buffer
    void encode();
};

}  // namespace Nn

#endif  // ENCODER_DECODER_LABEL_SCORER_HH
