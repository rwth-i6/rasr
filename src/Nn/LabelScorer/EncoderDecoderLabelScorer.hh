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
#include "Encoder.hh"
#include "LabelScorer.hh"

namespace Nn {

/*
 * Glue class to represent encoder-decoder model architectures. It consists of an
 * encoder component that does history-independent feature encoding once in the beginning
 * and a decoder component which is another internal LabelScorer that receives the encoder
 * states as its inputs.
 * This glue class automatically handles the information flow between its encoder and
 * decoder components.
 */
class EncoderDecoderLabelScorer : public LabelScorer {
public:
    EncoderDecoderLabelScorer(const Core::Configuration& config, const Core::Ref<Encoder> encoder, const Core::Ref<LabelScorer> decoder);
    virtual ~EncoderDecoderLabelScorer() = default;

    // Resets both encoder and decoder component
    void reset() override;

    // Signal end of feature stream to encoder, then encode features, pass them to the decoder and
    // finally signal end of feature stream to decoder.
    void signalNoMoreFeatures() override;

    // Get start context from decoder component
    ScoringContextRef getInitialScoringContext() override;

    // Get extended context from decoder component
    ScoringContextRef extendedScoringContext(Request request) override;

    // Add an input feature to the encoder component and if possible forward the encoder and add
    // the encoder states as inputs to the decoder component
    void addInput(std::shared_ptr<const f32> const& data, size_t F) override;

    // Same as `addInput` but adds features for multiple timesteps at once
    void addInputs(std::shared_ptr<const f32> const& data, size_t T, size_t F) override;

    // Run request through decoder component
    std::optional<LabelScorer::ScoreWithTime> getScoreWithTime(const LabelScorer::Request request) override;

    // Run requests through decoder component
    std::optional<LabelScorer::ScoresWithTimes> getScoresWithTimes(const std::vector<LabelScorer::Request>& requests) override;

protected:
    Core::Ref<Encoder>     encoder_;
    Core::Ref<LabelScorer> decoder_;

private:
    // Fetch as many outputs as possible from the encoder given its available features and pass
    // these outputs over to the decoder
    void encode();
};

}  // namespace Nn

#endif  // ENCODER_DECODER_LABEL_SCORER_HH
