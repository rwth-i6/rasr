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

#ifndef ENCODER_DECODER_LABEL_SCORER_HH
#define ENCODER_DECODER_LABEL_SCORER_HH

#include <functional>
#include <unordered_map>

#include <Core/Component.hh>

#include "Encoder.hh"
#include "LabelScorer.hh"
#include "ScaledLabelScorer.hh"

namespace Nn {

/*
 * Glue class to represent encoder-decoder model architectures. It consists of an
 * encoder component that computes feature encodings without requiring a ScoringContext
 * and a decoder component which is an arbitrary sub-LabelScorer that receives the encoded
 * features as its inputs.
 * This glue class automatically handles the information flow between its encoder and
 * decoder components.
 *
 * Multiple segments are supported if the decoder supports them and an encoder creation function is given.
 * Every segment gets its own encoder instance since encoders keep internal state for streaming.
 * Encoders of removed segments are kept for reuse.
 */
class EncoderDecoderLabelScorer : public LabelScorer {
public:
    typedef std::function<Core::Ref<Encoder>()> EncoderCreationFunction;

    EncoderDecoderLabelScorer(Core::Configuration const&          config,
                              Core::Ref<Encoder> const&           encoder,
                              Core::Ref<ScaledLabelScorer> const& decoder,
                              EncoderCreationFunction             createEncoder = {});
    virtual ~EncoderDecoderLabelScorer() = default;

    // Return the decoder label scorer
    Core::Ref<ScaledLabelScorer> getDecoderLabelScorer() const;

    // Forwarded to the encoder and the decoder scorer. The statistics of the encoders of all segments are
    // added up in the encoder of the default segment.
    void logStatistics() const override;

    // Reset the statistics of all encoders and the decoder
    void resetStatistics() override;

    // Signal end of feature stream to encoder, then encode features, pass them to the decoder and
    // finally signal end of feature stream to decoder.
    void signalNoMoreFeatures() override;

    // Get start context from decoder component
    ScoringContextRef getInitialScoringContext() override;

    // Get extended context from decoder component
    ScoringContextRef extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) override;

    // Cleanup decoder component. Encoder is "self-cleaning" already in that it only stores outputs until they are
    // retrieved.
    void cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) override;

    // Add an input feature to the encoder component and if possible forward the encoder and add
    // the encoder states as inputs to the decoder component
    void addInput(DataView const& input) override;

    // Same as `addInput` but adds features for multiple timesteps at once
    void addInputs(DataView const& input, size_t nTimesteps) override;

    // Resets both encoder and decoder component and releases the encoders of all other segments
    void reset() override;

    // Return accessor from decoder component
    std::optional<ScoreAccessorRef> getScoreAccessor(ScoringContextRef scoringContext) override;

    // Return accessors from decoder component
    std::vector<std::optional<ScoreAccessorRef>> getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) override;

    // Supported if the decoder supports multiple segments and further encoders can be created
    bool supportsMultipleSegments() const override;

    // Assign a (reset) encoder to the segment and register the segment in the decoder
    void addSegment(SegmentId segment) override;

    // Release the encoder of the segment and remove the segment from the decoder
    void removeSegment(SegmentId segment) override;

    // Same as `signalNoMoreFeatures` for the given segment
    void signalNoMoreSegmentFeatures(SegmentId segment) override;

    // Get start context of the segment from decoder component
    ScoringContextRef getInitialSegmentScoringContext(SegmentId segment) override;

    // Same as `addInput` for the given segment
    void addSegmentInput(SegmentId segment, DataView const& input) override;

    // Same as `addInputs` for the given segment
    void addSegmentInputs(SegmentId segment, DataView const& input, size_t nTimesteps) override;

private:
    Core::Ref<Encoder>           encoder_;  // Encoder of the default segment
    Core::Ref<ScaledLabelScorer> decoder_;

    EncoderCreationFunction                           createEncoder_;
    std::unordered_map<SegmentId, Core::Ref<Encoder>> segmentEncoders_;  // Encoders of all non-default segments
    std::vector<Core::Ref<Encoder>>                   unusedEncoders_;   // Encoders of removed segments for reuse

    // Return the encoder of the segment; creates one if the segment doesn't have one yet
    Core::Ref<Encoder> const& segmentEncoder(SegmentId segment);

    // Add the statistics of the encoders of all other segments to the encoder of the default segment
    void collectEncoderStatistics() const;

    // Fetch as many outputs as possible from the encoder of the segment given its available features and pass
    // these outputs over to the decoder
    void passEncoderOutputsToDecoder(SegmentId segment);
};

}  // namespace Nn

#endif  // ENCODER_DECODER_LABEL_SCORER_HH
