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

#ifndef DECODER_HH
#define DECODER_HH

#include <Core/Component.hh>
#include <Core/Configuration.hh>
#include <Core/ReferenceCounting.hh>
#include <Mm/FeatureScorer.hh>
#include <Speech/Feature.hh>
#include <optional>
#include "Encoder.hh"
#include "LabelHistory.hh"
#include "Search/Types.hh"

namespace Nn {

typedef Search::Score Score;

// Request that is passed as argument to `getDecoderStepScorer`.
struct ScoreRequest {
    Core::Ref<const LabelHistory> history;
    LabelIndex                    labelIndex;
    bool                          isLoop;
    std::optional<Score>          score;
};

// Base class for models that can score hypotheses based on history and encoder states
class Decoder : public virtual Core::Component,
                public Core::ReferenceCounted {
public:
    Decoder(const Core::Configuration& config);
    virtual ~Decoder() = default;

    // Reset buffer and segment end flag.
    virtual void reset();

    // Get initial history to be used at segment begin
    virtual LabelHistory getStartHistory() = 0;

    // (Maybe) extend a given history using the next label
    // `isLoop` may affect whether the history is updated or not, depending on the specific model
    virtual void extendHistory(LabelHistory& history, LabelIndex labelIndex, bool isLoop) = 0;

    // Add a single encoder state to buffer
    virtual void addEncoderOutput(FeatureVectorRef encoderOutput);

    // Tells the Decoder that there will be no more encoder states coming in the current segment
    // Relevant for e.g. Attention models that require all encoder states of a segment before decoding can begin
    virtual void signalSegmentEnd();

    // Decoder will compute score for label given a history ref and a boolean flag that says whether
    // it should score for a loop or forward transition.
    // The result will be set in the `score` member of the requests
    virtual void getDecoderStepScores(std::vector<ScoreRequest>& requests) = 0;

protected:
    std::vector<FeatureVectorRef> encoderOutputBuffer_;
    bool                          segmentEnd_;
};

// Dummy decoder that just returns back the encoder output at the current step
class NoOpDecoder : public Decoder {
    using Precursor   = Decoder;
    using HistoryType = StepLabelHistory;

public:
    NoOpDecoder(const Core::Configuration& config);

    LabelHistory getStartHistory() override;
    void         extendHistory(LabelHistory& history, LabelIndex labelIndex, bool isLoop) override;
    void         getDecoderStepScores(std::vector<ScoreRequest>& requests) override;
};

// Wrapper around legacy Mm::FeatureScorer.
// Encoder outputs are treated as features for the FeatureScorer.
// When adding encoder outputs, whenever possible (depending on FeatureScorer buffering)
// directly prepare ContextScorers based on them and cache these.
// Thus, the normal encoder output buffer is not used.
// Upon receiving segment end signal, all available ContextScorers are flushed.
class LegacyFeatureScorerDecoder : public Decoder {
    using Precursor   = Decoder;
    using HistoryType = StepLabelHistory;

public:
    LegacyFeatureScorerDecoder(const Core::Configuration& config);
    void         reset() override;
    void         addEncoderOutput(FeatureVectorRef encoderOutput) override;
    void         signalSegmentEnd() override;
    LabelHistory getStartHistory() override;
    void         extendHistory(LabelHistory& history, LabelIndex labelIndex, bool isLoop) override;
    void         getDecoderStepScores(std::vector<ScoreRequest>& requests) override;

private:
    Core::Ref<Mm::FeatureScorer>           featureScorer_;
    std::vector<Mm::FeatureScorer::Scorer> scoreCache_;
};

}  // namespace Nn

#endif  // DECODER_HH
