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

#ifndef STATEFUL_ONNX_LABEL_SCORER_HH
#define STATEFUL_ONNX_LABEL_SCORER_HH

#include <optional>
#include <unordered_map>

#include <Core/Component.hh>
#include <Core/Configuration.hh>
#include <Core/FIFOCache.hh>
#include <Core/ReferenceCounting.hh>
#include <Mm/FeatureScorer.hh>
#include <Onnx/OnnxInitUpdateScoreBundle.hh>
#include <Onnx/PersistentInput.hh>
#include <Speech/Feature.hh>

#include "BufferedLabelScorer.hh"
#include "ModelCache.hh"
#include "ScoringContext.hh"

namespace Nn {

/*
 * Scoring context consisting of a hidden state.
 * Assumes that two hidden states are equal if and only if they were created
 * from the same label history.
 */
struct OnnxHiddenStateScoringContext : public ScoringContext {
    std::vector<LabelIndex>          labelSeq;  // Used for hashing
    mutable Onnx::OnnxNamedValuesRef hiddenState;
    mutable bool                     requiresFinalize;

    OnnxHiddenStateScoringContext();
    OnnxHiddenStateScoringContext(std::vector<LabelIndex> const& labelSeq, Onnx::OnnxNamedValuesRef state, bool requiresFinalize);

    bool   isEqual(ScoringContextRef const& other) const override;
    size_t hash() const override;
};

typedef Core::Ref<OnnxHiddenStateScoringContext const> OnnxHiddenStateScoringContextRef;

/*
 * Label Scorer that performs scoring by forwarding hidden states through an ONNX model.
 * This Label Scorer requires three ONNX models:
 *  - A State Initializer which produces the hidden states for the first step (optionally based on the input sequence)
 *  - A State Updater which produces updated hidden states based on the previous hidden states and optionally also the input sequence and the next token
 *  - A Scorer which computes scores based on the hidden states
 *
 * The models themselves as well as the mapping between their inputs/outputs and the hidden states are
 * handled by `Onnx::OnnxInitUpdateScoreBundle`; see there for the metadata convention that the models have to follow.
 *
 * A common use case for this Label Scorer would be an AED model with cross-attention over the encoder output.
 * Since the encoder state inputs are optional, it can also be used for stateful language models without acoustic input.
 *
 * Note: This LabelScorer is similar to the `StatefulTransducerOnnxLabelScorer`. The difference is that in this it is assumed that the
 * input features are processed into the hidden states and they are not directly fed into the scorer. For this, the state initializer
 * and updater here also take input features in addition to tokens.
 *
 * Multiple segments: Encoder states, initial hidden state and caches are kept per segment. Contexts of different
 * segments are batched together if the state updater doesn't receive encoder states. Otherwise, they are forwarded
 * per segment unless `stack-encoder-states` is enabled: then the encoder states of the segments are stacked to shape
 * [B, T_max, E] (zero-padded) together with their sizes [B]. This requires a state updater with a dynamic batch dimension
 * of the encoder states that masks the padding via the encoder states size.
 * If the state updater has the optional input `encoder-states-index` [B], the encoder states are stacked only once per
 * segment ([S, T_max, E] with sizes [S]) and the index contains the row of each batch entry.
 */
class StatefulOnnxLabelScorer : public BufferedLabelScorer {
public:
    StatefulOnnxLabelScorer(Core::Configuration const& config, ModelCache& modelCache);
    virtual ~StatefulOnnxLabelScorer() = default;

    // Initial context with empty label sequence. The initial hidden state is only computed once it is needed.
    ScoringContextRef getInitialSegmentScoringContext(SegmentId segment) override;

    // Append the new token to the label sequence; does not update the hidden-state. This is only done once the scoringContext is used for scoring again.
    ScoringContextRef extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) override;

    // Add a single input feature to buffer of the segment
    void addSegmentInput(SegmentId segment, DataView const& input) override;

    // Create or remove buffered inputs, encoder states and caches of the segment
    void addSegment(SegmentId segment) override;
    void removeSegment(SegmentId segment) override;

protected:
    void resetInternal() override;
    void resetStatisticsInternal() override;

    // Update hidden state, run scorer and get an accessor for the output score vector
    std::optional<ScoreAccessorRef> computeScoreAccessor(ScoringContextRef scoringContext) override;

    // Update hidden states (batched), run scorers (batched) and get accessor for the output score vectors
    std::vector<std::optional<ScoreAccessorRef>> computeScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) override;
    size_t                                       getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const override;

private:
    using Precursor = BufferedLabelScorer;

    static const Core::ParameterBool paramBlankUpdatesHistory;
    static const Core::ParameterBool paramSilenceUpdatesHistory;
    static const Core::ParameterBool paramLoopUpdatesHistory;
    static const Core::ParameterInt  paramMaxBatchSize;
    static const Core::ParameterInt  paramMaxCachedScores;
    static const Core::ParameterBool paramStackEncoderStates;

    bool   blankUpdatesHistory_;
    bool   silenceUpdatesHistory_;
    bool   loopUpdatesHistory_;
    size_t maxBatchSize_;
    size_t maxCachedScores_;
    bool   stackEncoderStates_;

    Onnx::OnnxInitUpdateScoreBundle initUpdateScoreBundle_;

    std::string initializerEncoderStatesName_;
    std::string initializerEncoderStatesSizeName_;

    std::string updaterEncoderStatesName_;
    std::string updaterEncoderStatesSizeName_;
    std::string updaterEncoderStatesIndexName_;
    std::string updaterTokenName_;

    // Preparing the session inputs that are not hidden states and moving hidden states between the
    // caches and the batches. The session times and the marshalling done inside `initUpdateScoreBundle_`
    // are tracked there.
    Core::StopWatch stateMarshallingTime_;
    Core::StopWatch contextPreparationTime_;

    struct SegmentData {
        // Store the onnx values with all encoder states and lengths inside so that it doesn't have to be recomputed every time
        // The encoder states are passed to the session runs as views to avoid copies (and uploaded to the device only once for CUDA sessions)
        Onnx::PersistentInput encoderStatesValue;
        Onnx::Value           encoderStatesSizeValue;

        Onnx::OnnxNamedValuesRef initialHiddenState;

        Core::FIFOCache<OnnxHiddenStateScoringContextRef, std::shared_ptr<std::vector<Score>>, ScoringContextHash, ScoringContextEq> scoreCache;
        Core::FIFOCache<OnnxHiddenStateScoringContextRef, Onnx::OnnxNamedValuesRef, ScoringContextHash, ScoringContextEq>            stateCache;

        SegmentData(size_t maxCachedScores);
    };

    std::unordered_map<SegmentId, SegmentData> segmentData_;

    // Data of the segment, created if it doesn't exist yet
    SegmentData& segmentData(SegmentId segment);

    // Whether the state updater or state initializer receive encoder states (or their size)
    bool usesEncoderStates() const;

    // Forward a batch of scoringContexts through the ONNX scorer model and put the resulting scores into the score caches
    void cacheScores(std::vector<OnnxHiddenStateScoringContextRef> const& scoringContextBatch);

    // Computes new hidden state based on previous hidden state and next token with batched state-updater call
    // `segmentsBatch` contains the segment of each batch entry
    std::vector<Onnx::OnnxNamedValuesRef> updatedHiddenStates(std::vector<Onnx::OnnxNamedValuesRef> const& hiddenStatesBatch, std::vector<s32> nextTokensBatch, std::vector<SegmentId> const& segmentsBatch);

    // Compute updated states for all non-finalized scoring contexts and put them into the state caches
    void cacheStates(std::vector<OnnxHiddenStateScoringContextRef> const& scoringContextBatch);

    // Since the hidden-state matrix depends on the encoder time axis, we cannot create properly create hidden-states until all encoder states have been passed.
    // So getInitialScoringContext sets the initial hidden-state to a sentinel value (empty Ref) and when other functions such as `extendedScoringContext` and `getScoresWithTime`
    // encounter this sentinel value they call `computeInitialHiddenState` instead to get a usable hidden-state.
    Onnx::OnnxNamedValuesRef computeInitialHiddenState(SegmentId segment);

    void setupEncoderStatesValue(SegmentId segment);
    void setupEncoderStatesSizeValue(SegmentId segment);

    void logScoringBreakdown() const override;
};

}  // namespace Nn

#endif  // STATEFUL_ONNX_LABEL_SCORER_HH
