/** Copyright 2026 RWTH Aachen University. All rights reserved.
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

#ifndef PREFIX_SPEECH_LM_ONNX_LABEL_SCORER_HH
#define PREFIX_SPEECH_LM_ONNX_LABEL_SCORER_HH

#include <optional>
#include <unordered_map>

#include <Core/FIFOCache.hh>
#include <Onnx/Model.hh>

#include "BufferedLabelScorer.hh"
#include "ModelCache.hh"
#include "ScoringContext.hh"

namespace Nn {

/*
 * Scoring context for prefix speech LMs.
 * `labelSeq` contains only labels emitted by search, not from the prompt and speech-embeddings.
 * `historyLength` contains the total LM cache length, including prompt and speech-embedding positions.
 * Hashing and equality operators are based only on the `labelSeq`.
 */
struct PrefixSpeechLmScoringContext : public ScoringContext {
    using StateManager = AbstractStateManager<Onnx::Value, Onnx::OnnxStateVariable>;
    using HistoryState = StateManager::HistoryState;

    std::vector<LabelIndex>                               labelSeq;
    mutable size_t                                        historyLength;
    mutable Core::Ref<PrefixSpeechLmScoringContext const> parent;

    mutable std::shared_ptr<HistoryState> state;

    PrefixSpeechLmScoringContext();
    PrefixSpeechLmScoringContext(std::vector<LabelIndex>&& labelSeq, size_t historyLength, Core::Ref<PrefixSpeechLmScoringContext const> parent, std::shared_ptr<HistoryState> state = {});

    bool   isEqual(ScoringContextRef const& other) const override;
    size_t hash() const override;
};

typedef Core::Ref<PrefixSpeechLmScoringContext const> PrefixSpeechLmScoringContextRef;

/*
 * LabelScorer for prefix speech LMs that first initialize with
 * [initial prompt tokens, speech encoder states in LM embedding space, suffix prompt tokens]
 * before scoring hypothesis tokens one-by-one as usual.
 *
 * The LabelScorer requires two ONNX models, one which takes the prompts and encoder states
 * to produce the initial states and score distribution and one which takes a new token and
 * existing states to output updated states and scores after incorporating the token.
 * The states are managed with a StateManager.
 * With `prefill-with-step-model`, a single combined model is used for both (one ONNX session, the LM weights only once):
 * it takes several new positions at once (tokens and/or vectors), and the prefill runs it twice per segment.
 *
 * Multiple segments: encoder states, the initial context and caches are kept per segment. The initializer runs
 * once per segment, step contexts of all segments are batched together.
 */
class PrefixSpeechLmOnnxLabelScorer : public BufferedLabelScorer {
    using Precursor    = BufferedLabelScorer;
    using StateManager = AbstractStateManager<Onnx::Value, Onnx::OnnxStateVariable>;
    using HistoryState = StateManager::HistoryState;

    static const Core::ParameterIntVector paramInitialPromptLabels;
    static const Core::ParameterIntVector paramSuffixPromptLabels;
    static const Core::ParameterBool      paramBlankUpdatesHistory;
    static const Core::ParameterBool      paramSilenceUpdatesHistory;
    static const Core::ParameterBool      paramLoopUpdatesHistory;
    static const Core::ParameterBool      paramPrefillWithStepModel;
    static const Core::ParameterInt       paramMaxBatchSize;
    static const Core::ParameterInt       paramMaxCachedScores;

public:
    PrefixSpeechLmOnnxLabelScorer(Core::Configuration const& config, ModelCache& modelCache);
    virtual ~PrefixSpeechLmOnnxLabelScorer() = default;

    void addSegment(SegmentId segment) override;
    void removeSegment(SegmentId segment) override;
    void addSegmentInput(SegmentId segment, DataView const& input) override;

    ScoringContextRef getInitialSegmentScoringContext(SegmentId segment) override;

    // Override the initial or suffix prompt of the config for the given segment (default: the segment of single-segment search).
    // Call after the segment was started (`enterSegment` / `addSegment`) and before its features are passed.
    void setInitialPrompt(std::vector<s32> const& prompt, SegmentId segment = defaultSegmentId);
    void setSuffixPrompt(std::vector<s32> const& prompt, SegmentId segment = defaultSegmentId);
    ScoringContextRef extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) override;

protected:
    void resetInternal() override;
    void resetStatisticsInternal() override;

    std::optional<ScoreAccessorRef>              computeScoreAccessor(ScoringContextRef scoringContext) override;
    std::vector<std::optional<ScoreAccessorRef>> computeScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) override;
    size_t                                       getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const override;

private:
    struct SegmentData {
        Onnx::Value encoderStatesValue;
        Onnx::Value encoderStatesSizeValue;

        PrefixSpeechLmScoringContextRef initialContext;

        std::optional<std::vector<s32>> initialPrompt;  // Overrides of the prompts of the config
        std::optional<std::vector<s32>> suffixPrompt;

        Core::FIFOCache<PrefixSpeechLmScoringContextRef, std::shared_ptr<std::vector<Score>>, ScoringContextHash, ScoringContextEq> scoreCache;
        Core::FIFOCache<PrefixSpeechLmScoringContextRef, std::shared_ptr<HistoryState>, ScoringContextHash, ScoringContextEq>       stateCache;

        SegmentData(size_t maxCachedScores);
    };

    // Data of the segment, created if it doesn't exist yet
    SegmentData& segmentData(SegmentId segment);

    // Result of the prefill: scores of the first output token, the states of the prefix positions (one per position if
    // the state manager requires all parent states, else only the last) and the number of prefix positions
    struct PrefillResult {
        std::shared_ptr<std::vector<Score>> scores;
        std::vector<HistoryState>           states;
        size_t                              length = 0ul;
    };

    void          setupInitialStates(SegmentId segment);
    PrefillResult prefillWithInitializerModel(SegmentId segment, std::vector<s32> const& initialPrompt, std::vector<s32> const& suffixPrompt);
    // Prefill with the (combined) step model: initial prompt + encoder states with an empty cache, then the suffix prompt
    PrefillResult prefillWithStepModel(SegmentId segment, std::vector<s32> const& initialPrompt, std::vector<s32> const& suffixPrompt);
    // One call of the step model for one segment: the new positions `tokens` + `embeds` (may be nullptr) against the
    // given prefix states of `prefixLength` positions (0: empty cache)
    PrefillResult runStepModelChunk(std::vector<s32> const& tokens, Onnx::Value const* embeds, std::vector<HistoryState const*> const& prefixStates, size_t prefixLength);
    void          cacheStatesAndScores(std::vector<PrefixSpeechLmScoringContextRef> const& scoringContextBatch);

    void setupEncoderStatesValue(SegmentId segment);
    void setupEncoderStatesSizeValue(SegmentId segment);

    void logScoringBreakdown() const override;

    std::vector<s32> initialPromptLabels_;
    std::vector<s32> suffixPromptLabels_;
    bool             blankUpdatesHistory_;
    bool             silenceUpdatesHistory_;
    bool             loopUpdatesHistory_;
    size_t           maxBatchSize_;
    size_t           maxCachedScores_;
    bool             prefillWithStepModel_;
    int64_t          stepEmbedsDim_;  // feature dim of the "embeds" input of the step model, if it has one

    std::shared_ptr<Onnx::Model> initializerOnnxModel_;
    std::shared_ptr<Onnx::Model> stepOnnxModel_;

    std::unique_ptr<StateManager>        stateManager_;
    std::vector<Onnx::OnnxStateVariable> stateVariables_;
    CompressedVectorFactoryPtr<float>    stateVectorFactory_;

    std::string initializerInitialPromptName_;
    std::string initializerInitialPromptLengthName_;
    std::string initializerEncoderStatesName_;
    std::string initializerEncoderStatesSizeName_;
    std::string initializerSuffixPromptName_;
    std::string initializerSuffixPromptLengthName_;
    std::string initializerScoresName_;

    std::string stepTokenName_;
    std::string stepTokenLengthName_;
    std::string stepEmbedsName_;
    std::string stepEmbedsLengthName_;
    std::string stepPrefixLengthName_;
    std::string stepScoresName_;


    // Includes both the initializer and the step session runs
    Core::StopWatch onnxSessionTime_;
    // Merging the prefix states into the batched session inputs and splitting the outputs back up
    Core::StopWatch stateMarshallingTime_;
    Core::StopWatch contextPreparationTime_;

    std::unordered_map<SegmentId, SegmentData> segmentData_;
};

}  // namespace Nn

#endif  // PREFIX_SPEECH_LM_ONNX_LABEL_SCORER_HH
