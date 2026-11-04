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

#include <Core/FIFOCache.hh>
#include <Onnx/Model.hh>

#include "BufferedLabelScorer.hh"
#include "ModelCache.hh"
#include "ScoringContext.hh"

namespace Tokenizer {
class AbstractTokenizer;
}  // namespace Tokenizer

namespace Nn {

/*
 * Scoring context for prefix speech LMs.
 * `labelSeq` contains only labels emitted by search, not from the prompt and speech-embeddings.
 * `historyLength` contains the total LM cache length, including prompt and speech-embedding positions.
 * It is only known once the context has been scored and is 0 before.
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
    PrefixSpeechLmScoringContext(std::vector<LabelIndex>&& labelSeq, Core::Ref<PrefixSpeechLmScoringContext const> parent);

    bool   isEqual(ScoringContextRef const& other) const override;
    size_t hash() const override;
};

typedef Core::Ref<PrefixSpeechLmScoringContext const> PrefixSpeechLmScoringContextRef;

/*
 * LabelScorer for prefix speech LMs that first initialize with
 * [initial prompt tokens, speech encoder states in LM embedding space, suffix prompt tokens]
 * before scoring hypothesis tokens one-by-one as usual.
 *
 * The prompts are given either as text, which is tokenized with the configured tokenizer, or directly as token ids.
 *
 * The LabelScorer requires two ONNX models, one which takes the prompts and encoder states
 * to produce the initial states and score distribution and one which takes a new token and
 * existing states to output updated states and scores after incorporating the token.
 * The states are managed with a StateManager.
 */
class PrefixSpeechLmOnnxLabelScorer : public BufferedLabelScorer {
public:
    PrefixSpeechLmOnnxLabelScorer(Core::Configuration const& config, ModelCache& modelCache);
    virtual ~PrefixSpeechLmOnnxLabelScorer() = default;

    void addInput(DataView const& input) override;

    ScoringContextRef getInitialScoringContext() override;
    ScoringContextRef extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) override;

protected:
    void resetInternal() override;

    std::optional<ScoreAccessorRef>              computeScoreAccessor(ScoringContextRef scoringContext) override;
    std::vector<std::optional<ScoreAccessorRef>> computeScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) override;
    size_t                                       getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const override;

private:
    using Precursor    = BufferedLabelScorer;
    using StateManager = AbstractStateManager<Onnx::Value, Onnx::OnnxStateVariable>;
    using HistoryState = StateManager::HistoryState;

    static const Core::ParameterString    paramInitialPrompt;
    static const Core::ParameterString    paramSuffixPrompt;
    static const Core::ParameterIntVector paramInitialPromptLabels;
    static const Core::ParameterIntVector paramSuffixPromptLabels;
    static const Core::ParameterBool      paramBlankUpdatesHistory;
    static const Core::ParameterBool      paramSilenceUpdatesHistory;
    static const Core::ParameterBool      paramLoopUpdatesHistory;
    static const Core::ParameterInt       paramMaxBatchSize;
    static const Core::ParameterInt       paramMaxCachedScores;

    std::vector<s32> initialPromptLabels_;
    std::vector<s32> suffixPromptLabels_;
    bool             blankUpdatesHistory_;
    bool             silenceUpdatesHistory_;
    bool             loopUpdatesHistory_;
    size_t           maxBatchSize_;

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
    std::string stepPrefixLengthName_;
    std::string stepScoresName_;

    PrefixSpeechLmScoringContextRef initialContext_;
    // Kept outside of `scoreCache_` since the prefill can't be recomputed once the inputs are cleaned up
    std::shared_ptr<std::vector<Score>> initialScores_;

    // Session runs of the initializer model that prefill the prompt and encoder states
    Core::StopWatch initializerSessionTime_;
    Core::StopWatch stepSessionTime_;
    // Merging the prefix states into the batched session inputs and splitting the outputs back up
    Core::StopWatch stateMarshallingTime_;
    Core::StopWatch contextPreparationTime_;

    Core::FIFOCache<PrefixSpeechLmScoringContextRef, std::shared_ptr<std::vector<Score>>, ScoringContextHash, ScoringContextEq> scoreCache_;
    Core::FIFOCache<PrefixSpeechLmScoringContextRef, std::shared_ptr<HistoryState>, ScoringContextHash, ScoringContextEq>       stateCache_;

    // Get the token ids of a prompt from either its text parameter (tokenized with `tokenizer`) or its label parameter.
    // `tokenizer` may be null if the text parameter is not set.
    std::vector<s32> getPromptLabels(Core::ParameterString const&        textParam,
                                     Core::ParameterIntVector const&     labelsParam,
                                     Tokenizer::AbstractTokenizer const* tokenizer) const;

    // Replace the escape sequences \n, \t, \r, \\ and \xHH in a text prompt
    std::string unescapePrompt(std::string const& text) const;

    // Whether the prompt and encoder states have been prefilled for the current segment
    bool isPrefilled() const;

    void        setupInitialStates();
    Onnx::Value createEncoderStatesValue() const;

    // Forward a batch of scoringContexts through the step ONNX model and put the resulting states and scores into the caches
    void cacheStatesAndScores(std::vector<PrefixSpeechLmScoringContextRef> const& scoringContextBatch);

    void logScoringBreakdown() const override;
};

}  // namespace Nn

#endif  // PREFIX_SPEECH_LM_ONNX_LABEL_SCORER_HH
