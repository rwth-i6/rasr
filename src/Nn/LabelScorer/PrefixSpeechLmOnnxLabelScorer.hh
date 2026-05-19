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
#include "ScoringContext.hh"

namespace Nn {

/*
 * LabelScorer for prefix speech LMs that first initialize with
 * [initial prompt tokens, speech encoder states in LM embedding space, suffix prompt tokens]
 * before scoring hypothesis tokens one-by-one as usual.
 *
 * The LabelScorer requires two ONNX models, one which takes the prompts and encoder states
 * to produce the initial states and score distribution and one which takes a new token and
 * existing states to output updated states and scores after incorporating the token.
 * The states are managed with a StateManager.
 */
class PrefixSpeechLmOnnxLabelScorer : public BufferedLabelScorer {
    using Precursor    = BufferedLabelScorer;
    using StateManager = AbstractStateManager<Onnx::Value, Onnx::OnnxStateVariable>;
    using HistoryState = StateManager::HistoryState;

    static const Core::ParameterIntVector paramInitialPromptLabels;
    static const Core::ParameterIntVector paramSuffixPromptLabels;
    static const Core::ParameterBool      paramBlankUpdatesHistory;
    static const Core::ParameterBool      paramLoopUpdatesHistory;
    static const Core::ParameterInt       paramMaxBatchSize;
    static const Core::ParameterInt       paramMaxCachedScores;

public:
    PrefixSpeechLmOnnxLabelScorer(Core::Configuration const& config);
    virtual ~PrefixSpeechLmOnnxLabelScorer() = default;

    void reset() override;
    void addInput(DataView const& input) override;

    ScoringContextRef getInitialScoringContext() override;
    ScoringContextRef extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) override;

    std::optional<ScoreAccessorRef>              getScoreAccessor(ScoringContextRef scoringContext) override;
    std::vector<std::optional<ScoreAccessorRef>> getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) override;

protected:
    size_t getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const override;

private:
    void setupInitialStates();
    void cacheStatesAndScores(std::vector<PrefixSpeechLmScoringContextRef> const& scoringContextBatch);

    void setupEncoderStatesValue();
    void setupEncoderStatesSizeValue();

    std::vector<s32> initialPromptLabels_;
    std::vector<s32> suffixPromptLabels_;
    bool             blankUpdatesHistory_;
    bool             loopUpdatesHistory_;
    size_t           maxBatchSize_;

    Onnx::Model initializerOnnxModel_;
    Onnx::Model stepOnnxModel_;

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

    Onnx::Value encoderStatesValue_;
    Onnx::Value encoderStatesSizeValue_;

    PrefixSpeechLmScoringContextRef initialContext_;

    Core::FIFOCache<PrefixSpeechLmScoringContextRef, std::shared_ptr<std::vector<Score>>, ScoringContextHash, ScoringContextEq> scoreCache_;
    Core::FIFOCache<PrefixSpeechLmScoringContextRef, std::shared_ptr<HistoryState>, ScoringContextHash, ScoringContextEq>       stateCache_;
};

}  // namespace Nn

#endif  // PREFIX_SPEECH_LM_ONNX_LABEL_SCORER_HH
