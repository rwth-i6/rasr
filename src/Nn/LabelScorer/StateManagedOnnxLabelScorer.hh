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

#ifndef STATE_MANAGED_ONNX_LABEL_SCORER_HH
#define STATE_MANAGED_ONNX_LABEL_SCORER_HH

#include <Core/FIFOCache.hh>
#include <Onnx/Model.hh>

#include "BufferedLabelScorer.hh"
#include "ScoringContext.hh"

namespace Nn {

/*
 * LabelScorer for ONNX models whose hidden-state management done by a Rasr StateManager.
 * Each scoring context stores only the state slice produced
 * for its most recent token plus a parent pointer, which allows transformer KV caches to
 * be represented as a tree instead of duplicating the full prefix state per context.
 */
class StateManagedOnnxLabelScorer : public BufferedLabelScorer {
    using Precursor    = BufferedLabelScorer;
    using StateManager = AbstractStateManager<Onnx::Value, Onnx::OnnxStateVariable>;
    using HistoryState = StateManager::HistoryState;

    static const Core::ParameterInt  paramStartLabelIndex;
    static const Core::ParameterBool paramBlankUpdatesHistory;
    static const Core::ParameterBool paramLoopUpdatesHistory;
    static const Core::ParameterInt  paramMaxBatchSize;
    static const Core::ParameterInt  paramMaxCachedScores;

public:
    StateManagedOnnxLabelScorer(Core::Configuration const& config);
    virtual ~StateManagedOnnxLabelScorer() = default;

    void reset() override;
    void addInput(DataView const& input) override;

    ScoringContextRef getInitialScoringContext() override;
    ScoringContextRef extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) override;

    std::optional<ScoreAccessorRef>              getScoreAccessor(ScoringContextRef scoringContext) override;
    std::vector<std::optional<ScoreAccessorRef>> getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) override;

protected:
    size_t getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const override;

private:
    void setupEncoderStatesValue();
    void setupEncoderStatesSizeValue();

    // Forward a batch of scoringContexts through the ONNX model and put the resulting states and scores into the caches
    void cacheStatesAndScores(std::vector<StateManagedOnnxScoringContextRef> const& scoringContextBatch);

    size_t startLabelIndex_;
    bool   blankUpdatesHistory_;
    bool   loopUpdatesHistory_;
    size_t maxBatchSize_;

    Onnx::Model onnxModel_;

    std::unique_ptr<StateManager>        stateManager_;
    std::vector<Onnx::OnnxStateVariable> stateVariables_;
    CompressedVectorFactoryPtr<float>    stateVectorFactory_;

    std::string tokenName_;
    std::string tokenLengthName_;
    std::string prefixLengthName_;
    std::string scoresName_;
    std::string encoderStatesName_;
    std::string encoderStatesSizeName_;

    Onnx::Value encoderStatesValue_;
    Onnx::Value encoderStatesSizeValue_;

    Core::FIFOCache<StateManagedOnnxScoringContextRef, std::shared_ptr<std::vector<Score>>, ScoringContextHash, ScoringContextEq> scoreCache_;
    Core::FIFOCache<StateManagedOnnxScoringContextRef, std::shared_ptr<HistoryState>, ScoringContextHash, ScoringContextEq>       stateCache_;
};

}  // namespace Nn

#endif  // STATE_MANAGED_ONNX_LABEL_SCORER_HH
