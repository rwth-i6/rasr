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

#include <unordered_map>

#include <Core/FIFOCache.hh>
#include <Onnx/Model.hh>
#include <Onnx/PersistentInput.hh>

#include "BufferedLabelScorer.hh"
#include "ModelCache.hh"
#include "ScoringContext.hh"

namespace Nn {

/*
 * Scoring context for ONNX models with state managed by a StateManager.
 * The state stores only the slice produced for the most recent token.
 */
struct StateManagedOnnxScoringContext : public ScoringContext {
    using StateManager = AbstractStateManager<Onnx::Value, Onnx::OnnxStateVariable>;
    using HistoryState = StateManager::HistoryState;

    std::vector<LabelIndex>                                 labelSeq;
    mutable Core::Ref<StateManagedOnnxScoringContext const> parent;

    mutable std::shared_ptr<HistoryState> state;

    StateManagedOnnxScoringContext(HistoryState&& initialState);
    StateManagedOnnxScoringContext(std::vector<LabelIndex>&&                       labelSeq,
                                   Core::Ref<StateManagedOnnxScoringContext const> parent);

    bool   isEqual(ScoringContextRef const& other) const override;
    size_t hash() const override;
};

typedef Core::Ref<StateManagedOnnxScoringContext const> StateManagedOnnxScoringContextRef;

/*
 * LabelScorer for ONNX models whose hidden-state management is done by a RASR StateManager.
 * Each scoring context stores only the state slice produced
 * for its most recent token plus a parent pointer, which allows transformer KV caches to
 * be represented as a tree instead of duplicating the full prefix state per context.
 *
 * With multiple segments, contexts of different segments are forwarded together if the model doesn't need encoder
 * states or `stack-encoder-states` is enabled, otherwise per segment.
 */
class StateManagedOnnxLabelScorer : public BufferedLabelScorer {
public:
    StateManagedOnnxLabelScorer(Core::Configuration const& config, ModelCache& modelCache);
    virtual ~StateManagedOnnxLabelScorer() = default;

    void addSegment(SegmentId segment) override;
    void removeSegment(SegmentId segment) override;
    void addSegmentInput(SegmentId segment, DataView const& input) override;

    ScoringContextRef getInitialSegmentScoringContext(SegmentId segment) override;
    ScoringContextRef extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) override;

protected:
    void resetInternal() override;
    void resetStatisticsInternal() override;

    std::optional<ScoreAccessorRef>              computeScoreAccessor(ScoringContextRef scoringContext) override;
    std::vector<std::optional<ScoreAccessorRef>> computeScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) override;
    size_t                                       getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const override;

private:
    using Precursor    = BufferedLabelScorer;
    using StateManager = AbstractStateManager<Onnx::Value, Onnx::OnnxStateVariable>;
    using HistoryState = StateManager::HistoryState;

    static const Core::ParameterIntVector paramStartLabels;
    static const Core::ParameterBool      paramBlankUpdatesHistory;
    static const Core::ParameterBool      paramSilenceUpdatesHistory;
    static const Core::ParameterBool      paramLoopUpdatesHistory;
    static const Core::ParameterInt       paramMaxBatchSize;
    static const Core::ParameterInt       paramMaxCachedScores;
    static const Core::ParameterBool      paramStackEncoderStates;

    std::vector<size_t> startLabels_;
    bool                blankUpdatesHistory_;
    bool                silenceUpdatesHistory_;
    bool                loopUpdatesHistory_;
    size_t              maxBatchSize_;
    size_t              maxCachedScores_;
    bool                stackEncoderStates_;

    std::shared_ptr<Onnx::Model> onnxModel_;

    std::unique_ptr<StateManager>        stateManager_;
    std::vector<Onnx::OnnxStateVariable> stateVariables_;
    CompressedVectorFactoryPtr<float>    stateVectorFactory_;

    std::string tokenName_;
    std::string tokenLengthName_;
    std::string prefixLengthName_;
    std::string scoresName_;
    std::string encoderStatesName_;
    std::string encoderStatesSizeName_;
    std::string encoderStatesIndexName_;

    Core::StopWatch onnxSessionTime_;
    // Merging the prefix states into the batched session inputs and splitting the outputs back up
    Core::StopWatch stateMarshallingTime_;
    Core::StopWatch contextPreparationTime_;

    struct SegmentData {
        // Passed to every session run, as views to avoid copies (and uploaded to the device only once for CUDA sessions)
        Onnx::PersistentInput encoderStatesValue;
        Onnx::Value           encoderStatesSizeValue;

        Core::FIFOCache<StateManagedOnnxScoringContextRef, std::shared_ptr<std::vector<Score>>, ScoringContextHash, ScoringContextEq> scoreCache;
        Core::FIFOCache<StateManagedOnnxScoringContextRef, std::shared_ptr<HistoryState>, ScoringContextHash, ScoringContextEq>       stateCache;

        SegmentData(size_t maxCachedScores);
    };

    std::unordered_map<SegmentId, SegmentData> segmentData_;

    // Data of the segment, created if it doesn't exist yet
    SegmentData& segmentData(SegmentId segment);

    bool usesEncoderStates() const;

    void setupEncoderStatesValue(SegmentId segment);
    void setupEncoderStatesSizeValue(SegmentId segment);

    // Forward a batch of scoringContexts through the ONNX model and put the resulting states and scores into the caches
    void cacheStatesAndScores(std::vector<StateManagedOnnxScoringContextRef> const& scoringContextBatch);

    void logScoringBreakdown() const override;
};

}  // namespace Nn

#endif  // STATE_MANAGED_ONNX_LABEL_SCORER_HH
