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

#ifndef STATEFUL_ONNX_LABEL_SCORER_HH
#define STATEFUL_ONNX_LABEL_SCORER_HH

#include <Core/Component.hh>
#include <Core/Configuration.hh>
#include <Core/FIFOCache.hh>
#include <Core/ReferenceCounting.hh>
#include <Mm/FeatureScorer.hh>
#include <Onnx/Model.hh>
#include <Speech/Feature.hh>
#include <optional>
#include "BufferedLabelScorer.hh"
#include "ScoringContext.hh"

#include <Onnx/IOSpecification.hh>
#include <Onnx/Session.hh>

namespace Nn {

/*
 * Label Scorer that performs scoring by forwarding a collection of hidden state tensors
 * through an ONNX model. The hidden state is initialized and updated with separate ONNX
 * sessions. For state initialization, the session receives all input encoder states
 * and for state update the session gets the previous states, all input features and the
 * next token index
 */
class StatefulOnnxLabelScorer : public BufferedLabelScorer {
    using Precursor = BufferedLabelScorer;

    static const Core::ParameterBool paramBlankUpdatesHistory;
    static const Core::ParameterBool paramLoopUpdatesHistory;
    static const Core::ParameterInt  paramMaxBatchSize;
    static const Core::ParameterInt  paramMaxCachedScores;

public:
    StatefulOnnxLabelScorer(const Core::Configuration& config);
    virtual ~StatefulOnnxLabelScorer() = default;

    void reset() override;

    // Hardcoded to use all-zero tensors as first hidden state
    // If startLabelIndex is set, forward that through the state updater to obtain the start history
    Core::Ref<const ScoringContext> getInitialScoringContext() override;

    // Forward hidden-state through state-updater ONNX model
    Core::Ref<const ScoringContext> extendedScoringContext(LabelScorer::Request const& request) override;

    // Add a single encoder outputs to buffer
    void addInput(DataView const& input) override;

    std::optional<LabelScorer::ScoreWithTime>   computeScoreWithTime(LabelScorer::Request const& request) override;
    std::optional<LabelScorer::ScoresWithTimes> computeScoresWithTimes(std::vector<LabelScorer::Request> const& requests) override;

protected:
    size_t getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const override;

private:
    // Forward a batch of histories through the ONNX model and put the resulting scores into the score cache
    void forwardBatch(std::vector<HiddenStateScoringContextRef> const& historyBatch);

    HiddenStateRef updatedHiddenState(HiddenStateRef const& hiddenState, LabelIndex nextToken);

    // Since the hidden-state matrix depends on the encoder time axis, we cannot create properly create hidden-states until all encoder states have been passed.
    // So getStartHistory sets the initial hidden-state to a sentinel value (empty Ref) and when other functions such as `extendedHistory` and `getScoresWithTime`
    // encounter this sentinel value they call `computeInitialHiddenState` instead to get a usable hidden-state.
    HiddenStateRef computeInitialHiddenState();

    void setupEncoderStatesValue();
    void setupEncoderStatesSizeValue();

    HiddenStateRef initialHiddenState_;

    // Map input/output names of onnx models to hidden state names taken from state initializer model
    std::unordered_map<std::string, std::string> initializerOutputToStateNameMap_;
    std::unordered_map<std::string, std::string> updaterInputToStateNameMap_;
    std::unordered_map<std::string, std::string> updaterOutputToStateNameMap_;
    std::unordered_map<std::string, std::string> scorerInputToStateNameMap_;

    bool   blankUpdatesHistory_;
    bool   loopUpdatesHistory_;
    size_t maxBatchSize_;

    Onnx::Model scorerOnnxModel_;
    Onnx::Model stateInitializerOnnxModel_;
    Onnx::Model stateUpdaterOnnxModel_;

    std::string scoresName_;

    std::string initializerEncoderStatesName_;
    std::string initializerEncoderStatesSizeName_;

    std::string updaterEncoderStatesName_;
    std::string updaterEncoderStatesSizeName_;
    std::string updaterTokenName_;

    // Store the onnx values with all encoder states and lengths inside so that it doesn't have to be recomputed every time
    Onnx::Value encoderStatesValue_;
    Onnx::Value encoderStatesSizeValue_;

    Core::FIFOCache<HiddenStateScoringContextRef, std::vector<Score>, ScoringContextHash, ScoringContextEq> scoreCache_;
};

}  // namespace Nn

#endif  // STATEFUL_ONNX_LABEL_SCORER_HH
