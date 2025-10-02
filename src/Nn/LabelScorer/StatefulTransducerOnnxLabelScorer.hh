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

#ifndef STATEFUL_TRANSDUCER_ONNX_LABEL_SCORER_HH
#define STATEFUL_TRANSDUCER_ONNX_LABEL_SCORER_HH

#include <optional>

#include <Core/Component.hh>
#include <Core/Configuration.hh>
#include <Core/FIFOCache.hh>
#include <Core/ReferenceCounting.hh>
#include <Mm/FeatureScorer.hh>
#include <Onnx/IOSpecification.hh>
#include <Onnx/Model.hh>
#include <Onnx/Session.hh>
#include <Speech/Feature.hh>

#include "BufferedLabelScorer.hh"
#include "ScoringContext.hh"

namespace Nn {

/*
 * Label Scorer that performs scoring by forwarding hidden states through an ONNX model.
 * This Label Scorer requires three ONNX models:
 *  - A State Initializer which produces the hidden states for the first step
 *  - A State Updater which produces updated hidden states based on the previous hidden states and the next token
 *  - A Scorer which computes scores based on the current input feature and the hidden states
 *
 * The hidden states can be any number of ONNX tensors of any shape and type.
 * Each ONNX model must have metadata that specifies the mapping of its input and output names to the corresponding state names.
 * These state names need to be consistent over all three models.
 *
 * For example:
 *   - The State Initializer has output called "lstm_c" and {"lstm_c": "LSTM_C"} in its metadata
 *   - The State Updater has input "lstm_c_in", output "lstm_c_out" and {"lstm_c_in": "LSTM_C", "lstm_c_out": "LSTM_C"} in its metadata
 *   - The Scorer has input "lstm_c" and {"lstm_c": "LSTM_C"} in its metadata
 * Here, "LSTM_C" is the state name and the same across all three models while the specific input/output names are arbitrary.
 *
 * The State Initializer must have all states as output.
 * The State Updater must have a subset of states as input and all states as output.
 * The Scorer must have a subset of states and a feature as input.
 *
 * A common use case for this Label Scorer would be a Transducer model with unlimited context.
 *
 * Note: This LabelScorer is similar to the `StatefulOnnxLabelScorer`. The difference is that in this one the ScoringContext also
 * contains the current step and the input feature at the current step is fed to the Scorer. Furthermore, the state initializer
 * and updater here only take tokens and no input features.
 */
class StatefulTransducerOnnxLabelScorer : public BufferedLabelScorer {
    using Precursor = BufferedLabelScorer;

    static const Core::ParameterBool paramBlankUpdatesHistory;
    static const Core::ParameterBool paramLoopUpdatesHistory;
    static const Core::ParameterBool paramVerticalLabelTransition;
    static const Core::ParameterInt  paramMaxBatchSize;
    static const Core::ParameterInt  paramMaxCachedScores;

public:
    StatefulTransducerOnnxLabelScorer(const Core::Configuration& config);
    virtual ~StatefulTransducerOnnxLabelScorer() = default;

    void reset() override;

    // If startLabelIndex is set, forward that through the state updater to obtain the start history
    Core::Ref<const ScoringContext> getInitialScoringContext() override;

    // Forward hidden-state through state-updater ONNX model
    Core::Ref<const ScoringContext> extendedScoringContext(LabelScorer::Request const& request) override;

    std::optional<LabelScorer::ScoreWithTime>   computeScoreWithTime(LabelScorer::Request const& request) override;
    std::optional<LabelScorer::ScoresWithTimes> computeScoresWithTimes(std::vector<LabelScorer::Request> const& requests) override;

protected:
    size_t getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const override;

private:
    // Forward a batch of histories through the ONNX model and put the resulting scores into the score cache
    void forwardBatch(std::vector<StepOnnxHiddenStateScoringContextRef> const& historyBatch);

    // Computes new hidden state based on previous hidden state and next token through state-updater call
    OnnxHiddenStateRef updatedHiddenState(OnnxHiddenStateRef const& hiddenState, LabelIndex nextToken);

    // Replace hidden-state in scoringContext with an updated version that includes the last label
    void finalizeScoringContext(StepOnnxHiddenStateScoringContextRef const& scoringContext);

    void setupEncoderStatesValue();
    void setupEncoderStatesSizeValue();

    bool   blankUpdatesHistory_;
    bool   loopUpdatesHistory_;
    bool   verticalLabelTransition_;
    size_t maxBatchSize_;

    Onnx::Model scorerOnnxModel_;
    Onnx::Model stateInitializerOnnxModel_;
    Onnx::Model stateUpdaterOnnxModel_;

    StepOnnxHiddenStateScoringContextRef initialScoringContext_;

    // Map input/output names of onnx models to hidden state names taken from state initializer model
    std::unordered_map<std::string, std::string> initializerOutputToStateNameMap_;
    std::unordered_map<std::string, std::string> updaterInputToStateNameMap_;
    std::unordered_map<std::string, std::string> updaterOutputToStateNameMap_;
    std::unordered_map<std::string, std::string> scorerInputToStateNameMap_;

    std::string scorerInputFeatureName_;
    std::string scorerScoresName_;

    std::string updaterTokenName_;

    Core::FIFOCache<StepOnnxHiddenStateScoringContextRef, std::vector<Score>, ScoringContextHash, ScoringContextEq> scoreCache_;
};

}  // namespace Nn

#endif  // STATEFUL_ONNX_LABEL_SCORER_HH
