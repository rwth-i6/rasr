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

#ifndef STATEFUL_FULL_ENC_ONNX_DECODER_HH
#define STATEFUL_FULL_ENC_ONNX_DECODER_HH

#include <Core/Component.hh>
#include <Core/Configuration.hh>
#include <Core/FIFOCache.hh>
#include <Core/ReferenceCounting.hh>
#include <Mm/FeatureScorer.hh>
#include <Speech/Feature.hh>
#include <optional>
#include "Decoder.hh"
#include "LabelHistory.hh"
#include "LabelScorer.hh"

#include <Onnx/IOSpecification.hh>
#include <Onnx/Session.hh>

namespace Nn {

class StatefulFullEncOnnxDecoder : public Decoder {
    using Precursor = Decoder;

    static const Core::ParameterBool paramBlankUpdatesHistory;
    static const Core::ParameterBool paramLoopUpdatesHistory;
    static const Core::ParameterInt  paramMaxBatchSize;
    static const Core::ParameterInt  paramMaxCachedScores;

public:
    StatefulFullEncOnnxDecoder(const Core::Configuration& config);
    virtual ~StatefulFullEncOnnxDecoder() = default;

    void reset() override;

    // Hardcoded to use all-zero tensors as first hidden state
    // If startLabelIndex is set, forward that through the state updater to obtain the start history
    Core::Ref<const LabelHistory> getStartHistory() override;

    // Forward hidden-state through state-updater ONNX model
    Core::Ref<const LabelHistory> extendedHistory(LabelScorer::Request request) override;

    // Add a single encoder outputs to buffer
    void addEncoderOutput(FeatureVectorRef encoderOutput) override;

    std::optional<std::pair<Score, Speech::TimeframeIndex>>                                     getScoreWithTime(const LabelScorer::Request request) override;
    std::optional<std::pair<std::vector<Score>, Core::CollapsedVector<Speech::TimeframeIndex>>> getScoresWithTime(const std::vector<LabelScorer::Request>& requests) override;

private:
    // Forward a batch of histories through the ONNX model and put the resulting scores into the score cache
    void forwardBatch(const std::vector<HiddenStateLabelHistoryRef> historyBatch);

    HiddenStateRef updatedHiddenState(HiddenStateRef hiddenState, LabelIndex nextToken);

    // Since the hidden-state matrix depends on the encoder time axis, we cannot create properly create hidden-states until all encoder states have been passed.
    // So getStartHistory sets the initial hidden-state to a sentinel value (empty Ref) and when other functions such as `extendedHistory` and `getScoresWithTime`
    // encounter this sentinel value they call `computeInitialHiddenState` instead to get a usable hidden-state.
    HiddenStateRef computeInitialHiddenState();
    HiddenStateRef initialHiddenState_;

    bool   blankUpdatesHistory_;
    bool   loopUpdatesHistory_;
    size_t maxBatchSize_;

    size_t hiddenStateVecSize_;
    size_t hiddenStateMatSize_;

    Onnx::Session                                   decoderSession_;
    static const std::vector<Onnx::IOSpecification> decoderIoSpec_;  // fixed to "encoder-states", "hidden-state", "scores"
    Onnx::IOValidator                               decoderValidator_;
    const Onnx::IOMapping                           decoderMapping_;

    Onnx::Session                                   stateInitializerSession_;
    static const std::vector<Onnx::IOSpecification> stateInitializerIoSpec_;  // fixed to "hidden-state-in", "token", "hidden-state-out"
    Onnx::IOValidator                               stateInitializerValidator_;
    const Onnx::IOMapping                           stateInitializerMapping_;

    Onnx::Session                                   stateUpdaterSession_;
    static const std::vector<Onnx::IOSpecification> stateUpdaterIoSpec_;  // fixed to "hidden-state-in", "token", "hidden-state-out"
    Onnx::IOValidator                               stateUpdaterValidator_;
    const Onnx::IOMapping                           stateUpdaterMapping_;

    std::string scoresName_;

    std::string initEncoderStatesName_;
    std::string initEncoderSizeName_;

    std::string updaterEncoderStatesName_;
    std::string updaterEncoderSizeName_;
    std::string updaterTokenName_;

    // Store the onnx values with all encoder states and lengths inside so that it doesn't have to be recomputed every time
    Onnx::Value encoderStatesValue_;
    Onnx::Value encoderStatesSizeValue_;

    Core::FIFOCache<HiddenStateLabelHistoryRef, std::vector<Score>, HiddenStateLabelHistoryHash, HiddenStateLabelHistoryEq> scoreCache_;
};

}  // namespace Nn

#endif  // STATEFUL_FULL_ENC_ONNX_DECODER_HH
