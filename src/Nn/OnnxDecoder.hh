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

#ifndef ONNX_DECODER_HH
#define ONNX_DECODER_HH

#include <Core/Component.hh>
#include <Core/Configuration.hh>
#include <Core/ReferenceCounting.hh>
#include <Mm/FeatureScorer.hh>
#include <Speech/Feature.hh>
#include <optional>
#include "Decoder.hh"
#include "LabelHistory.hh"
#include "LabelScorer.hh"

#include <Onnx/IOSpecification.hh>
#include <Onnx/Session.hh>
#include <unordered_map>

namespace Nn {

class LimitedCtxOnnxDecoder : public Decoder {
    using Precursor = Decoder;

    static const Core::ParameterInt  paramStartLabelIndex;
    static const Core::ParameterInt  paramHistoryLength;
    static const Core::ParameterBool paramBlankUpdatesHistory;
    static const Core::ParameterBool paramLoopUpdatesHistory;
    static const Core::ParameterBool paramVerticalLabelTransition;
    static const Core::ParameterInt  paramMaxBatchSize;
    static const Core::ParameterInt  paramMaxCachedScores;

public:
    LimitedCtxOnnxDecoder(const Core::Configuration& config);
    virtual ~LimitedCtxOnnxDecoder() = default;

    void                                                                                        reset() override;
    Core::Ref<const LabelHistory>                                                               getStartHistory() override;
    Core::Ref<const LabelHistory>                                                               extendedHistory(LabelScorer::Request request) override;
    std::optional<std::pair<Score, Speech::TimeframeIndex>>                                     getScoreWithTime(const LabelScorer::Request request) override;
    std::optional<std::pair<std::vector<Score>, Core::CollapsedVector<Speech::TimeframeIndex>>> getScoresWithTime(const std::vector<LabelScorer::Request>& requests) override;

private:
    // Forward a batch of histories through the ONNX model and put the resulting scores into the score cache
    // Assumes that all histories in the batch are based on the same timestep
    void forwardBatch(const std::vector<SeqStepLabelHistoryRef> historyBatch);

    size_t startLabelIndex_;
    size_t historyLength_;
    bool   blankUpdatesHistory_;
    bool   loopUpdatesHistory_;
    bool   verticalLabelTransition_;
    size_t maxBatchSize_;
    size_t maxCachedScores_;

    Onnx::Session                                   session_;
    static const std::vector<Onnx::IOSpecification> ioSpec_;  // fixed to "encoder-state", "history", "scores"
    Onnx::IOValidator                               validator_;
    const Onnx::IOMapping                           mapping_;

    std::string encoderStateName_;
    std::string historyName_;
    std::string scoresName_;

    std::unordered_map<SeqStepLabelHistoryRef, std::vector<Score>, SeqStepLabelHistoryHash, SeqStepLabelHistoryEq> scoreCache_;  // TODO: Re-do as ring buffer
};

}  // namespace Nn

#endif  // ONNX_DECODER_HH
