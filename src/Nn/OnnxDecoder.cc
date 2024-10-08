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

#include "OnnxDecoder.hh"
#include <Core/Assertions.hh>
#include <Core/ReferenceCounting.hh>
#include <Flow/Timestamp.hh>
#include <Math/FastMatrix.hh>
#include <Mm/Module.hh>
#include <Speech/Types.hh>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include "LabelHistory.hh"
#include "LabelScorer.hh"

namespace Nn {

/*
 * =============================
 * === LimitedCtxOnnxDecoder ===
 * =============================
 */

const Core::ParameterInt LimitedCtxOnnxDecoder::paramStartLabelIndex(
        "start-label-index",
        "Initial history in the first step is filled with this label index.",
        0);

const Core::ParameterInt LimitedCtxOnnxDecoder::paramHistoryLength(
        "history-length",
        "Number of previous labels that are passed as history.",
        1);

const Core::ParameterBool LimitedCtxOnnxDecoder::paramBlankUpdatesHistory(
        "blank-updates-history",
        "Whether previously emitted blank labels should be included in the history.",
        false);

const Core::ParameterBool LimitedCtxOnnxDecoder::paramLoopUpdatesHistory(
        "loop-updates-history",
        "Whether in the case of loop transitions every repeated emission should be separately included in the history.",
        false);

const Core::ParameterBool LimitedCtxOnnxDecoder::paramVerticalLabelTransition(
        "vertical-label-transition",
        "Whether (non-blank) label transitions should be vertical, i.e. not increase the time step.",
        false);

const Core::ParameterInt LimitedCtxOnnxDecoder::paramMaxBatchSize(
        "max-batch-size",
        "Max number of histories that can be fed into the ONNX model at once.",
        Core::Type<int>::max);

const Core::ParameterInt LimitedCtxOnnxDecoder::paramMaxCachedScores(
        "max-cached-scores",
        "Maximum size of cache that maps histories to scores. This prevents memory overflow in case of very long audio segments.",
        1000);

LimitedCtxOnnxDecoder::LimitedCtxOnnxDecoder(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config),
          startLabelIndex_(paramStartLabelIndex(config)),
          historyLength_(paramHistoryLength(config)),
          blankUpdatesHistory_(paramBlankUpdatesHistory(config)),
          loopUpdatesHistory_(paramLoopUpdatesHistory(config)),
          verticalLabelTransition_(paramVerticalLabelTransition(config)),
          maxBatchSize_(paramMaxBatchSize(config)),
          maxCachedScores_(paramMaxCachedScores(config)),
          session_(select("session")),
          validator_(select("validator")),
          mapping_(select("io-map"), ioSpec_),
          encoderStateName_(mapping_.getOnnxName("encoder-state")),
          historyName_(mapping_.getOnnxName("history")),
          scoresName_(mapping_.getOnnxName("scores")) {
    validator_.validate(ioSpec_, mapping_, session_);
}

const std::vector<Onnx::IOSpecification> LimitedCtxOnnxDecoder::ioSpec_ = {
        Onnx::IOSpecification{
                "encoder-state",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -2}, {1, -2}}},
        Onnx::IOSpecification{
                "history",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{-1, -2}}},
        Onnx::IOSpecification{
                "scores",
                Onnx::IODirection::OUTPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -2}}}};

void LimitedCtxOnnxDecoder::reset() {
    Precursor::reset();
    scoreCache_.clear();
}

Core::Ref<const LabelHistory> LimitedCtxOnnxDecoder::getStartHistory() {
    auto hist = Core::ref(new SeqStepLabelHistory());
    hist->labelSeq.resize(historyLength_, startLabelIndex_);
    return hist;
}

Core::Ref<const LabelHistory> LimitedCtxOnnxDecoder::extendedHistory(LabelScorer::Request request) {
    SeqStepLabelHistoryRef history(dynamic_cast<const SeqStepLabelHistory*>(request.history.get()));

    bool pushToken     = false;
    bool incrementTime = false;
    switch (request.transitionType) {
        case LabelScorer::TransitionType::BLANK_LOOP:
            pushToken     = blankUpdatesHistory_ and loopUpdatesHistory_;
            incrementTime = true;
            break;
        case LabelScorer::TransitionType::LABEL_TO_BLANK:
            pushToken     = blankUpdatesHistory_;
            incrementTime = true;
            break;
        case LabelScorer::TransitionType::LABEL_LOOP:
            pushToken     = loopUpdatesHistory_;
            incrementTime = not verticalLabelTransition_;
            break;
        case LabelScorer::TransitionType::BLANK_TO_LABEL:
        case LabelScorer::TransitionType::LABEL_TO_LABEL:
            pushToken     = true;
            incrementTime = not verticalLabelTransition_;
            break;
        default:
            error() << "Unknown transition type " << request.transitionType;
    }

    // If history is not going to be modified, return the original one to avoid copying
    if (not pushToken and not incrementTime) {
        return request.history;
    }

    Core::Ref<SeqStepLabelHistory> newHistory(new SeqStepLabelHistory(history->labelSeq, history->currentStep));
    if (pushToken) {
        newHistory->labelSeq.push_back(request.nextToken);
        newHistory->labelSeq.erase(newHistory->labelSeq.begin());
    }
    if (incrementTime) {
        ++newHistory->currentStep;
    }
    return newHistory;
}

std::optional<std::pair<std::vector<Score>, Core::CollapsedVector<Speech::TimeframeIndex>>> LimitedCtxOnnxDecoder::getScoresWithTime(const std::vector<LabelScorer::Request>& requests) {
    /*
     * Collect all requests that are based on the same timestep (-> same encoder state) and
     * group them together
     */
    std::unordered_map<size_t, std::vector<size_t>> requestsWithTimestep;  // Maps timestep to list of all indices of requests with that timestep

    // Timeframe return value can already be set along the way
    Core::CollapsedVector<Speech::TimeframeIndex> timeframeResults;
    timeframeResults.reserve(requests.size());

    for (size_t b = 0ul; b < requests.size(); ++b) {
        SeqStepLabelHistoryRef history(dynamic_cast<const SeqStepLabelHistory*>(requests[b].history.get()));
        auto                   step = history->currentStep;
        if (step >= encoderOutputBuffer_.size()) {
            // Early exit if at least one of the histories is not scorable yet
            return {};
        }
        timeframeResults.push_back(step);

        // Create new vector if step value isn't present in map yet
        auto [it, inserted] = requestsWithTimestep.emplace(step, std::vector<size_t>());
        it->second.push_back(b);
    }

    /*
     * Iterate over distinct timesteps
     */
    for (const auto& [timestep, requestIndices] : requestsWithTimestep) {
        /*
         * Identify unique histories that still need session runs
         */
        std::unordered_set<SeqStepLabelHistoryRef, SeqStepLabelHistoryHash, SeqStepLabelHistoryEq> uniqueUncachedHistories;

        for (auto requestIndex : requestIndices) {
            SeqStepLabelHistoryRef historyPtr(dynamic_cast<const SeqStepLabelHistory*>(requests[requestIndex].history.get()));
            if (scoreCache_.find(historyPtr) == scoreCache_.end()) {
                // Group by unique history
                uniqueUncachedHistories.emplace(historyPtr);
            }
        }

        if (uniqueUncachedHistories.empty()) {
            continue;
        }

        std::vector<SeqStepLabelHistoryRef> historyBatch;
        historyBatch.reserve(std::min(uniqueUncachedHistories.size(), maxBatchSize_));
        for (auto history : uniqueUncachedHistories) {
            historyBatch.push_back(history);
            if (historyBatch.size() == maxBatchSize_) {  // Batch is full -> forward now
                forwardBatch(historyBatch);
                historyBatch.clear();
            }
        }

        forwardBatch(historyBatch);  // Forward remaining histories
    }

    /*
     * Assign from cache map to result vector
     */
    std::vector<Score> scoreResults;
    scoreResults.reserve(requests.size());
    for (const auto& request : requests) {
        SeqStepLabelHistoryRef history(dynamic_cast<const SeqStepLabelHistory*>(request.history.get()));

        auto cacheResult = scoreCache_.find(history);
        verify(cacheResult != scoreCache_.end());

        scoreResults.push_back(cacheResult->second.at(request.nextToken));
    }

    // Avoid memory overflow due to score cache
    if (scoreCache_.size() > maxCachedScores_) {
        scoreCache_.clear();
    }

    return std::make_pair(scoreResults, timeframeResults);
}

std::optional<std::pair<Score, Speech::TimeframeIndex>> LimitedCtxOnnxDecoder::getScoreWithTime(LabelScorer::Request request) {
    auto result = getScoresWithTime({request});
    if (not result.has_value()) {
        return {};
    }
    return std::make_pair(result->first.front(), result->second.front());
}

void LimitedCtxOnnxDecoder::forwardBatch(const std::vector<SeqStepLabelHistoryRef> historyBatch) {
    if (historyBatch.empty()) {
        return;
    }

    /*
     * Create session inputs
     */

    // All requests in this iteration share the same encoder state which is set up here
    Math::FastMatrix<f32>
            encoderMat(encoderOutputBuffer_.front()->size(), 1);
    auto&   encoderState = encoderOutputBuffer_[historyBatch.front()->currentStep];
    std::copy(encoderState->begin(), encoderState->end(), encoderMat.begin());

    // Create batched history input
    Math::FastMatrix<s32> historyMat(historyLength_, historyBatch.size());
    for (size_t b = 0ul; b < historyBatch.size(); ++b) {
        auto history = historyBatch[b];
        std::copy(history->labelSeq.begin(), history->labelSeq.end(), &(historyMat.at(0, b)));  // Pointer to first element in column b
    }

    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;
    sessionInputs.emplace_back(encoderStateName_, Onnx::Value::create(encoderMat, true));
    sessionInputs.emplace_back(historyName_, Onnx::Value::create(historyMat, true));

    /*
     * Run session
     */
    // auto t_start = std::chrono::steady_clock::now();

    std::vector<Onnx::Value> sessionOutputs;
    session_.run(std::move(sessionInputs), {scoresName_}, sessionOutputs);

    // auto t_end     = std::chrono::steady_clock::now();
    // auto t_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();  // in seconds
    // log("Ran decoder with encoder state of shape (%u x %u) and history batch of shape (%u x %u) in %.4f seconds.", encoderMat.nColumns(), encoderMat.nRows(), historyMat.nColumns(), historyMat.nRows(), t_elapsed);

    /*
     * Put resulting scores into cache map
     */
    for (size_t b = 0ul; b < historyBatch.size(); ++b) {
        std::vector<f32> scoreVec;
        sessionOutputs.front().get(b, scoreVec);
        scoreCache_.emplace(historyBatch[b], std::move(scoreVec));
    }
}
}  // namespace Nn
