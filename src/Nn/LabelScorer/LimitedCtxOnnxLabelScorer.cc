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

#include "LimitedCtxOnnxLabelScorer.hh"
#include <Core/Assertions.hh>
#include <Core/ReferenceCounting.hh>
#include <Flow/Timestamp.hh>
#include <Math/FastMatrix.hh>
#include <Mm/Module.hh>
#include <Speech/Types.hh>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include "LabelScorer.hh"
#include "ScoringContext.hh"

namespace Nn {

/*
 * =============================
 * === LimitedCtxOnnxDecoder ===
 * =============================
 */

const Core::ParameterInt LimitedCtxOnnxLabelScorer::paramStartLabelIndex(
        "start-label-index",
        "Initial history in the first step is filled with this label index.",
        0);

const Core::ParameterInt LimitedCtxOnnxLabelScorer::paramHistoryLength(
        "history-length",
        "Number of previous labels that are passed as history.",
        1);

const Core::ParameterBool LimitedCtxOnnxLabelScorer::paramBlankUpdatesHistory(
        "blank-updates-history",
        "Whether previously emitted blank labels should be included in the history.",
        false);

const Core::ParameterBool LimitedCtxOnnxLabelScorer::paramLoopUpdatesHistory(
        "loop-updates-history",
        "Whether in the case of loop transitions every repeated emission should be separately included in the history.",
        false);

const Core::ParameterBool LimitedCtxOnnxLabelScorer::paramVerticalLabelTransition(
        "vertical-label-transition",
        "Whether (non-blank) label transitions should be vertical, i.e. not increase the time step.",
        false);

const Core::ParameterInt LimitedCtxOnnxLabelScorer::paramMaxBatchSize(
        "max-batch-size",
        "Max number of histories that can be fed into the ONNX model at once.",
        Core::Type<int>::max);

const Core::ParameterInt LimitedCtxOnnxLabelScorer::paramMaxCachedScores(
        "max-cached-scores",
        "Maximum size of cache that maps histories to scores. This prevents memory overflow in case of very long audio segments.",
        1000);

static const std::vector<Onnx::IOSpecification> ioSpec = {
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

LimitedCtxOnnxLabelScorer::LimitedCtxOnnxLabelScorer(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config),
          startLabelIndex_(paramStartLabelIndex(config)),
          historyLength_(paramHistoryLength(config)),
          blankUpdatesHistory_(paramBlankUpdatesHistory(config)),
          loopUpdatesHistory_(paramLoopUpdatesHistory(config)),
          verticalLabelTransition_(paramVerticalLabelTransition(config)),
          maxBatchSize_(paramMaxBatchSize(config)),
          onnxModel_(select("onnx-model"), ioSpec),
          encoderStateName_(onnxModel_.mapping.getOnnxName("encoder-state")),
          historyName_(onnxModel_.mapping.getOnnxName("history")),
          scoresName_(onnxModel_.mapping.getOnnxName("scores")),
          scoreCache_(paramMaxCachedScores(config)) {
}

void LimitedCtxOnnxLabelScorer::reset() {
    Precursor::reset();
    scoreCache_.clear();
}

Core::Ref<const ScoringContext> LimitedCtxOnnxLabelScorer::getInitialScoringContext() {
    auto hist = Core::ref(new SeqStepScoringContext());
    hist->labelSeq.resize(historyLength_, startLabelIndex_);
    return hist;
}

Core::Ref<const ScoringContext> LimitedCtxOnnxLabelScorer::extendedScoringContext(LabelScorer::Request request) {
    SeqStepScoringContextRef context(dynamic_cast<const SeqStepScoringContext*>(request.context.get()));

    bool pushToken     = false;
    bool incrementTime = false;
    switch (request.transitionType) {
        case TransitionType::BLANK_LOOP:
            pushToken     = blankUpdatesHistory_ and loopUpdatesHistory_;
            incrementTime = true;
            break;
        case TransitionType::LABEL_TO_BLANK:
            pushToken     = blankUpdatesHistory_;
            incrementTime = true;
            break;
        case TransitionType::LABEL_LOOP:
            pushToken     = loopUpdatesHistory_;
            incrementTime = not verticalLabelTransition_;
            break;
        case TransitionType::BLANK_TO_LABEL:
        case TransitionType::LABEL_TO_LABEL:
            pushToken     = true;
            incrementTime = not verticalLabelTransition_;
            break;
        default:
            error() << "Unknown transition type " << request.transitionType;
    }

    // If context is not going to be modified, return the original one to avoid copying
    if (not pushToken and not incrementTime) {
        return request.context;
    }

    Core::Ref<SeqStepScoringContext> newContext(new SeqStepScoringContext(context->labelSeq, context->currentStep));
    if (pushToken) {
        newContext->labelSeq.push_back(request.nextToken);
        newContext->labelSeq.erase(newContext->labelSeq.begin());
    }
    if (incrementTime) {
        ++newContext->currentStep;
    }
    return newContext;
}

std::optional<LabelScorer::ScoresWithTimes> LimitedCtxOnnxLabelScorer::getScoresWithTimes(const std::vector<LabelScorer::Request>& requests) {
    ScoresWithTimes result;
    result.scores.reserve(requests.size());

    /*
     * Collect all requests that are based on the same timestep (-> same encoder state) and
     * group them together
     */
    std::unordered_map<size_t, std::vector<size_t>> requestsWithTimestep;  // Maps timestep to list of all indices of requests with that timestep

    for (size_t b = 0ul; b < requests.size(); ++b) {
        SeqStepScoringContextRef context(dynamic_cast<const SeqStepScoringContext*>(requests[b].context.get()));
        auto                     step = context->currentStep;
        if (step >= inputBuffer_.size()) {
            // Early exit if at least one of the histories is not scorable yet
            return {};
        }
        result.timesteps.push_back(step);

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
        std::unordered_set<SeqStepScoringContextRef, SeqStepScoringContextHash, SeqStepScoringContextEq> uniqueUncachedContexts;

        for (auto requestIndex : requestIndices) {
            SeqStepScoringContextRef contextPtr(dynamic_cast<const SeqStepScoringContext*>(requests[requestIndex].context.get()));
            if (not scoreCache_.contains(contextPtr)) {
                // Group by unique context
                uniqueUncachedContexts.emplace(contextPtr);
            }
        }

        if (uniqueUncachedContexts.empty()) {
            continue;
        }

        std::vector<SeqStepScoringContextRef> contextBatch;
        contextBatch.reserve(std::min(uniqueUncachedContexts.size(), maxBatchSize_));
        for (auto context : uniqueUncachedContexts) {
            contextBatch.push_back(context);
            if (contextBatch.size() == maxBatchSize_) {  // Batch is full -> forward now
                forwardBatch(contextBatch);
                contextBatch.clear();
            }
        }

        forwardBatch(contextBatch);  // Forward remaining histories
    }

    /*
     * Assign from cache map to result vector
     */
    for (const auto& request : requests) {
        SeqStepScoringContextRef context(dynamic_cast<const SeqStepScoringContext*>(request.context.get()));
        result.scores.push_back(scoreCache_.get(context).at(request.nextToken));
    }

    return result;
}

std::optional<LabelScorer::ScoreWithTime> LimitedCtxOnnxLabelScorer::getScoreWithTime(LabelScorer::Request request) {
    auto result = getScoresWithTimes({request});
    if (not result.has_value()) {
        return {};
    }
    return ScoreWithTime{result->scores.front(), result->timesteps.front()};
}

void LimitedCtxOnnxLabelScorer::forwardBatch(const std::vector<SeqStepScoringContextRef> contextBatch) {
    if (contextBatch.empty()) {
        return;
    }

    /*
     * Create session inputs
     */

    // All requests in this iteration share the same encoder state which is set up here
    f32 const*           encoderState = inputBuffer_[contextBatch.front()->currentStep].get();
    std::vector<int64_t> encoderShape = {1, static_cast<int64_t>(featureSize_)};

    // Create batched context input
    Math::FastMatrix<s32> historyMat(historyLength_, contextBatch.size());
    for (size_t b = 0ul; b < contextBatch.size(); ++b) {
        auto context = contextBatch[b];
        std::copy(context->labelSeq.begin(), context->labelSeq.end(), &(historyMat.at(0, b)));  // Pointer to first element in column b
    }

    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;
    sessionInputs.emplace_back(encoderStateName_, Onnx::Value::create(encoderState, encoderShape));
    sessionInputs.emplace_back(historyName_, Onnx::Value::create(historyMat, true));

    /*
     * Run session
     */

    std::vector<Onnx::Value> sessionOutputs;
    onnxModel_.session.run(std::move(sessionInputs), {scoresName_}, sessionOutputs);

    /*
     * Put resulting scores into cache map
     */
    for (size_t b = 0ul; b < contextBatch.size(); ++b) {
        std::vector<f32> scoreVec;
        sessionOutputs.front().get(b, scoreVec);
        scoreCache_.put(contextBatch[b], std::move(scoreVec));
    }
}
}  // namespace Nn
