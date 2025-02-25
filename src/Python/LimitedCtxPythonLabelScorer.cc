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

#include "LimitedCtxPythonLabelScorer.hh"

#include <Core/Assertions.hh>
#include <Core/ReferenceCounting.hh>
#include <Flow/Timestamp.hh>
#include <Math/FastMatrix.hh>
#include <Mm/Module.hh>
#include <Speech/Types.hh>
#undef ensure  // macro duplication in pybind11/numpy.h
#include <pybind11/buffer_info.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace Python {

/*
 * =============================
 * == LimitedCtxPythonDecoder ==
 * =============================
 */

const Core::ParameterString LimitedCtxPythonLabelScorer::paramCallbackName(
        "callback-name",
        "Name of python callback for forwarding of encoder state and history. Callback must be registered separately under exactly this name.",
        "");

const Core::ParameterInt LimitedCtxPythonLabelScorer::paramStartLabelIndex(
        "start-label-index",
        "Initial history in the first step is filled with this label index.",
        0);

const Core::ParameterInt LimitedCtxPythonLabelScorer::paramHistoryLength(
        "history-length",
        "Number of previous labels that are passed as history.",
        1);

const Core::ParameterBool LimitedCtxPythonLabelScorer::paramBlankUpdatesHistory(
        "blank-updates-history",
        "Whether previously emitted blank labels should be included in the history.",
        false);

const Core::ParameterBool LimitedCtxPythonLabelScorer::paramLoopUpdatesHistory(
        "loop-updates-history",
        "Whether in the case of loop transitions every repeated emission should be separately included in the history.",
        false);

const Core::ParameterBool LimitedCtxPythonLabelScorer::paramVerticalLabelTransition(
        "vertical-label-transition",
        "Whether (non-blank) label transitions should be vertical, i.e. not increase the time step.",
        false);

const Core::ParameterInt LimitedCtxPythonLabelScorer::paramMaxBatchSize(
        "max-batch-size",
        "Max number of histories that can be fed into the ONNX model at once.",
        Core::Type<int>::max);

const Core::ParameterInt LimitedCtxPythonLabelScorer::paramMaxCachedScores(
        "max-cached-scores",
        "Maximum size of cache that maps histories to scores. This prevents memory overflow in case of very long audio segments.",
        1000);

LimitedCtxPythonLabelScorer::LimitedCtxPythonLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          callbackName_(paramCallbackName(config)),
          callback_(),
          startLabelIndex_(paramStartLabelIndex(config)),
          historyLength_(paramHistoryLength(config)),
          blankUpdatesHistory_(paramBlankUpdatesHistory(config)),
          loopUpdatesHistory_(paramLoopUpdatesHistory(config)),
          verticalLabelTransition_(paramVerticalLabelTransition(config)),
          maxBatchSize_(paramMaxBatchSize(config)),
          scoreCache_(paramMaxCachedScores(config)) {
    log() << "Create LimitedCtxOnnxLabelScorer with context size " << historyLength_;
}

void LimitedCtxPythonLabelScorer::reset() {
    Precursor::reset();
    scoreCache_.clear();
}

Nn::ScoringContextRef LimitedCtxPythonLabelScorer::getInitialScoringContext() {
    auto hist = Core::ref(new Nn::SeqStepScoringContext());
    hist->labelSeq.resize(historyLength_, startLabelIndex_);
    return hist;
}

Nn::ScoringContextRef LimitedCtxPythonLabelScorer::extendedScoringContext(LabelScorer::Request const& request) {
    Nn::SeqStepScoringContextRef context(dynamic_cast<const Nn::SeqStepScoringContext*>(request.context.get()));

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

    Core::Ref<Nn::SeqStepScoringContext> newContext(new Nn::SeqStepScoringContext(context->labelSeq, context->currentStep));
    if (pushToken) {
        newContext->labelSeq.push_back(request.nextToken);
        newContext->labelSeq.erase(newContext->labelSeq.begin());
    }
    if (incrementTime) {
        ++newContext->currentStep;
    }
    return newContext;
}

std::optional<Nn::LabelScorer::ScoresWithTimes> LimitedCtxPythonLabelScorer::computeScoresWithTimes(std::vector<LabelScorer::Request> const& requests) {
    ScoresWithTimes result;
    result.scores.reserve(requests.size());

    /*
     * Collect all requests that are based on the same timestep (-> same encoder state) and
     * group them together
     */
    std::unordered_map<size_t, std::vector<size_t>> requestsWithTimestep;  // Maps timestep to list of all indices of requests with that timestep

    for (size_t b = 0ul; b < requests.size(); ++b) {
        Nn::SeqStepScoringContextRef context(dynamic_cast<const Nn::SeqStepScoringContext*>(requests[b].context.get()));
        auto                         step = context->currentStep;
        if (step >= inputBuffer_.size()) {
            // Early exit if at least one of the histories is not scorable yet
            return {};
        }
        result.timeframes.push_back(step);

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
        std::unordered_set<Nn::SeqStepScoringContextRef, Nn::ScoringContextHash, Nn::ScoringContextEq> uniqueUncachedContexts;

        for (auto requestIndex : requestIndices) {
            Nn::SeqStepScoringContextRef contextPtr(dynamic_cast<const Nn::SeqStepScoringContext*>(requests[requestIndex].context.get()));
            if (not scoreCache_.contains(contextPtr)) {
                // Group by unique context
                uniqueUncachedContexts.emplace(contextPtr);
            }
        }

        if (uniqueUncachedContexts.empty()) {
            continue;
        }

        std::vector<Nn::SeqStepScoringContextRef> contextBatch;
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
        Nn::SeqStepScoringContextRef context(dynamic_cast<const Nn::SeqStepScoringContext*>(request.context.get()));

        auto scores = scoreCache_.get(context);
        if (request.nextToken < scores->get().size()) {
            result.scores.push_back(scores->get()[request.nextToken]);
        }
        else {
            result.scores.push_back(0);
        }
    }

    return result;
}

std::optional<Nn::LabelScorer::ScoreWithTime> LimitedCtxPythonLabelScorer::computeScoreWithTime(LabelScorer::Request const& request) {
    auto result = computeScoresWithTimes({request});
    if (not result.has_value()) {
        return {};
    }
    return ScoreWithTime{result->scores.front(), result->timeframes.front()};
}

void LimitedCtxPythonLabelScorer::registerPythonCallback(std::string const& name, py::function const& callback) {
    if (name == callbackName_) {
        callback_ = callback;
        log() << "Registered new python callback named \"" << name << "\" for LimitedCtxPythonLabelScorer";
    }
}

void LimitedCtxPythonLabelScorer::forwardBatch(std::vector<Nn::SeqStepScoringContextRef> const& contextBatch) {
    if (contextBatch.empty()) {
        return;
    }

    /*
     * Create session inputs
     */
    // All requests in this iteration share the same encoder state which is set up here
    std::vector<int64_t> encoderStateShape   = {1ul, static_cast<int64_t>(featureSize_)};
    std::vector<int64_t> encoderStateStrides = {static_cast<int64_t>(sizeof(f32) * featureSize_), sizeof(f32)};
    const f32*           encoderStateData    = inputBuffer_[contextBatch.front()->currentStep].get();

    py::array_t<f32> encoderState(encoderStateShape, encoderStateStrides, encoderStateData);

    // Create batched context input
    py::array_t<s32> historyBatch({contextBatch.size(), historyLength_});
    auto             historyBuf = historyBatch.mutable_unchecked<2>();

    for (size_t b = 0ul; b < contextBatch.size(); ++b) {
        for (size_t h = 0ul; h < historyLength_; ++h) {
            historyBuf(b, h) = contextBatch[b]->labelSeq[h];
        }
    }

    /*
     * Run session
     */
    if (not callback_) {
        warning() << "LabelScorer expects callback named \"" << callbackName_ << "\" to be registered before running";
        return;
    }

    py::gil_scoped_acquire gil;
    py::array_t<f32>       result = callback_(encoderState, historyBatch).cast<py::array_t<f32>>();

    /*
     * Put resulting scores into cache map
     */
    for (size_t b = 0ul; b < contextBatch.size(); ++b) {
        py::buffer_info  buf     = result.request();
        f32*             dataPtr = static_cast<f32*>(buf.ptr);
        std::vector<f32> scoreVec(dataPtr, dataPtr + buf.size);
        scoreCache_.put(contextBatch[b], std::move(scoreVec));
    }
}
}  // namespace Python
