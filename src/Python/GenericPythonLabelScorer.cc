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

#include "GenericPythonLabelScorer.hh"

#include <Core/Assertions.hh>
#include <Core/ReferenceCounting.hh>
#include <Flow/Timestamp.hh>
#include <Math/FastMatrix.hh>
#include <Mm/Module.hh>
#include <Speech/Types.hh>
#include "Nn/LabelScorer/ScoringContext.hh"
#undef ensure  // macro duplication in pybind11/numpy.h
#include <pybind11/buffer_info.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace Python {

/*
 * =============================
 * == GenericPythonDecoder ==
 * =============================
 */

const Core::ParameterString GenericPythonLabelScorer::paramInitScoringContextCallbackName(
        "init-context-callback-name",
        "Name of python callback for forwarding of encoder state and history. Callback must be registered separately under exactly this name.",
        "");

const Core::ParameterString GenericPythonLabelScorer::paramExtendScoringContextCallbackName(
        "extend-context-callback-name",
        "Name of python callback for forwarding of encoder state and history. Callback must be registered separately under exactly this name.",
        "");

const Core::ParameterString GenericPythonLabelScorer::paramScoreCallbackName(
        "score-callback-name",
        "Name of python callback for forwarding of encoder state and history. Callback must be registered separately under exactly this name.",
        "");

const Core::ParameterString GenericPythonLabelScorer::paramFinishCheckCallbackName(
        "finish-check-callback-name",
        "Name of python callback for forwarding of encoder state and history. Callback must be registered separately under exactly this name.",
        "");

const Core::ParameterBool GenericPythonLabelScorer::paramBlankUpdatesHistory(
        "blank-updates-history",
        "Whether previously emitted blank labels should be included in the history.",
        false);

const Core::ParameterBool GenericPythonLabelScorer::paramLoopUpdatesHistory(
        "loop-updates-history",
        "Whether in the case of loop transitions every repeated emission should be separately included in the history.",
        false);

const Core::ParameterInt GenericPythonLabelScorer::paramMaxCachedScores(
        "max-cached-scores",
        "Maximum size of cache that maps histories to scores. This prevents memory overflow in case of very long audio segments.",
        1000);

GenericPythonLabelScorer::GenericPythonLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          initScoringContextCallbackName_(paramInitScoringContextCallbackName(config)),
          extendScoringContextCallbackName_(paramExtendScoringContextCallbackName(config)),
          scoreCallbackName_(paramScoreCallbackName(config)),
          finishCheckCallbackName_(paramFinishCheckCallbackName(config)),
          initScoringContextCallback_(),
          extendScoringContextCallback_(),
          scoreCallback_(),
          finishCheckCallback_(),
          blankUpdatesHistory_(paramBlankUpdatesHistory(config)),
          loopUpdatesHistory_(paramLoopUpdatesHistory(config)),
          encoderStates_(),
          initialState_(py::none()),
          scoreCache_(paramMaxCachedScores(config)) {
}

void GenericPythonLabelScorer::reset() {
    Precursor::reset();
    scoreCache_.clear();
    encoderStates_ = py::array_t<f32>();
    initialState_  = py::none();
}

Nn::ScoringContextRef GenericPythonLabelScorer::getInitialScoringContext() {
    return Core::ref(new Nn::PythonScoringContext());
}

Nn::ScoringContextRef GenericPythonLabelScorer::extendedScoringContext(LabelScorer::Request const& request) {
    Nn::PythonScoringContextRef context(dynamic_cast<const Nn::PythonScoringContext*>(request.context.get()));

    bool pushToken = false;
    switch (request.transitionType) {
        case TransitionType::BLANK_LOOP:
            pushToken = blankUpdatesHistory_ and loopUpdatesHistory_;
            break;
        case TransitionType::LABEL_TO_BLANK:
            pushToken = blankUpdatesHistory_;
            break;
        case TransitionType::LABEL_LOOP:
            pushToken = loopUpdatesHistory_;
            break;
        case TransitionType::BLANK_TO_LABEL:
        case TransitionType::LABEL_TO_LABEL:
            pushToken = true;
            break;
        default:
            error() << "Unknown transition type " << request.transitionType;
    }

    // If context is not going to be modified, return the original one to avoid copying
    if (not pushToken) {
        return request.context;
    }

    if (not extendScoringContextCallback_) {
        warning() << "LabelScorer expects callback named \"" << extendScoringContextCallbackName_ << "\" to be registered before running";
        return {};
    }
    Core::Ref<Nn::PythonScoringContext> newContext(new Nn::PythonScoringContext(extendScoringContextCallback_((context->step == 0ul ? computeInitialState() : context->object), request.nextToken), context->step + 1ul));
    return newContext;
}

std::optional<Nn::LabelScorer::ScoresWithTimes> GenericPythonLabelScorer::computeScoresWithTimes(std::vector<LabelScorer::Request> const& requests) {
    if (expectMoreFeatures_ or bufferSize() == 0ul) {  // Only allow scoring once all encoder states have been passed
        return {};
    }

    if (not finishCheckCallback_) {
        warning() << "LabelScorer expects callback named \"" << finishCheckCallbackName_ << "\" to be registered before running";
        return {};
    }

    ScoresWithTimes result;
    result.scores.reserve(requests.size());

    /*
     * Identify unique histories that still need session runs
     */
    std::unordered_set<Nn::PythonScoringContextRef, Nn::ScoringContextHash, Nn::ScoringContextEq> uniqueUncachedHistories;

    for (auto& request : requests) {
        Nn::PythonScoringContextRef historyPtr(dynamic_cast<const Nn::PythonScoringContext*>(request.context.get()));
        if (finishCheckCallback_(historyPtr->object.is_none() ? computeInitialState() : historyPtr->object).cast<bool>()) {
            return {};
        }
        if (not scoreCache_.contains(historyPtr)) {
            // Group by unique history
            uniqueUncachedHistories.emplace(historyPtr);
        }
    }

    std::vector<Nn::PythonScoringContextRef> historyBatch;
    historyBatch.reserve(std::min(uniqueUncachedHistories.size(), maxBatchSize_));
    for (auto history : uniqueUncachedHistories) {
        historyBatch.push_back(history);
        if (historyBatch.size() == maxBatchSize_) {  // Batch is full -> forward now
            forwardBatch(historyBatch);
            historyBatch.clear();
        }
    }

    forwardBatch(historyBatch);  // Forward remaining histories

    /*
     * Assign from cache map to result vector
     */
    for (const auto& request : requests) {
        Nn::PythonScoringContextRef history(dynamic_cast<const Nn::PythonScoringContext*>(request.context.get()));
        auto                        scores = scoreCache_.get(history);
        if (request.nextToken < scores->get().size()) {
            result.scores.push_back(scores->get()[request.nextToken]);
        }
        else {
            result.scores.push_back(0);
        }

        result.timeframes.push_back(history->step);
    }

    return result;
}

std::optional<Nn::LabelScorer::ScoreWithTime> GenericPythonLabelScorer::computeScoreWithTime(LabelScorer::Request const& request) {
    auto result = computeScoresWithTimes({request});
    if (not result.has_value()) {
        return {};
    }
    return ScoreWithTime{result->scores.front(), result->timeframes.front()};
}

void GenericPythonLabelScorer::registerPythonCallback(std::string const& name, py::function const& callback) {
    if (name == initScoringContextCallbackName_) {
        initScoringContextCallback_ = callback;
        log() << "Registered new python callback named \"" << name << "\" for scoring context initialization in GenericPythonLabelScorer";
    }
    if (name == extendScoringContextCallbackName_) {
        extendScoringContextCallback_ = callback;
        log() << "Registered new python callback named \"" << name << "\" for scoring context extension in GenericPythonLabelScorer";
    }
    if (name == scoreCallbackName_) {
        scoreCallback_ = callback;
        log() << "Registered new python callback named \"" << name << "\" for score computation in GenericPythonLabelScorer";
    }
    if (name == finishCheckCallbackName_) {
        finishCheckCallback_ = callback;
        log() << "Registered new python callback named \"" << name << "\" for finish checking in GenericPythonLabelScorer";
    }
}

size_t GenericPythonLabelScorer::getMinActiveInputIndex(Core::CollapsedVector<Nn::ScoringContextRef> const& activeContexts) const {
    return 0u;
}

void GenericPythonLabelScorer::forwardBatch(std::vector<Nn::PythonScoringContextRef> const& contextBatch) {
    if (contextBatch.empty()) {
        return;
    }

    std::vector<py::object> states;
    states.reserve(contextBatch.size());
    for (auto const& context : contextBatch) {
        if (context->step == 0ul) {
            states.push_back(computeInitialState());
        }
        else {
            states.push_back(context->object);
        }
    }
    /*
     * Run session
     */
    if (not scoreCallback_) {
        warning() << "LabelScorer expects callback named \"" << scoreCallbackName_ << "\" to be registered before running";
        return;
    }

    py::gil_scoped_acquire gil;
    py::array_t<f32>       result = scoreCallback_(encoderStates_, states).cast<py::array_t<f32>>();

    /*
     * Put resulting scores into cache map
     */
    for (size_t b = 0ul; b < contextBatch.size(); ++b) {
        auto             scoreBuf = result.unchecked<2>();
        std::vector<f32> scoreVec;
        scoreVec.reserve(scoreBuf.shape(1));
        for (size_t f = 0ul; f < scoreBuf.shape(1); ++f) {
            scoreVec.push_back(scoreBuf(b, f));
        }
        scoreCache_.put(contextBatch[b], std::move(scoreVec));
    }
}

py::object GenericPythonLabelScorer::computeInitialState() {
    verify(not expectMoreFeatures_);

    if (initialState_.is_none()) {  // initialHiddenState_ is still sentinel value -> compute it
        setupEncoderStatesValue();

        if (not initScoringContextCallback_) {
            warning() << "LabelScorer expects callback named \"" << initScoringContextCallbackName_ << "\" to be registered before running";
            return py::none();
        }

        py::gil_scoped_acquire gil;
        initialState_ = initScoringContextCallback_(encoderStates_);
    }

    return initialState_;
}

void GenericPythonLabelScorer::setupEncoderStatesValue() {
    if (encoderStates_.size() > 0) {
        return;
    }

    int64_t T                    = bufferSize();
    auto    inputFeatureDataView = getInput(0);
    encoderStates_               = py::array_t<f32>({1l, T, static_cast<int64_t>(inputFeatureDataView->size())});
    auto encoderStatesBuf        = encoderStates_.mutable_unchecked<3>();
    for (size_t t = 0ul; t < T; ++t) {
        inputFeatureDataView = getInput(t);
        for (size_t f = 0ul; f < inputFeatureDataView->size(); ++f) {
            encoderStatesBuf(0, t, f) = (*getInput(t))[f];
        }
    }
}

}  // namespace Python
