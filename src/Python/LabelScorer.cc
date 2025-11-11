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

#include "LabelScorer.hh"

#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ScoringContext.hh"

namespace py = pybind11;

namespace Python {

PythonLabelScorer::PythonLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config) {
}

void PythonLabelScorer::reset() {
    PYBIND11_OVERRIDE_PURE(void, LabelScorer, reset);
}

void PythonLabelScorer::signalNoMoreFeatures() {
    PYBIND11_OVERRIDE_PURE_NAME(
            void,
            LabelScorer,
            "signal_no_more_features",
            signalNoMoreFeatures);
}

Nn::ScoringContextRef PythonLabelScorer::getInitialScoringContext() {
    py::gil_scoped_acquire gil;
    // Store `py::object` from virtual python call in a `PythonScoringContext`
    return Core::ref(new PythonScoringContext(getInitialPythonScoringContext()));
}

py::object PythonLabelScorer::getInitialPythonScoringContext() {
    PYBIND11_OVERRIDE_PURE_NAME(
            py::object,
            Nn::LabelScorer,
            "get_initial_scoring_context",
            getInitialPythonScoringContext);
}

void PythonLabelScorer::addInput(Nn::DataView const& input) {
    // Call batched version
    addInputs(input, 1);
}

void PythonLabelScorer::addInputs(Nn::DataView const& input, size_t nTimesteps) {
    py::gil_scoped_acquire gil;

    // Convert `input` to a `py::array` for virtual python call
    ssize_t featureDimSize = input.size() / nTimesteps;

    py::array_t<f32> inputArray(
            {static_cast<ssize_t>(nTimesteps), featureDimSize},
            {sizeof(f32) * featureDimSize, sizeof(f32)},
            input.data());

    addPythonInputs(inputArray);
}

void PythonLabelScorer::addPythonInputs(py::array const& inputs) {
    PYBIND11_OVERRIDE_PURE_NAME(
            void,
            Nn::LabelScorer,
            "add_inputs",
            addPythonInputs,
            inputs);
}

void PythonLabelScorer::setInstance(py::object const& instance) {
    py::gil_scoped_acquire gil;
    pyInstance_ = instance;
}

Nn::ScoringContextRef PythonLabelScorer::extendedScoringContextInternal(Request const& request) {
    auto*                  pythonScoringContext = dynamic_cast<PythonScoringContext const*>(request.context.get());
    py::gil_scoped_acquire gil;
    // Store `py::object` from virtual python call in a `PythonScoringContext`
    auto newScoringContext = extendedPythonScoringContextInternal(pythonScoringContext->object, request.nextToken, request.transitionType);
    return Core::ref(new PythonScoringContext(std::move(newScoringContext)));
}

py::object PythonLabelScorer::extendedPythonScoringContextInternal(py::object const& context, Nn::LabelIndex nextToken, TransitionType transitionType) {
    PYBIND11_OVERRIDE_PURE_NAME(
            py::object,
            Nn::LabelScorer,
            "extended_scoring_context_internal",
            extendedPythonScoringContext,
            context,
            nextToken,
            transitionType);
}

std::optional<Nn::LabelScorer::ScoreWithTime> PythonLabelScorer::computeScoreWithTimeInternal(Request const& request) {
    // Extract the underlying `py::object` from ScoringContext in `request` to supply them to the virtual python call
    auto* pythonScoringContext = dynamic_cast<PythonScoringContext const*>(request.context.get());

    std::vector<py::object>     contexts        = {pythonScoringContext->object};
    std::vector<Nn::LabelIndex> nextTokens      = {request.nextToken};
    std::vector<TransitionType> transitionTypes = {request.transitionType};

    py::gil_scoped_acquire gil;

    // Call batched version
    if (auto result = computePythonScoresWithTimesInternal(contexts, nextTokens, transitionTypes)) {
        verify(result->size() == 1);
        ScoreWithTime scoreWithTime{result->front().first, result->front().second};
        return scoreWithTime;
    }

    return {};
}

std::optional<Nn::LabelScorer::ScoresWithTimes> PythonLabelScorer::computeScoresWithTimesInternal(std::vector<Request> const& requests) {
    std::vector<py::object>     contexts;
    std::vector<Nn::LabelIndex> nextTokens;
    std::vector<TransitionType> transitionTypes;

    contexts.reserve(requests.size());
    nextTokens.reserve(requests.size());
    transitionTypes.reserve(requests.size());

    // Extract the underlying `py::object`s from ScoringContexts in `requests` to supply them to the virtual python call
    for (auto const& request : requests) {
        auto* pythonScoringContext = dynamic_cast<PythonScoringContext const*>(request.context.get());
        contexts.push_back(pythonScoringContext->object);
        nextTokens.push_back(request.nextToken);
        transitionTypes.push_back(request.transitionType);
    }

    py::gil_scoped_acquire gil;

    if (auto result = computePythonScoresWithTimesInternal(contexts, nextTokens, transitionTypes)) {
        verify(result->size() == requests.size());
        ScoresWithTimes scoresWithTimes;
        scoresWithTimes.scores.reserve(result->size());
        for (auto const& [score, timeframe] : *result) {
            scoresWithTimes.scores.push_back(score);
            scoresWithTimes.timeframes.push_back(timeframe);
        }
        return scoresWithTimes;
    }

    return {};
}

std::optional<std::vector<std::pair<Nn::LabelScorer::Score, Speech::TimeframeIndex>>> PythonLabelScorer::computePythonScoresWithTimesInternal(std::vector<py::object> const& contexts, std::vector<Nn::LabelIndex> const& nextTokens, std::vector<TransitionType> const& transitionTypes) {
    using returnType = std::optional<std::vector<std::pair<Nn::LabelScorer::Score, Speech::TimeframeIndex>>>;  // Macro can't handle types with commas inside properly
    PYBIND11_OVERRIDE_PURE_NAME(
            returnType,
            Nn::LabelScorer,
            "compute_scores_with_times_internal",
            computePythonScoresWithTimes,
            contexts,
            nextTokens,
            transitionTypes);
}

}  // namespace Python
