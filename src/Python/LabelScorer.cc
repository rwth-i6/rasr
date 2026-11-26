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

#include "Nn/LabelScorer/ScoreAccessor.hh"
#include "ScoringContext.hh"

namespace py = pybind11;

namespace Python {

PythonLabelScorer::PythonLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config) {
}

Core::Status PythonLabelScorer::signalNoMoreFeatures() {
    PYBIND11_OVERRIDE_PURE_NAME(
            Core::Status,
            LabelScorer,
            "signal_no_more_features",
            signalNoMoreFeatures);
}

Core::Status PythonLabelScorer::getInitialScoringContext(Nn::ScoringContextRef& initialScoringContext) {
    py::gil_scoped_acquire gil;
    auto [status, pythonContext] = getInitialPythonScoringContext();
    if (status.ok()) {
        // Store `py::object` from virtual python call in a `PythonScoringContext`
        initialScoringContext = Core::ref(new PythonScoringContext(std::move(pythonContext)));
    }
    return status;
}

std::pair<Core::Status, py::object> PythonLabelScorer::getInitialPythonScoringContext() {
    using ReturnType = std::pair<Core::Status, py::object>;  // Macro can't handle types with commas inside properly
    PYBIND11_OVERRIDE_PURE_NAME(
            ReturnType,
            Nn::LabelScorer,
            "get_initial_scoring_context",
            getInitialPythonScoringContext);
}

Core::Status PythonLabelScorer::addInput(Nn::DataView const& input) {
    // Call batched version
    return addInputs(input, 1);
}

Core::Status PythonLabelScorer::addInputs(Nn::DataView const& input, size_t nTimesteps) {
    py::gil_scoped_acquire gil;

    // Convert `input` to a `py::array` for virtual python call
    ssize_t featureDimSize = input.size() / nTimesteps;

    py::array_t<f32> inputArray(
            {static_cast<ssize_t>(nTimesteps), featureDimSize},
            {sizeof(f32) * featureDimSize, sizeof(f32)},
            input.data());

    return addPythonInputs(inputArray);
}

Core::Status PythonLabelScorer::addPythonInputs(py::array const& inputs) {
    PYBIND11_OVERRIDE_PURE_NAME(
            Core::Status,
            Nn::LabelScorer,
            "add_inputs",
            addPythonInputs,
            inputs);
}

void PythonLabelScorer::setInstance(py::object const& instance) {
    py::gil_scoped_acquire gil;
    pyInstance_ = instance;

    py::object allowedTransitionTypes = getPythonAllowedTransitionTypes();
    for (py::handle transitionType : allowedTransitionTypes) {
        enabledTransitions_.enable(transitionType.cast<Nn::TransitionType>());
    }
}

py::object PythonLabelScorer::getPythonAllowedTransitionTypes() {
    PYBIND11_OVERRIDE_PURE_NAME(
            py::object,
            Nn::LabelScorer,
            "allowed_transition_types",
            getPythonAllowedTransitionTypes);
}

Core::Status PythonLabelScorer::extendedScoringContext(Nn::ScoringContextRef scoringContext, Nn::LabelIndex nextToken, Nn::TransitionType transitionType, Nn::ScoringContextRef& extendedContext) {
    auto*                  pythonScoringContext = dynamic_cast<PythonScoringContext const*>(scoringContext.get());
    py::gil_scoped_acquire gil;
    auto [status, newPythonContext] = extendedPythonScoringContext(pythonScoringContext->object, nextToken, transitionType);
    if (status.ok()) {
        // Store `py::object` from virtual python call in a `PythonScoringContext`
        extendedContext = Core::ref(new PythonScoringContext(std::move(newPythonContext)));
    }
    return status;
}

std::pair<Core::Status, py::object> PythonLabelScorer::extendedPythonScoringContext(py::object const& pythonContext, Nn::LabelIndex nextToken, Nn::TransitionType transitionType) {
    using ReturnType = std::pair<Core::Status, py::object>;  // Macro can't handle types with commas inside properly
    PYBIND11_OVERRIDE_PURE_NAME(
            ReturnType,
            Nn::LabelScorer,
            "extended_scoring_context",
            extendedPythonScoringContext,
            pythonContext,
            nextToken,
            transitionType);
}

std::pair<Core::Status, PythonLabelScorer::PythonScoresWithTimes> PythonLabelScorer::getPythonScoresWithTimes(std::vector<py::object> const& pythonContexts) {
    using ReturnType = std::pair<Core::Status, PythonScoresWithTimes>;  // Macro can't handle types with commas inside properly
    PYBIND11_OVERRIDE_PURE_NAME(
            ReturnType,
            Nn::LabelScorer,
            "compute_scores_with_times",
            computePythonScoresWithTimes,
            pythonContexts);
}

void PythonLabelScorer::resetInternal() {
    PYBIND11_OVERRIDE_PURE_NAME(
            void,
            LabelScorer,
            "reset",
            resetInternal);
}

Core::Status PythonLabelScorer::computeScoreAccessor(Nn::ScoringContextRef scoringContext, std::optional<Nn::ScoreAccessorRef>& scoreAccessor) {
    std::vector<std::optional<Nn::ScoreAccessorRef>> scoreAccessors;
    auto                                             status = computeScoreAccessors({scoringContext}, scoreAccessors);
    scoreAccessor                                           = scoreAccessors.front();
    return status;
}

Core::Status PythonLabelScorer::computeScoreAccessors(std::vector<Nn::ScoringContextRef> const& scoringContexts, std::vector<std::optional<Nn::ScoreAccessorRef>>& scoreAccessors) {
    scoreAccessors.assign(scoringContexts.size(), std::nullopt);

    std::vector<py::object> pythonContexts;

    pythonContexts.reserve(scoringContexts.size());

    // Extract the underlying `py::object`s from ScoringContexts in `requests` to supply them to the virtual python call
    for (auto const& scoringContext : scoringContexts) {
        auto* pythonScoringContext = dynamic_cast<PythonScoringContext const*>(scoringContext.get());
        pythonContexts.push_back(pythonScoringContext->object);
    }

    py::gil_scoped_acquire gil;

    auto [status, pythonScoresWithTimes] = getPythonScoresWithTimes(pythonContexts);
    if (not status.ok()) {
        return status;
    }
    for (size_t contextIndex = 0ul; contextIndex < scoringContexts.size(); ++contextIndex) {
        if (not pythonScoresWithTimes[contextIndex]) {
            countUncomputedAccessors();
            continue;
        }
        auto scoreVec                = std::make_shared<std::vector<Nn::Score>>(std::move(pythonScoresWithTimes[contextIndex]->first));
        scoreAccessors[contextIndex] = Core::ref(new Nn::VectorScoreAccessor(scoreVec, pythonScoresWithTimes[contextIndex]->second));
    }

    return Core::Status();
}

}  // namespace Python
