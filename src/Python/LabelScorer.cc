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

namespace {

/*
 * Helpers to call the Python overrides that return a `librasr.Status`, alone or as the first element of a
 * `(Status, value)` tuple. Unlike `PYBIND11_OVERRIDE_PURE_NAME`, they check the type of the returned object and raise a
 * `TypeError` that names the override. Otherwise a wrong return value, e.g. `None` from an override written before
 * these methods returned a Status, surfaces as a misleading "incompatible function arguments" error of whichever
 * `librasr` function the search was called from. The caller must hold the GIL.
 */

py::function getPythonOverride(Nn::LabelScorer const* self, char const* name) {
    py::function override = py::get_override(self, name);
    if (not override) {
        py::pybind11_fail(std::string("Tried to call pure virtual function \"LabelScorer.") + name + "\"");
    }
    return override;
}

[[noreturn]] void throwReturnTypeError(py::function const& override, char const* expected, py::handle result) {
    throw py::type_error(py::str(override.attr("__qualname__")).cast<std::string>() + "() must return " + expected +
                         ", but returned " + py::str(py::type::handle_of(result).attr("__name__")).cast<std::string>() +
                         ". Python label scorers report errors by returning a librasr.Status, see \"Writing a custom label "
                         "scorer in Python\" in the SearchV2 documentation.");
}

template<typename... Args>
Core::Status callStatusOverride(Nn::LabelScorer const* self, char const* name, Args const&... args) {
    auto       override = getPythonOverride(self, name);
    py::object result   = override(args...);
    if (not py::isinstance<Core::Status>(result)) {
        throwReturnTypeError(override, "a librasr.Status", result);
    }
    return result.cast<Core::Status>();
}

template<typename... Args>
std::pair<Core::Status, py::object> callStatusAndValueOverride(Nn::LabelScorer const* self, char const* name, Args const&... args) {
    auto       override = getPythonOverride(self, name);
    py::object result   = override(args...);
    if (py::isinstance<py::tuple>(result)) {
        auto tuple = result.cast<py::tuple>();
        if (tuple.size() == 2 and py::isinstance<Core::Status>(tuple[0])) {
            return {tuple[0].cast<Core::Status>(), tuple[1]};
        }
    }
    throwReturnTypeError(override, "a tuple (librasr.Status, value)", result);
}

}  // namespace

PythonLabelScorer::PythonLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config) {
}

Core::Status PythonLabelScorer::signalNoMoreFeatures() {
    py::gil_scoped_acquire gil;
    return callStatusOverride(this, "signal_no_more_features");
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
    py::gil_scoped_acquire gil;
    return callStatusAndValueOverride(this, "get_initial_scoring_context");
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
    py::gil_scoped_acquire gil;
    return callStatusOverride(this, "add_inputs", inputs);
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
    py::gil_scoped_acquire gil;
    return callStatusAndValueOverride(this, "extended_scoring_context", pythonContext, nextToken, transitionType);
}

std::pair<Core::Status, PythonLabelScorer::PythonScoresWithTimes> PythonLabelScorer::getPythonScoresWithTimes(std::vector<py::object> const& pythonContexts) {
    py::gil_scoped_acquire gil;
    auto [status, scoresWithTimes] = callStatusAndValueOverride(this, "compute_scores_with_times", pythonContexts);
    if (not status.ok()) {
        // The value accompanying an error status is ignored, so it doesn't have to be a valid score list
        return {status, {}};
    }
    return {status, scoresWithTimes.cast<PythonScoresWithTimes>()};
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
