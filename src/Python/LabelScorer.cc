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

Nn::ScoringContextRef PythonLabelScorer::extendedScoringContext(Nn::ScoringContextRef scoringContext, Nn::LabelIndex nextToken, Nn::TransitionType transitionType) {
    auto*                  pythonScoringContext = dynamic_cast<PythonScoringContext const*>(scoringContext.get());
    py::gil_scoped_acquire gil;
    // Store `py::object` from virtual python call in a `PythonScoringContext`
    auto newScoringContext = extendedPythonScoringContext(pythonScoringContext->object, nextToken, transitionType);
    return Core::ref(new PythonScoringContext(std::move(newScoringContext)));
}

py::object PythonLabelScorer::extendedPythonScoringContext(py::object const& pythonContext, Nn::LabelIndex nextToken, Nn::TransitionType transitionType) {
    PYBIND11_OVERRIDE_PURE_NAME(
            py::object,
            Nn::LabelScorer,
            "extended_scoring_context",
            extendedPythonScoringContext,
            pythonContext,
            nextToken,
            transitionType);
}

std::optional<Nn::ScoreAccessorRef> PythonLabelScorer::getScoreAccessor(Nn::ScoringContextRef scoringContext) {
    return getScoreAccessors({scoringContext})[0];
}

std::vector<std::optional<Nn::ScoreAccessorRef>> PythonLabelScorer::getScoreAccessors(std::vector<Nn::ScoringContextRef> const& scoringContexts) {
    std::vector<py::object> pythonContexts;

    pythonContexts.reserve(scoringContexts.size());

    // Extract the underlying `py::object`s from ScoringContexts in `requests` to supply them to the virtual python call
    for (auto const& scoringContext : scoringContexts) {
        auto* pythonScoringContext = dynamic_cast<PythonScoringContext const*>(scoringContext.get());
        pythonContexts.push_back(pythonScoringContext->object);
    }

    py::gil_scoped_acquire gil;

    std::vector<std::optional<Nn::ScoreAccessorRef>> scoreAccessors(scoringContexts.size(), std::nullopt);
    auto                                             pythonScoresWithTimes = getPythonScoresWithTimes(pythonContexts);
    for (size_t contextIndex = 0ul; contextIndex < scoringContexts.size(); ++contextIndex) {
        if (not pythonScoresWithTimes[contextIndex]) {
            continue;
        }
        auto scoreVec                = std::make_shared<std::vector<Nn::Score>>(std::move(pythonScoresWithTimes[contextIndex]->first));
        scoreAccessors[contextIndex] = Core::ref(new Nn::VectorScoreAccessor(scoreVec, pythonScoresWithTimes[contextIndex]->second));
    }

    return scoreAccessors;
}

std::vector<std::optional<std::pair<std::vector<Nn::Score>, Nn::TimeframeIndex>>> PythonLabelScorer::getPythonScoresWithTimes(std::vector<py::object> const& pythonContexts) {
    using returnType = std::vector<std::optional<std::pair<std::vector<Nn::Score>, Nn::TimeframeIndex>>>;  // Macro can't handle types with commas inside properly
    PYBIND11_OVERRIDE_PURE_NAME(
            returnType,
            Nn::LabelScorer,
            "compute_scores_with_times",
            computePythonScoresWithTimes,
            pythonContexts);
}

}  // namespace Python
