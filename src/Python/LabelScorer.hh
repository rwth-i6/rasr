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

#ifndef PYTHON_LABEL_SCORER_HH
#define PYTHON_LABEL_SCORER_HH

#include <pybind11/pybind11.h>
#include "Nn/LabelScorer/Types.hh"

#include <Nn/LabelScorer/LabelScorer.hh>

namespace py = pybind11;

namespace Python {

/*
 * Trampoline class that is used in order to expose the LabelScorer class via pybind.
 * It mainly specifies the signatures of abstract methods that need to be implemented in python
 * and performs conversion between "C++ types" such as `DataView` and `ScoringContext`
 * and "Python types" such as `py::array` and `py::object`.
 *
 * See https://pybind11.readthedocs.io/en/stable/advanced/classes.html for official documentation
 * on the "trampoline" pattern.
 */
class PythonLabelScorer : public Nn::StatTrackingLabelScorer {
public:
    using Precursor = Nn::StatTrackingLabelScorer;

    PythonLabelScorer(Core::Configuration const& config);
    virtual ~PythonLabelScorer() = default;

    // Must be overridden in python by name "signal_no_more_features"
    virtual Core::Status signalNoMoreFeatures() override;

    // Must be overridden in python by name "get_initial_scoring_context"
    virtual Core::Status                        getInitialScoringContext(Nn::ScoringContextRef& initialScoringContext) override;
    virtual std::pair<Core::Status, py::object> getInitialPythonScoringContext();

    // Calls batched version with `nTimesteps = 1`
    virtual Core::Status addInput(Nn::DataView const& input) override;

    // Must be overridden in python by name "add_inputs"
    virtual Core::Status addInputs(Nn::DataView const& input, size_t nTimesteps) override;
    virtual Core::Status addPythonInputs(py::array const& inputs);

    // Keep track of python object as a member to make sure it doesn't get garbage collected
    void setInstance(py::object const& instance);

    // Must be overridden in python by name "allowed_transition_types"
    virtual py::object getPythonAllowedTransitionTypes();

    // the following methods are protected in the base class

    // Must be overridden in python by name "extended_scoring_context"
    virtual Core::Status                        extendedScoringContext(Nn::ScoringContextRef scoringContext, Nn::LabelIndex nextToken, Nn::TransitionType transitionType, Nn::ScoringContextRef& extendedContext) override;
    virtual std::pair<Core::Status, py::object> extendedPythonScoringContext(py::object const& pythonContext, Nn::LabelIndex nextToken, Nn::TransitionType transitionType);

    // Must be overridden in python by name "compute_scores_with_times"
    using PythonScoresWithTimes = std::vector<std::optional<std::pair<std::vector<Nn::Score>, Nn::TimeframeIndex>>>;
    virtual std::pair<Core::Status, PythonScoresWithTimes> getPythonScoresWithTimes(std::vector<py::object> const& pythonContexts);

protected:
    py::object pyInstance_;  // Hold the Python wrapper

    // Must be overridden in python by name "reset"
    virtual void resetInternal() override;

    // Calls batched version
    virtual Core::Status computeScoreAccessor(Nn::ScoringContextRef scoringContext, std::optional<Nn::ScoreAccessorRef>& scoreAccessor) override;

    // Forwards to `getPythonScoresWithTimes`
    virtual Core::Status computeScoreAccessors(std::vector<Nn::ScoringContextRef> const& scoringContexts, std::vector<std::optional<Nn::ScoreAccessorRef>>& scoreAccessors) override;
};

}  // namespace Python

#endif  // PYTHON_LABEL_SCORER_HH
