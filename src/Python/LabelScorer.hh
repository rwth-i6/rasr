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
class PythonLabelScorer : public Nn::LabelScorer {
public:
    using Precursor = Nn::LabelScorer;

    PythonLabelScorer(Core::Configuration const& config);
    virtual ~PythonLabelScorer() = default;

    // Must be overridden in python by name "reset"
    virtual void reset() override;

    // Can be overridden in python. No-op per default.
    virtual void signalNoMoreFeatures() override;

    // Must be overridden in python by name "get_initial_scoring_context"
    virtual Nn::ScoringContextRef getInitialScoringContext() override;
    virtual py::object            getInitialPythonScoringContext();

    // Must be overridden in python by name "extended_scoring_context"
    virtual Nn::ScoringContextRef extendedScoringContext(Request const& request) override;
    virtual py::object            extendedPythonScoringContext(py::object const& context, Nn::LabelIndex nextToken, TransitionType transitionType);

    // Calls batched version with `nTimesteps = 1`
    virtual void addInput(Nn::DataView const& input) override;

    // Must be overridden in python by name "add_inputs"
    virtual void addInputs(Nn::DataView const& input, size_t nTimesteps) override;
    virtual void addPythonInputs(py::array const& inputs);

    // Calls batched version
    virtual std::optional<ScoreWithTime> computeScoreWithTime(Request const& request) override;

    // Must be overridden in python by name "compute_scores_with_times"
    virtual std::optional<ScoresWithTimes>                                       computeScoresWithTimes(std::vector<Request> const& requests) override;
    virtual std::optional<std::vector<std::pair<Score, Speech::TimeframeIndex>>> computePythonScoresWithTimes(std::vector<py::object> const& contexts, std::vector<Nn::LabelIndex> const& nextTokens, std::vector<TransitionType> const& transitionTypes);

    // Keep track of python object as a member to make sure it doesn't get garbage collected
    void setInstance(py::object const& instance);

protected:
    py::object pyInstance_;  // Hold the Python wrapper
};

}  // namespace Python

#endif  // PYTHON_LABEL_SCORER_HH
