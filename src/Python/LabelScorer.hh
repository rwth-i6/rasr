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

#include <Nn/LabelScorer/CombineLabelScorer.hh>
#include <Nn/LabelScorer/CtcPrefixLabelScorer.hh>
#include <Nn/LabelScorer/EncoderDecoderLabelScorer.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScaledLabelScorer.hh>

namespace py = pybind11;

// Return the sub-scorer that is wrapped by the given label scorer
// If the label scorer is a CombineLabelScorer, return the sub-scorer at the specified index
static Nn::ScaledLabelScorer* getSubScorer(Nn::ScaledLabelScorer* labelScorer, std::optional<size_t> index = 0) {
    Core::Ref<Nn::LabelScorer> wrappedLabelScorer = labelScorer->labelScorer();

    auto* combineScorer = dynamic_cast<Nn::CombineLabelScorer*>(wrappedLabelScorer.get());
    if (combineScorer) {
        return combineScorer->getSubScorer(*index).get();
    }

    auto* encoderDecoderScorer = dynamic_cast<Nn::EncoderDecoderLabelScorer*>(wrappedLabelScorer.get());
    if (encoderDecoderScorer) {
        return encoderDecoderScorer->getDecoderLabelScorer().get();
    }

    auto* ctcPrefixScorer = dynamic_cast<Nn::CtcPrefixLabelScorer*>(wrappedLabelScorer.get());
    if (ctcPrefixScorer) {
        return ctcPrefixScorer->getCtcLabelScorer().get();
    }

    throw py::value_error("Label scorer does not a have a sub-scorer");
}

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

    // Calls batched version with `nTimesteps = 1`
    virtual void addInput(Nn::DataView const& input) override;

    // Must be overridden in python by name "add_inputs"
    virtual void addInputs(Nn::DataView const& input, size_t nTimesteps) override;
    virtual void addPythonInputs(py::array const& inputs);

    // Keep track of python object as a member to make sure it doesn't get garbage collected
    void setInstance(py::object const& instance);

    // Must be overridden in python by name "allowed_transition_types"
    virtual py::object getPythonAllowedTransitionTypes();

    // the following methods are protected in the base class

    // Must be overridden in python by name "extended_scoring_context_internal"
    virtual Nn::ScoringContextRef extendedScoringContext(Nn::ScoringContextRef scoringContext, Nn::LabelIndex nextToken, Nn::TransitionType transitionType) override;
    virtual py::object            extendedPythonScoringContext(py::object const& pythonContext, Nn::LabelIndex nextToken, Nn::TransitionType transitionType);

    // Calls batched version
    virtual std::optional<Nn::ScoreAccessorRef> getScoreAccessor(Nn::ScoringContextRef scoringContext) override;

    // Must be overridden in python by name "compute_scores_with_times_internal"
    virtual std::vector<std::optional<Nn::ScoreAccessorRef>>                                  getScoreAccessors(std::vector<Nn::ScoringContextRef> const& scoringContexts) override;
    virtual std::vector<std::optional<std::pair<std::vector<Nn::Score>, Nn::TimeframeIndex>>> getPythonScoresWithTimes(std::vector<py::object> const& pythonContexts);

protected:
    py::object pyInstance_;  // Hold the Python wrapper
};

}  // namespace Python

#endif  // PYTHON_LABEL_SCORER_HH
