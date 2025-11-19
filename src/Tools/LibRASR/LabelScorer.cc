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

#include <string>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Nn/Module.hh>
#include <Python/LabelScorer.hh>

// Make it so that a `py::object` can use `Core::Ref` as a holder type instead of the usual `std::unique_ptr`.
// See https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html#custom-smart-pointers for official documentation.
PYBIND11_DECLARE_HOLDER_TYPE(T, Core::Ref<T>, true);

void registerPythonLabelScorer(std::string const& name, py::object const& pyLabelScorerClass) {
    Nn::Module::instance().labelScorerFactory().registerLabelScorer(
            name.c_str(),
            [pyLabelScorerClass](Core::Configuration const& config) {
                py::gil_scoped_acquire gil;
                // Call constructor of `pyLabelScorerClass`
                py::object inst = pyLabelScorerClass(config);
                inst.cast<Python::PythonLabelScorer*>()->setInstance(inst);
                return inst.cast<Core::Ref<Nn::LabelScorer>>();
            });
}

void bindLabelScorer(py::module_& module) {
    module.def(
            "register_label_scorer_type",
            &registerPythonLabelScorer,
            py::arg("name"),
            py::arg("label_scorer_cls"),
            "Register a custom label scorer type in the internal label scorer factory of RASR.\n\n"
            "Args:\n"
            "    name: The name under which the label scorer type is registered. The same name must be used in the RASR config\n"
            "          in order to make RASR instantiate a label scorer of this type later.\n"
            "    label_scorer_cls: A class that inherits from `librasr.LabelScorer` and implements the abstract methods.");

    py::enum_<Nn::LabelScorer::TransitionType>(module, "TransitionType")
            .value("LABEL_TO_LABEL", Nn::LabelScorer::TransitionType::LABEL_TO_LABEL)
            .value("LABEL_LOOP", Nn::LabelScorer::TransitionType::LABEL_LOOP)
            .value("LABEL_TO_BLANK", Nn::LabelScorer::TransitionType::LABEL_TO_BLANK)
            .value("BLANK_TO_LABEL", Nn::LabelScorer::TransitionType::BLANK_TO_LABEL)
            .value("BLANK_LOOP", Nn::LabelScorer::TransitionType::BLANK_LOOP)
            .value("INITIAL_LABEL", Nn::LabelScorer::TransitionType::INITIAL_LABEL)
            .value("INITIAL_BLANK", Nn::LabelScorer::TransitionType::INITIAL_BLANK)
            .value("WORD_EXIT", Nn::LabelScorer::TransitionType::WORD_EXIT)
            .value("SILENCE_EXIT", Nn::LabelScorer::TransitionType::SILENCE_EXIT);

    // Specify `Python::LabelScorer` as trampoline class and `Core::Ref<Nn::LabelScorer>` as holder type
    py::class_<Nn::LabelScorer, Python::PythonLabelScorer, Core::Ref<Nn::LabelScorer>> pyLabelScorer(
            module,
            "LabelScorer",
            "Abstract base class for label scorers. A label scorer is responsible for initializing and updating a 'scoring context'\n"
            "and then computing scores for tokens given a scoring context. This scoring context can be an arbitrary (hashable)\n"
            "python object depending on the needs of the model.\n"
            "For example for a CTC model, the scoring context could just be the current timestep. For a transducer model with\n"
            "LSTM prediction network it could be the timestep together with an LSTM hidden state tensor.\n"
            "Label scorers implemented in python can be used in conjunction with native RASR label scorers such as\n"
            "`CombineLabelScorer`, `EncoderDecoderLabelScorer` + `OnnxEncoder`, etc.\n"
            "A label scorer instance can be used by RASR in order to perform procedures such as search or forced alignment.\n"
            "Concrete subclasses need to implement the following methods:\n"
            " - `reset`\n"
            " - `signal_no_more_features`\n"
            " - `get_initial_scoring_context`\n"
            " - `extended_scoring_context`\n"
            " - `add_inputs`\n"
            " - `compute_scores_with_times`");

    pyLabelScorer.def(
            py::init<Core::Configuration const&>(),
            py::arg("config"),
            "Construct a label scorer from a RASR config.");

    pyLabelScorer.def(
            "reset",
            &Nn::LabelScorer::reset,
            "Reset any internal buffers and flags related to the current segment in order to prepare the label scorer for a new segment.");

    pyLabelScorer.def(
            "signal_no_more_features",
            &Nn::LabelScorer::signalNoMoreFeatures,
            "Signal to the label scorer that all features for the current segment have been passed.");

    pyLabelScorer.def(
            "get_initial_scoring_context",
            [](Python::PythonLabelScorer& self) { return self.getInitialPythonScoringContext(); },
            "Create some arbitrary (hashable) python object which symbolizes the scoring context in the first search step");

    pyLabelScorer.def(
            "extended_scoring_context",
            [](Python::PythonLabelScorer&      self,
               py::object const&               context,
               Nn::LabelIndex                  nextToken,
               Nn::LabelScorer::TransitionType transitionType) { return self.extendedPythonScoringContextInternal(context, nextToken, transitionType); },
            py::arg("context"),
            py::arg("next_token"),
            py::arg("transition_type"),
            "Create a new extended scoring context given the previous context and next token.\n\n"
            "Args:\n"
            "    context: The previous scoring context. The type of the object is the same as the one returned by `get_initial_scoring_context`.\n"
            "    next_token: The most recent token that has been hypothesized and can now be integrated into the scoring context.\n"
            "    transition_type: The kind of transition that has just been performed.\n\n"
            "Returns:\n"
            "    An extended scoring context. The type of this object should be the same as the type of the input `context`.");

    pyLabelScorer.def(
            "add_inputs",
            [](
                    Python::PythonLabelScorer& self,
                    py::array const&           inputs) { self.addPythonInputs(inputs); },
            py::arg("inputs"),
            "Feed an array of input features to the label scorer.\n\n"
            "Args:\n"
            "    inputs: A numpy array of shape [T, F] containing the input features for `T` time steps.");

    pyLabelScorer.def(
            "compute_scores_with_times",
            [](Python::PythonLabelScorer&                   self,
               std::vector<py::object> const&               contexts,
               std::vector<Nn::LabelIndex>                  nextTokens,
               std::vector<Nn::LabelScorer::TransitionType> transitionTypes) { return self.computePythonScoresWithTimesInternal(contexts, nextTokens, transitionTypes); },
            py::arg("contexts"),
            py::arg("next_tokens"),
            py::arg("transition_types"),
            "Compute the scores and timestamps of tokens given the current scoring contexts. Timestamps need to be computed because\n"
            "each label scorer may implement custom logic about how much time is advanced depending on the situation\n"
            "(e.g. vertical vs. diagonal blank transitions in transducer).\n\n"
            "Args:\n"
            "    contexts: A list of length `B` containing current scoring contexts for all requests. The type is the same as the one returned by"
            "              `get_initial_scoring_context` and `extended_scoring_context`.\n"
            "    next_tokens: A list of length `B` containing the tokens for which the score should be computed.\n"
            "    transition_types: A list of length `B` containing the types of the hypothesized transitions.\n\n"
            "Returns:\n"
            "    Either `None` if the label scorer is not ready to process the requests (e.g. expects more features or segment end signal)\n"
            "    or a list of length `B` containing the scores and timestamps for each request. The returned timestamps will be used\n"
            "    to form word boundaries in the search traceback.");
}
