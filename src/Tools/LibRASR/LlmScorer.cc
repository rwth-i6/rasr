/** Copyright 2026 RWTH Aachen University. All rights reserved.
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

#include "LlmScorer.hh"

#include <string>
#include <pybind11/stl.h>

#include <Python/LlmScorer.hh>
#include <Search/Module.hh>

// Make it so that a `py::object` can use `Core::Ref` as a holder type instead of the usual `std::unique_ptr`.
// See https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html#custom-smart-pointers for official documentation.
PYBIND11_DECLARE_HOLDER_TYPE(T, Core::Ref<T>, true);

void registerPythonLlmScorer(std::string const& name, py::object const& pyLlmScorerClass) {
    Search::Module::instance().llmScorerFactory().registerLlmScorer(
            name.c_str(),
            [pyLlmScorerClass](Core::Configuration const& config) {
                py::gil_scoped_acquire gil;
                // Call constructor of `pyLlmScorerClass`
                py::object inst = pyLlmScorerClass(config);
                inst.cast<Python::PythonLlmScorer*>()->setInstance(inst);
                return inst.cast<Core::Ref<Search::LlmScorer>>();
            });
}

void bindLlmScorer(py::module_& module) {
    module.def(
            "register_llm_scorer_type",
            &registerPythonLlmScorer,
            py::arg("name"),
            py::arg("llm_scorer_cls"),
            "Register a subclass of `LlmScorer` under a name, which `llm-timesync-beam-search` then accepts as `llm.type`.");

    py::class_<Search::LlmScoringRequest> pyRequest(
            module,
            "LlmScoringRequest",
            "Tokens to score after a history. Histories are integer handles; equal handles mean equal token sequences\n"
            "within a segment, so states such as key/value caches can be kept per handle.");
    pyRequest.def_readonly("history", &Search::LlmScoringRequest::history, "Handle of the history to continue.");
    pyRequest.def_readonly("prefix", &Search::LlmScoringRequest::prefix, "Token ids of the history, starting with `initial_tokens()`.");
    pyRequest.def_readonly("tokens", &Search::LlmScoringRequest::tokens, "Token ids to score after the history.");
    pyRequest.def_readonly("token_histories", &Search::LlmScoringRequest::tokenHistories, "Handle of the history after each of `tokens`.");

    // `Python::PythonLlmScorer` is the trampoline class and `Core::Ref` the holder type
    py::class_<Search::LlmScorer, Python::PythonLlmScorer, Core::Ref<Search::LlmScorer>> pyLlmScorer(
            module,
            "LlmScorer",
            "Token-level LM with its own tokenizer, e.g. an LLM, for `llm-timesync-beam-search`. Subclasses implement\n"
            "`reset`, `initial_tokens`, `sentence_end_tokens`, `tokenize` and `score`, and may override\n"
            "`spelling_variants` and `cleanup`.");

    pyLlmScorer.def(py::init<Core::Configuration const&>(), py::arg("config"));

    pyLlmScorer.def("reset", &Search::LlmScorer::reset, "Start a new segment; handles of the previous one are not used again.");

    pyLlmScorer.def(
            "initial_tokens",
            &Search::LlmScorer::initialTokens,
            "Token ids every history starts with, e.g. a begin-of-sequence token or a prompt. Must not be empty.");

    pyLlmScorer.def(
            "sentence_end_tokens",
            &Search::LlmScorer::sentenceEndTokens,
            "Token ids scored at the end of every segment, e.g. an end-of-sequence token. May be empty.");

    pyLlmScorer.def(
            "spelling_variants",
            &Search::LlmScorer::spellingVariants,
            py::arg("words"),
            "Non-empty list of spellings to score for each word, e.g. in different casing; the cheapest one is used.\n"
            "Defaults to the word itself.");

    pyLlmScorer.def(
            "tokenize",
            &Search::LlmScorer::tokenize,
            py::arg("texts"),
            "Non-empty list of token ids for each text. A text is a spelling, preceded by `word-separator` unless it is\n"
            "the first word of a segment.");

    pyLlmScorer.def(
            "score",
            &Search::LlmScorer::score,
            py::arg("requests"),
            "Cost (negative natural log-probability) of every token of every `LlmScoringRequest`, each given its\n"
            "history and the preceding tokens of the request.");

    pyLlmScorer.def(
            "cleanup",
            &Search::LlmScorer::cleanup,
            py::arg("active_histories"),
            "Only the given history handles and their extensions are requested from now on, so states of all others\n"
            "can be dropped. Does nothing by default.");
}
