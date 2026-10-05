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
            "Register a custom LLM scorer type for `llm-timesync-beam-search`.\n\n"
            "Args:\n"
            "    name: The name under which the LLM scorer type is registered. The same name must be used as `type` in the\n"
            "          `llm` selection of the search algorithm in the RASR config.\n"
            "    llm_scorer_cls: A class that inherits from `librasr.LlmScorer` and implements the abstract methods.");

    // Specify `Python::PythonLlmScorer` as trampoline class and `Core::Ref<Search::LlmScorer>` as holder type
    py::class_<Search::LlmScorer, Python::PythonLlmScorer, Core::Ref<Search::LlmScorer>> pyLlmScorer(
            module,
            "LlmScorer",
            "Abstract base class for a token-level language model with its own tokenizer, e.g. an LLM such as Qwen,\n"
            "used by `llm-timesync-beam-search`.\n"
            "The search finishes words on the fly, tokenizes their surface spelling with `tokenize` and scores the resulting\n"
            "tokens with `score_continuations`. A spelling may be tokenized into several variants, e.g. of different casing;\n"
            "the search scores all of them and greedily continues with the cheapest one.\n"
            "Tokenizations and token scores are cached by the search, so the same text or\n"
            "the same (history, tokens) pair is not requested twice within a segment. All requests of one search step are\n"
            "passed in one call.\n"
            "Concrete subclasses need to implement the following methods:\n"
            " - `reset`\n"
            " - `initial_tokens`\n"
            " - `sentence_end_tokens`\n"
            " - `tokenize`\n"
            " - `score_continuations`");

    pyLlmScorer.def(
            py::init<Core::Configuration const&>(),
            py::arg("config"),
            "Construct an LLM scorer from a RASR config.");

    pyLlmScorer.def(
            "reset",
            &Search::LlmScorer::reset,
            "Prepare for a new segment, e.g. drop key/value caches of the previous one.");

    pyLlmScorer.def(
            "initial_tokens",
            &Search::LlmScorer::initialTokens,
            "Return the list of token ids every history starts with, e.g. a begin-of-sequence token and/or a prompt.");

    pyLlmScorer.def(
            "sentence_end_tokens",
            &Search::LlmScorer::sentenceEndTokens,
            "Return the list of token ids that is scored once at the end of every segment, e.g. an end-of-sequence token.\n"
            "May be empty.");

    pyLlmScorer.def(
            "tokenize",
            &Search::LlmScorer::tokenize,
            py::arg("texts"),
            "Tokenize each text independently into one or more variants.\n\n"
            "Args:\n"
            "    texts: A list of strings. Each is the surface spelling of one finished word, preceded by the configured\n"
            "           `word-separator` unless it is the first word of the segment.\n"
            "Returns:\n"
            "    A list of the same length containing, per text, a non-empty list of variants, each a non-empty list of\n"
            "    token ids, e.g. the tokenizations of the text as is, lowercased and capitalized. The search scores all\n"
            "    variants and continues with the cheapest one. Return a single variant to disable this.");

    pyLlmScorer.def(
            "score_continuations",
            &Search::LlmScorer::scoreContinuations,
            py::arg("prefixes"),
            py::arg("continuations"),
            "Score token continuations of token prefixes.\n\n"
            "Args:\n"
            "    prefixes: A list of length `B` of token id lists. Each is a full history, starting with `initial_tokens()`.\n"
            "    continuations: A list of length `B` of non-empty token id lists to score after the respective prefix.\n"
            "Returns:\n"
            "    A list of length `B` where entry `i` is a list with one cost (negative natural log-probability) per token of\n"
            "    `continuations[i]`, each given `prefixes[i]` and all preceding tokens of `continuations[i]`.");
}
