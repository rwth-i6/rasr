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

#include <pybind11/gil.h>
#include <pybind11/stl.h>

namespace Python {

PythonLlmScorer::PythonLlmScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config) {
}

void PythonLlmScorer::setInstance(py::object const& instance) {
    py::gil_scoped_acquire gil;
    pyInstance_ = instance;
}

void PythonLlmScorer::reset() {
    PYBIND11_OVERRIDE_PURE(void, Search::LlmScorer, reset);
}

Search::LlmTokenSequence PythonLlmScorer::initialTokens() {
    PYBIND11_OVERRIDE_PURE_NAME(
            Search::LlmTokenSequence,
            Search::LlmScorer,
            "initial_tokens",
            initialTokens);
}

Search::LlmTokenSequence PythonLlmScorer::sentenceEndTokens() {
    PYBIND11_OVERRIDE_PURE_NAME(
            Search::LlmTokenSequence,
            Search::LlmScorer,
            "sentence_end_tokens",
            sentenceEndTokens);
}

std::vector<Search::LlmTokenSequenceVariants> PythonLlmScorer::tokenize(std::vector<std::string> const& texts) {
    PYBIND11_OVERRIDE_PURE(
            std::vector<Search::LlmTokenSequenceVariants>,
            Search::LlmScorer,
            tokenize,
            texts);
}

std::vector<std::vector<Search::Score>> PythonLlmScorer::scoreContinuations(std::vector<Search::LlmTokenSequence> const& prefixes,
                                                                            std::vector<Search::LlmTokenSequence> const& continuations) {
    using returnType = std::vector<std::vector<Search::Score>>;  // Macro can't handle types with commas inside properly
    PYBIND11_OVERRIDE_PURE_NAME(
            returnType,
            Search::LlmScorer,
            "score_continuations",
            scoreContinuations,
            prefixes,
            continuations);
}

}  // namespace Python
