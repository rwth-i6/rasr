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
    PYBIND11_OVERRIDE_PURE_NAME(Search::LlmTokenSequence, Search::LlmScorer, "initial_tokens", initialTokens);
}

Search::LlmTokenSequence PythonLlmScorer::sentenceEndTokens() {
    PYBIND11_OVERRIDE_PURE_NAME(Search::LlmTokenSequence, Search::LlmScorer, "sentence_end_tokens", sentenceEndTokens);
}

std::vector<std::vector<std::string>> PythonLlmScorer::spellingVariants(std::vector<std::string> const& words) {
    using returnType = std::vector<std::vector<std::string>>;
    PYBIND11_OVERRIDE_NAME(returnType, Search::LlmScorer, "spelling_variants", spellingVariants, words);
}

std::vector<Search::LlmTokenSequence> PythonLlmScorer::tokenize(std::vector<std::string> const& texts) {
    PYBIND11_OVERRIDE_PURE(std::vector<Search::LlmTokenSequence>, Search::LlmScorer, tokenize, texts);
}

std::vector<std::vector<Search::Score>> PythonLlmScorer::score(std::vector<Search::LlmScoringRequest> const& requests) {
    using returnType = std::vector<std::vector<Search::Score>>;
    PYBIND11_OVERRIDE_PURE(returnType, Search::LlmScorer, score, requests);
}

void PythonLlmScorer::cleanup(std::vector<Search::LlmHistory> const& activeHistories) {
    PYBIND11_OVERRIDE(void, Search::LlmScorer, cleanup, activeHistories);
}

}  // namespace Python
