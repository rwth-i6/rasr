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

#ifndef PYTHON_LLM_SCORER_HH
#define PYTHON_LLM_SCORER_HH

#include <pybind11/pybind11.h>

#include <Search/LlmTimesyncBeamSearch/LlmScorer.hh>

namespace py = pybind11;

namespace Python {

/*
 * Trampoline class that is used in order to expose the LlmScorer class via pybind,
 * so that it can be implemented in Python, e.g. with HuggingFace transformers.
 * Every call into Python acquires the GIL.
 *
 * See https://pybind11.readthedocs.io/en/stable/advanced/classes.html for official documentation
 * on the "trampoline" pattern.
 */
class PythonLlmScorer : public Search::LlmScorer {
public:
    using Precursor = Search::LlmScorer;

    PythonLlmScorer(Core::Configuration const& config);
    virtual ~PythonLlmScorer() = default;

    // Keep track of python object as a member to make sure it doesn't get garbage collected
    void setInstance(py::object const& instance);

    // Must be overridden in python by name "reset"
    void reset() override;

    // Must be overridden in python by name "initial_tokens"
    Search::LlmTokenSequence initialTokens() override;

    // Must be overridden in python by name "sentence_end_tokens"
    Search::LlmTokenSequence sentenceEndTokens() override;

    // Must be overridden in python by name "tokenize"
    std::vector<Search::LlmTokenSequenceVariants> tokenize(std::vector<std::string> const& texts) override;

    // Must be overridden in python by name "score_continuations"
    std::vector<std::vector<Search::Score>> scoreContinuations(std::vector<Search::LlmTokenSequence> const& prefixes,
                                                               std::vector<Search::LlmTokenSequence> const& continuations) override;

protected:
    py::object pyInstance_;  // Hold the Python wrapper
};

}  // namespace Python

#endif  // PYTHON_LLM_SCORER_HH
