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
 * Trampoline that lets `Search::LlmScorer` be implemented in Python, see
 * https://pybind11.readthedocs.io/en/stable/advanced/classes.html
 */
class PythonLlmScorer : public Search::LlmScorer {
public:
    using Precursor = Search::LlmScorer;

    PythonLlmScorer(Core::Configuration const& config);
    virtual ~PythonLlmScorer() = default;

    // Keep the Python object alive as long as the scorer
    void setInstance(py::object const& instance);

    void                                    reset() override;
    Search::LlmTokenSequence                initialTokens() override;
    Search::LlmTokenSequence                sentenceEndTokens() override;
    std::vector<std::vector<std::string>>   spellingVariants(std::vector<std::string> const& words) override;
    std::vector<Search::LlmTokenSequence>   tokenize(std::vector<std::string> const& texts) override;
    std::vector<std::vector<Search::Score>> score(std::vector<Search::LlmScoringRequest> const& requests) override;
    void                                    cleanup(std::vector<Search::LlmHistory> const& activeHistories) override;

protected:
    py::object pyInstance_;
};

}  // namespace Python

#endif  // PYTHON_LLM_SCORER_HH
