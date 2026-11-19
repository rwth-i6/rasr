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

#ifndef _PYTHON_TOKENIZER_HH
#define _PYTHON_TOKENIZER_HH

#include <pybind11/pybind11.h>

#include <Tokenizer/AbstractTokenizer.hh>

namespace py = pybind11;

namespace Python {

/*
 * Trampoline class that allows implementing tokenizers in python and registering them in the TokenizerFactory.
 * See https://pybind11.readthedocs.io/en/stable/advanced/classes.html for the "trampoline" pattern.
 */
class PythonTokenizer : public Tokenizer::AbstractTokenizer {
public:
    using Precursor = Tokenizer::AbstractTokenizer;

    PythonTokenizer(Core::Configuration const& config);
    virtual ~PythonTokenizer() = default;

    // Must be overridden in python by the snake_case names of these methods
    std::vector<std::string> tokenize(std::string const& text) const override;
    std::string              detokenize(std::vector<std::string> const& tokens) const override;
    u32                      vocabularySize() const override;
    TokenId                  tokenStrToId(std::string const& token) const override;
    std::string              tokenIdToStr(TokenId id) const override;
    WordBoundaryMarks        wordBoundaryMarks(TokenId id) const override;

    // Can be overridden in python, otherwise uses the default implementations
    std::vector<TokenId> encode(std::string const& text) const override;
    std::string          decode(std::vector<TokenId> const& ids) const override;

    // Keep track of python object as a member to make sure it doesn't get garbage collected
    void setInstance(py::object const& instance);

protected:
    py::object pyInstance_;
};

}  // namespace Python

#endif  // _PYTHON_TOKENIZER_HH
