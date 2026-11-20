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
#include <pybind11/trampoline_self_life_support.h>

#include <Tokenizer/AbstractTokenizer.hh>

namespace py = pybind11;

namespace Python {

/*
 * Trampoline class that allows implementing tokenizers in python and registering them in the TokenizerFactory.
 * See https://pybind11.readthedocs.io/en/stable/advanced/classes.html for the "trampoline" pattern.
 *
 * The python methods return their results and raise exceptions, which are turned into a status with code `Unknown`.
 *
 * Hidden like its base `py::trampoline_self_life_support`, which `py::smart_holder` requires.
 */
class __attribute__((visibility("hidden"))) PythonTokenizer : public Tokenizer::AbstractTokenizer,
                                                              public py::trampoline_self_life_support {
public:
    using Precursor = Tokenizer::AbstractTokenizer;

    PythonTokenizer(Core::Configuration const& config);
    virtual ~PythonTokenizer() = default;

    // Must be overridden in python by the snake_case names of these methods
    Core::Status encode(std::string const& text, std::vector<TokenId>& ids) const override;
    Core::Status decode(std::vector<TokenId> const& ids, std::string& text) const override;
    u32          vocabularySize() const override;
    TokenId      tokenStrToId(std::string const& token) const override;
    Core::Status tokenIdToStr(TokenId id, std::string& token) const override;
    Core::Status wordBoundaryMarks(TokenId id, WordBoundaryMarks& marks) const override;

    // Can be overridden in python, otherwise uses the default implementations
    Core::Status tokenize(std::string const& text, std::vector<std::string>& tokens) const override;
    Core::Status detokenize(std::vector<std::string> const& tokens, std::string& text) const override;

private:
    // Calls the python method `name` and stores its result, `Unimplemented` if the method is not overridden
    template<typename T, typename... Args>
    Core::Status callOverride(char const* name, T& result, Args const&... args) const;
};

}  // namespace Python

#endif  // _PYTHON_TOKENIZER_HH
