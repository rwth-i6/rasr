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

#include "Tokenizer.hh"

#include <pybind11/stl.h>

namespace Python {

PythonTokenizer::PythonTokenizer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config) {
}

std::vector<std::string> PythonTokenizer::tokenize(std::string const& text) const {
    PYBIND11_OVERRIDE_PURE_NAME(std::vector<std::string>, Precursor, "tokenize", tokenize, text);
}

std::string PythonTokenizer::detokenize(std::vector<std::string> const& tokens) const {
    PYBIND11_OVERRIDE_PURE_NAME(std::string, Precursor, "detokenize", detokenize, tokens);
}

u32 PythonTokenizer::vocabularySize() const {
    PYBIND11_OVERRIDE_PURE_NAME(u32, Precursor, "vocabulary_size", vocabularySize);
}

PythonTokenizer::TokenId PythonTokenizer::tokenStrToId(std::string const& token) const {
    PYBIND11_OVERRIDE_PURE_NAME(TokenId, Precursor, "token_str_to_id", tokenStrToId, token);
}

std::string PythonTokenizer::tokenIdToStr(TokenId id) const {
    PYBIND11_OVERRIDE_PURE_NAME(std::string, Precursor, "token_id_to_str", tokenIdToStr, id);
}

std::vector<PythonTokenizer::TokenId> PythonTokenizer::encode(std::string const& text) const {
    PYBIND11_OVERRIDE_NAME(std::vector<TokenId>, Precursor, "encode", encode, text);
}

std::string PythonTokenizer::decode(std::vector<TokenId> const& ids) const {
    PYBIND11_OVERRIDE_NAME(std::string, Precursor, "decode", decode, ids);
}

void PythonTokenizer::setInstance(py::object const& instance) {
    pyInstance_ = instance;
}

}  // namespace Python
