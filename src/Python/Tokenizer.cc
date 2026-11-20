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

Core::Expected<std::vector<PythonTokenizer::TokenId>> PythonTokenizer::encode(std::string const& text) const {
    return callOverride<std::vector<TokenId>>("encode", text);
}

Core::Expected<std::string> PythonTokenizer::decode(std::vector<TokenId> const& ids) const {
    return callOverride<std::string>("decode", ids);
}

u32 PythonTokenizer::vocabularySize() const {
    PYBIND11_OVERRIDE_PURE_NAME(u32, Precursor, "vocabulary_size", vocabularySize);
}

PythonTokenizer::TokenId PythonTokenizer::tokenStrToId(std::string const& token) const {
    PYBIND11_OVERRIDE_PURE_NAME(TokenId, Precursor, "token_str_to_id", tokenStrToId, token);
}

Core::Expected<std::string> PythonTokenizer::tokenIdToStr(TokenId id) const {
    return callOverride<std::string>("token_id_to_str", id);
}

Core::Expected<PythonTokenizer::WordBoundaryMarks> PythonTokenizer::wordBoundaryMarks(TokenId id) const {
    return callOverride<WordBoundaryMarks>("word_boundary_marks", id);
}

Core::Expected<std::vector<std::string>> PythonTokenizer::tokenize(std::string const& text) const {
    {
        py::gil_scoped_acquire gil;
        if (not py::get_override(static_cast<Precursor const*>(this), "tokenize")) {
            return Precursor::tokenize(text);
        }
    }
    return callOverride<std::vector<std::string>>("tokenize", text);
}

Core::Expected<std::string> PythonTokenizer::detokenize(std::vector<std::string> const& tokens) const {
    {
        py::gil_scoped_acquire gil;
        if (not py::get_override(static_cast<Precursor const*>(this), "detokenize")) {
            return Precursor::detokenize(tokens);
        }
    }
    return callOverride<std::string>("detokenize", tokens);
}

template<typename T, typename... Args>
Core::Expected<T> PythonTokenizer::callOverride(char const* name, Args const&... args) const {
    py::gil_scoped_acquire gil;
    py::function           method = py::get_override(static_cast<Precursor const*>(this), name);
    if (not method) {
        return std::unexpected(Core::Status(Core::StatusCode::Unimplemented, std::string("Python tokenizer does not implement `") + name + "`"));
    }
    try {
        return method(args...).template cast<T>();
    }
    catch (std::exception const& exception) {  // Python exceptions and pybind11 cast errors
        return std::unexpected(Core::Status(Core::StatusCode::Unknown, std::string("Python tokenizer error in `") + name + "`: " + exception.what()));
    }
}

}  // namespace Python
