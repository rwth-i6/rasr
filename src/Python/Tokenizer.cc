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

Core::Status PythonTokenizer::encode(std::string const& text, std::vector<TokenId>& ids) const {
    return callOverride("encode", ids, text);
}

Core::Status PythonTokenizer::decode(std::vector<TokenId> const& ids, std::string& text) const {
    return callOverride("decode", text, ids);
}

Core::Status PythonTokenizer::vocabularySize(u32& size) const {
    return callOverride("vocabulary_size", size);
}

Core::Status PythonTokenizer::tokenStrToId(std::string const& token, TokenId& id) const {
    Core::Status status = callOverride("token_str_to_id", id, token);
    if (status.ok() and id == invalidTokenId) {
        return Core::Status(Core::StatusCode::NotFound, "Token '" + token + "' is not part of the vocabulary");
    }
    return status;
}

Core::Status PythonTokenizer::tokenIdToStr(TokenId id, std::string& token) const {
    return callOverride("token_id_to_str", token, id);
}

Core::Status PythonTokenizer::wordBoundaryMarks(TokenId id, WordBoundaryMarks& marks) const {
    return callOverride("word_boundary_marks", marks, id);
}

Core::Status PythonTokenizer::tokenize(std::string const& text, std::vector<std::string>& tokens) const {
    {
        py::gil_scoped_acquire gil;
        if (not py::get_override(static_cast<Precursor const*>(this), "tokenize")) {
            return Precursor::tokenize(text, tokens);
        }
    }
    return callOverride("tokenize", tokens, text);
}

Core::Status PythonTokenizer::detokenize(std::vector<std::string> const& tokens, std::string& text) const {
    {
        py::gil_scoped_acquire gil;
        if (not py::get_override(static_cast<Precursor const*>(this), "detokenize")) {
            return Precursor::detokenize(tokens, text);
        }
    }
    return callOverride("detokenize", text, tokens);
}

template<typename T, typename... Args>
Core::Status PythonTokenizer::callOverride(char const* name, T& result, Args const&... args) const {
    py::gil_scoped_acquire gil;
    py::function           method = py::get_override(static_cast<Precursor const*>(this), name);
    if (not method) {
        return Core::Status(Core::StatusCode::Unimplemented, std::string("Python tokenizer does not implement `") + name + "`");
    }
    try {
        result = method(args...).template cast<T>();
        return {};
    }
    catch (std::exception const& exception) {  // Python exceptions and pybind11 cast errors
        return Core::Status(Core::StatusCode::Unknown, std::string("Python tokenizer error in `") + name + "`: " + exception.what());
    }
}

}  // namespace Python
