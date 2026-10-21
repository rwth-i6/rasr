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

#include "HuggingFaceTokenizer.hh"

#include <pybind11/gil.h>
#include <pybind11/stl.h>

#include <Core/Application.hh>
#include <Python/Init.hh>

namespace py = pybind11;

namespace {

// Initializes python unless it is already running (e.g. when RASR is used via librasr).
// The interpreter is kept alive until the application exits because python cannot reliably be re-initialized
// after finalization (e.g. re-importing `_datetime` crashes).
void initPython() {
    if (Py_IsInitialized()) {
        return;
    }
    static Python::Initializer initializer;
    // Registered before `init` so that it runs before the exit handler of Python::Initializer, which would warn otherwise
    if (Core::Application::us()) {
        Core::Application::us()->atexit([] { initializer.uninit(); });
    }
    initializer.init();
}

// Runs `f` while holding the GIL. Python exceptions (e.g. for invalid UTF-8 input) are turned into errors, `fallback` is returned then.
template<typename T, typename F>
T callPython(Core::Component const& component, T fallback, F&& f) {
    py::gil_scoped_acquire gil;
    try {
        return f();
    }
    catch (std::exception const& e) {  // Python exceptions and pybind11 cast errors
        component.error("HuggingFace tokenizer error: %s", e.what());
        return fallback;
    }
}

}  // namespace

namespace Tokenizer {

const Core::ParameterString HuggingFaceTokenizer::paramTokenizerFile(
        "tokenizer-file",
        "HuggingFace tokenizer.json file",
        "");

const Core::ParameterBool HuggingFaceTokenizer::paramAddSpecialTokens(
        "add-special-tokens",
        "whether to add special tokens (e.g. BOS) as defined by the tokenizer's post-processor when encoding",
        false);

const Core::ParameterBool HuggingFaceTokenizer::paramSkipSpecialTokens(
        "skip-special-tokens",
        "whether to drop special tokens when decoding",
        true);

HuggingFaceTokenizer::HuggingFaceTokenizer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          addSpecialTokens_(paramAddSpecialTokens(config)),
          skipSpecialTokens_(paramSkipSpecialTokens(config)),
          vocabularySize_(0u) {
    initPython();
    std::string            tokenizerFile = paramTokenizerFile(config);
    py::gil_scoped_acquire gil;
    try {
        py::object tokenizer = py::module_::import("tokenizers").attr("Tokenizer").attr("from_file")(tokenizerFile);
        vocabularySize_      = tokenizer.attr("get_vocab_size")(py::arg("with_added_tokens") = true).cast<u32>();
        tokenizer_.takeOver(tokenizer.release().ptr());
    }
    catch (py::error_already_set const& e) {
        criticalError("Failed to load HuggingFace tokenizer '%s': %s", tokenizerFile.c_str(), e.what());
        return;
    }
    log("Loaded HuggingFace tokenizer with vocabulary size %u from '%s'", vocabularySize_, tokenizerFile.c_str());
}

HuggingFaceTokenizer::~HuggingFaceTokenizer() {
    // After interpreter shutdown the python object can only be leaked
    if (not Py_IsInitialized()) {
        tokenizer_.release();
    }
}

std::vector<std::string> HuggingFaceTokenizer::tokenize(std::string const& text) const {
    return callPython(*this, std::vector<std::string>(), [&] {
        return py::handle(tokenizer_).attr("encode")(text, py::arg("add_special_tokens") = addSpecialTokens_).attr("tokens").cast<std::vector<std::string>>();
    });
}

std::string HuggingFaceTokenizer::detokenize(std::vector<std::string> const& tokens) const {
    std::vector<TokenId> ids;
    ids.reserve(tokens.size());
    for (auto const& t : tokens) {
        TokenId id = tokenId(t);
        if (id == invalidTokenId) {
            error("Token '%s' is not part of the vocabulary", t.c_str());
            continue;
        }
        ids.push_back(id);
    }
    return decode(ids);
}

u32 HuggingFaceTokenizer::vocabularySize() const {
    return vocabularySize_;
}

AbstractTokenizer::TokenId HuggingFaceTokenizer::tokenId(std::string const& token) const {
    return callPython(*this, invalidTokenId, [&] {
        py::object id = py::handle(tokenizer_).attr("token_to_id")(token);
        return id.is_none() ? invalidTokenId : id.cast<TokenId>();
    });
}

std::string HuggingFaceTokenizer::token(TokenId id) const {
    if (not checkTokenId(id)) {
        return {};
    }
    return callPython(*this, std::string(), [&] {
        return py::handle(tokenizer_).attr("id_to_token")(id).cast<std::string>();
    });
}

std::vector<AbstractTokenizer::TokenId> HuggingFaceTokenizer::encode(std::string const& text) const {
    return callPython(*this, std::vector<TokenId>(), [&] {
        return py::handle(tokenizer_).attr("encode")(text, py::arg("add_special_tokens") = addSpecialTokens_).attr("ids").cast<std::vector<TokenId>>();
    });
}

std::string HuggingFaceTokenizer::decode(std::vector<TokenId> const& ids) const {
    // `tokenizers` silently skips invalid ids
    for (auto id : ids) {
        if (not checkTokenId(id)) {
            return {};
        }
    }
    return callPython(*this, std::string(), [&] {
        return py::handle(tokenizer_).attr("decode")(ids, py::arg("skip_special_tokens") = skipSpecialTokens_).cast<std::string>();
    });
}

}  // namespace Tokenizer
