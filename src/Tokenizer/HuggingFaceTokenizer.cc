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
#include <Core/StringUtilities.hh>
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
    catch (std::exception const& exception) {  // Python exceptions and pybind11 cast errors
        component.error("HuggingFace tokenizer error: %s", exception.what());
        return fallback;
    }
}

// Byte-level BPE maps the whitespace bytes tab, newline, vertical tab, form feed, carriage return and blank to these
// characters, each of them is two bytes long in UTF-8
bool isByteLevelWhitespace(std::string const& character) {
    static std::unordered_set<std::string> const whitespace = {"\u0109", "\u010a", "\u010b", "\u010c", "\u010d", "\u0120"};
    return whitespace.count(character) > 0ul;
}

bool startsWithByteLevelWhitespace(std::string const& token) {
    return token.size() >= 2ul and isByteLevelWhitespace(token.substr(0ul, 2ul));
}

bool endsWithByteLevelWhitespace(std::string const& token) {
    return token.size() >= 2ul and isByteLevelWhitespace(token.substr(token.size() - 2ul));
}

}  // namespace

namespace Tokenizer {

const Core::ParameterString HuggingFaceTokenizer::paramTokenizerFile(
        "tokenizer-file",
        "HuggingFace tokenizer.json file",
        "");

const Core::ParameterBool HuggingFaceTokenizer::paramAddSpecialTokensOnEncode(
        "add-special-tokens-on-encode",
        "whether `tokenize` and `encode` insert the special tokens defined by the post-processor of the tokenizer, "
        "e.g. \"hello\" -> [\"<s>\", \"hello\"] for a tokenizer that prepends a BOS token",
        false);

const Core::ParameterBool HuggingFaceTokenizer::paramRemoveSpecialTokensOnDecode(
        "remove-special-tokens-on-decode",
        "whether `decode` and `detokenize` drop special tokens from the text, "
        "e.g. [\"<s>\", \"hello\", \"</s>\"] -> \"hello\" instead of \"<s>hello</s>\"",
        true);

HuggingFaceTokenizer::HuggingFaceTokenizer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          addSpecialTokensOnEncode_(paramAddSpecialTokensOnEncode(config)),
          removeSpecialTokensOnDecode_(paramRemoveSpecialTokensOnDecode(config)),
          vocabularySize_(0u),
          wordBoundaryMarking_({WordBoundaryMarking::Type::Unknown, ""}) {
    initPython();
    std::string            tokenizerFile = paramTokenizerFile(config);
    py::gil_scoped_acquire gil;
    try {
        py::object tokenizer = py::module_::import("tokenizers").attr("Tokenizer").attr("from_file")(tokenizerFile);
        vocabularySize_      = tokenizer.attr("get_vocab_size")(py::arg("with_added_tokens") = true).cast<u32>();
        wordBoundaryMarking_ = parseWordBoundaryMarking(py::module_::import("json").attr("loads")(tokenizer.attr("to_str")())["decoder"]);
        for (auto item : tokenizer.attr("get_added_tokens_decoder")().cast<py::dict>()) {
            if (item.second.attr("special").cast<bool>()) {
                specialTokenIds_.insert(item.first.cast<TokenId>());
            }
        }
        tokenizer_.takeOver(tokenizer.release().ptr());
    }
    catch (py::error_already_set const& exception) {
        criticalError("Failed to load HuggingFace tokenizer '%s': %s", tokenizerFile.c_str(), exception.what());
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
        return py::handle(tokenizer_).attr("encode")(text, py::arg("add_special_tokens") = addSpecialTokensOnEncode_).attr("tokens").cast<std::vector<std::string>>();
    });
}

std::string HuggingFaceTokenizer::detokenize(std::vector<std::string> const& tokens) const {
    std::vector<TokenId> ids;
    ids.reserve(tokens.size());
    for (auto const& token : tokens) {
        TokenId id = tokenStrToId(token);
        if (id == invalidTokenId) {
            error("Token '%s' is not part of the vocabulary", token.c_str());
            continue;
        }
        ids.push_back(id);
    }
    return decode(ids);
}

u32 HuggingFaceTokenizer::vocabularySize() const {
    return vocabularySize_;
}

AbstractTokenizer::TokenId HuggingFaceTokenizer::tokenStrToId(std::string const& token) const {
    return callPython(*this, invalidTokenId, [&] {
        py::object id = py::handle(tokenizer_).attr("token_to_id")(token);
        return id.is_none() ? invalidTokenId : id.cast<TokenId>();
    });
}

std::string HuggingFaceTokenizer::tokenIdToStr(TokenId id) const {
    if (not checkTokenId(id)) {
        return {};
    }
    return callPython(*this, std::string(), [&] {
        return py::handle(tokenizer_).attr("id_to_token")(id).cast<std::string>();
    });
}

AbstractTokenizer::WordBoundaryMarks HuggingFaceTokenizer::wordBoundaryMarks(TokenId id) const {
    if (not checkTokenId(id)) {
        return {false, false};
    }
    if (specialTokenIds_.count(id) > 0ul) {
        return {true, true};
    }
    std::string        token  = tokenIdToStr(id);
    std::string const& marker = wordBoundaryMarking_.marker;
    switch (wordBoundaryMarking_.type) {
        case WordBoundaryMarking::Type::ByteLevel:
            return {startsWithByteLevelWhitespace(token), endsWithByteLevelWhitespace(token)};
        case WordBoundaryMarking::Type::Prefix:
            return {Core::startsWith(token, marker), false};
        case WordBoundaryMarking::Type::ContinuationPrefix:
            return {not Core::startsWith(token, marker), false};
        case WordBoundaryMarking::Type::Suffix:
            return {false, token.size() >= marker.size() and token.compare(token.size() - marker.size(), marker.size(), marker) == 0};
        case WordBoundaryMarking::Type::Unknown:
            break;
    }
    error("Word boundaries are not supported for the decoder of this HuggingFace tokenizer");
    return {false, false};
}

std::vector<AbstractTokenizer::TokenId> HuggingFaceTokenizer::encode(std::string const& text) const {
    return callPython(*this, std::vector<TokenId>(), [&] {
        return py::handle(tokenizer_).attr("encode")(text, py::arg("add_special_tokens") = addSpecialTokensOnEncode_).attr("ids").cast<std::vector<TokenId>>();
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
        return py::handle(tokenizer_).attr("decode")(ids, py::arg("skip_special_tokens") = removeSpecialTokensOnDecode_).cast<std::string>();
    });
}

HuggingFaceTokenizer::WordBoundaryMarking HuggingFaceTokenizer::parseWordBoundaryMarking(py::handle decoder) {
    if (decoder.is_none()) {
        return {WordBoundaryMarking::Type::Unknown, ""};
    }
    std::string type = decoder["type"].cast<std::string>();
    if (type == "ByteLevel") {
        return {WordBoundaryMarking::Type::ByteLevel, ""};
    }
    if (type == "Metaspace") {
        return {WordBoundaryMarking::Type::Prefix, decoder["replacement"].cast<std::string>()};
    }
    if (type == "WordPiece") {
        return {WordBoundaryMarking::Type::ContinuationPrefix, decoder["prefix"].cast<std::string>()};
    }
    if (type == "BPEDecoder") {
        return {WordBoundaryMarking::Type::Suffix, decoder["suffix"].cast<std::string>()};
    }
    // E.g. Llama 2 replaces "▁" by a blank
    if (type == "Replace" and decoder["content"].cast<std::string>() == " " and decoder["pattern"].contains("String")) {
        return {WordBoundaryMarking::Type::Prefix, decoder["pattern"]["String"].cast<std::string>()};
    }
    if (type == "Sequence") {
        for (auto element : decoder["decoders"]) {
            WordBoundaryMarking marking = parseWordBoundaryMarking(element);
            if (marking.type != WordBoundaryMarking::Type::Unknown) {
                return marking;
            }
        }
    }
    return {WordBoundaryMarking::Type::Unknown, ""};
}

}  // namespace Tokenizer
