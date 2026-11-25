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

#include <fstream>

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

// Runs `f` while holding the GIL. Python exceptions (e.g. for invalid UTF-8 input) are turned into a status.
template<typename F>
Core::Status callPython(F&& f) {
    py::gil_scoped_acquire gil;
    try {
        f();
        return {};
    }
    catch (std::exception const& exception) {  // Python exceptions and pybind11 cast errors
        return Core::Status(Core::StatusCode::Unknown, std::string("HuggingFace tokenizer error: ") + exception.what());
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

const Core::ParameterBool HuggingFaceTokenizer::paramEncodeSpecialTokensAsText(
        "encode-special-tokens-as-text",
        "whether `tokenize` and `encode` treat special tokens in the text as ordinary text, "
        "e.g. \"<s>hi\" -> [\"<\", \"s\", \">\", \"hi\"] instead of [\"<s>\", \"hi\"]",
        true);

Core::Status HuggingFaceTokenizer::create(Core::Configuration const& config, TokenizerRef& tokenizer) {
    std::shared_ptr<HuggingFaceTokenizer> huggingFace(new HuggingFaceTokenizer(config));
    Core::Status                          status = huggingFace->load();
    if (status.ok()) {
        tokenizer = std::move(huggingFace);
    }
    return status;
}

HuggingFaceTokenizer::~HuggingFaceTokenizer() {
    // After interpreter shutdown the python object can only be leaked
    if (not Py_IsInitialized()) {
        tokenizer_.release();
    }
}

Core::Status HuggingFaceTokenizer::encode(std::string const& text, std::vector<TokenId>& ids) const {
    return callPython([&] {
        ids = py::handle(tokenizer_).attr("encode")(text, py::arg("add_special_tokens") = addSpecialTokensOnEncode_).attr("ids").cast<std::vector<TokenId>>();
    });
}

Core::Status HuggingFaceTokenizer::decode(std::vector<TokenId> const& ids, std::string& text) const {
    // `tokenizers` silently skips invalid ids
    for (auto id : ids) {
        Core::Status status = checkTokenId(id);
        if (not status.ok()) {
            return status;
        }
    }
    return callPython([&] {
        text = py::handle(tokenizer_).attr("decode")(ids, py::arg("skip_special_tokens") = removeSpecialTokensOnDecode_).cast<std::string>();
    });
}

Core::Status HuggingFaceTokenizer::vocabularySize(u32& size) const {
    size = vocabularySize_;
    return {};
}

Core::Status HuggingFaceTokenizer::tokenStrToId(std::string const& token, TokenId& id) const {
    bool         found  = false;
    Core::Status status = callPython([&] {
        py::object pyToken = py::reinterpret_steal<py::object>(PyUnicode_DecodeUTF8(token.data(), token.size(), nullptr));
        if (not pyToken) {  // Invalid UTF-8 cannot be part of the vocabulary
            PyErr_Clear();
            return;
        }
        py::object result = py::handle(tokenizer_).attr("token_to_id")(pyToken);
        found             = not result.is_none();
        if (found) {
            id = result.cast<TokenId>();
        }
    });
    if (status.ok() and not found) {
        return Core::Status(Core::StatusCode::NotFound, "Token '" + token + "' is not part of the vocabulary");
    }
    return status;
}

Core::Status HuggingFaceTokenizer::tokenIdToStr(TokenId id, std::string& token) const {
    Core::Status status = checkTokenId(id);
    if (not status.ok()) {
        return status;
    }
    return callPython([&] {
        token = py::handle(tokenizer_).attr("id_to_token")(id).cast<std::string>();
    });
}

Core::Status HuggingFaceTokenizer::wordBoundaryMarks(TokenId id, WordBoundaryMarks& marks) const {
    std::string  token;
    Core::Status status = tokenIdToStr(id, token);
    if (not status.ok()) {
        return status;
    }
    if (specialTokenIds_.count(id) > 0ul) {
        marks = {true, true};
        return {};
    }
    std::string const& marker = wordBoundaryMarking_.marker;
    switch (wordBoundaryMarking_.type) {
        case WordBoundaryMarking::Type::ByteLevel:
            marks = {startsWithByteLevelWhitespace(token), endsWithByteLevelWhitespace(token)};
            return {};
        case WordBoundaryMarking::Type::Prefix:
            marks = {Core::startsWith(token, marker), false};
            return {};
        case WordBoundaryMarking::Type::ContinuationPrefix:
            marks = {not Core::startsWith(token, marker), false};
            return {};
        case WordBoundaryMarking::Type::Suffix:
            marks = {false, token.size() >= marker.size() and token.compare(token.size() - marker.size(), marker.size(), marker) == 0};
            return {};
        case WordBoundaryMarking::Type::Unknown:
            break;
    }
    return Core::Status(Core::StatusCode::Unimplemented, "Word boundaries are not supported for the decoder of this HuggingFace tokenizer");
}

Core::Status HuggingFaceTokenizer::tokenize(std::string const& text, std::vector<std::string>& tokens) const {
    return callPython([&] {
        tokens = py::handle(tokenizer_).attr("encode")(text, py::arg("add_special_tokens") = addSpecialTokensOnEncode_).attr("tokens").cast<std::vector<std::string>>();
    });
}

HuggingFaceTokenizer::HuggingFaceTokenizer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          addSpecialTokensOnEncode_(paramAddSpecialTokensOnEncode(config)),
          removeSpecialTokensOnDecode_(paramRemoveSpecialTokensOnDecode(config)),
          vocabularySize_(0u),
          wordBoundaryMarking_({WordBoundaryMarking::Type::Unknown, ""}) {}

Core::Status HuggingFaceTokenizer::load() {
    std::string tokenizerFile = paramTokenizerFile(config);
    if (not std::ifstream(tokenizerFile).good()) {
        return Core::Status(Core::StatusCode::NotFound, "Failed to open HuggingFace tokenizer file '" + tokenizerFile + "'");
    }
    initPython();
    py::gil_scoped_acquire gil;
    py::object             tokenizers;
    try {
        tokenizers = py::module_::import("tokenizers");
    }
    catch (py::error_already_set const& exception) {
        return Core::Status(Core::StatusCode::FailedPrecondition, std::string("Failed to import python package 'tokenizers': ") + exception.what());
    }
    try {
        py::object tokenizer                    = tokenizers.attr("Tokenizer").attr("from_file")(tokenizerFile);
        tokenizer.attr("encode_special_tokens") = paramEncodeSpecialTokensAsText(config);
        vocabularySize_                         = tokenizer.attr("get_vocab_size")(py::arg("with_added_tokens") = true).cast<u32>();
        wordBoundaryMarking_                    = parseWordBoundaryMarking(py::module_::import("json").attr("loads")(tokenizer.attr("to_str")())["decoder"]);
        for (auto item : tokenizer.attr("get_added_tokens_decoder")().cast<py::dict>()) {
            if (item.second.attr("special").cast<bool>()) {
                specialTokenIds_.insert(item.first.cast<TokenId>());
            }
        }
        tokenizer_.takeOver(tokenizer.release().ptr());
    }
    catch (py::error_already_set const& exception) {
        return Core::Status(Core::StatusCode::InvalidFileFormat, "Failed to load HuggingFace tokenizer '" + tokenizerFile + "': " + exception.what());
    }
    log("Loaded HuggingFace tokenizer with vocabulary size %u from '%s'", vocabularySize_, tokenizerFile.c_str());
    return {};
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
