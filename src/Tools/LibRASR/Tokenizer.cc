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

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <pybind11/stl.h>

#include <Python/Tokenizer.hh>
#include <Tokenizer/Module.hh>

namespace {

using Tokenizer::AbstractTokenizer;
using TokenId = AbstractTokenizer::TokenId;

void throwOnError(Core::Status const& status) {
    if (not status.ok()) {
        throw std::runtime_error(status.message());
    }
}

}  // namespace

void registerPythonTokenizer(std::string const& name, py::object const& pyTokenizerClass) {
    // The factory is destroyed with the static objects of RASR, which happens after the python interpreter has been
    // finalized, when the reference can only be leaked
    std::shared_ptr<py::object> callable(new py::object(pyTokenizerClass), [](py::object* object) {
        if (not Py_IsInitialized()) {
            object->release();
        }
        delete object;
    });
    Tokenizer::Module::instance().tokenizerFactory().registerTokenizer(
            name.c_str(),
            [callable](Core::Configuration const& config, Tokenizer::TokenizerRef& tokenizer) {
                py::gil_scoped_acquire gil;
                try {
                    // Call constructor of `pyTokenizerClass`. The returned pointer keeps the python object alive.
                    tokenizer = (*callable)(config).cast<Tokenizer::TokenizerRef>();
                    return Core::Status();
                }
                catch (std::exception const& exception) {  // Python exceptions and pybind11 cast errors
                    return Core::Status(Core::StatusCode::Unknown, std::string("Failed to create python tokenizer: ") + exception.what());
                }
            });
}

void bindTokenizer(py::module_& module) {
    module.def(
            "register_tokenizer_type",
            &registerPythonTokenizer,
            py::arg("name"),
            py::arg("tokenizer_cls"),
            "Register a custom tokenizer type in the internal tokenizer factory of RASR.\n\n"
            "Args:\n"
            "    name: The name under which the tokenizer type is registered. The same name must be used as `type` in the RASR config.\n"
            "          Registering an existing name, also of a built-in type such as \"bpe\", replaces it.\n"
            "    tokenizer_cls: A class that inherits from `librasr.Tokenizer` and implements the abstract methods, or a function that\n"
            "                   creates an instance of such a class from the config.");

    module.def(
            "create_tokenizer",
            [](Core::Configuration const& config) {
                Tokenizer::TokenizerRef tokenizer;
                throwOnError(Tokenizer::Module::instance().tokenizerFactory().createTokenizer(Core::Configuration(config, "tokenizer"), tokenizer));
                return tokenizer;
            },
            py::arg("config"),
            "Create the tokenizer configured via `*.tokenizer.*`, e.g. `*.tokenizer.type`. Works for native as well as registered python tokenizer types.");

    // Specify `Python::PythonTokenizer` as trampoline class and `py::smart_holder` as holder type, which allows passing
    // python implementations to RASR as `std::shared_ptr`
    py::class_<AbstractTokenizer, Python::PythonTokenizer, py::smart_holder> pyTokenizer(
            module,
            "Tokenizer",
            "Abstract base class for tokenizers that convert between text and token sequences.\n"
            "Concrete subclasses need to implement the following methods:\n"
            " - `encode`\n"
            " - `decode`\n"
            " - `vocabulary_size`\n"
            " - `token_str_to_id`\n"
            " - `token_id_to_str`\n"
            " - `word_boundary_marks`\n"
            "`tokenize` and `detokenize` can optionally be overridden.\n"
            "Errors raise a `RuntimeError`.");

    // Registered before `word_boundary_marks`, which returns it
    using WordBoundaryMarks = AbstractTokenizer::WordBoundaryMarks;
    py::class_<WordBoundaryMarks>(
            pyTokenizer,
            "WordBoundaryMarks",
            "Word boundaries that a token marks before and after itself and whether it is whitespace, see `Tokenizer.word_boundary_marks`.")
            .def(py::init([](bool before, bool after, bool whitespace) { return WordBoundaryMarks{before, after, whitespace}; }),
                 py::arg("before"),
                 py::arg("after"),
                 py::arg("whitespace") = false)
            .def_readwrite("before", &WordBoundaryMarks::before)
            .def_readwrite("after", &WordBoundaryMarks::after)
            .def_readwrite("whitespace", &WordBoundaryMarks::whitespace)
            .def("__repr__", [](WordBoundaryMarks const& self) {
                return std::string("WordBoundaryMarks(before=") + (self.before ? "True" : "False") + ", after=" + (self.after ? "True" : "False") +
                       ", whitespace=" + (self.whitespace ? "True" : "False") + ")";
            });

    pyTokenizer.def(
            py::init<Core::Configuration const&>(),
            py::arg("config"),
            "Construct a tokenizer from a RASR config.");

    pyTokenizer.def(
            "encode",
            [](AbstractTokenizer const& self, std::string const& text) {
                std::vector<TokenId> ids;
                throwOnError(self.encode(text, ids));
                return ids;
            },
            py::arg("text"),
            "Convert text into a list of token ids.");

    pyTokenizer.def(
            "decode",
            [](AbstractTokenizer const& self, std::vector<TokenId> const& ids) {
                std::string text;
                throwOnError(self.decode(ids, text));
                return text;
            },
            py::arg("ids"),
            "Convert a list of token ids back to text.");

    pyTokenizer.def(
            "vocabulary_size",
            [](AbstractTokenizer const& self) {
                u32 size;
                throwOnError(self.vocabularySize(size));
                return size;
            },
            "Return the number of token ids.");

    pyTokenizer.def(
            "token_str_to_id",
            [](AbstractTokenizer const& self, std::string const& token) -> std::optional<TokenId> {
                TokenId      id;
                Core::Status status = self.tokenStrToId(token, id);
                if (status.code() == Core::StatusCode::NotFound) {
                    return std::nullopt;
                }
                throwOnError(status);
                return id;
            },
            py::arg("token"),
            "Return the id of a token string or `None` if it is not part of the vocabulary.");

    pyTokenizer.def(
            "token_id_to_str",
            [](AbstractTokenizer const& self, TokenId id) {
                std::string token;
                throwOnError(self.tokenIdToStr(id, token));
                return token;
            },
            py::arg("id"),
            "Return the token string of an id.");

    pyTokenizer.def(
            "word_boundary_marks",
            [](AbstractTokenizer const& self, TokenId id) {
                WordBoundaryMarks marks;
                throwOnError(self.wordBoundaryMarks(id, marks));
                return marks;
            },
            py::arg("id"),
            "Return the word boundaries that a token marks before and after itself. A word boundary lies between two consecutive\n"
            "tokens if the first one marks it after itself or the second one before itself. Most tokenizers only mark one side,\n"
            "e.g. BPE tokens without \"@@\" mark a boundary after themselves and SentencePiece pieces with \"▁\" one before themselves.\n"
            "`whitespace` is set for tokens that decode to only whitespace or nothing on their own, e.g. \"Ġ\" of byte-level BPE.\n"
            "Such tokens can form words of their own that have no text.");

    pyTokenizer.def(
            "tokenize",
            [](AbstractTokenizer const& self, std::string const& text) {
                std::vector<std::string> tokens;
                throwOnError(self.tokenize(text, tokens));
                return tokens;
            },
            py::arg("text"),
            "Split text into a list of token strings.");

    pyTokenizer.def(
            "detokenize",
            [](AbstractTokenizer const& self, std::vector<std::string> const& tokens) {
                std::string text;
                throwOnError(self.detokenize(tokens, text));
                return text;
            },
            py::arg("tokens"),
            "Convert a list of token strings back to text.");
}
