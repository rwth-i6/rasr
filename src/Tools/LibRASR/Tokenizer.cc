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

#include <string>

#include <pybind11/stl.h>

#include <Python/Tokenizer.hh>
#include <Tokenizer/Module.hh>

// Make it so that a `py::object` can use `Core::Ref` as a holder type instead of the usual `std::unique_ptr`.
PYBIND11_DECLARE_HOLDER_TYPE(T, Core::Ref<T>, true);

void registerPythonTokenizer(std::string const& name, py::object const& pyTokenizerClass) {
    Tokenizer::Module::instance().tokenizerFactory().registerTokenizer(
            name.c_str(),
            [pyTokenizerClass](Core::Configuration const& config) {
                py::gil_scoped_acquire gil;
                // Call constructor of `pyTokenizerClass`
                py::object inst = pyTokenizerClass(config);
                inst.cast<Python::PythonTokenizer*>()->setInstance(inst);
                return inst.cast<Tokenizer::TokenizerRef>();
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
            "    tokenizer_cls: A class that inherits from `librasr.Tokenizer` and implements the abstract methods.");

    module.def(
            "create_tokenizer",
            [](Core::Configuration const& config) {
                return Tokenizer::Module::instance().tokenizerFactory().createTokenizer(Core::Configuration(config, "tokenizer"));
            },
            py::arg("config"),
            "Create the tokenizer configured via `*.tokenizer.*`, e.g. `*.tokenizer.type`. Works for native as well as registered python tokenizer types.");

    // Specify `Python::PythonTokenizer` as trampoline class and `Core::Ref<Tokenizer::AbstractTokenizer>` as holder type
    py::class_<Tokenizer::AbstractTokenizer, Python::PythonTokenizer, Tokenizer::TokenizerRef> pyTokenizer(
            module,
            "Tokenizer",
            "Abstract base class for tokenizers that convert between text and token sequences.\n"
            "Concrete subclasses need to implement the following methods:\n"
            " - `tokenize`\n"
            " - `detokenize`\n"
            " - `vocabulary_size`\n"
            " - `token_str_to_id`\n"
            " - `token_id_to_str`\n"
            "`encode` and `decode` can optionally be overridden.");

    pyTokenizer.attr("INVALID_TOKEN_ID") = Tokenizer::AbstractTokenizer::invalidTokenId;

    pyTokenizer.def(
            py::init<Core::Configuration const&>(),
            py::arg("config"),
            "Construct a tokenizer from a RASR config.");

    pyTokenizer.def(
            "tokenize",
            &Tokenizer::AbstractTokenizer::tokenize,
            py::arg("text"),
            "Split text into a list of token strings.");

    pyTokenizer.def(
            "detokenize",
            &Tokenizer::AbstractTokenizer::detokenize,
            py::arg("tokens"),
            "Convert a list of token strings back to text.");

    pyTokenizer.def(
            "vocabulary_size",
            &Tokenizer::AbstractTokenizer::vocabularySize,
            "Return the number of token ids.");

    pyTokenizer.def(
            "token_str_to_id",
            &Tokenizer::AbstractTokenizer::tokenStrToId,
            py::arg("token"),
            "Return the id of a token string or `INVALID_TOKEN_ID` if it is not part of the vocabulary.");

    pyTokenizer.def(
            "token_id_to_str",
            &Tokenizer::AbstractTokenizer::tokenIdToStr,
            py::arg("id"),
            "Return the token string of an id.");

    pyTokenizer.def(
            "encode",
            &Tokenizer::AbstractTokenizer::encode,
            py::arg("text"),
            "Convert text into a list of token ids.");

    pyTokenizer.def(
            "decode",
            &Tokenizer::AbstractTokenizer::decode,
            py::arg("ids"),
            "Convert a list of token ids back to text.");
}
