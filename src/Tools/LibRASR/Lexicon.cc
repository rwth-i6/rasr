/** Copyright 2025 RWTH Aachen University. All rights reserved.
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

#include "Lexicon.hh"

#include <Bliss/Fsa.hh>
#include <Bliss/Lexicon.hh>

PYBIND11_DECLARE_HOLDER_TYPE(T, Core::Ref<T>, true);

namespace {

// Defines __eq__ and __ne__ for comparisons with any object, which is what type checkers expect, and a consistent
// __hash__, as pybind11 would otherwise make the class unhashable
template<typename Class, typename Hash>
void defineEquality(Class& cls, Hash hash) {
    using T = typename Class::type;
    cls.def(
               "__eq__", [](T const& self, py::object const& other) { return py::isinstance<T>(other) and self == other.cast<T const&>(); }, py::arg("other"))
            .def(
                    "__ne__", [](T const& self, py::object const& other) { return not(py::isinstance<T>(other) and self == other.cast<T const&>()); }, py::arg("other"))
            .def("__hash__", hash);
}

// An empty Symbol has no string, which would crash the conversions to std::string
std::string symbolString(Bliss::Symbol const& symbol) {
    return symbol ? std::string(symbol.str()) : std::string();
}

std::string symbolRepresentation(Bliss::Symbol const& symbol) {
    return symbol ? "Symbol(\"" + symbolString(symbol) + "\")" : "Symbol()";
}

}  // namespace

void bindLexicon(py::module_& m) {
    // Classes that are referenced in signatures before their methods are defined are registered first,
    // otherwise pybind11 shows their C++ types in the signatures and type stubs
    py::class_<Bliss::Token>                                                                    token(m, "Token");
    py::class_<Bliss::SyntacticToken, Bliss::Token>                                             syntacticToken(m, "SyntacticToken");
    py::class_<Bliss::Pronunciation, std::unique_ptr<Bliss::Pronunciation, py::nodelete>>       pronunciation(m, "Pronunciation");
    py::class_<Bliss::LemmaPronunciation>                                                       lemmaPronunciation(m, "LemmaPronunciation");
    py::class_<Bliss::PhonemeInventory, Core::Ref<Bliss::PhonemeInventory>>                     phonemeInventory(m, "PhonemeInventory");
    py::class_<Fsa::Alphabet, Core::Ref<Fsa::Alphabet>>                                         alphabet(m, "Alphabet");
    py::class_<Bliss::TokenAlphabet, Fsa::Alphabet, Core::Ref<Bliss::TokenAlphabet>>            tokenAlphabet(m, "TokenAlphabet", py::multiple_inheritance());
    py::class_<Bliss::PhonemeAlphabet, Bliss::TokenAlphabet, Core::Ref<Bliss::PhonemeAlphabet>> phonemeAlphabet(m, "PhonemeAlphabet", py::multiple_inheritance());

    py::class_<Bliss::Symbol> symbol(m, "Symbol");
    symbol
            .def(py::init<>())
            .def(py::init<const Bliss::Symbol&>(), py::arg("symbol"))
            .def("__bool__", &Bliss::Symbol::operator bool)
            .def("to_string", &symbolString)
            .def("to_cstring", &Bliss::Symbol::str)
            .def("__len__", [](Bliss::Symbol const& self) { return symbolString(self).size(); })
            .def("__str__", &symbolRepresentation)
            .def("__repr__", &symbolRepresentation);
    defineEquality(symbol, [](Bliss::Symbol const& self) { return Bliss::Symbol::Hash()(self); });

    py::class_<Bliss::Symbol::Hash>(symbol, "Hash")
            .def("__call__", &Bliss::Symbol::Hash::operator(), py::arg("symbol"), py::is_operator());

    py::class_<Bliss::Symbol::Equality>(symbol, "Equality")
            .def("__call__", &Bliss::Symbol::Equality::operator(), py::arg("first"), py::arg("second"), py::is_operator());

    py::class_<Bliss::OrthographicFormList>(m, "OrthographicFormList")
            .def(py::init<>())
            .def(py::init<const Bliss::Symbol*, const Bliss::Symbol*>(), py::arg("begin"), py::arg("end"))
            .def(py::init<const Bliss::OrthographicFormList&>(), py::arg("other"))
            .def("valid", &Bliss::OrthographicFormList::valid)
            .def("is_epsilon", &Bliss::OrthographicFormList::isEpsilon)
            .def("front", &Bliss::OrthographicFormList::front, py::return_value_policy::reference_internal)
            .def("__getitem__", &Bliss::OrthographicFormList::operator[], py::arg("index"), py::is_operator())
            .def("__len__", &Bliss::OrthographicFormList::size)
            .def("__iter__", [](Bliss::OrthographicFormList const& self) { return py::make_iterator(self.begin(), self.end()); }, py::keep_alive<0, 1>());

    py::class_<Bliss::SyntacticTokenSequence>(m, "SyntacticTokenSequence")
            .def(py::init<>())
            .def(py::init<const Bliss::SyntacticTokenSequence&>(), py::arg("other"))
            .def("valid", &Bliss::SyntacticTokenSequence::valid)
            .def("is_epsilon", &Bliss::SyntacticTokenSequence::isEpsilon)
            .def("front", &Bliss::SyntacticTokenSequence::front, py::return_value_policy::reference_internal)
            .def("__len__", &Bliss::SyntacticTokenSequence::size)
            .def("__getitem__", &Bliss::SyntacticTokenSequence::operator[], py::arg("index"), py::is_operator());

    token
            .def("symbol", &Bliss::Token::symbol)
            .def("id", &Bliss::Token::id)
            .def_readonly_static("invalid_id", &Bliss::Token::invalidId);

    py::class_<Bliss::Lemma, Bliss::Token, std::unique_ptr<Bliss::Lemma, py::nodelete>> lemma(m, "Lemma");
    lemma
            .def("hasName", &Bliss::Lemma::hasName)
            .def("name", &Bliss::Lemma::name)
            .def("num_orthographic_forms", &Bliss::Lemma::nOrthographicForms)
            .def("orthographic_forms", &Bliss::Lemma::orthographicForms, py::return_value_policy::reference_internal)
            .def("preferred_orthographic_form", &Bliss::Lemma::preferredOrthographicForm)
            .def("num_pronunciations", &Bliss::Lemma::nPronunciations)
            .def("has_pronunciation", &Bliss::Lemma::hasPronunciation, py::arg("pronunciation"))
            .def("has_syntactic_token_sequence", &Bliss::Lemma::hasSyntacticTokenSequence)
            .def("syntactic_token_sequence", &Bliss::Lemma::syntacticTokenSequence, py::return_value_policy::reference_internal)
            .def("has_evaluation_token_sequence", &Bliss::Lemma::hasEvaluationTokenSequence)
            .def("num_evaluation_token_sequences", &Bliss::Lemma::nEvaluationTokenSequences);

    py::class_<Bliss::Lemma::PronunciationIterator>(lemma, "PronunciationIterator")
            .def("to_lemma_pronunciation", &Bliss::Lemma::PronunciationIterator::operator const Bliss::LemmaPronunciation*)
            .def("__next__", &Bliss::Lemma::PronunciationIterator::operator++, py::return_value_policy::reference_internal, py::is_operator());

    pronunciation
            .def("num_lemmas", &Bliss::Pronunciation::nLemmas)
            .def("format", &Bliss::Pronunciation::format, py::arg("phoneme_inventory"))
            .def("__len__", &Bliss::Pronunciation::length)
            .def("__iter__", [](Bliss::Pronunciation const& self) { return py::make_iterator(self.phonemes(), self.phonemes() + self.length()); }, py::keep_alive<0, 1>())
            .def("__getitem__", &Bliss::Pronunciation::operator[], py::arg("index"), py::is_operator());

    py::class_<Bliss::Pronunciation::LemmaIterator> lemmaIterator(pronunciation, "LemmaIterator");
    defineEquality(lemmaIterator, [](Bliss::Pronunciation::LemmaIterator const& self) { return std::hash<Bliss::LemmaPronunciation const*>()(self); });
    lemmaIterator
            .def(py::init<>())
            .def("to_lemma_pronunciation", &Bliss::Pronunciation::LemmaIterator::operator const Bliss::LemmaPronunciation*)
            .def("__next__", &Bliss::Pronunciation::LemmaIterator::operator++, py::return_value_policy::reference_internal, py::is_operator());

    py::class_<Bliss::Pronunciation::Hash>(pronunciation, "Hash")
            .def("__call__", (u32(Bliss::Pronunciation::Hash::*)(const Bliss::Phoneme::Id*) const) & Bliss::Pronunciation::Hash::operator(), py::arg("phonemes"), py::is_operator())
            .def("__call__", (u32(Bliss::Pronunciation::Hash::*)(const Bliss::Pronunciation*) const) & Bliss::Pronunciation::Hash::operator(), py::arg("pronunciation"), py::is_operator());

    py::class_<Bliss::Pronunciation::Equality>(pronunciation, "Equality")
            .def("__call__", (bool(Bliss::Pronunciation::Equality::*)(const Bliss::Phoneme::Id*, const Bliss::Phoneme::Id*) const) & Bliss::Pronunciation::Equality::operator(), py::arg("first"), py::arg("second"), py::is_operator())
            .def("__call__", (bool(Bliss::Pronunciation::Equality::*)(const Bliss::Pronunciation*, const Bliss::Pronunciation*) const) & Bliss::Pronunciation::Equality::operator(), py::arg("first"), py::arg("second"), py::is_operator());

    lemmaPronunciation
            .def("id", &Bliss::LemmaPronunciation::id)
            .def("lemma", &Bliss::LemmaPronunciation::lemma, py::return_value_policy::reference_internal)
            .def("pronunciation", &Bliss::LemmaPronunciation::pronunciation, py::return_value_policy::reference_internal)
            .def("pronunciation_probability", &Bliss::LemmaPronunciation::pronunciationProbability)
            .def("pronunciation_score", &Bliss::LemmaPronunciation::pronunciationScore)
            .def("set_pronunciation_probability", &Bliss::LemmaPronunciation::setPronunciationProbability, py::arg("probability"))
            .def("next_for_this_lemma", &Bliss::LemmaPronunciation::nextForThisLemma, py::return_value_policy::reference_internal)
            .def("next_for_this_pronunciation", &Bliss::LemmaPronunciation::nextForThisPronunciation, py::return_value_policy::reference_internal);

    syntacticToken
            .def("num_lemmas", &Bliss::SyntacticToken::nLemmas)
            .def("class_emission_score", &Bliss::SyntacticToken::classEmissionScore);

    py::class_<Bliss::Letter, Bliss::Token>(m, "Letter");

    py::class_<Bliss::Phoneme, Bliss::Token>(m, "Phoneme")
            .def("set_context_dependent", &Bliss::Phoneme::setContextDependent, py::arg("context_dependent"))
            .def("is_context_dependent", &Bliss::Phoneme::isContextDependent)
            .def_readonly_static("term", &Bliss::Phoneme::term);

    phonemeInventory
            .def(py::init<>())
            .def("num_phonemes", &Bliss::PhonemeInventory::nPhonemes)
            .def("phoneme", (const Bliss::Phoneme* (Bliss::PhonemeInventory::*)(const std::string&) const) & Bliss::PhonemeInventory::phoneme, py::arg("symbol"), py::return_value_policy::reference_internal)
            .def("phoneme", (const Bliss::Phoneme* (Bliss::PhonemeInventory::*)(Bliss::Phoneme::Id) const) & Bliss::PhonemeInventory::phoneme, py::arg("id"), py::return_value_policy::reference_internal)
            .def("is_valid_phoneme_id", &Bliss::PhonemeInventory::isValidPhonemeId, py::arg("id"))
            .def("new_phoneme", &Bliss::PhonemeInventory::newPhoneme, py::return_value_policy::reference_internal)
            .def("assign_symbol", &Bliss::PhonemeInventory::assignSymbol, py::arg("phoneme"), py::arg("symbol"))
            .def("phoneme_alphabet", &Bliss::PhonemeInventory::phonemeAlphabet, py::return_value_policy::take_ownership)
            .def("parse_selection", &Bliss::PhonemeInventory::parseSelection, py::arg("selection"))
            .def("write_xml", [](Bliss::PhonemeInventory& self, const std::string& name) {
                std::ofstream file(name + ".xml");
                if (file) {
                    Core::XmlWriter writer(file);
                    self.writeXml(writer);
                }
                file.close(); }, py::arg("name"));

    alphabet
            .def("special_symbol", &Fsa::Alphabet::specialSymbol, py::arg("index"))
            .def("special_index", &Fsa::Alphabet::specialIndex, py::arg("symbol"))
            .def("next", &Fsa::Alphabet::next, py::arg("index"))
            .def("symbol", &Fsa::Alphabet::symbol, py::arg("index"))
            .def("index", &Fsa::Alphabet::index, py::arg("symbol"))
            .def("is_disambiguator", &Fsa::Alphabet::isDisambiguator, py::arg("index"))
            .def("tag", &Fsa::Alphabet::tag, py::arg("index"))
            .def("get_memory_used", &Fsa::Alphabet::getMemoryUsed)
            .def("__iter__", [](Fsa::Alphabet const& self) { return py::make_iterator(self.begin(), self.end()); }, py::keep_alive<0, 1>())
            .def("write_xml", [](Fsa::Alphabet& self, const std::string& name) {
                std::ofstream file(name + ".xml");
                if (file) {
                    Core::XmlWriter writer(file);
                    self.writeXml(writer);
                }
                file.close(); }, py::arg("name"));

    py::class_<Fsa::Alphabet::const_iterator> alphabetIterator(alphabet, "const_iterator");
    defineEquality(alphabetIterator, [](Fsa::Alphabet::const_iterator const& self) { return static_cast<Fsa::LabelId>(self); });
    alphabetIterator
            .def(py::init<Core::Ref<const Fsa::Alphabet>, Fsa::LabelId>(), py::arg("alphabet"), py::arg("index"))
            .def("get_symbol", &Fsa::Alphabet::const_iterator::operator*)
            .def("to_label_id", (Fsa::LabelId(Fsa::Alphabet::const_iterator::*)() const) & Fsa::Alphabet::const_iterator::operator Fsa::LabelId)
            .def("__next__", &Fsa::Alphabet::const_iterator::operator++, py::return_value_policy::reference_internal, py::is_operator());

    tokenAlphabet
            .def("symbol", &Bliss::TokenAlphabet::symbol, py::arg("index"))
            .def("index", (Fsa::LabelId(Bliss::TokenAlphabet::*)(const std::string&) const) & Bliss::TokenAlphabet::index, py::arg("symbol"))
            .def("index", (Fsa::LabelId(Bliss::TokenAlphabet::*)(const Bliss::Token*) const) & Bliss::TokenAlphabet::index, py::arg("token"))
            .def("token", &Bliss::TokenAlphabet::token, py::arg("index"), py::return_value_policy::reference_internal)
            .def("num_disambiguators", &Bliss::TokenAlphabet::nDisambiguators)
            .def("disambiguator", &Bliss::TokenAlphabet::disambiguator, py::arg("number"))
            .def("is_disambiguator", &Bliss::TokenAlphabet::isDisambiguator, py::arg("index"))
            .def("write_xml", [](Bliss::TokenAlphabet& self, const std::string& name) {
                std::ofstream file(name + ".xml");
                if (file) {
                    Core::XmlWriter writer(file);
                    self.writeXml(writer);
                }
                file.close(); }, py::arg("name"));

    phonemeAlphabet
            .def("phoneme_inventory", &Bliss::PhonemeAlphabet::phonemeInventory, py::return_value_policy::take_ownership)
            .def("phoneme", &Bliss::PhonemeAlphabet::phoneme, py::arg("index"), py::return_value_policy::reference_internal)
            .def("symbol", &Bliss::PhonemeAlphabet::symbol, py::arg("index"))
            .def("index", (Fsa::LabelId(Bliss::PhonemeAlphabet::*)(const std::string&) const) & Bliss::PhonemeAlphabet::index, py::arg("symbol"))
            .def("index", (Fsa::LabelId(Bliss::PhonemeAlphabet::*)(const Bliss::Token*) const) & Bliss::PhonemeAlphabet::index, py::arg("token"))
            .def("__iter__", [](Bliss::PhonemeAlphabet const& self) { return py::make_iterator(self.begin(), self.end()); }, py::keep_alive<0, 1>())
            .def("write_xml", [](Bliss::PhonemeAlphabet& self, const std::string& name) {
                std::ofstream file(name + ".xml");
                if (file) {
                    Core::XmlWriter writer(file);
                    self.writeXml(writer);
                }
                file.close(); }, py::arg("name"));

    py::class_<Bliss::LemmaAlphabet, Bliss::TokenAlphabet, Core::Ref<Bliss::LemmaAlphabet>>(m, "LemmaAlphabet", py::multiple_inheritance())
            .def("lemma", &Bliss::LemmaAlphabet::lemma, py::arg("index"), py::return_value_policy::reference_internal);

    py::class_<Bliss::LemmaPronunciationAlphabet, Fsa::Alphabet, Core::Ref<Bliss::LemmaPronunciationAlphabet>>(m, "LemmaPronunciationAlphabet", py::multiple_inheritance())
            .def("index", (Fsa::LabelId(Bliss::LemmaPronunciationAlphabet::*)(const std::string&) const) & Bliss::LemmaPronunciationAlphabet::index, py::arg("symbol"))
            .def("index", (Fsa::LabelId(Bliss::LemmaPronunciationAlphabet::*)(const Bliss::LemmaPronunciation*) const) & Bliss::LemmaPronunciationAlphabet::index, py::arg("lemma_pronunciation"))
            .def("lemma_pronunciation", &Bliss::LemmaPronunciationAlphabet::lemmaPronunciation, py::arg("index"), py::return_value_policy::reference_internal)
            .def("symbol", &Bliss::LemmaPronunciationAlphabet::symbol, py::arg("index"))
            .def("end", &Bliss::LemmaPronunciationAlphabet::end)
            .def("num_disambiguators", &Bliss::LemmaPronunciationAlphabet::nDisambiguators)
            .def("disambiguator", &Bliss::LemmaPronunciationAlphabet::disambiguator, py::arg("number"))
            .def("is_disambiguator", &Bliss::LemmaPronunciationAlphabet::isDisambiguator, py::arg("index"))
            .def("write_xml", [](Bliss::LemmaPronunciationAlphabet& self, const std::string& name) {
                std::ofstream file(name + ".xml");
                if (file) {
                    Core::XmlWriter writer(file);
                    self.writeXml(writer);
                }
                file.close(); }, py::arg("name"));

    py::class_<Bliss::SyntacticTokenAlphabet, Bliss::TokenAlphabet, Core::Ref<Bliss::SyntacticTokenAlphabet>>(m, "SyntacticTokenAlphabet", py::multiple_inheritance())
            .def("syntactic_token", &Bliss::SyntacticTokenAlphabet::syntacticToken, py::arg("index"), py::return_value_policy::reference_internal);

    py::class_<Bliss::TokenInventory>(m, "TokenInventory")
            .def("insert", &Bliss::TokenInventory::insert, py::arg("token"))
            .def("link", &Bliss::TokenInventory::link, py::arg("symbol"), py::arg("token"))
            .def("add", &Bliss::TokenInventory::add, py::arg("token"))
            .def("__len__", &Bliss::TokenInventory::size)
            .def("__iter__", [](Bliss::TokenInventory const& self) { return py::make_iterator(self.begin(), self.end()); }, py::keep_alive<0, 1>())
            .def("__getitem__", (Bliss::Token * (Bliss::TokenInventory::*)(Bliss::Token::Id) const) & Bliss::TokenInventory::operator[], py::arg("id"), py::return_value_policy::reference_internal, py::is_operator())
            .def("__getitem__", (Bliss::Token * (Bliss::TokenInventory::*)(const std::string&) const) & Bliss::TokenInventory::operator[], py::arg("symbol"), py::return_value_policy::reference_internal, py::is_operator())
            .def("__getitem__", (Bliss::Token * (Bliss::TokenInventory::*)(Bliss::Symbol) const) & Bliss::TokenInventory::operator[], py::arg("symbol"), py::return_value_policy::reference_internal, py::is_operator());

    py::class_<Bliss::EvaluationToken, Bliss::Token>(m, "EvaluationToken");

    py::class_<Bliss::EvaluationTokenAlphabet, Bliss::TokenAlphabet, Core::Ref<Bliss::EvaluationTokenAlphabet>>(m, "EvaluationTokenAlphabet")
            .def("evaluation_token", &Bliss::EvaluationTokenAlphabet::evaluationToken, py::arg("index"), py::return_value_policy::reference_internal);

    py::class_<Bliss::LetterAlphabet, Bliss::TokenAlphabet, Core::Ref<Bliss::LetterAlphabet>>(m, "LetterAlphabet")
            .def("letter", &Bliss::LetterAlphabet::letter, py::arg("index"), py::return_value_policy::reference_internal);

    py::class_<Bliss::Lexicon, Core::Ref<Bliss::Lexicon>>(m, "Lexicon", py::multiple_inheritance())
            .def(py::init<const Core::Configuration&>(), py::arg("config"))
            .def("new_lemma", (Bliss::Lemma * (Bliss::Lexicon::*)()) & Bliss::Lexicon::newLemma, py::return_value_policy::reference_internal)
            .def("new_lemma", (Bliss::Lemma * (Bliss::Lexicon::*)(const std::string&)) & Bliss::Lexicon::newLemma, py::arg("name"), py::return_value_policy::reference_internal)
            .def("set_orthographic_forms", &Bliss::Lexicon::setOrthographicForms, py::arg("lemma"), py::arg("orthographic_forms"))
            .def("set_default_lemma_name", &Bliss::Lexicon::setDefaultLemmaName, py::arg("lemma"))
            //.def("get_pronunciation", &Bliss::Lexicon::getPronunciation, py::return_value_policy::reference_internal)
            .def("add_pronunciation", &Bliss::Lexicon::addPronunciation, py::arg("lemma"), py::arg("pronunciation"), py::arg("weight") = 1.0f, py::return_value_policy::reference_internal)
            .def("normalize_pronunciation_weights", &Bliss::Lexicon::normalizePronunciationWeights, py::arg("lemma"))
            .def("set_syntactic_token_sequence", py::overload_cast<Bliss::Lemma*, std::vector<std::string> const&>(&Bliss::Lexicon::setSyntacticTokenSequence), py::arg("lemma"), py::arg("symbols"))
            .def("set_syntactic_token_sequence", py::overload_cast<Bliss::Lemma*, std::vector<Bliss::Token::Id> const&>(&Bliss::Lexicon::setSyntacticTokenSequence), py::arg("lemma"), py::arg("ids"))
            .def("set_default_syntactic_token", &Bliss::Lexicon::setDefaultSyntacticToken, py::arg("lemma"))
            .def("add_evaluation_token_sequence", py::overload_cast<Bliss::Lemma*, std::vector<std::string> const&>(&Bliss::Lexicon::addEvaluationTokenSequence), py::arg("lemma"), py::arg("symbols"))
            .def("add_evaluation_token_sequence", py::overload_cast<Bliss::Lemma*, std::vector<Bliss::Token::Id> const&>(&Bliss::Lexicon::addEvaluationTokenSequence), py::arg("lemma"), py::arg("ids"))
            .def("set_default_evaluation_token", &Bliss::Lexicon::setDefaultEvaluationToken, py::arg("lemma"))
            .def("define_special_lemma", &Bliss::Lexicon::defineSpecialLemma, py::arg("name"), py::arg("lemma"))
            .def("load", &Bliss::Lexicon::load, py::arg("filename"))
            .def("write_xml", [](Bliss::Lexicon& self, const std::string& name) {
                std::ofstream file(name + ".xml");
                if (file) {
                    Core::XmlWriter writer(file);
                    self.writeXml(writer);
                }
                file.close(); }, py::arg("name"))
            .def("log_statistics", &Bliss::Lexicon::logStatistics)
            .def_static("create", &Bliss::Lexicon::create, py::arg("config"))
            .def("num_lemmas", &Bliss::Lexicon::nLemmas)
            .def("lemmas", [](Bliss::Lexicon const& self) { auto begin_end = self.lemmas(); return py::make_iterator(begin_end.first, begin_end.second); }, py::keep_alive<0, 1>())
            .def("special_lemma", &Bliss::Lexicon::specialLemma, py::arg("name"), py::return_value_policy::reference_internal)
            .def("lemma_alphabet", &Bliss::Lexicon::lemmaAlphabet, py::return_value_policy::take_ownership)
            .def("set_phoneme_inventory", &Bliss::Lexicon::setPhonemeInventory, py::arg("phoneme_inventory"))
            .def("phoneme_inventory", &Bliss::Lexicon::phonemeInventory, py::return_value_policy::take_ownership)
            .def("num_pronunciations", &Bliss::Lexicon::nPronunciations)
            .def("pronunciations", [](Bliss::Lexicon const& self) { auto begin_end = self.pronunciations(); return py::make_iterator(begin_end.first, begin_end.second); }, py::keep_alive<0, 1>())
            .def("num_lemma_pronunciations", &Bliss::Lexicon::nLemmaPronunciations)
            .def("lemma_pronunciations", [](Bliss::Lexicon const& self) { auto begin_end = self.lemmaPronunciations(); return py::make_iterator(begin_end.first, begin_end.second); }, py::keep_alive<0, 1>())
            .def("lemma_pronunciation_alphabet", &Bliss::Lexicon::lemmaPronunciationAlphabet, py::return_value_policy::take_ownership)
            .def("lemma_pronunciation", &Bliss::Lexicon::lemmaPronunciation, py::arg("id"), py::return_value_policy::reference_internal)
            .def("num_syntactic_tokens", &Bliss::Lexicon::nSyntacticTokens)
            .def("syntactic_token", &Bliss::Lexicon::syntacticToken, py::arg("symbol"), py::return_value_policy::reference_internal)
            .def("syntactic_token_inventory", &Bliss::Lexicon::syntacticTokenInventory, py::return_value_policy::reference_internal)
            .def("syntactic_token_alphabet", &Bliss::Lexicon::syntacticTokenAlphabet, py::return_value_policy::take_ownership)
            .def("num_evaluation_tokens", &Bliss::Lexicon::nEvaluationTokens)
            .def("evaluation_token_inventory", &Bliss::Lexicon::evaluationTokenInventory, py::return_value_policy::reference_internal)
            .def("evaluation_token_alphabet", &Bliss::Lexicon::evaluationTokenAlphabet, py::return_value_policy::take_ownership)
            .def("letter", &Bliss::Lexicon::letter, py::arg("symbol"), py::return_value_policy::reference_internal)
            .def("letter_inventory", &Bliss::Lexicon::letterInventory, py::return_value_policy::reference_internal)
            .def("letter_alphabet", &Bliss::Lexicon::letterAlphabet, py::return_value_policy::take_ownership);
}
