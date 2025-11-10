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

void bindLexicon(py::module_& m) {
    py::class_<Bliss::Symbol> symbol(m, "Symbol");
    symbol
            .def(py::init<>())
            .def(py::init<const Bliss::Symbol&>())
            .def("__eq__", &Bliss::Symbol::operator==, py::is_operator())
            .def("__ne__", &Bliss::Symbol::operator!=, py::is_operator())
            .def("_bool_", &Bliss::Symbol::operator bool, py::is_operator())
            .def("to_string", &Bliss::Symbol::operator Bliss::Symbol::String)
            .def("to_cstring", &Bliss::Symbol::str)
            .def("__len__", &Bliss::Symbol::length)
            .def("__str__", [](Bliss::Symbol const& self) { return std::string("Symbol(\"") + self.str() + "\")"; })
            .def("__repr__", [](Bliss::Symbol const& self) { return std::string("Symbol(\"") + self.str() + "\")"; })
            .def_static("cast", &Bliss::Symbol::cast);

    py::class_<Bliss::Symbol::Hash>(symbol, "Hash")
            .def("__call__", &Bliss::Symbol::Hash::operator(), py::is_operator());

    py::class_<Bliss::Symbol::Equality>(symbol, "Equality")
            .def("__call__", &Bliss::Symbol::Equality::operator(), py::is_operator());

    py::class_<Bliss::OrthographicFormList>(m, "OrthographicFormList")
            .def(py::init<>())
            .def(py::init<const Bliss::Symbol*, const Bliss::Symbol*>())
            .def(py::init<const Bliss::OrthographicFormList&>())
            .def("valid", &Bliss::OrthographicFormList::valid)
            .def("is_epsilon", &Bliss::OrthographicFormList::isEpsilon)
            .def("front", &Bliss::OrthographicFormList::front, py::return_value_policy::reference_internal)
            .def("__getitem__", &Bliss::OrthographicFormList::operator[], py::is_operator())
            .def("__len__", &Bliss::OrthographicFormList::size)
            .def("__iter__", [](Bliss::OrthographicFormList const& self) { return py::make_iterator(self.begin(), self.end()); }, py::keep_alive<0, 1>());

    py::class_<Bliss::SyntacticTokenSequence>(m, "SyntacticTokenSequence")
            .def(py::init<>())
            .def(py::init<const Bliss::SyntacticTokenSequence&>())
            .def("valid", &Bliss::SyntacticTokenSequence::valid)
            .def("is_epsilon", &Bliss::SyntacticTokenSequence::isEpsilon)
            .def("front", &Bliss::SyntacticTokenSequence::front, py::return_value_policy::reference_internal)
            .def("__len__", &Bliss::SyntacticTokenSequence::size)
            .def("__getitem__", &Bliss::SyntacticTokenSequence::operator[], py::is_operator());

    py::class_<Bliss::Token>(m, "Token")
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
            .def("has_pronunciation", &Bliss::Lemma::hasPronunciation)
            .def("has_syntactic_token_sequence", &Bliss::Lemma::hasSyntacticTokenSequence)
            .def("syntactic_token_sequence", &Bliss::Lemma::syntacticTokenSequence, py::return_value_policy::reference_internal)
            .def("has_evaluation_token_sequence", &Bliss::Lemma::hasEvaluationTokenSequence)
            .def("num_evaluation_token_sequences", &Bliss::Lemma::nEvaluationTokenSequences);

    py::class_<Bliss::Lemma::PronunciationIterator>(lemma, "PronunciationIterator")
            .def("to_lemma_pronunciation", &Bliss::Lemma::PronunciationIterator::operator const Bliss::LemmaPronunciation*)
            .def("__next__", &Bliss::Lemma::PronunciationIterator::operator++, py::return_value_policy::reference_internal, py::is_operator());

    py::class_<Bliss::Pronunciation, std::unique_ptr<Bliss::Pronunciation, py::nodelete>> pronunciation(m, "Pronunciation");
    pronunciation
            .def("num_lemmas", &Bliss::Pronunciation::nLemmas)
            .def("format", &Bliss::Pronunciation::format)
            .def("__len__", &Bliss::Pronunciation::length)
            .def("__iter__", [](Bliss::Pronunciation const& self) { return py::make_iterator(self.phonemes(), self.phonemes() + self.length()); }, py::keep_alive<0, 1>())
            .def("__getitem__", &Bliss::Pronunciation::operator[], py::is_operator());

    py::class_<Bliss::Pronunciation::LemmaIterator>(pronunciation, "LemmaIterator")
            .def(py::init<>())
            .def("__eq__", &Bliss::Pronunciation::LemmaIterator::operator==, py::is_operator())
            .def("__ne__", &Bliss::Pronunciation::LemmaIterator::operator!=, py::is_operator())
            .def("to_lemma_pronunciation", &Bliss::Pronunciation::LemmaIterator::operator const Bliss::LemmaPronunciation*)
            .def("__next__", &Bliss::Pronunciation::LemmaIterator::operator++, py::return_value_policy::reference_internal, py::is_operator());

    py::class_<Bliss::Pronunciation::Hash>(pronunciation, "Hash")
            .def("__call__", (u32(Bliss::Pronunciation::Hash::*)(const Bliss::Phoneme::Id*) const) & Bliss::Pronunciation::Hash::operator(), py::is_operator())
            .def("__call__", (u32(Bliss::Pronunciation::Hash::*)(const Bliss::Pronunciation*) const) & Bliss::Pronunciation::Hash::operator(), py::is_operator());

    py::class_<Bliss::Pronunciation::Equality>(pronunciation, "Equality")
            .def("__call__", (bool(Bliss::Pronunciation::Equality::*)(const Bliss::Phoneme::Id*, const Bliss::Phoneme::Id*) const) & Bliss::Pronunciation::Equality::operator(), py::is_operator())
            .def("__call__", (bool(Bliss::Pronunciation::Equality::*)(const Bliss::Pronunciation*, const Bliss::Pronunciation*) const) & Bliss::Pronunciation::Equality::operator(), py::is_operator());

    py::class_<Bliss::LemmaPronunciation>(m, "LemmaPronunciation")
            .def("id", &Bliss::LemmaPronunciation::id)
            .def("lemma", &Bliss::LemmaPronunciation::lemma, py::return_value_policy::reference_internal)
            .def("pronunciation", &Bliss::LemmaPronunciation::pronunciation, py::return_value_policy::reference_internal)
            .def("pronunciation_probability", &Bliss::LemmaPronunciation::pronunciationProbability)
            .def("pronunciation_score", &Bliss::LemmaPronunciation::pronunciationScore)
            .def("set_pronunciation_probability", &Bliss::LemmaPronunciation::setPronunciationProbability)
            .def("next_for_this_lemma", &Bliss::LemmaPronunciation::nextForThisLemma, py::return_value_policy::reference_internal)
            .def("next_for_this_pronunciation", &Bliss::LemmaPronunciation::nextForThisPronunciation, py::return_value_policy::reference_internal);

    py::class_<Bliss::SyntacticToken, Bliss::Token>(m, "SyntacticToken")
            .def("num_lemmas", &Bliss::SyntacticToken::nLemmas)
            .def("class_emission_score", &Bliss::SyntacticToken::classEmissionScore);

    py::class_<Bliss::Letter, Bliss::Token>(m, "Letter");

    py::class_<Bliss::Phoneme, Bliss::Token>(m, "Phoneme")
            .def("set_context_dependent", &Bliss::Phoneme::setContextDependent)
            .def("is_context_dependent", &Bliss::Phoneme::isContextDependent)
            .def_readonly_static("term", &Bliss::Phoneme::term);

    py::class_<Bliss::PhonemeInventory, Core::Ref<Bliss::PhonemeInventory>>(m, "PhonemeInventory")
            .def(py::init<>())
            .def("num_phonemes", &Bliss::PhonemeInventory::nPhonemes)
            .def("phoneme", (const Bliss::Phoneme* (Bliss::PhonemeInventory::*)(const std::string&) const) & Bliss::PhonemeInventory::phoneme, py::return_value_policy::reference_internal)
            .def("phoneme", (const Bliss::Phoneme* (Bliss::PhonemeInventory::*)(Bliss::Phoneme::Id) const) & Bliss::PhonemeInventory::phoneme, py::return_value_policy::reference_internal)
            .def("is_valid_phoneme_id", &Bliss::PhonemeInventory::isValidPhonemeId)
            .def("new_phoneme", &Bliss::PhonemeInventory::newPhoneme, py::return_value_policy::reference_internal)
            .def("assign_symbol", &Bliss::PhonemeInventory::assignSymbol)
            .def("phoneme_alphabet", &Bliss::PhonemeInventory::phonemeAlphabet, py::return_value_policy::take_ownership)
            .def("parse_selection", &Bliss::PhonemeInventory::parseSelection)
            .def("write_xml", [](Bliss::PhonemeInventory& self, const std::string& name) {
                std::ofstream file(name + ".xml");
                if (file) {
                    Core::XmlWriter writer(file);
                    self.writeXml(writer);
                }
                file.close();
            });

    py::class_<Fsa::Alphabet, Core::Ref<Fsa::Alphabet>> alphabet(m, "Alphabet");
    alphabet
            .def("special_symbol", &Fsa::Alphabet::specialSymbol)
            .def("special_index", &Fsa::Alphabet::specialIndex)
            .def("next", &Fsa::Alphabet::next)
            .def("symbol", &Fsa::Alphabet::symbol)
            .def("index", &Fsa::Alphabet::index)
            .def("is_disambiguator", &Fsa::Alphabet::isDisambiguator)
            .def("tag", &Fsa::Alphabet::tag)
            .def("get_memory_used", &Fsa::Alphabet::getMemoryUsed)
            .def("__iter__", [](Fsa::Alphabet const& self) { return py::make_iterator(self.begin(), self.end()); }, py::keep_alive<0, 1>())
            .def("write_xml", [](Fsa::Alphabet& self, const std::string& name) {
                std::ofstream file(name + ".xml");
                if (file) {
                    Core::XmlWriter writer(file);
                    self.writeXml(writer);
                }
                file.close(); });

    py::class_<Fsa::Alphabet::const_iterator>(alphabet, "const_iterator")
            .def(py::init<Core::Ref<const Fsa::Alphabet>, Fsa::LabelId>())
            .def("__eq__", &Fsa::Alphabet::const_iterator::operator==, py::is_operator())
            .def("__ne__", &Fsa::Alphabet::const_iterator::operator!=, py::is_operator())
            .def("get_symbol", &Fsa::Alphabet::const_iterator::operator*)
            .def("to_label_id", (Fsa::LabelId(Fsa::Alphabet::const_iterator::*)()) & Fsa::Alphabet::const_iterator::operator Fsa::LabelId)
            .def("to_label_id", (Fsa::LabelId(Fsa::Alphabet::const_iterator::*)() const) & Fsa::Alphabet::const_iterator::operator Fsa::LabelId)
            .def("__next__", &Fsa::Alphabet::const_iterator::operator++, py::return_value_policy::reference_internal, py::is_operator());

    py::class_<Bliss::TokenAlphabet, Fsa::Alphabet, Core::Ref<Bliss::TokenAlphabet>>(m, "TokenAlphabet", py::multiple_inheritance())
            .def("symbol", &Bliss::TokenAlphabet::symbol)
            .def("index", (Fsa::LabelId(Bliss::TokenAlphabet::*)(const std::string&) const) & Bliss::TokenAlphabet::index)
            .def("index", (Fsa::LabelId(Bliss::TokenAlphabet::*)(const Bliss::Token*) const) & Bliss::TokenAlphabet::index)
            .def("token", &Bliss::TokenAlphabet::token, py::return_value_policy::reference_internal)
            .def("num_disambiguators", &Bliss::TokenAlphabet::nDisambiguators)
            .def("disambiguator", &Bliss::TokenAlphabet::disambiguator)
            .def("is_disambiguator", &Bliss::TokenAlphabet::isDisambiguator)
            .def("write_xml", [](Bliss::TokenAlphabet& self, const std::string& name) {
                std::ofstream file(name + ".xml");
                if (file) {
                    Core::XmlWriter writer(file);
                    self.writeXml(writer);
                }
                file.close();
            });

    py::class_<Bliss::PhonemeAlphabet, Bliss::TokenAlphabet, Core::Ref<Bliss::PhonemeAlphabet>>(m, "PhonemeAlphabet", py::multiple_inheritance())
            .def("phoneme_inventory", &Bliss::PhonemeAlphabet::phonemeInventory, py::return_value_policy::take_ownership)
            .def("phoneme", &Bliss::PhonemeAlphabet::phoneme, py::return_value_policy::reference_internal)
            .def("symbol", &Bliss::PhonemeAlphabet::symbol)
            .def("index", (Fsa::LabelId(Bliss::PhonemeAlphabet::*)(const std::string&) const) & Bliss::PhonemeAlphabet::index)
            .def("index", (Fsa::LabelId(Bliss::PhonemeAlphabet::*)(const Bliss::Token*) const) & Bliss::PhonemeAlphabet::index)
            .def("__iter__", [](Bliss::PhonemeAlphabet const& self) { return py::make_iterator(self.begin(), self.end()); }, py::keep_alive<0, 1>())
            .def("write_xml", [](Bliss::PhonemeAlphabet& self, const std::string& name) {
                std::ofstream file(name + ".xml");
                if (file) {
                    Core::XmlWriter writer(file);
                    self.writeXml(writer);
                }
                file.close(); });

    py::class_<Bliss::LemmaAlphabet, Bliss::TokenAlphabet, Core::Ref<Bliss::LemmaAlphabet>>(m, "LemmaAlphabet", py::multiple_inheritance())
            .def("lemma", &Bliss::LemmaAlphabet::lemma, py::return_value_policy::reference_internal);

    py::class_<Bliss::LemmaPronunciationAlphabet, Fsa::Alphabet, Core::Ref<Bliss::LemmaPronunciationAlphabet>>(m, "LemmaPronunciationAlphabet", py::multiple_inheritance())
            .def("index", (Fsa::LabelId(Bliss::LemmaPronunciationAlphabet::*)(const std::string&) const) & Bliss::LemmaPronunciationAlphabet::index)
            .def("index", (Fsa::LabelId(Bliss::LemmaPronunciationAlphabet::*)(const Bliss::LemmaPronunciation*) const) & Bliss::LemmaPronunciationAlphabet::index)
            .def("lemma_pronunciation", &Bliss::LemmaPronunciationAlphabet::lemmaPronunciation, py::return_value_policy::reference_internal)
            .def("symbol", &Bliss::LemmaPronunciationAlphabet::symbol)
            .def("end", &Bliss::LemmaPronunciationAlphabet::end)
            .def("num_disambiguators", &Bliss::LemmaPronunciationAlphabet::nDisambiguators)
            .def("disambiguator", &Bliss::LemmaPronunciationAlphabet::disambiguator)
            .def("is_disambiguator", &Bliss::LemmaPronunciationAlphabet::isDisambiguator)
            .def("write_xml", [](Bliss::LemmaPronunciationAlphabet& self, const std::string& name) {
                std::ofstream file(name + ".xml");
                if (file) {
                    Core::XmlWriter writer(file);
                    self.writeXml(writer);
                }
                file.close();
            });

    py::class_<Bliss::SyntacticTokenAlphabet, Bliss::TokenAlphabet, Core::Ref<Bliss::SyntacticTokenAlphabet>>(m, "SyntacticTokenAlphabet", py::multiple_inheritance())
            .def("syntactic_token", &Bliss::SyntacticTokenAlphabet::syntacticToken, py::return_value_policy::reference_internal);

    py::class_<Bliss::TokenInventory>(m, "TokenInventory")
            .def("insert", &Bliss::TokenInventory::insert)
            .def("link", &Bliss::TokenInventory::link)
            .def("add", &Bliss::TokenInventory::add)
            .def("insert", &Bliss::TokenInventory::insert)
            .def("__len__", &Bliss::TokenInventory::size)
            .def("__iter__", [](Bliss::TokenInventory const& self) { return py::make_iterator(self.begin(), self.end()); }, py::keep_alive<0, 1>())
            .def("__getitem__", (Bliss::Token * (Bliss::TokenInventory::*)(Bliss::Token::Id) const) & Bliss::TokenInventory::operator[], py::return_value_policy::reference_internal, py::is_operator())
            .def("__getitem__", (Bliss::Token * (Bliss::TokenInventory::*)(const std::string&) const) & Bliss::TokenInventory::operator[], py::return_value_policy::reference_internal, py::is_operator())
            .def("__getitem__", (Bliss::Token * (Bliss::TokenInventory::*)(Bliss::Symbol) const) & Bliss::TokenInventory::operator[], py::return_value_policy::reference_internal, py::is_operator());

    py::class_<Bliss::EvaluationToken, Bliss::Token>(m, "EvaluationToken");

    py::class_<Bliss::EvaluationTokenAlphabet, Bliss::TokenAlphabet, Core::Ref<Bliss::EvaluationTokenAlphabet>>(m, "EvaluationTokenAlphabet")
            .def("evaluation_token", &Bliss::EvaluationTokenAlphabet::evaluationToken, py::return_value_policy::reference_internal);

    py::class_<Bliss::LetterAlphabet, Bliss::TokenAlphabet, Core::Ref<Bliss::LetterAlphabet>>(m, "LetterAlphabet")
            .def("letter", &Bliss::LetterAlphabet::letter, py::return_value_policy::reference_internal);

    py::class_<Bliss::Lexicon, Core::Ref<Bliss::Lexicon>>(m, "Lexicon", py::multiple_inheritance())
            .def(py::init<const Core::Configuration&>())
            .def("new_lemma", (Bliss::Lemma * (Bliss::Lexicon::*)()) & Bliss::Lexicon::newLemma, py::return_value_policy::reference_internal)
            .def("new_lemma", (Bliss::Lemma * (Bliss::Lexicon::*)(const std::string&)) & Bliss::Lexicon::newLemma, py::return_value_policy::reference_internal)
            .def("set_orthographic_forms", &Bliss::Lexicon::setOrthographicForms)
            .def("set_default_lemma_name", &Bliss::Lexicon::setDefaultLemmaName)
            //.def("get_pronunciation", &Bliss::Lexicon::getPronunciation, py::return_value_policy::reference_internal)
            .def("add_pronunciation", &Bliss::Lexicon::addPronunciation, py::return_value_policy::reference_internal)
            .def("normalize_pronunciation_weights", &Bliss::Lexicon::normalizePronunciationWeights)
            .def("set_syntactic_token_sequence", py::overload_cast<Bliss::Lemma*, std::vector<std::string> const&>(&Bliss::Lexicon::setSyntacticTokenSequence))
            .def("set_syntactic_token_sequence", py::overload_cast<Bliss::Lemma*, std::vector<Bliss::Token::Id> const&>(&Bliss::Lexicon::setSyntacticTokenSequence))
            .def("set_default_syntactic_token", &Bliss::Lexicon::setDefaultSyntacticToken)
            .def("add_evaluation_token_sequence", py::overload_cast<Bliss::Lemma*, std::vector<std::string> const&>(&Bliss::Lexicon::addEvaluationTokenSequence))
            .def("add_evaluation_token_sequence", py::overload_cast<Bliss::Lemma*, std::vector<Bliss::Token::Id> const&>(&Bliss::Lexicon::addEvaluationTokenSequence))
            .def("set_default_evaluation_token", &Bliss::Lexicon::setDefaultEvaluationToken)
            .def("define_special_lemma", &Bliss::Lexicon::defineSpecialLemma)
            .def("load", &Bliss::Lexicon::load)
            .def("write_xml", [](Bliss::Lexicon& self, const std::string& name) {
                std::ofstream file(name + ".xml");
                if (file) {
                    Core::XmlWriter writer(file);
                    self.writeXml(writer);
                }
                file.close();
            })
            .def("log_statistics", &Bliss::Lexicon::logStatistics)
            .def_static("create", &Bliss::Lexicon::create)
            .def("num_lemmas", &Bliss::Lexicon::nLemmas)
            .def("lemmas", [](Bliss::Lexicon const& self) { auto begin_end = self.lemmas(); return py::make_iterator(begin_end.first, begin_end.second); }, py::keep_alive<0, 1>())
            .def("special_lemma", &Bliss::Lexicon::specialLemma, py::return_value_policy::reference_internal)
            .def("lemma_alphabet", &Bliss::Lexicon::lemmaAlphabet, py::return_value_policy::take_ownership)
            .def("set_phoneme_inventory", &Bliss::Lexicon::setPhonemeInventory)
            .def("phoneme_inventory", &Bliss::Lexicon::phonemeInventory, py::return_value_policy::take_ownership)
            .def("num_pronunciations", &Bliss::Lexicon::nPronunciations)
            .def("pronunciations", [](Bliss::Lexicon const& self) { auto begin_end = self.pronunciations(); return py::make_iterator(begin_end.first, begin_end.second); }, py::keep_alive<0, 1>())
            .def("num_lemma_pronunciations", &Bliss::Lexicon::nLemmaPronunciations)
            .def("lemma_pronunciations", [](Bliss::Lexicon const& self) { auto begin_end = self.lemmaPronunciations(); return py::make_iterator(begin_end.first, begin_end.second); }, py::keep_alive<0, 1>())
            .def("lemma_pronunciation_alphabet", &Bliss::Lexicon::lemmaPronunciationAlphabet, py::return_value_policy::take_ownership)
            .def("lemma_pronunciation", &Bliss::Lexicon::lemmaPronunciation, py::return_value_policy::reference_internal)
            .def("num_syntactic_tokens", &Bliss::Lexicon::nSyntacticTokens)
            .def("syntactic_token", &Bliss::Lexicon::syntacticToken, py::return_value_policy::reference_internal)
            .def("syntactic_token_inventory", &Bliss::Lexicon::syntacticTokenInventory, py::return_value_policy::reference_internal)
            .def("syntactic_token_alphabet", &Bliss::Lexicon::syntacticTokenAlphabet, py::return_value_policy::take_ownership)
            .def("num_evaluation_tokens", &Bliss::Lexicon::nEvaluationTokens)
            .def("evaluation_token_inventory", &Bliss::Lexicon::evaluationTokenInventory, py::return_value_policy::reference_internal)
            .def("evaluation_token_alphabet", &Bliss::Lexicon::evaluationTokenAlphabet, py::return_value_policy::take_ownership)
            .def("letter", &Bliss::Lexicon::letter, py::return_value_policy::reference_internal)
            .def("letter_inventory", &Bliss::Lexicon::letterInventory, py::return_value_policy::reference_internal)
            .def("letter_alphabet", &Bliss::Lexicon::letterAlphabet, py::return_value_policy::take_ownership);
}
