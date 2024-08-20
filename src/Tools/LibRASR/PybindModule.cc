#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/iostream.h>

#include <Am/Module.hh>
#include <Audio/Module.hh>
#include <Bliss/CorpusDescription.hh>
#include <Bliss/CorpusParser.hh>
#include <Bliss/Lexicon.hh>
#include <Core/Application.hh>
#include <Flf/Module.hh>
#include <Flow/Module.hh>
#include <Lm/Module.hh>
#include <Math/Module.hh>
#include <Mm/Module.hh>
#include <Modules.hh>
#ifdef MODULE_NN
#include <Nn/Module.hh>
#endif
#ifdef MODULE_ONNX
#include <Onnx/Module.hh>
#endif
#include <Signal/Module.hh>
#include <Speech/Module.hh>
#ifdef MODULE_TENSORFLOW
#include <Tensorflow/Module.hh>
#endif

#include <Python/AllophoneStateFsaBuilder.hh>
#include <Python/Configuration.hh>

#include "LibRASR.hh"

namespace py = pybind11;

class PublicCorpusVisitor : public Speech::CorpusVisitor {

public:
    using Speech::CorpusVisitor::enterCorpus;
    using Speech::CorpusVisitor::leaveCorpus;
    using Speech::CorpusVisitor::enterRecording;
    using Speech::CorpusVisitor::leaveRecording;
    using Speech::CorpusVisitor::visitSegment;
    using Speech::CorpusVisitor::visitSpeechSegment;
    using Speech::CorpusVisitor::CorpusVisitor;
};

class PyCorpusVisitor : public Speech::CorpusVisitor {

public:
	using Speech::CorpusVisitor::CorpusVisitor;
	void enterCorpus(Bliss::Corpus* corpus) override { PYBIND11_OVERRIDE(void, Speech::CorpusVisitor, enterCorpus, corpus); }
	void leaveCorpus(Bliss::Corpus* corpus) override { PYBIND11_OVERRIDE(void, Speech::CorpusVisitor, leaveCorpus, corpus); }
	void enterRecording(Bliss::Recording* recording) override { PYBIND11_OVERRIDE(void, Speech::CorpusVisitor, enterRecording, recording); }
	void leaveRecording(Bliss::Recording* recording) override { PYBIND11_OVERRIDE(void, Speech::CorpusVisitor, leaveRecording, recording); }
	void visitSegment(Bliss::Segment* segment) override { PYBIND11_OVERRIDE(void, Speech::CorpusVisitor, visitSegment, segment); }
	void visitSpeechSegment(Bliss::SpeechSegment* segment) override { PYBIND11_OVERRIDE(void, Speech::CorpusVisitor, visitSpeechSegment, segment); }

};


class _DummyApplication : Core::Application {
public:
    _DummyApplication();
    virtual ~_DummyApplication();

    int main(std::vector<std::string> const& arguments);
};

_DummyApplication::_DummyApplication() : Core::Application() {
    setTitle("lib-rasr");
    config.set("*.encoding", "utf-8");
    openLogging();

    INIT_MODULE(Flf);
    INIT_MODULE(Am);
    INIT_MODULE(Audio);
    INIT_MODULE(Flow);
    INIT_MODULE(Math);
    INIT_MODULE(Mm);
    INIT_MODULE(Lm);
    INIT_MODULE(Signal);
    INIT_MODULE(Speech);
#ifdef MODULE_NN
    INIT_MODULE(Nn);
#endif
#ifdef MODULE_ONNX
    INIT_MODULE(Onnx);
#endif
#ifdef MODULE_STREAMING
    INIT_MODULE(Streaming);
#endif
#ifdef MODULE_TENSORFLOW
    INIT_MODULE(Tensorflow);
#endif
}

_DummyApplication::~_DummyApplication() {
    closeLogging();
}

int _DummyApplication::main(std::vector<std::string> const& arguments) {
    return EXIT_SUCCESS;
}

PYBIND11_DECLARE_HOLDER_TYPE(T, Core::Ref<T>, true);
PYBIND11_DECLARE_HOLDER_TYPE(T, Flow::DataPtr<T>, true); // confirm

PYBIND11_MODULE(librasr, m) {
    static DummyApplication app;

    m.doc() = "RASR python module";

    py::class_<Core::Configuration> baseConfigClass(m, "_BaseConfig");

    py::class_<PyConfiguration> pyRasrConfig(m, "Configuration", baseConfigClass);
    pyRasrConfig.def(py::init<>());
    pyRasrConfig.def("set_from_file",
                     (bool (Core::Configuration::*)(const std::string&)) &Core::Configuration::setFromFile);

    py::class_<AllophoneStateFsaBuilder> pyFsaBuilder(m, "AllophoneStateFsaBuilder");
    pyFsaBuilder.def(py::init<const Core::Configuration&>());
    pyFsaBuilder.def("build_by_orthography",
                     &AllophoneStateFsaBuilder::buildByOrthography);
    pyFsaBuilder.def("build_by_segment_name",
                     &AllophoneStateFsaBuilder::buildBySegmentName);

    py::class_<Bliss::Symbol> symbol(m, "Symbol");
    symbol
    .def(py::init<>())
    .def(py::init<const Bliss::Symbol&>())
    .def("length", &Bliss::Symbol::length)
    .def("__eq__", &Bliss::Symbol::operator==)
    .def("__ne__", &Bliss::Symbol::operator!=)
    .def("__bool__", [](const Bliss::Symbol &self) { return !self.operator!(); })
    .def("to_bool", &Bliss::Symbol::operator bool)
    .def("to_string", &Bliss::Symbol::operator Bliss::Symbol::String)
    .def("str", &Bliss::Symbol::str)
    .def_static("cast", &Bliss::Symbol::cast);

    py::class_<Bliss::Symbol::Hash>(symbol, "Hash")
    .def("__call__", &Bliss::Symbol::Hash::operator());

    py::class_<Bliss::Symbol::Equality>(symbol, "Equality")
    .def("__call__", &Bliss::Symbol::Equality::operator());

    py::class_<Bliss::OrthographicFormList>(m, "OrthographicFormList")
    .def(py::init<const Bliss::Symbol*, const Bliss::Symbol*>())
    .def(py::init<>())
    .def(py::init<const Bliss::SymbolSequence<Bliss::OrthographicForm>&>())
    .def("valid", &Bliss::SymbolSequence<Bliss::OrthographicForm>::valid)
    .def("size", &Bliss::SymbolSequence<Bliss::OrthographicForm>::size)
    .def("length", &Bliss::SymbolSequence<Bliss::OrthographicForm>::length)
    .def("is_epsilon", &Bliss::SymbolSequence<Bliss::OrthographicForm>::isEpsilon)
    .def("front", &Bliss::SymbolSequence<Bliss::OrthographicForm>::front, py::return_value_policy::reference_internal)
    .def("begin", &Bliss::SymbolSequence<Bliss::OrthographicForm>::begin, py::return_value_policy::reference_internal)
    .def("end", &Bliss::SymbolSequence<Bliss::OrthographicForm>::end, py::return_value_policy::reference_internal)
    .def("__getitem__", &Bliss::OrthographicFormList::operator[]);

    py::class_<Bliss::SyntacticTokenSequence>(m, "SyntacticTokenSequence")
    .def(py::init<>())
    .def(py::init<const Bliss::SymbolSequence<const Bliss::SyntacticToken*>&>())
    .def("valid", &Bliss::SymbolSequence<const Bliss::SyntacticToken*>::valid)
    .def("size", &Bliss::SymbolSequence<const Bliss::SyntacticToken*>::size)
    .def("length", &Bliss::SymbolSequence<const Bliss::SyntacticToken*>::length)
    .def("is_epsilon", &Bliss::SymbolSequence<const Bliss::SyntacticToken*>::isEpsilon)
    .def("front", &Bliss::SymbolSequence<const Bliss::SyntacticToken*>::front, py::return_value_policy::reference_internal)
    .def("__getitem__", &Bliss::SymbolSequence<const Bliss::SyntacticToken*>::operator[]);

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
    .def(py::init<>())
    .def("to_lemma_pronunciation", &Bliss::Lemma::PronunciationIterator::operator const Bliss::LemmaPronunciation*)
    .def("increment", &Bliss::Lemma::PronunciationIterator::operator++, py::return_value_policy::reference_internal);

    py::class_<Bliss::Pronunciation, std::unique_ptr<Bliss::Pronunciation, py::nodelete>> pronunciation(m, "Pronunciation");
    pronunciation
    .def("num_lemmas", &Bliss::Pronunciation::nLemmas)
    .def("length", &Bliss::Pronunciation::length)
    .def("format", &Bliss::Pronunciation::format)
    .def("phonemes", &Bliss::Pronunciation::phonemes, py::return_value_policy::reference_internal)
    .def("__getitem__", &Bliss::Pronunciation::operator[]);

    py::class_<Bliss::Pronunciation::LemmaIterator>(pronunciation, "LemmaIterator")
    .def(py::init<>())
    .def("__eq__", &Bliss::Pronunciation::LemmaIterator::operator==)
    .def("__ne__", &Bliss::Pronunciation::LemmaIterator::operator!=)
    .def("to_lemma_pronunciation", &Bliss::Pronunciation::LemmaIterator::operator const Bliss::LemmaPronunciation*)
    .def("increment", &Bliss::Pronunciation::LemmaIterator::operator++, py::return_value_policy::reference_internal);

    py::class_<Bliss::Pronunciation::Hash>(pronunciation, "Hash")
    .def("__call__", (u32 (Bliss::Pronunciation::Hash::*)(const Bliss::Phoneme::Id*) const) &Bliss::Pronunciation::Hash::operator())
    .def("__call__", (u32 (Bliss::Pronunciation::Hash::*)(const Bliss::Pronunciation*) const) &Bliss::Pronunciation::Hash::operator());

    py::class_<Bliss::Pronunciation::Equality>(pronunciation, "Equality")
    .def("__call__", (bool (Bliss::Pronunciation::Equality::*)(const Bliss::Phoneme::Id*, const Bliss::Phoneme::Id*) const) &Bliss::Pronunciation::Equality::operator())
    .def("__call__", (bool (Bliss::Pronunciation::Equality::*)(const Bliss::Pronunciation*, const Bliss::Pronunciation*) const) &Bliss::Pronunciation::Equality::operator());

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
    .def("phoneme", (const Bliss::Phoneme* (Bliss::PhonemeInventory::*)(const std::string&) const) &Bliss::PhonemeInventory::phoneme, py::return_value_policy::reference_internal)
    .def("phoneme", (const Bliss::Phoneme* (Bliss::PhonemeInventory::*)(Bliss::Phoneme::Id) const) &Bliss::PhonemeInventory::phoneme, py::return_value_policy::reference_internal)
    .def("is_valid_phoneme_id", &Bliss::PhonemeInventory::isValidPhonemeId)
    .def("new_phoneme", &Bliss::PhonemeInventory::newPhoneme, py::return_value_policy::reference_internal)
    .def("assign_symbol", &Bliss::PhonemeInventory::assignSymbol)
    .def("phoneme_alphabet", &Bliss::PhonemeInventory::phonemeAlphabet, py::return_value_policy::take_ownership)
    .def("parse_selection", &Bliss::PhonemeInventory::parseSelection)
    .def("write_xml", [](Bliss::PhonemeInventory &self, const std::string& name) {
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
    .def("begin", &Fsa::Alphabet::begin)
    .def("end", &Fsa::Alphabet::end)
    .def("get_memory_used", &Fsa::Alphabet::getMemoryUsed)
    .def("write_xml", [](Fsa::Alphabet &self, const std::string& name) {
    std::ofstream file(name + ".xml");
    if (file) {
        Core::XmlWriter writer(file);
        self.writeXml(writer);
    }
    file.close();
    });

    py::class_<Fsa::Alphabet::const_iterator>(alphabet, "const_iterator")
    .def(py::init<Core::Ref<const Fsa::Alphabet>, Fsa::LabelId>())
    .def("__eq__", &Fsa::Alphabet::const_iterator::operator==)
    .def("__ne__", &Fsa::Alphabet::const_iterator::operator!=)
    .def("get_symbol", &Fsa::Alphabet::const_iterator::operator*)
    .def("to_label_id", (Fsa::LabelId (Fsa::Alphabet::const_iterator::*)()) &Fsa::Alphabet::const_iterator::operator Fsa::LabelId)
    .def("to_label_id", (Fsa::LabelId (Fsa::Alphabet::const_iterator::*)() const) &Fsa::Alphabet::const_iterator::operator Fsa::LabelId)
    .def("increment", &Fsa::Alphabet::const_iterator::operator++, py::return_value_policy::reference_internal);

    py::class_<Bliss::TokenAlphabet, Fsa::Alphabet, Core::Ref<Bliss::TokenAlphabet>>(m, "TokenAlphabet")
    .def("symbol", &Bliss::TokenAlphabet::symbol)
    .def("end", &Bliss::TokenAlphabet::end)
    .def("index", (Fsa::LabelId (Bliss::TokenAlphabet::*)(const std::string&) const) &Bliss::TokenAlphabet::index)
    .def("index", (Fsa::LabelId (Bliss::TokenAlphabet::*)(const Bliss::Token*) const) &Bliss::TokenAlphabet::index)
    .def("token", &Bliss::TokenAlphabet::token, py::return_value_policy::reference_internal)
    .def("num_disambiguators", &Bliss::TokenAlphabet::nDisambiguators)
    .def("disambiguator", &Bliss::TokenAlphabet::disambiguator)
    .def("is_disambiguator", &Bliss::TokenAlphabet::isDisambiguator)
    .def("write_xml", [](Bliss::TokenAlphabet &self, const std::string& name) {
    std::ofstream file(name + ".xml");
    if (file) {
        Core::XmlWriter writer(file);
        self.writeXml(writer);
    }
    file.close();
    });

    py::class_<Bliss::PhonemeAlphabet, Bliss::TokenAlphabet, Core::Ref<Bliss::PhonemeAlphabet>>(m, "PhonemeAlphabet")
    .def("phoneme_inventory", &Bliss::PhonemeAlphabet::phonemeInventory, py::return_value_policy::take_ownership)
    .def("phoneme", &Bliss::PhonemeAlphabet::phoneme, py::return_value_policy::reference_internal)
    .def("symbol", &Bliss::PhonemeAlphabet::symbol)
    .def("begin", &Bliss::PhonemeAlphabet::begin)
    .def("end", &Bliss::PhonemeAlphabet::end)
    .def("index", (Fsa::LabelId (Bliss::PhonemeAlphabet::*)(const std::string&) const) &Bliss::PhonemeAlphabet::index)
    .def("index", (Fsa::LabelId (Bliss::PhonemeAlphabet::*)(const Bliss::Token*) const) &Bliss::PhonemeAlphabet::index)
    .def("write_xml", [](Bliss::PhonemeAlphabet &self, const std::string& name) {
    std::ofstream file(name + ".xml");
    if (file) {
        Core::XmlWriter writer(file);
        self.writeXml(writer);
    }
    file.close();
    });

    py::class_<Bliss::LemmaAlphabet, Bliss::TokenAlphabet, Core::Ref<Bliss::LemmaAlphabet>>(m, "LemmaAlphabet")
    .def("lemma", &Bliss::LemmaAlphabet::lemma, py::return_value_policy::reference_internal);

    py::class_<Bliss::LemmaPronunciationAlphabet, Fsa::Alphabet, Core::Ref<Bliss::LemmaPronunciationAlphabet>>(m, "LemmaPronunciationAlphabet")
    .def("index", (Fsa::LabelId (Bliss::LemmaPronunciationAlphabet::*)(const std::string&) const) &Bliss::LemmaPronunciationAlphabet::index)
    .def("index", (Fsa::LabelId (Bliss::LemmaPronunciationAlphabet::*)(const Bliss::LemmaPronunciation*) const) &Bliss::LemmaPronunciationAlphabet::index)
    .def("lemma_pronunciation", &Bliss::LemmaPronunciationAlphabet::lemmaPronunciation, py::return_value_policy::reference_internal)
    .def("symbol", &Bliss::LemmaPronunciationAlphabet::symbol)
    .def("end", &Bliss::LemmaPronunciationAlphabet::end)
    .def("num_disambiguators", &Bliss::LemmaPronunciationAlphabet::nDisambiguators)
    .def("disambiguator", &Bliss::LemmaPronunciationAlphabet::disambiguator)
    .def("is_disambiguator", &Bliss::LemmaPronunciationAlphabet::isDisambiguator)
    .def("write_xml", [](Bliss::LemmaPronunciationAlphabet &self, const std::string& name) {
    std::ofstream file(name + ".xml");
    if (file) {
        Core::XmlWriter writer(file);
        self.writeXml(writer);
    }
    file.close();
    });

    py::class_<Bliss::SyntacticTokenAlphabet, Bliss::TokenAlphabet, Core::Ref<Bliss::SyntacticTokenAlphabet>>(m, "SyntacticTokenAlphabet")
    .def("syntactic_token", &Bliss::SyntacticTokenAlphabet::syntacticToken, py::return_value_policy::reference_internal);

    py::class_<Bliss::TokenInventory>(m, "TokenInventory")
    .def("insert", &Bliss::TokenInventory::insert)
    .def("link", &Bliss::TokenInventory::link)
    .def("add", &Bliss::TokenInventory::add)
    .def("size", &Bliss::TokenInventory::size)
    .def("insert", &Bliss::TokenInventory::insert)
    .def("__getitem__", (Bliss::Token* (Bliss::TokenInventory::*)(Bliss::Token::Id) const) &Bliss::TokenInventory::operator[], py::return_value_policy::reference_internal)
    .def("__getitem__", (Bliss::Token* (Bliss::TokenInventory::*)(const std::string&) const) &Bliss::TokenInventory::operator[], py::return_value_policy::reference_internal)
    .def("__getitem__", (Bliss::Token* (Bliss::TokenInventory::*)(Bliss::Symbol) const) &Bliss::TokenInventory::operator[], py::return_value_policy::reference_internal);

    py::class_<Bliss::EvaluationToken, Bliss::Token>(m, "EvaluationToken");

    py::class_<Bliss::EvaluationTokenAlphabet, Core::Ref<Bliss::EvaluationTokenAlphabet>>(m, "EvaluationTokenAlphabet")
    .def("evaluation_token", &Bliss::EvaluationTokenAlphabet::evaluationToken, py::return_value_policy::reference_internal);

    py::class_<Bliss::LetterAlphabet, Bliss::TokenAlphabet, Core::Ref<Bliss::LetterAlphabet>>(m, "LetterAlphabet")
    .def("letter", &Bliss::LetterAlphabet::letter, py::return_value_policy::reference_internal);

    py::class_<Bliss::Lexicon>(m, "Lexicon")
    .def(py::init<const Core::Configuration&>())
    .def("new_lemma", (Bliss::Lemma* (Bliss::Lexicon::*)()) &Bliss::Lexicon::newLemma, py::return_value_policy::reference_internal)
    .def("new_lemma", (Bliss::Lemma* (Bliss::Lexicon::*)(const std::string&)) &Bliss::Lexicon::newLemma, py::return_value_policy::reference_internal)
    .def("set_orthographic_forms", &Bliss::Lexicon::setOrthographicForms)
    .def("set_default_lemma_name", &Bliss::Lexicon::setDefaultLemmaName)
    .def("get_pronunciation", &Bliss::Lexicon::getPronunciation, py::return_value_policy::reference_internal)
    .def("add_pronunciation", &Bliss::Lexicon::addPronunciation, py::return_value_policy::reference_internal)
    .def("normalize_pronunciation_weights", &Bliss::Lexicon::normalizePronunciationWeights)
    .def("set_syntactic_token_sequence", &Bliss::Lexicon::setSyntacticTokenSequence)
    .def("set_default_syntactic_token", &Bliss::Lexicon::setDefaultSyntacticToken)
    .def("add_evaluation_token_sequence", &Bliss::Lexicon::addEvaluationTokenSequence)
    .def("set_default_evaluation_token", &Bliss::Lexicon::setDefaultEvaluationToken)
    .def("define_special_lemma", &Bliss::Lexicon::defineSpecialLemma)
    .def("load", &Bliss::Lexicon::load)
    .def("write_xml", [](Bliss::Lexicon &self, const std::string& name) {
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
    .def("special_lemma", &Bliss::Lexicon::specialLemma, py::return_value_policy::reference_internal)
    .def("lemma_alphabet", &Bliss::Lexicon::lemmaAlphabet, py::return_value_policy::take_ownership)
    .def("set_phoneme_inventory", &Bliss::Lexicon::setPhonemeInventory)
    .def("phoneme_inventory", &Bliss::Lexicon::phonemeInventory, py::return_value_policy::take_ownership)
    .def("num_pronunciations", &Bliss::Lexicon::nPronunciations)
    .def("pronunciations", &Bliss::Lexicon::pronunciations)
    .def("num_lemma_pronunciations", &Bliss::Lexicon::nLemmaPronunciations)
    .def("lemma_pronunciations", &Bliss::Lexicon::lemmaPronunciations)
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

    /////////////////////////////////////////////////////////

    py::class_<Bliss::NamedCorpusEntity>(m, "NamedCorpusEntity")
    .def("parent", &Bliss::NamedCorpusEntity::parent, py::return_value_policy::reference_internal)
    .def("set_parent", &Bliss::NamedCorpusEntity::setParent)
    .def("name", &Bliss::NamedCorpusEntity::name, py::return_value_policy::reference_internal)
    .def("full_name", &Bliss::NamedCorpusEntity::fullName)
    .def("set_name", &Bliss::NamedCorpusEntity::setName)
    .def("set_remove_prefix", &Bliss::NamedCorpusEntity::setRemovePrefix)
    .def("is_anonymous", &Bliss::NamedCorpusEntity::isAnonymous);

    py::class_<Bliss::Speaker, Bliss::NamedCorpusEntity> speaker(m, "Speaker");
    speaker
    .def(py::init<Bliss::ParentEntity*>())
    .def("gender", &Bliss::Speaker::gender)
    .def("set_parent", &Bliss::Speaker::setParent);

    // static const char* genderId[nGenders]

    py::enum_<Bliss::Speaker::Gender>(speaker, "Gender")
    .value("unknown", Bliss::Speaker::Gender::unknown)
    .value("male", Bliss::Speaker::Gender::male)
    .value("female", Bliss::Speaker::Gender::female)
    .value("num_genders", Bliss::Speaker::Gender::nGenders)
    .export_values();

    py::class_<Bliss::AcousticCondition, Bliss::NamedCorpusEntity>(m, "AcousticCondition")
    .def(py::init<Bliss::ParentEntity*>())
    .def("set_parent", &Bliss::AcousticCondition::setParent);

    py::class_<Bliss::ParentEntity, Bliss::NamedCorpusEntity>(m, "ParentEntity")
    .def("is_name_reserved", &Bliss::ParentEntity::isNameReserved)
    .def("reserve_name", &Bliss::ParentEntity::reserveName);

    py::class_<Bliss::Directory<Bliss::Speaker>>(m, "DirectorySpeaker")
    .def(py::init<>())
    .def(py::init<const Bliss::Directory<Bliss::Speaker>&>())
    .def("has_key", &Bliss::Directory<Bliss::Speaker>::hasKey)
    .def("add", &Bliss::Directory<Bliss::Speaker>::add)
    .def("lookup", &Bliss::Directory<Bliss::Speaker>::lookup, py::return_value_policy::reference_internal);

    py::class_<Bliss::Directory<Bliss::AcousticCondition>>(m, "DirectoryAcousticCondition")
    .def(py::init<>())
    .def(py::init<const Bliss::Directory<Bliss::AcousticCondition>&>())
    .def("has_key", &Bliss::Directory<Bliss::AcousticCondition>::hasKey)
    .def("add", &Bliss::Directory<Bliss::AcousticCondition>::add)
    .def("lookup", &Bliss::Directory<Bliss::AcousticCondition>::lookup, py::return_value_policy::reference_internal);

    py::class_<Bliss::CorpusSection, Bliss::ParentEntity>(m, "CorpusSection")
    .def(py::init<Bliss::CorpusSection*>())
    .def("parent", &Bliss::CorpusSection::parent, py::return_value_policy::reference_internal)
    .def("level", &Bliss::CorpusSection::level)
    .def("speaker", &Bliss::CorpusSection::speaker)
    .def("default_speaker", &Bliss::CorpusSection::defaultSpeaker)
    .def("condition", &Bliss::CorpusSection::condition)
    .def("default_condition", &Bliss::CorpusSection::defaultCondition);

    py::class_<Bliss::Corpus, Bliss::CorpusSection>(m, "Corpus")
    .def(py::init<Bliss::Corpus*>());

    py::class_<Bliss::Recording, Bliss::CorpusSection>(m, "Recording")
    .def(py::init<Bliss::Corpus*>())
    .def("audio", &Bliss::Recording::audio, py::return_value_policy::reference_internal)
    .def("set_audio", &Bliss::Recording::setAudio)
    .def("video", &Bliss::Recording::video, py::return_value_policy::reference_internal)
    .def("set_video", &Bliss::Recording::setVideo)
    .def("duration", &Bliss::Recording::duration)
    .def("set_duration", &Bliss::Recording::setDuration);

    py::class_<Bliss::Segment, Bliss::ParentEntity> segment(m, "Segment");
    segment
    .def(py::init<Bliss::Segment::Type, Bliss::Recording*>())
    .def("recording", &Bliss::Segment::recording, py::return_value_policy::reference_internal)
    .def("set_recording", &Bliss::Segment::setRecording)
    .def("parent", &Bliss::Segment::parent, py::return_value_policy::reference_internal)
    .def("type", &Bliss::Segment::type)
    .def("set_type", &Bliss::Segment::setType)
    .def("start", &Bliss::Segment::start)
    .def("set_start", &Bliss::Segment::setStart)
    .def("end", &Bliss::Segment::end)
    .def("set_end", &Bliss::Segment::setEnd)
    .def("track", &Bliss::Segment::track)
    .def("set_track", &Bliss::Segment::setTrack)
    .def("condition", &Bliss::Segment::condition, py::return_value_policy::reference_internal)
    .def("set_condition", &Bliss::Segment::setCondition)
    .def("accept", &Bliss::Segment::accept);

    // static const char* typeId[nTypes]

    py::enum_<Bliss::Segment::Type>(segment, "Type")
    .value("type_speech", Bliss::Segment::Type::typeSpeech)
    .value("type_other", Bliss::Segment::Type::typeOther)
    .value("num_types", Bliss::Segment::Type::nTypes)
    .export_values();

    py::class_<Bliss::SpeechSegment, Bliss::Segment>(m, "SpeechSegment")
    .def(py::init<Bliss::Recording*>())
    .def("orth", &Bliss::SpeechSegment::orth, py::return_value_policy::reference_internal)
    .def("set_orth", &Bliss::SpeechSegment::setOrth)
    .def("left_context_orth", &Bliss::SpeechSegment::leftContextOrth, py::return_value_policy::reference_internal)
    .def("set_left_context_orth", &Bliss::SpeechSegment::setLeftContextOrth)
    .def("right_context_orth", &Bliss::SpeechSegment::rightContextOrth, py::return_value_policy::reference_internal)
    .def("set_right_context_orth", &Bliss::SpeechSegment::setRightContextOrth)
    .def("speaker", &Bliss::SpeechSegment::speaker, py::return_value_policy::reference_internal)
    .def("set_speaker", &Bliss::SpeechSegment::setSpeaker)
    .def("accept", &Bliss::SpeechSegment::accept);

    py::class_<Bliss::SegmentVisitor>(m, "SegmentVisitor")
    .def("visit_segment", &Bliss::SegmentVisitor::visitSegment)
    .def("visit_speech_segment", &Bliss::SegmentVisitor::visitSpeechSegment);

    py::class_<Bliss::CorpusVisitor, Bliss::SegmentVisitor>(m, "BlissCorpusVisitor")
    .def("enter_recording", &Bliss::CorpusVisitor::enterRecording)
    .def("leave_recording", &Bliss::CorpusVisitor::leaveRecording)
    .def("enter_corpus", &Bliss::CorpusVisitor::enterCorpus)
    .def("leave_corpus", &Bliss::CorpusVisitor::leaveCorpus);

    py::class_<Bliss::CorpusDescription>(m, "CorpusDescription")
    .def(py::init<const Core::Configuration&>())
    .def("file", &Bliss::CorpusDescription::file, py::return_value_policy::reference_internal)
    .def("accept", &Bliss::CorpusDescription::accept)
    .def("total_segment_count", &Bliss::CorpusDescription::totalSegmentCount);

     /* static const Core::ParameterString       paramFilename
      * static const Core::ParameterBool         paramAllowEmptyWhitelist
      * static const Core::ParameterString       paramEncoding
      * static const Core::ParameterInt          paramPartition
      * static const Core::ParameterInt          paramPartitionSelection
      * static const Core::ParameterInt          paramSkipFirstSegments
      * static const Core::ParameterStringVector paramSegmentsToSkip
      * static const Core::ParameterBool         paramRecordingBasedPartition
      * static const Core::ParameterString       paramSegmentOrder
      * static const Core::ParameterBool         paramSegmentOrderLookupName
      * static const Core::ParameterBool         paramSegmentOrderShuffle
      * static const Core::ParameterInt          paramSegmentOrderShuffleSeed
      * static const Core::ParameterBool         paramSegmentOrderSortByTimeLength
      * static const Core::ParameterInt          paramSegmentOrderSortByTimeLengthChunkSize
      * static const Core::ParameterBool         paramTheanoSegmentOrder
      * static const Core::ParameterBool         paramPythonSegmentOrder
      * static const Core::ParameterString       paramPythonSegmentOrderModPath
      * static const Core::ParameterString       paramPythonSegmentOrderModName
      * static const Core::ParameterString       paramPythonSegmentOrderConfig
      */

     py::class_<Bliss::ProgressReportingVisitorAdaptor, Bliss::CorpusVisitor>(m, "ProgressReportingVisitorAdaptor")
    .def("set_visitor", &Bliss::ProgressReportingVisitorAdaptor::setVisitor)
    .def("enter_corpus", &Bliss::ProgressReportingVisitorAdaptor::enterCorpus)
    .def("leave_corpus", &Bliss::ProgressReportingVisitorAdaptor::leaveCorpus)
    .def("enter_recording", &Bliss::ProgressReportingVisitorAdaptor::enterRecording)
    .def("leave_recording", &Bliss::ProgressReportingVisitorAdaptor::leaveRecording)
    .def("visit_segment", &Bliss::ProgressReportingVisitorAdaptor::visitSegment)
    .def("visit_speech_segment", &Bliss::ProgressReportingVisitorAdaptor::visitSpeechSegment);

     // ProgressReportingVisitorAdaptor(Core::XmlChannel& ch, bool reportOrth = false)

    py::class_<Core::StringExpression>(m, "StringExpression")
    .def(py::init<>())
    .def("is_constant", &Core::StringExpression::isConstant)
    .def("has_variable", &Core::StringExpression::hasVariable)
    .def("value", &Core::StringExpression::value)
    .def("is_constant", &Core::StringExpression::isConstant)
    .def("set_variable", &Core::StringExpression::setVariable)
    .def("set_variables", &Core::StringExpression::setVariables)
    .def("clear", (bool (Core::StringExpression::*)(const std::string&)) &Core::StringExpression::clear)
    .def("clear", (void (Core::StringExpression::*)()) &Core::StringExpression::clear);

    py::class_<Bliss::CorpusKey, Core::StringExpression/*, Core::Ref<Bliss::CorpusKey>*/>(m, "CorpusKey")
    .def(py::init<const Core::Configuration&>())
    .def_readonly_static("open_tag", &Bliss::CorpusKey::openTag)
    .def_readonly_static("close_tag", &Bliss::CorpusKey::closeTag)
    .def("resolve", &Bliss::CorpusKey::resolve);

     // static const Core::ParameterString paramTemplate

     py::class_<Bliss::CorpusDescriptionParser>(m, "CorpusDescriptionParser")
    .def(py::init<const Core::Configuration&>())
    .def("accept", &Bliss::CorpusDescriptionParser::accept);

     /*
      * class CorpusDescriptionParser : public Core::XmlSchemaParser
      * static const Core::ParameterString paramAudioDir
      * static const Core::ParameterString paramVideoDir
      * static const Core::ParameterString paramRemoveCorpusNamePrefix
      * static const Core::ParameterBool   paramCaptializeTranscriptions
      * static const Core::ParameterBool   paramGemenizeTranscriptions
      * static const Core::ParameterBool   paramProgress
      */

     ///////////////////////////////////////////////////////////////////////////////////

    /*py::class_<Speech::CorpusVisitor, Bliss::CorpusVisitor>(m, "SpeechCorpusVisitor") // Core::Component
    .def(py::init<const Core::Configuration&>())
    .def("sign_on", (void (Speech::CorpusVisitor::*)(Speech::CorpusProcessor*)) &Speech::CorpusVisitor::signOn)
    .def("sign_on", (void (Speech::CorpusVisitor::*)(Core::Ref<Speech::DataSource>)) &Speech::CorpusVisitor::signOn)
    .def("sign_on", (void (Speech::CorpusVisitor::*)(Core::Ref<Bliss::CorpusKey>)) &Speech::CorpusVisitor::signOn)
    .def("clear_registrations", &Speech::CorpusVisitor::clearRegistrations)
    .def("is_signed_on", (bool (Speech::CorpusVisitor::*)(Core::Ref<Speech::DataSource>) const) &Speech::CorpusVisitor::isSignedOn);

    py::class_<SpeechCorpusVisitor, Speech::CorpusVisitor>(m, "CorpusVisitor")
    .def(py::init<const Core::Configuration&>())
    .def("enter_corpus", &SpeechCorpusVisitor::enterCorpus)
    .def("leave_corpus", &SpeechCorpusVisitor::leaveCorpus)
    .def("enter_recording", &SpeechCorpusVisitor::enterRecording)
    .def("leave_recording", &SpeechCorpusVisitor::leaveRecording)
    .def("visitSegment", &SpeechCorpusVisitor::visitSegment)
    .def("visitSpeechSegment", &SpeechCorpusVisitor::visitSpeechSegment)
    .def("sign_on", (void (SpeechCorpusVisitor::*)(Speech::CorpusProcessor*)) &SpeechCorpusVisitor::signOn)
    .def("sign_on", (void (SpeechCorpusVisitor::*)(Core::Ref<Speech::DataSource>)) &SpeechCorpusVisitor::signOn)
    .def("sign_on", (void (SpeechCorpusVisitor::*)(Core::Ref<Bliss::CorpusKey>)) &SpeechCorpusVisitor::signOn)
    .def("clear_registrations", &SpeechCorpusVisitor::clearRegistrations)
    .def("is_signed_on", (bool (SpeechCorpusVisitor::*)(Core::Ref<Speech::DataSource>) const) &SpeechCorpusVisitor::isSignedOn);
    */

    py::class_<Speech::CorpusVisitor, Bliss::CorpusVisitor, PyCorpusVisitor>(m, "SpeechCorpusVisitor")
    .def(py::init<const Core::Configuration&>())
    .def("enter_corpus", &PublicCorpusVisitor::enterCorpus)
    .def("leave_corpus", &PublicCorpusVisitor::leaveCorpus)
    .def("enter_recording", &PublicCorpusVisitor::enterRecording)
    .def("leave_recording", &PublicCorpusVisitor::leaveRecording)
    .def("visitSegment", &PublicCorpusVisitor::visitSegment)
    .def("visitSpeechSegment", &PublicCorpusVisitor::visitSpeechSegment)
    //.def("sign_on", (void (PublicCorpusVisitor::*)(Speech::CorpusProcessor*)) &PublicCorpusVisitor::signOn);
        .def("sign_on", [](Speech::CorpusVisitor &self, Speech::CorpusProcessor* processor) {self.signOn(processor);});

   // .def("sign_on", (void (PublicCorpusVisitor::*)(Core::Ref<Speech::DataSource>)) &PublicCorpusVisitor::signOn)
   // .def("sign_on", (void (PublicCorpusVisitor::*)(Core::Ref<Bliss::CorpusKey>)) &PublicCorpusVisitor::signOn)
   // .def("clear_registrations", &PublicCorpusVisitor::clearRegistrations)
   // .def("is_signed_on", (bool (PublicCorpusVisitor::*)(Core::Ref<Speech::DataSource>) const) &PublicCorpusVisitor::isSignedOn);

    py::class_<Speech::CorpusProcessor>(m, "CorpusProcessor")
    .def(py::init<const Core::Configuration&>())
    .def("sign_on", &Speech::CorpusProcessor::signOn)
    .def("enter_corpus", &Speech::CorpusProcessor::enterCorpus)
    .def("leave_corpus", &Speech::CorpusProcessor::leaveCorpus)
    .def("enter_recording", &Speech::CorpusProcessor::enterRecording)
    .def("leave_recording", &Speech::CorpusProcessor::leaveRecording)
    .def("enter_segment", &Speech::CorpusProcessor::enterSegment)
    .def("process_segment", &Speech::CorpusProcessor::processSegment)
    .def("leave_segment", &Speech::CorpusProcessor::leaveSegment)
    .def("enter_speech_segment", &Speech::CorpusProcessor::enterSpeechSegment)
    .def("process_speech_segment", &Speech::CorpusProcessor::processSpeechSegment)
    .def("leave_speech_segment", &Speech::CorpusProcessor::leaveSpeechSegment);

 //    py::class_<Mm::Feature::Vector, Mm::FeatureVector, Core::Ref<Mm::Feature::Vector>>(m, "MmFeatureVector")
//    .def(py::init<>())
//    .def(py::init<size_t>())
//    .def(py::init<const Mm::FeatureVector&>());

    py::class_<Mm::Feature, Core::Ref<Mm::Feature>>(m, "MmFeature")
    .def(py::init<size_t>())
    .def("add", (size_t (Mm::Feature::*)(const Core::Ref<const Mm::Feature::Vector>&)) &Mm::Feature::add)
    .def("add", (void (Mm::Feature::*)(size_t, const Core::Ref<const Mm::Feature::Vector>&)) &Mm::Feature::add)
    .def("set", (void (Mm::Feature::*)(size_t, const Core::Ref<const Mm::Feature::Vector>&)) &Mm::Feature::set)
    .def("set", (void (Mm::Feature::*)(const std::vector<size_t>&, const Core::Ref<const Mm::Feature::Vector>&)) &Mm::Feature::set)
    .def("clear", &Mm::Feature::clear)
    .def("main_stream", &Mm::Feature::mainStream, py::return_value_policy::take_ownership)
    .def("set_number_of_streams", &Mm::Feature::setNumberOfStreams)
    .def("num_streams", &Mm::Feature::nStreams);

    //    typedef std::vector<Core::Ref<const Vector>>::const_iterator Iterator;

    //    static Core::Ref<const Vector> convert(const FeatureVector& f)
    //    explicit Feature(const Core::Ref<const Vector>& f)
    //    explicit Feature(const FeatureVector& f)
    //    bool operator==(const Feature& r) const
    //    Core::Ref<const Vector> operator[](size_t streamIndex) const
    //    Iterator begin() const
    //    Iterator end() const

    py::class_<Flow::Data, Flow::DataPtr<Flow::Data>>(m, "Data")
    .def(py::init<>())
    .def(py::init<const Flow::Data&>());

    /*
     * virtual Data* clone() const
     * const Datatype* datatype() const
     * virtual bool operator==(const Data& other) const
     * virtual Core::XmlWriter& dump(Core::XmlWriter&) const
     * virtual bool             read(Core::BinaryInputStream&)
     * virtual bool write(Core::BinaryOutputStream&) const
     * static inline Data* eos()
     * static inline Data* ood()
     * static bool isSentinel(const ThreadSafeReferenceCounted* object)
     * static bool isNotSentinel(const ThreadSafeReferenceCounted* object)
     */

    py::class_<Flow::Timestamp, Flow::Data, Flow::DataPtr<Flow::Timestamp>>(m, "Timestamp")
    .def(py::init<Flow::Time, Flow::Time>())
    .def("set_start_time", &Flow::Timestamp::setStartTime)
    .def("get_start_time", &Flow::Timestamp::getStartTime)
    .def("start_time", &Flow::Timestamp::startTime)
    .def("set_end_time", &Flow::Timestamp::setEndTime)
    .def("get_end_time", &Flow::Timestamp::getEndTime)
    .def("end_time", &Flow::Timestamp::endTime)
    .def("set_timestamp", &Flow::Timestamp::setTimestamp)
    .def("expand_timestamp", &Flow::Timestamp::expandTimestamp)
    .def("invalidate_timestamp", &Flow::Timestamp::invalidateTimestamp)
    .def("is_valid_timestamp", &Flow::Timestamp::isValidTimestamp)
    .def("equals_to_start_time", &Flow::Timestamp::equalsToStartTime)
    .def("equal_start_time", &Flow::Timestamp::equalStartTime)
    .def("equals_to_end_time", &Flow::Timestamp::equalsToEndTime)
    .def("equal_end_time", &Flow::Timestamp::equalEndTime)
    .def("contains", (bool (Flow::Timestamp::*)(Flow::Time) const) &Flow::Timestamp::contains)
    .def("contains", (bool (Flow::Timestamp::*)(const Flow::Timestamp&) const) &Flow::Timestamp::contains)
    .def("overlap", &Flow::Timestamp::overlap);

    //    static const Datatype* type()
    //    virtual Core::XmlWriter& dump(Core::XmlWriter& o) const
    //    bool read(Core::BinaryInputStream& i)
    //    bool write(Core::BinaryOutputStream& o) const

    py::class_<Speech::Feature, Mm::Feature, Core::Ref<Speech::Feature>>(m, "Feature")
    .def(py::init<>())
    .def(py::init<Flow::DataPtr<Speech::Feature::FlowVector>&>()) // must be tested
    .def(py::init<Flow::DataPtr<Speech::Feature::FlowFeature>&>())
    .def("set_timestamp", &Speech::Feature::setTimestamp)
    .def("timestamp", &Speech::Feature::timestamp, py::return_value_policy::reference_internal)
    .def("take", (void (Speech::Feature::*)(Flow::DataPtr<Speech::Feature::FlowVector>&)) &Speech::Feature::take)
    .def("take", (void (Speech::Feature::*)(Flow::DataPtr<Speech::Feature::FlowFeature>&)) &Speech::Feature::take)
    .def("take", (bool (Speech::Feature::*)(Flow::DataPtr<Flow::Timestamp>&)) &Speech::Feature::take);

    // virtual Mm::FeatureDescription* getDescription(const Core::Configurable& parent) const

    py::class_<Flow::DataSource, Core::Ref<Flow::DataSource>>(m, "FlowDataSource") // Flow::Network
    .def(py::init<const Core::Configuration&, bool>())
    .def("get_data", (bool (Flow::DataSource::*)(Flow::PortId, Flow::DataPtr<Flow::Data>&)) &Flow::DataSource::getData);

    py::class_<Speech::DataSource, Flow::DataSource, Core::Ref<Speech::DataSource>>(m, "DataSource")
    .def(py::init<const Core::Configuration&, bool>())
    .def("initialize", &Speech::DataSource::initialize)
    .def("finalize", &Speech::DataSource::finalize)
    .def("get_data", (bool (Speech::DataSource::*)(Flow::PortId, Core::Ref<Speech::Feature>&)) &Speech::DataSource::getData)
    .def("get_data", (bool (Speech::DataSource::*)(Core::Ref<Speech::Feature>&)) &Speech::DataSource::getData)
    .def("get_data", (bool (Speech::DataSource::*)()) &Speech::DataSource::getData)
    .def("convert", &Speech::DataSource::convert)
    .def("main_port_id", &Speech::DataSource::mainPortId)
    .def("nun_frames", &Speech::DataSource::nFrames, py::return_value_policy::reference_internal)
    .def("real_time", &Speech::DataSource::realTime)
    .def("set_progress_indication", &Speech::DataSource::setProgressIndication)
    .def("get_data", (bool (Speech::DataSource::*)(Flow::PortId, Flow::DataPtr<Flow::Data>&)) &Speech::DataSource::getData)
    .def("get_data", (bool (Speech::DataSource::*)(Flow::DataPtr<Flow::Data>&)) &Speech::DataSource::getData); // be sure of template issue

    py::class_<Speech::DataExtractor, Speech::CorpusProcessor>(m, "DataExtractor")
    .def(py::init<const Core::Configuration&, bool>())
    .def("sign_on", &Speech::DataExtractor::signOn)
    .def("enter_corpus", &Speech::DataExtractor::enterCorpus)
    .def("leave_corpus", &Speech::DataExtractor::leaveCorpus)
    .def("enter_recording", &Speech::DataExtractor::enterRecording)
    .def("enter_segment", &Speech::DataExtractor::enterSegment)
    .def("leave_segment", &Speech::DataExtractor::leaveSegment)
    .def("process_segment", &Speech::DataExtractor::processSegment);

    py::class_<Speech::FeatureExtractor, Speech::DataExtractor>(m, "FeatureExtractor")
    .def(py::init<const Core::Configuration&, bool>())
    .def("process_segment", &Speech::FeatureExtractor::processSegment);

    py::class_<Speech::FeatureVectorExtractor, Speech::FeatureExtractor>(m, "FeatureVectorExtractor")
    .def(py::init<const Core::Configuration&>());

}
