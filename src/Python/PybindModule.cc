#include <string>

#include <pybind11/pybind11.h>

#include <Am/Module.hh>
#include <Audio/Module.hh>
#include <Bliss/CorpusDescription.hh>
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

#include "AllophoneStateFsaBuilder.hh"
#include "Configuration.hh"

namespace py = pybind11;

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

PYBIND11_MODULE(librasr, m) {
    static _DummyApplication app;

    m.doc() = "RASR python module";

    py::class_<Core::Configuration> baseConfigClass(m, "_BaseConfig");

    py::class_<PyConfiguration> pyRasrConfig(m, "Configuration", baseConfigClass);
    pyRasrConfig.def(py::init<>());
    pyRasrConfig.def("set_from_file",
        (bool (Core::Configuration::*)(const std::string&)) &Core::Configuration::setFromFile);

    py::class_<AllophoneStateFsaBuilder> pyFsaBuilder(m, "AllophoneStateFsaBuilder");
    pyFsaBuilder.def(py::init<const Core::Configuration&>());
    pyFsaBuilder.def("build_by_orthography",
        &AllophoneStateFsaBuilder::buildByOrthography
    );
    pyFsaBuilder.def("build_by_segment_name",
        &AllophoneStateFsaBuilder::buildBySegmentName
    );

    // py::class_<Core::XmlWriter> xmlWriter(m, "XmlWriter");

    py::class_<Bliss::Symbol>(m, "Symbol");

    py::class_<Bliss::OrthographicFormList::Size>(m, "OrthographicFormList::Size");

    py::class_<Bliss::Lemma::PronunciationIterator>(m, "Lemma::PronunciationIterator");

    py::class_<Bliss::OrthographicFormList>(m, "OrthographicFormList");

    py::class_<Bliss::SyntacticTokenSequence>(m, "SyntacticTokenSequence");

    py::class_<Bliss::Token>(m, "Token");

    py::class_<Bliss::Lemma, std::unique_ptr<Bliss::Lemma, py::nodelete>>(m, "Lemma")
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

    /*
     * typedef EvaluationTokenSequenceList::const_iterator EvaluationTokenSequenceIterator
     * std::pair<EvaluationTokenSequenceIterator, EvaluationTokenSequenceIterator> evaluationTokenSequences() const
     * typedef std::pair<PronunciationIterator, PronunciationIterator> LemmaPronunciationRange
     * LemmaPronunciationRange pronunciations() const
     */


    py::class_<Bliss::Pronunciation::LemmaIterator>(m, "Pronunciation::LemmaIterator");

    py::class_<Bliss::Pronunciation, std::unique_ptr<Bliss::Pronunciation, py::nodelete>>(m, "Pronunciation")
    .def("num_lemmas", &Bliss::Pronunciation::nLemmas)
    .def("length", &Bliss::Pronunciation::length)
    .def("format", &Bliss::Pronunciation::format)
    .def("phonemes", &Bliss::Pronunciation::phonemes, py::return_value_policy::reference_internal);

    /*
     * struct Hash
     * struct Equality
     * std::pair<LemmaIterator, LemmaIterator> lemmas() const
     * Phoneme::Id operator[](unsigned i) const
     */

    py::class_<Bliss::LemmaPronunciation>(m, "LemmaPronunciation")
    //.def_readonly_static("invalid_id", &Bliss::LemmaPronunciation::invalidId) n// not working
    .def("id", &Bliss::LemmaPronunciation::id)
    .def("lemma", &Bliss::LemmaPronunciation::lemma, py::return_value_policy::reference_internal)
    .def("pronunciation", &Bliss::LemmaPronunciation::pronunciation, py::return_value_policy::reference_internal)
    .def("pronunciation_probability", &Bliss::LemmaPronunciation::pronunciationProbability)
    .def("pronunciation_score", &Bliss::LemmaPronunciation::pronunciationScore)
    .def("set_pronunciation_probability", &Bliss::LemmaPronunciation::setPronunciationProbability)
    .def("next_for_this_lemma", &Bliss::LemmaPronunciation::nextForThisLemma, py::return_value_policy::reference_internal)
    .def("next_for_this_pronunciation", &Bliss::LemmaPronunciation::nextForThisPronunciation, py::return_value_policy::reference_internal);

    py::class_<Bliss::SyntacticToken>(m, "SyntacticToken")
    .def("num_lemmas", &Bliss::SyntacticToken::nLemmas)
    .def("class_emission_score", &Bliss::SyntacticToken::classEmissionScore);

    /*
     * typedef LemmaList::const_iterator LemmaIterator
     * std::pair<LemmaIterator, LemmaIterator> lemmas() const
     */

    py::class_<Bliss::Letter>(m, "Letter");

    py::class_<Bliss::Phoneme>(m, "Phoneme");

    py::class_<Bliss::PhonemeAlphabet, Core::Ref<Bliss::PhonemeAlphabet>>(m, "PhonemeAlphabet");

    py::class_<Bliss::PhonemeInventory, Core::Ref<Bliss::PhonemeInventory>> blissPhonemeInventory(m, "PhonemeInventory");
    blissPhonemeInventory
    .def(py::init<>())
    .def("num_phonemes", &Bliss::PhonemeInventory::nPhonemes)
    .def("phoneme", (const Bliss::Phoneme* (Bliss::PhonemeInventory::*)(const std::string&) const) &Bliss::PhonemeInventory::phoneme, py::return_value_policy::reference_internal)
    .def("phoneme", (const Bliss::Phoneme* (Bliss::PhonemeInventory::*)(Bliss::Phoneme::Id) const) &Bliss::PhonemeInventory::phoneme, py::return_value_policy::reference_internal)
    .def("is_valid_phoneme_id", &Bliss::PhonemeInventory::isValidPhonemeId)
    .def("new_phoneme", &Bliss::PhonemeInventory::newPhoneme, py::return_value_policy::reference_internal)
    .def("assign_symbol", &Bliss::PhonemeInventory::assignSymbol)
    .def("phoneme_alphabet", &Bliss::PhonemeInventory::phonemeAlphabet, py::return_value_policy::take_ownership);

    /*
     * typedef const Phoneme* const* PhonemeIterator
     * std::pair<PhonemeIterator, PhonemeIterator> phonemes() const
     * void writeBinary(Core::BinaryOutputStream&) const
     * void writeXml(Core::XmlWriter&) const
     * std::set<Bliss::Phoneme::Id> parseSelection(std::string selection) const
     */

    py::class_<Bliss::TokenAlphabet>(m, "TokenAlphabet");

    py::class_<Bliss::LemmaAlphabet, Core::Ref<Bliss::LemmaAlphabet>>(m, "LemmaAlphabet")
    .def("lemma", &Bliss::LemmaAlphabet::lemma, py::return_value_policy::reference_internal);

    /*
     * virtual void describe(Core::XmlWriter&) const
     */

    py::class_<Fsa::Alphabet>(m, "Alphabet");

    py::class_<Bliss::LemmaPronunciationAlphabet, Core::Ref<Bliss::LemmaPronunciationAlphabet>>(m, "LemmaPronunciationAlphabet")
    .def("index", (Fsa::LabelId (Bliss::LemmaPronunciationAlphabet::*)(const std::string&) const) &Bliss::LemmaPronunciationAlphabet::index)
    .def("index", (Fsa::LabelId (Bliss::LemmaPronunciationAlphabet::*)(const Bliss::LemmaPronunciation*) const) &Bliss::LemmaPronunciationAlphabet::index)
    .def("lemma_pronunciation", &Bliss::LemmaPronunciationAlphabet::lemmaPronunciation, py::return_value_policy::reference_internal)
    .def("symbol", &Bliss::LemmaPronunciationAlphabet::symbol)
    .def("num_disambiguators", &Bliss::LemmaPronunciationAlphabet::nDisambiguators)
    .def("disambiguator", &Bliss::LemmaPronunciationAlphabet::disambiguator)
    .def("is_disambiguator", &Bliss::LemmaPronunciationAlphabet::isDisambiguator);

    /*
     * virtual const_iterator end() const
     * virtual void         writeXml(Core::XmlWriter& os) const
     */

    py::class_<Bliss::SyntacticTokenAlphabet, Core::Ref<Bliss::SyntacticTokenAlphabet>>(m, "SyntacticTokenAlphabet")
    .def("syntactic_token", &Bliss::SyntacticTokenAlphabet::syntacticToken, py::return_value_policy::reference_internal);

    /*
     * virtual void describe(Core::XmlWriter& os) const;
     */

    py::class_<Bliss::TokenInventory>(m, "TokenInventory")
    .def("insert", &Bliss::TokenInventory::insert)
    .def("link", &Bliss::TokenInventory::link)
    .def("add", &Bliss::TokenInventory::add)
    .def("size", &Bliss::TokenInventory::size)
    .def("insert", &Bliss::TokenInventory::insert);

    /*
     * Token* operator[](Token::Id id) const
     * Token* operator[](const std::string& sym) const
     * Token* operator[](Symbol sym) const
     * typedef Token* const* Iterator
     * Iterator              begin() const
     * Iterator end() const
     */


    py::class_<Bliss::EvaluationToken>(m, "EvaluationToken");

    py::class_<Bliss::EvaluationTokenAlphabet, Core::Ref<Bliss::EvaluationTokenAlphabet>>(m, "EvaluationTokenAlphabet")
    .def("evaluation_token", &Bliss::EvaluationTokenAlphabet::evaluationToken, py::return_value_policy::reference_internal);

    /*
     * virtual void describe(Core::XmlWriter& os) const
     */

    py::class_<Bliss::LetterAlphabet, Core::Ref<Bliss::LetterAlphabet>>(m, "LetterAlphabet")
    .def("letter", &Bliss::LetterAlphabet::letter, py::return_value_policy::reference_internal);

    /*
     * virtual void describe(Core::XmlWriter& os) const
     */

    py::class_<Core::Dependency>(m, "Dependency");

    py::class_<Bliss::Lexicon>(m, "Lexicon")
        .def(py::init<const Core::Configuration&>())
        .def("get_dependency", &Bliss::Lexicon::getDependency, py::return_value_policy::reference_internal)
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
        // blissLexicon.def("write_xml", &Bliss::Lexicon::writeXml);
        .def("log_statistics", &Bliss::Lexicon::logStatistics)
        // static LexiconRef create(const Core::Configuration&);
        .def("num_lemmas", &Bliss::Lexicon::nLemmas)
        //typedef const Lemma* const* LemmaIterator;
        //std::pair<LemmaIterator, LemmaIterator> lemmas()
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
        // typedef const SyntacticToken* const* SyntacticTokenIterator;
        // std::pair<SyntacticTokenIterator, SyntacticTokenIterator> syntacticTokens()
        .def("syntactic_token", &Bliss::Lexicon::syntacticToken, py::return_value_policy::reference_internal)
        .def("syntactic_token_inventory", &Bliss::Lexicon::syntacticTokenInventory, py::return_value_policy::reference_internal)
        .def("syntactic_token_alphabet", &Bliss::Lexicon::syntacticTokenAlphabet, py::return_value_policy::take_ownership)
        .def("num_evaluation_tokens", &Bliss::Lexicon::nEvaluationTokens)
        // typedef const EvaluationToken* const* EvaluationTokenIterator;
        // std::pair<EvaluationTokenIterator, EvaluationTokenIterator> evaluationTokens()
        .def("evaluation_token_inventory", &Bliss::Lexicon::evaluationTokenInventory, py::return_value_policy::reference_internal)
        .def("evaluation_token_alphabet", &Bliss::Lexicon::evaluationTokenAlphabet, py::return_value_policy::take_ownership)
        .def("letter", &Bliss::Lexicon::letter, py::return_value_policy::reference_internal)
        .def("letter_inventory", &Bliss::Lexicon::letterInventory, py::return_value_policy::reference_internal)
        .def("letter_alphabet", &Bliss::Lexicon::letterAlphabet, py::return_value_policy::take_ownership);
}