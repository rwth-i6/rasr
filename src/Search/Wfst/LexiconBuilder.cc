/** Copyright 2018 RWTH Aachen University. All rights reserved.
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
#include <Bliss/Fsa.hh>
#include <Fsa/AlphabetUtility.hh>
#include <OpenFst/SymbolTable.hh>
#include <Search/Wfst/LexiconBuilder.hh>
#include <Search/Wfst/NonWordTokens.hh>

using namespace Search::Wfst;
using Bliss::LemmaPronunciation;
using Bliss::Lexicon;
using Bliss::Pronunciation;

struct LexiconBuilder::Options {
    Options(const Core::Configuration& c);

    Core::XmlWriter& write(Core::XmlWriter& o) const;

    bool addEmptyTokens_;
    bool emptyTokenOutput_;
    bool markInitialPhones_;
    bool addWordDisambiguators_;
    bool disambiguateHomophones_;
    bool initialNonWordLoop_, wordEndNonWordLoop_;
    bool initialNonWords_;
    bool initialIsFinal_;
    bool pronScores_;
    bool addWordBoundaryPhones_;
    bool closureOutput_;
    bool addSentenceEndMark_;
    bool addNonWords_;
    bool addNonWordLoops_;
    bool removeEmptyProns_;

    static const Core::ParameterBool         paramAddEmptyTokens;
    static const Core::ParameterBool         paramEmptyTokenOutput;
    static const Core::ParameterBool         paramMarkInitialPhones;
    static const Core::ParameterBool         paramAddWordDisambiguators;
    static const Core::ParameterBool         paramDisambiguateHomophones;
    static const Core::ParameterBool         paramWordEndEmptyTokenLoop;
    static const Core::ParameterBool         paramInitialEmptyTokenLoop;
    static const Core::ParameterBool         paramInitialNonWords;
    static const Core::ParameterBool         paramInitialIsFinal;
    static const Core::ParameterBool         paramUsePronunciationScore;
    static const Core::ParameterBool         paramAddWordBoundaryPhones;
    static const Core::ParameterBool         paramClosureOutput;
    static const Core::ParameterBool         paramAddSentenceEndMark;
    static const Core::ParameterBool         paramAddNonWords;
    static const Core::ParameterBool         paramAddNonWordLoops;
    static const Core::ParameterBool         paramRemoveEmptyProns;
    static const Core::ParameterStringVector paramLemmasWithoutNonWords;
};

LexiconBuilder::Options::Options(const Core::Configuration& config)
        : addEmptyTokens_(paramAddEmptyTokens(config)),
          emptyTokenOutput_(paramEmptyTokenOutput(config)),
          markInitialPhones_(paramMarkInitialPhones(config)),
          addWordDisambiguators_(paramAddWordDisambiguators(config)),
          disambiguateHomophones_(paramDisambiguateHomophones(config)),
          initialNonWordLoop_(paramInitialEmptyTokenLoop(config)),
          wordEndNonWordLoop_(paramWordEndEmptyTokenLoop(config)),
          initialNonWords_(paramInitialNonWords(config)),
          initialIsFinal_(paramInitialIsFinal(config)),
          pronScores_(paramUsePronunciationScore(config)),
          addWordBoundaryPhones_(paramAddWordBoundaryPhones(config)),
          closureOutput_(paramClosureOutput(config)),
          addSentenceEndMark_(paramAddSentenceEndMark(config)),
          addNonWords_(paramAddNonWords(config)),
          addNonWordLoops_(addNonWords_ && paramAddNonWordLoops(config)),
          removeEmptyProns_(paramRemoveEmptyProns(config)) {}

namespace {
std::string boolToStr(bool b) {
    return b ? "true" : "false";
}
}  // namespace

Core::XmlWriter& LexiconBuilder::Options::write(Core::XmlWriter& out) const {
    out << "add empty tokens: " << boolToStr(addEmptyTokens_) << "\n"
        << "empty token output: " << boolToStr(emptyTokenOutput_) << "\n"
        << "mark initial phones: " << boolToStr(markInitialPhones_) << "\n"
        << "add word disambiguators: " << boolToStr(addWordDisambiguators_) << "\n"
        << "disambiguate homophones: " << boolToStr(disambiguateHomophones_) << "\n"
        << "word end non-word loop: " << boolToStr(wordEndNonWordLoop_) << "\n"
        << "initial non-word loop: " << boolToStr(initialNonWordLoop_) << "\n"
        << "initial non-word arcs: " << boolToStr(initialNonWords_) << "\n"
        << "final initial state: " << boolToStr(initialIsFinal_) << "\n"
        << "closure output: " << boolToStr(closureOutput_) << "\n"
        << "pronunciation scores: " << boolToStr(pronScores_) << "\n"
        << "add word boundary phones:" << boolToStr(addWordBoundaryPhones_) << "\n"
        << "add sentence end mark: " << boolToStr(addSentenceEndMark_) << "\n"
        << "add optional non-words: " << boolToStr(addNonWords_) << "\n"
        << "add non-word loops: " << boolToStr(addNonWordLoops_) << "\n"
        << "remove empty pronunciations: " << boolToStr(removeEmptyProns_) << "\n";
    return out;
}

const Core::ParameterBool LexiconBuilder::Options::paramAddEmptyTokens(
        "add-empty-tokens",
        "add pronunciations with empty syntatic token sequence", false);
const Core::ParameterBool LexiconBuilder::Options::paramEmptyTokenOutput(
        "empty-token-output",
        "add output labels for pronunciations with empty syntactic "
        "token sequence",
        true);
const Core::ParameterBool LexiconBuilder::Options::paramMarkInitialPhones(
        "mark-initial-phones",
        "add offset to initial phones", true);
const Core::ParameterBool LexiconBuilder::Options::paramAddWordDisambiguators(
        "add-word-disambiguators",
        "add a distinct disambiguation symbol for each lemma pronunciation", false);
const Core::ParameterBool LexiconBuilder::Options::paramDisambiguateHomophones(
        "add-disambiguators",
        "add disambiguators for homophones", true);
const Core::ParameterBool LexiconBuilder::Options::paramWordEndEmptyTokenLoop(
        "word-end-non-word-loop",
        "add loop transitions for non-word tokens at each word end", false);
const Core::ParameterBool LexiconBuilder::Options::paramInitialEmptyTokenLoop(
        "initial-non-word-loop",
        "add loop transitions for non-word tokens at the initial state", false);
const Core::ParameterBool LexiconBuilder::Options::paramInitialNonWords(
        "initial-non-words",
        "add optional transitions for non-word tokens from the initial state", false);
const Core::ParameterBool LexiconBuilder::Options::paramInitialIsFinal(
        "initial-final",
        "set initial state as final state", false);
const Core::ParameterBool LexiconBuilder::Options::paramUsePronunciationScore(
        "use-pron-score",
        "add pronunciation scores as arc weights", false);
const Core::ParameterBool LexiconBuilder::Options::paramAddWordBoundaryPhones(
        "add-word-boundary-phones",
        "add additional input labels for phones at word boundaries", false);
const Core::ParameterBool LexiconBuilder::Options::paramClosureOutput(
        "closure-output",
        "closure arcs have output", true);
const Core::ParameterBool LexiconBuilder::Options::paramAddSentenceEndMark(
        "add-sentence-end",
        "add a special label for the sentence end lemma", false);
const Core::ParameterBool LexiconBuilder::Options::paramAddNonWords(
        "add-non-words",
        "add optional non-word arcs without output labels at word ends", false);
const Core::ParameterBool LexiconBuilder::Options::paramAddNonWordLoops(
        "non-word-loops",
        "add optional non-word loops (requires add-non-words)", false);
const Core::ParameterStringVector LexiconBuilder::Options::paramLemmasWithoutNonWords(
        "lemmas-without-non-words",
        "lemmas without optional non-word arcs at word end (used with add-non-words=true)",
        ",");
const Core::ParameterBool LexiconBuilder::Options::paramRemoveEmptyProns(
        "remove-empty-pronunciations",
        "remove lemma pronunciations with empty pronunciation", false);

const char* LexiconBuilder::initialSuffix     = "@i";
const char* LexiconBuilder::finalSuffix       = "@f";
const char* LexiconBuilder::sentenceEndSymbol = "#$";

LexiconBuilder::LexiconBuilder(const Core::Configuration& c, const Lexicon& lexicon)
        : Core::Component(c),
          options_(new Options(c)),
          result_(0),
          inputSymbols_(0),
          outputSymbols_(0),
          nonWordTokens_(new NonWordTokens(select("non-word-tokens"), lexicon)),
          lexicon_(lexicon),
          nGrammarDisambiguators_(0),
          nPhoneDisambiguators_(-1),
          initialPhoneOffset_(-1),
          wordLabelOffset_(-1),
          disambiguatorOffset_(-1),
          sentenceEndLemma_(Fsa::InvalidLabelId),
          silencePhone_(OpenFst::InvalidLabelId) {
    nonWordTokens_->init();
}

LexiconBuilder::~LexiconBuilder() {
    delete options_;
    delete inputSymbols_;
    delete outputSymbols_;
    delete nonWordTokens_;
}

void LexiconBuilder::logSettings(bool buildClosed) const {
    options_->write(log("building lexicon transducer\n"))
            << "build closed: " << boolToStr(buildClosed) << "\n"
            << "#disambiguators: " << nGrammarDisambiguators_ << "\n";
}

bool LexiconBuilder::addWordDisambiguators() const {
    return options_->addWordDisambiguators_;
}

/**
 * find the the sentence end lemma
 * sets sentenceEndLemma_
 */
void LexiconBuilder::getSentenceEnd() {
    const Bliss::Lemma* lemma = lexicon_.specialLemma("sentence-end");
    verify(lemma);
    sentenceEndLemma_ = lemma->id();
}

/**
 * sets pronsWithoutNonWords_
 */
void LexiconBuilder::getPronsWithoutNonWords(const std::vector<std::string>& lemmas) {
    Core::Ref<const Bliss::LemmaPronunciationAlphabet> lp = lexicon_.lemmaPronunciationAlphabet();
    for (std::vector<std::string>::const_iterator symbol = lemmas.begin(); symbol != lemmas.end(); ++symbol) {
        const Bliss::Lemma* lemma = lexicon_.lemma(*symbol);
        if (!lemma)
            error("unknown lemma symbol '%s'", symbol->c_str());
        else {
            Bliss::Lemma::LemmaPronunciationRange prons = lemma->pronunciations();
            for (Bliss::Lemma::PronunciationIterator p = prons.first; p != prons.second; ++p)
                pronsWithoutNonWords_.insert(OpenFst::convertLabelFromFsa(lp->index(p)));
        }
    }
}

/**
 * create input (phones) and output (pronunciation) symbol tables.
 * sets inputSymbols_, outputSymbols_ , disambiguatorOffset_,
 * initialPhoneOffset_, wordLabelOffset_
 */
void LexiconBuilder::createSymbolTables() {
    Core::Ref<const Bliss::PhonemeAlphabet> phones =
            lexicon_.phonemeInventory()->phonemeAlphabet();
    Core::Ref<const Bliss::LemmaPronunciationAlphabet> prons =
            lexicon_.lemmaPronunciationAlphabet();
    inputSymbols_        = OpenFst::convertAlphabet(phones, "phones", -1);
    outputSymbols_       = OpenFst::convertAlphabet(prons, "pronunciations");
    disambiguatorOffset_ = inputSymbols_->AvailableKey();
    if (options_->addWordBoundaryPhones_) {
        addBoundaryPhoneLabels(true);
    }
    if (options_->markInitialPhones_) {
        if (options_->addWordBoundaryPhones_) {
            error("cannot use mark-initial-phones and add-word-boundary-phones");
        }
        // no initial symbol for p = 0 = epsilon
        initialPhoneOffset_ = inputSymbols_->AvailableKey() - 1;
        addBoundaryPhoneLabels(false);
    }
    else {
        initialPhoneOffset_ = 0;
    }
    if (options_->addNonWords_) {
        const std::vector<Bliss::Phoneme::Id>& nonWordPhones = nonWordTokens_->phones();
        for (std::vector<Bliss::Phoneme::Id>::const_iterator nwp = nonWordPhones.begin();
             nwp != nonWordPhones.end(); ++nwp) {
            std::string symbol = nonWordTokens_->phoneSymbol(*nwp);
            verify(inputSymbols_->Find(symbol) < 0);
            inputSymbols_->AddSymbol(symbol);
        }
        disambiguatorOffset_ = inputSymbols_->AvailableKey();
    }
    if (options_->addSentenceEndMark_) {
        verify(inputSymbols_->Find(sentenceEndSymbol) < 0);
        inputSymbols_->AddSymbol(sentenceEndSymbol);
        disambiguatorOffset_ = inputSymbols_->AvailableKey();
    }
    wordLabelOffset_ = inputSymbols_->AvailableKey();
    if (options_->addWordDisambiguators_) {
        addWordDisambiguatorLabels();
    }
}

OpenFst::VectorFst* LexiconBuilder::build(bool buildClosed) {
    logSettings(buildClosed);
    if (options_->addSentenceEndMark_)
        getSentenceEnd();

    result_ = new OpenFst::VectorFst();

    createSymbolTables();
    log("initial phone offset: %d", initialPhoneOffset_);
    log("word label offset: %d", wordLabelOffset_);
    log("disambiguator offset: %d", disambiguatorOffset_);

    if (options_->addNonWords_) {
        getPronsWithoutNonWords(Options::paramLemmasWithoutNonWords(config));
        log("not using non-word arcs for %d lemmas", static_cast<u32>(pronsWithoutNonWords_.size()));
    }

    initialState_ = result_->AddState();
    result_->SetStart(initialState_);

    if (options_->initialNonWords_) {
        initialState_ = result_->AddState();
        addOptionalNonWordArcs(result_->Start(), initialState_);
        result_->AddArc(result_->Start(),
                        OpenFst::Arc(OpenFst::Epsilon, OpenFst::Epsilon, OpenFst::Weight::One(), initialState_));
    }

    if (options_->initialIsFinal_)
        result_->SetFinal(initialState_, OpenFst::Weight::One());

    PronunciationHashMap           pronHash;
    Lexicon::PronunciationIterator piBegin, piEnd;
    Core::tie(piBegin, piEnd) = lexicon_.pronunciations();
    for (Lexicon::PronunciationIterator pi = piBegin; pi != piEnd; ++pi) {
        Pronunciation::LemmaIterator liBegin, liEnd;
        Core::tie(liBegin, liEnd) = (*pi)->lemmas();
        for (Pronunciation::LemmaIterator li = liBegin; li != liEnd; ++li) {
            const LemmaPronunciation* lemmaPron    = li;
            const bool                isEmptyToken = lemmaPron->lemma()->syntacticTokenSequence().isEpsilon();
            if (options_->addSentenceEndMark_ && lemmaPron->lemma()->id() == sentenceEndLemma_) {
                addSentenceEnd(OpenFst::convertLabelFromFsa(lemmaPron->id()), buildClosed);
            }
            else if (!isEmptyToken || options_->addEmptyTokens_) {
                if (lemmaPron->pronunciation()->phonemes()[0] == Bliss::Phoneme::term) {
                    warning("empty pronunciation for lemma %s", lemmaPron->lemma()->symbol().str());
                    if (options_->removeEmptyProns_)
                        continue;
                }
                OpenFst::Label output;
                if (!isEmptyToken || options_->emptyTokenOutput_)
                    output = OpenFst::convertLabelFromFsa(lemmaPron->id());
                else
                    output = OpenFst::Epsilon;
                addPronunciation(lemmaPron, output, buildClosed);
            }
        }
    }
    if (options_->initialNonWordLoop_) {
        addNonWordLoop(initialState_, OpenFst::Weight::One());
    }
    if (nGrammarDisambiguators_) {
        log("adding disambiguator loop for grammar disambiguators");
        addDisambiguatorLoop();
    }
    if (options_->wordEndNonWordLoop_) {
        const OpenFst::StateId nStates = result_->NumStates();
        for (OpenFst::StateId sid = 0; sid < nStates; ++sid) {
            if (OpenFst::isFinalState(*result_, sid) && sid != initialState_)
                addNonWordLoop(sid, OpenFst::Weight::One());
        }
    }
    log("phone disambiguators: %d", nPhoneDisambiguators());
    homophones_.clear();
    result_->SetInputSymbols(inputSymbols_);
    result_->SetOutputSymbols(outputSymbols_);
    return result_;
}

void LexiconBuilder::addBoundaryPhoneLabels(bool addFinal) {
    verify(inputSymbols_);
    u32 nPhones = inputSymbols_->AvailableKey();
    for (u32 p = 1; p < nPhones; ++p) {
        std::string phone = inputSymbols_->Find(p);
        if (phone.empty()) {
            warning("empty phone symbol for index %d", p);
            phone = "_";
        }
        const Bliss::Phoneme* phoneme = lexicon_.phonemeInventory()->phoneme(phone);
        if (!phoneme->isContextDependent() && !options_->markInitialPhones_) {
            // do not create word boundary dependent phones for CI phones,
            // but add initial tags when marking initial phones
            continue;
        }
        inputSymbols_->AddSymbol(phone + initialSuffix);
        if (addFinal) {
            inputSymbols_->AddSymbol(phone + finalSuffix);
            inputSymbols_->AddSymbol(phone + initialSuffix + finalSuffix);
        }
    }
    disambiguatorOffset_ = inputSymbols_->AvailableKey();
}

void LexiconBuilder::addWordDisambiguatorLabels() {
    verify(inputSymbols_);
    verify(outputSymbols_);
    u32 nProns = outputSymbols_->AvailableKey();
    for (u32 l = 0; l < nProns; ++l) {
        std::string pron = outputSymbols_->Find(l);
        verify(!pron.empty());
        inputSymbols_->AddSymbol("#_" + pron);
    }
    disambiguatorOffset_ = inputSymbols_->AvailableKey();
}

std::string LexiconBuilder::phoneDisambiguatorSymbol(u32 disambiguator) {
    return Core::form("#%d", disambiguator);
}

OpenFst::Label LexiconBuilder::phoneDisambiguator(u32 disambiguator) {
    OpenFst::Label key = disambiguatorOffset_ + disambiguator;
    while (inputSymbols_->AvailableKey() <= key) {
        std::string symbol = phoneDisambiguatorSymbol(int(inputSymbols_->AvailableKey() - disambiguatorOffset_));
        inputSymbols_->AddSymbol(symbol);
    }
    return key;
}

OpenFst::Label LexiconBuilder::inputLabel(Bliss::Phoneme::Id phone,
                                          bool initial, bool final) const {
    // symbol index is not shifted, because phone indexes in Bliss
    // are in range [ 1 .. n ]
    OpenFst::Label input = phone;
    if (initial && (options_->markInitialPhones_ || options_->addWordBoundaryPhones_)) {
        OpenFst::Label newLabel = inputSymbols_->Find(inputSymbols_->Find(input) + initialSuffix);
        if (newLabel > 0)
            input = newLabel;
    }
    if (final && options_->addWordBoundaryPhones_) {
        OpenFst::Label newLabel = inputSymbols_->Find(inputSymbols_->Find(input) + finalSuffix);
        if (newLabel > 0)
            input = newLabel;
    }
    return input;
}

void LexiconBuilder::addNonWordLoop(OpenFst::StateId s, OpenFst::Weight weight) {
    const std::vector<const LemmaPronunciation*>&                  prons = nonWordTokens_->lemmaPronunciations();
    typedef std::vector<const LemmaPronunciation*>::const_iterator LpIter;
    for (LpIter lp = prons.begin(); lp != prons.end(); ++lp) {
        verify((*lp)->pronunciation()->length() == 1);
        const Bliss::Phoneme::Id phone  = (*lp)->pronunciation()->phonemes()[0];
        OpenFst::Label           input  = inputLabel(phone, true);
        OpenFst::Label           output = options_->emptyTokenOutput_ ? OpenFst::convertLabelFromFsa((*lp)->id()) : OpenFst::Epsilon;
        result_->AddArc(s, OpenFst::Arc(input, output, weight, s));
    }
}

void LexiconBuilder::addOptionalNonWordArcs(OpenFst::StateId from, OpenFst::StateId to) {
    const std::vector<Bliss::Phoneme::Id>&                  phones = nonWordTokens_->phones();
    typedef std::vector<Bliss::Phoneme::Id>::const_iterator Iter;
    for (Iter p = phones.begin(); p != phones.end(); ++p) {
        OpenFst::Label label = inputSymbols_->Find(nonWordTokens_->phoneSymbol(*p));
        verify(label > 0);
        result_->AddArc(from, OpenFst::Arc(label, OpenFst::Epsilon, OpenFst::Weight::One(), to));
    }
}

void LexiconBuilder::addPronunciation(const LemmaPronunciation* lemmaPron,
                                      OpenFst::Label output, bool closed) {
    Core::Ref<const Bliss::PhonemeAlphabet> phoneAlphabet = lexicon_.phonemeInventory()->phonemeAlphabet();
    const Pronunciation*                    pron          = lemmaPron->pronunciation();
    OpenFst::StateId                        s             = initialState_;
    OpenFst::Label                          arcOutput     = output;
    const Bliss::Phoneme::Id*               phones        = pron->phonemes();
    OpenFst::Weight                         weight;
    OpenFst::Weight                         one = OpenFst::Weight::One();
    if (options_->pronScores_)
        weight = OpenFst::Weight(lemmaPron->pronunciationScore());
    else
        weight = one;
    const bool addNonWordArcs   = options_->addNonWords_ && !pronsWithoutNonWords_.count(output) && !options_->initialNonWords_;
    const bool connectToInitial = closed && !(options_->disambiguateHomophones_ || (options_->addWordDisambiguators_ && output != OpenFst::Epsilon)) && !addNonWordArcs;
    bool       first            = true;
    while (*phones != Bliss::Phoneme::term) {
        const bool     lastPhone = *(phones + 1) == Bliss::Phoneme::term;
        OpenFst::Label input     = inputLabel(*phones, first, lastPhone);
        ++phones;
        OpenFst::StateId nextState = OpenFst::InvalidStateId;
        if (*phones == Bliss::Phoneme::term && connectToInitial) {
            nextState = result_->Start();
        }
        else {
            nextState = result_->AddState();
        }
        result_->AddArc(s, OpenFst::Arc(input, arcOutput, weight, nextState));
        arcOutput = OpenFst::Epsilon;
        weight    = one;
        s         = nextState;
        first     = false;
    }
    if (options_->disambiguateHomophones_) {
        int                            homophoneIndex = 0;
        PronunciationHashMap::iterator hi             = homophones_.find(pron);
        if (hi == homophones_.end()) {
            homophones_.insert(std::make_pair(pron, 0));
        }
        else {
            hi->second++;
            homophoneIndex = hi->second;
        }
        OpenFst::StateId nextState = OpenFst::InvalidStateId;
        if (closed && !options_->addWordDisambiguators_ && output != OpenFst::Epsilon && !addNonWordArcs)
            nextState = result_->Start();
        else
            nextState = result_->AddState();
        OpenFst::Label input = phoneDisambiguator(homophoneIndex);
        result_->AddArc(s, OpenFst::Arc(input, OpenFst::Epsilon, weight, nextState));
        s = nextState;
    }
    if (options_->addWordDisambiguators_ && output != OpenFst::Epsilon) {
        OpenFst::StateId nextState = OpenFst::InvalidStateId;
        if (closed && !addNonWordArcs)
            nextState = result_->Start();
        else
            nextState = result_->AddState();

        OpenFst::Label input = wordLabelOffset_ + lemmaPron->id();
        result_->AddArc(s, OpenFst::Arc(input, OpenFst::Epsilon, weight, nextState));
        s = nextState;
    }
    if (addNonWordArcs) {
        OpenFst::StateId nextState = OpenFst::InvalidStateId;
        if (closed)
            nextState = result_->Start();
        else if (!options_->addNonWordLoops_)
            nextState = result_->AddState();
        if (options_->addNonWordLoops_) {
            addOptionalNonWordArcs(s, s);
        }
        else {
            addOptionalNonWordArcs(s, nextState);
        }
        result_->AddArc(s, OpenFst::Arc(OpenFst::Epsilon, OpenFst::Epsilon, OpenFst::Weight::One(), nextState));
        s = nextState;
    }
    result_->SetFinal(s, OpenFst::Weight::One());
}

void LexiconBuilder::addSentenceEnd(OpenFst::Label output, bool close) {
    verify(sentenceEndLemma_ != Fsa::InvalidLabelId);
    OpenFst::Label input = inputSymbols_->Find(sentenceEndSymbol);
    verify(input >= 0);
    OpenFst::StateId next = result_->Start();
    if (!close) {
        next = result_->AddState();
        result_->SetFinal(next, OpenFst::Weight::One());
    }
    result_->AddArc(initialState_, OpenFst::Arc(input, output, OpenFst::Weight::One(), next));
}

void LexiconBuilder::addDisambiguatorLoop() {
    Core::Ref<const Bliss::LemmaPronunciationAlphabet> prons   = lexicon_.lemmaPronunciationAlphabet();
    OpenFst::StateId                                   initial = result_->Start();
    for (int d = 0; d < nGrammarDisambiguators_; ++d) {
        result_->AddArc(initial, OpenFst::Arc(phoneDisambiguator(d),
                                              OpenFst::convertLabelFromFsa(prons->disambiguator(d)),
                                              OpenFst::Weight::One(), initial));
    }
}

void LexiconBuilder::close(OpenFst::VectorFst* l, bool useEmptyTokens) {
    require(initialPhoneOffset_ >= 0);
    Core::Ref<const Bliss::PhonemeAlphabet> phoneAlphabet = lexicon_.phonemeInventory()->phonemeAlphabet();
    std::vector<const LemmaPronunciation*>  closureTokens;
    if (useEmptyTokens) {
        nonWordTokens_->getEmptySyntacticTokenProns(closureTokens);
        log("building closure with %d lemma pronunciations", (int)closureTokens.size());
    }
    OpenFst::StateId initial = l->Start();
    OpenFst::Weight  weight  = OpenFst::Weight::One();
    const u32        nStates = l->NumStates();
    for (OpenFst::StateId sid = 0; sid < nStates; ++sid) {
        if (OpenFst::isFinalState(*l, sid) && sid != initial) {
            l->AddArc(sid, OpenFst::Arc(OpenFst::Epsilon, OpenFst::Epsilon, weight, initial));
            std::vector<const LemmaPronunciation*>::const_iterator lp = closureTokens.begin();
            for (; lp != closureTokens.end(); ++lp) {
                OpenFst::Label phone = (*lp)->pronunciation()->phonemes()[0];
                if (options_->markInitialPhones_) {
                    phone = inputSymbols_->Find(inputSymbols_->Find(phone) + initialSuffix);
                    verify(phone > 0);
                }
                OpenFst::StateId to     = initial;
                OpenFst::Label   output = OpenFst::convertLabelFromFsa((*lp)->id());
                if (options_->addWordDisambiguators_ && options_->emptyTokenOutput_) {
                    to = l->AddState();
                }
                if (options_->addWordDisambiguators_ || !options_->emptyTokenOutput_ || !options_->closureOutput_)
                    output = OpenFst::Epsilon;
                l->AddArc(sid, OpenFst::Arc(phone, output, weight, to));
                if (options_->addWordDisambiguators_ && options_->emptyTokenOutput_) {
                    OpenFst::Label input = wordLabelOffset_ + (*lp)->id();
                    l->AddArc(to, OpenFst::Arc(input, OpenFst::Epsilon, weight, initial));
                }
            }
        }
        l->SetFinal(sid, OpenFst::Weight::Zero());
    }
    l->SetFinal(l->Start(), OpenFst::Weight::One());
}
