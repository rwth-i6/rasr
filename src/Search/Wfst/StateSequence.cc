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
#include <Am/ClassicAcousticModel.hh>
#include <Core/CompressedStream.hh>
#include <Core/Debug.hh>
#include <Fsa/Hash.hh>
#include <OpenFst/SymbolTable.hh>
#include <Search/Wfst/NonWordTokens.hh>
#include <Search/Wfst/StateSequence.hh>
#include <fst/relabel.h>

namespace Search {
namespace Wfst {

bool StateSequence::read(Core::BinaryInputStream& in) {
    u32 nStates = 0;
    in >> flags_;
    in >> nStates;
    states_.resize(nStates);
    for (std::vector<State>::iterator i = states_.begin(); i != states_.end(); ++i) {
        in >> i->emission_ >> i->transition_;
    }
    return in.good();
}

bool StateSequence::write(Core::BinaryOutputStream& out) const {
    out << flags_;
    out << u32(states_.size());
    for (std::vector<State>::const_iterator i = states_.begin(); i != states_.end(); ++i)
        out << i->emission_ << i->transition_;
    return out;
}

void StateSequence::createFromAllophone(Core::Ref<const Am::AcousticModel> model, const Am::Allophone* allophone) {
    const Am::ClassicHmmTopology* hmmTopology = model->hmmTopology(allophone->central());
    require(hmmTopology != 0);
    int nPhoneStates = hmmTopology->nPhoneStates();
    int nSubStates   = hmmTopology->nSubStates();
    int nStates      = nPhoneStates * nSubStates;
    flags_           = allophone->boundary;
    states_.clear();
    states_.reserve(nStates);
    for (int s = 0; s < nStates; ++s) {
        Am::AllophoneState                      as         = model->allophoneStateAlphabet()->allophoneState(allophone, s / nSubStates);
        Mm::MixtureIndex                        emission   = model->emissionIndex(as);
        Am::AcousticModel::StateTransitionIndex transition = model->stateTransitionIndex(as, s % nSubStates);
        appendState(emission, transition);
    }
}

// ============================================================================

const char* StateSequenceList::MAGIC = "RWTHESL";

bool StateSequenceList::write(const std::string& filename) const {
    Core::CompressedOutputStream o(filename);
    Core::BinaryOutputStream     bo(o);
    if (!bo.write(MAGIC, strlen(MAGIC) + 1))
        return false;
    bo << u32(size());
    for (const_iterator i = begin(); i != end(); ++i) {
        if (!i->write(bo)) {
            return false;
        }
    }
    return true;
}

bool StateSequenceList::read(const std::string& filename) {
    Core::CompressedInputStream ifs(filename);
    if (!ifs.good())
        return false;
    Core::BinaryInputStream bi(ifs);
    char*                   buf = new char[strlen(MAGIC) + 1];
    bi.read(buf, strlen(MAGIC) + 1);
    if (strcmp(buf, MAGIC) != 0) {
        Core::Application::us()->criticalError("cannot read header '%s', read '%s'", MAGIC, buf);
        return false;
    }
    delete[] buf;
    u32 s = 0, i = 0;
    bi >> s;
    resize(s);
    for (iterator sequence = begin(); sequence != end(); ++sequence, ++i) {
        if (!sequence->read(bi)) {
            Core::Application::us()->criticalError("cannot read sequence %d", i);
            return false;
        }
    }
    return !ifs.fail();
}

size_t StateSequenceList::memoryUsage() const {
    size_t sum = 0;
    for (const_iterator i = begin(); i != end(); ++i)
        sum += i->memoryUsage();
    sum = +capacity() * sizeof(StateSequence);
    return sum;
}

void StateSequenceList::dump(Core::Ref<const Am::AcousticModel> am, const Bliss::LexiconRef lexicon,
                             Core::Channel& output) const {
    const Am::ClassicAcousticModel* cam = dynamic_cast<const Am::ClassicAcousticModel*>(am.get());
    verify(cam);
    Am::ClassicStateTyingRef             stateTying = cam->stateTying();
    std::vector<std::list<Fsa::LabelId>> emissionToAllophoneState(stateTying->nClasses());
    Am::ConstAllophoneStateAlphabetRef   alloStateAlphabet = am->allophoneStateAlphabet();
    for (std::pair<Am::AllophoneStateIterator, Am::AllophoneStateIterator> it = alloStateAlphabet->allophoneStates();
         it.first != it.second; ++it.first) {
        Mm::MixtureIndex emission = stateTying->classify(it.first.allophoneState());
        emissionToAllophoneState[emission].push_back(it.first.id());
    }
    for (u32 s = 0; s < size(); ++s) {
        const StateSequence& ss = (*this)[s];
        output << s << " " << ss.isInitial() << " " << ss.isFinal() << " ";
        for (u32 hmmState = 0; hmmState < ss.nStates(); ++hmmState) {
            output << hmmState << "=(";
            u32 t = ss.state(hmmState).transition_;
            u32 m = ss.state(hmmState).emission_;
            output << "t:" << t << " e:" << m << " ";
            for (std::list<Fsa::LabelId>::const_iterator i = emissionToAllophoneState[m].begin();
                 i != emissionToAllophoneState[m].end(); ++i) {
                output << alloStateAlphabet->symbol(*i) << " ";
            }
            output << ") ";
        }
        output << std::endl;
    }
}

// ============================================================================

AllophoneToAlloponeStateSequenceMap::AllophoneToAlloponeStateSequenceMap(Core::Ref<const Am::AcousticModel> model,
                                                                         bool                               removeDisambiguators,
                                                                         bool                               tieAllophones,
                                                                         bool                               ignoreFlags)
        : model_(model), stateSequences_(0), removeDisambiguators_(removeDisambiguators), nDisambiguators_(0) {
    allophoneAlphabet_      = model_->allophoneAlphabet();
    allophoneStateAlphabet_ = model_->allophoneStateAlphabet();
    if (tieAllophones) {
        if (ignoreFlags)
            stateSequences_ = new FullyTiedStateSequenceMap;
        else
            stateSequences_ = new TiedStateSequenceMap;
    }
    else {
        stateSequences_ = new UniqueStateSequenceMap;
    }
}

AllophoneToAlloponeStateSequenceMap::~AllophoneToAlloponeStateSequenceMap() {
    delete stateSequences_;
}

Fsa::LabelId AllophoneToAlloponeStateSequenceMap::stateSequenceIndex(Fsa::LabelId allophoneIndex) {
    LabelTranslationMap::const_iterator i = labelMapping_.find(allophoneIndex);
    if (i != labelMapping_.end())
        return i->second;
    Fsa::LabelId index;
    if (allophoneAlphabet_->isDisambiguator(allophoneIndex)) {
        if (removeDisambiguators_)
            index = Fsa::Epsilon;
        else {
            index = getDisambiguator(nDisambiguators_);
            nDisambiguators_++;
        }
    }
    else {
        const Am::Allophone* allophone = allophoneAlphabet_->allophone(allophoneIndex);
        StateSequence        stateSequence;
        stateSequence.createFromAllophone(model_, allophone);
        index = stateSequences_->index(stateSequence);
    }
    labelMapping_[allophoneIndex] = index;
    return index;
}

// ============================================================================

const Core::ParameterBool StateSequenceBuilder::paramRemoveDisambiguators(
        "remove-disambiguators", "replace disambiguator labels by epsilon", false);

const Core::ParameterBool StateSequenceBuilder::paramTiedAllophones(
        "tied-allophones", "tie allophones with equal state sequence", true);

const Core::ParameterBool StateSequenceBuilder::paramIgnoreFlags(
        "ignore-flags", "ignore initial/final flags when tying allophones", false);

const Core::ParameterBool StateSequenceBuilder::paramAddNonWords(
        "add-non-words", "add state sequences for non-word models without output", false);

StateSequenceBuilder::StateSequenceBuilder(const Core::Configuration&         c,
                                           Core::Ref<const Am::AcousticModel> am,
                                           Bliss::LexiconRef                  lexicon)
        : Core::Component(c), am_(am), lexicon_(lexicon), nDisambiguators_(0), map_(new AllophoneToAlloponeStateSequenceMap(am, paramRemoveDisambiguators(config), paramTiedAllophones(config), paramIgnoreFlags(config))), addNonWords_(paramAddNonWords(config)), nonWordTokens_(0) {
    log("tied allophones: %s", paramTiedAllophones(config) ? "true" : "false");
    log("ignore flags: %s", paramIgnoreFlags(config) ? "true" : "false");
    log("add non words: %s", addNonWords_ ? "true" : "false");
    if (addNonWords_) {
        nonWordTokens_ = new NonWordTokens(select("non-word-tokens"), *lexicon_);
        nonWordTokens_->init();
    }
}

StateSequenceBuilder::~StateSequenceBuilder() {
    delete nonWordTokens_;
    delete map_;
}

void StateSequenceBuilder::build() {
    typedef Am::AllophoneAlphabet::AllophoneList AllophoneList;
    Core::Ref<const Am::AllophoneAlphabet>       alphabet = am_->allophoneAlphabet();
    allophoneToLabel_.resize(alphabet->nClasses());
    if (addNonWords_) {
        nonWordTokens_->createAllophones(alphabet);
        allophoneToLabel_.resize(allophoneToLabel_.size() + nonWordTokens_->allophones().size());
    }
    const AllophoneList& allophoneList = alphabet->allophones();
    for (AllophoneList::const_iterator ai = allophoneList.begin();
         ai != allophoneList.end(); ++ai) {
        Fsa::LabelId allophone = alphabet->index(*ai);
        verify(!addNonWords_ || !nonWordTokens_->isNonWordPhone((*ai)->central()));
        Fsa::LabelId label = map_->stateSequenceIndex(allophone);
        addToMap(allophone, label);
    }
    for (u32 d = 0; d < nDisambiguators_; ++d) {
        Fsa::LabelId amIndex = alphabet->disambiguator(d);
        map_->stateSequenceIndex(amIndex);
    }
    if (addNonWords_) {
        Fsa::LabelId                       label             = map_->size();
        const NonWordTokens::AllophoneMap& nonWordAllophones = nonWordTokens_->allophones();
        for (NonWordTokens::AllophoneMap::const_iterator a = nonWordAllophones.begin();
             a != nonWordAllophones.end(); ++a) {
            Fsa::LabelId allophone = nonWordTokens_->allophoneId(a->second);
            log("non word state sequences: %d -> label %d", allophone, label);
            addToMap(allophone, label++);
        }
    }
}

void StateSequenceBuilder::addToMap(Fsa::LabelId allophone, Fsa::LabelId label) {
    if (label >= labelToAllophones_.size())
        labelToAllophones_.resize(label + 1);
    labelToAllophones_[label].push_back(allophone);
    verify(allophone < allophoneToLabel_.size());
    allophoneToLabel_[allophone] = label;
}

StateSequenceList* StateSequenceBuilder::createStateSequenceList() const {
    StateSequenceList* ssl = new StateSequenceList();
    map_->stateSequences().createStateSequenceList(*ssl);
    if (addNonWords_) {
        addNonWordsToList(ssl);
    }
    return ssl;
}

void StateSequenceBuilder::addNonWordsToList(StateSequenceList* list) const {
    Core::Ref<const Am::AllophoneAlphabet> alphabet          = am_->allophoneAlphabet();
    u32                                    nSequences        = list->size();
    const NonWordTokens::AllophoneMap&     nonWordAllophones = nonWordTokens_->allophones();
    for (NonWordTokens::AllophoneMap::const_iterator a = nonWordAllophones.begin();
         a != nonWordAllophones.end(); ++a) {
        Bliss::Phoneme::Id origPhone = nonWordTokens_->sourcePhone(a->second->central());
        verify(origPhone != Bliss::Phoneme::term);
        Am::Allophone origAllophone(origPhone, Am::Allophone::isInitialPhone | Am::Allophone::isFinalPhone);
        Fsa::LabelId  origIndex = alphabet->index(alphabet->toString(origAllophone));
        verify(origIndex < allophoneToLabel_.size());
        Fsa::LabelId seqId = allophoneToLabel_[origIndex];
        verify(seqId < nSequences);
        StateSequence seq = (*list)[seqId];
        list->push_back(seq);
        ensure(labelToAllophones_[list->size() - 1].front() == nonWordTokens_->allophoneId(a->second));
    }
}

void StateSequenceBuilder::relabelTransducer(FstLib::MutableFst<OpenFst::Arc>* f) const {
    Fsa::LabelId                                           label    = 0;
    Core::Ref<const Am::AllophoneAlphabet>                 alphabet = am_->allophoneAlphabet();
    std::vector<std::pair<OpenFst::Label, OpenFst::Label>> mapping, dummy;
    mapping.reserve(labelToAllophones_.size());
    for (LabelToLabelsMap::const_iterator i = labelToAllophones_.begin();
         i != labelToAllophones_.end(); ++i, ++label) {
        for (LabelToLabelsMap::value_type::const_iterator a = i->begin(); a != i->end(); ++a) {
            mapping.push_back(std::make_pair(OpenFst::convertLabelFromFsa(*a),
                                             OpenFst::convertLabelFromFsa(label)));
        }
    }
    for (u32 d = 0; d < nDisambiguators_; ++d) {
        Fsa::LabelId amIndex = alphabet->disambiguator(d);
        Fsa::LabelId label   = map_->stateSequenceIndex(amIndex);
        mapping.push_back(std::make_pair(OpenFst::convertLabelFromFsa(amIndex),
                                         OpenFst::convertLabelFromFsa(label)));
    }
    FstLib::Relabel(f, mapping, dummy);
    f->SetInputSymbols(createSymbols());
}

OpenFst::SymbolTable* StateSequenceBuilder::createSymbols() const {
    Core::Ref<const Am::AllophoneAlphabet> alphabet = am_->allophoneAlphabet();
    OpenFst::SymbolTable*                  symbols  = new OpenFst::SymbolTable("state-sequences");
    symbols->AddSymbol("eps", 0);
    OpenFst::Label label = 0;
    for (LabelToLabelsMap::const_iterator i = labelToAllophones_.begin();
         i != labelToAllophones_.end(); ++i, ++label) {
        std::string symbol;
        for (LabelToLabelsMap::value_type::const_iterator a = i->begin(); a != i->end(); ++a) {
            if (addNonWords_ && *a >= alphabet->nClasses()) {
                symbol += alphabet->toString(*nonWordTokens_->allophone(*a));
            }
            else {
                symbol += alphabet->symbol(*a);
            }
            symbol += "_";
        }
        symbols->AddSymbol(symbol, OpenFst::convertLabelFromFsa(label));
    }
    for (u32 d = 0; d < nDisambiguators_; ++d) {
        Fsa::LabelId dId    = alphabet->disambiguator(d);
        std::string  symbol = alphabet->symbol(dId);
        symbols->AddSymbol(symbol, OpenFst::convertLabelFromFsa(dId));
    }
    return symbols;
}

bool StateSequenceBuilder::isFsaDisambiguator(Fsa::LabelId label) {
    return AllophoneToAlloponeStateSequenceMap::isDisambiguator(label);
}

bool StateSequenceBuilder::isDisambiguator(OpenFst::Label label) {
    return isFsaDisambiguator(OpenFst::convertLabelToFsa(label));
}

const Core::ParameterString HmmListConverter::paramSilencePhone(
        "silence-phone", "silence phone symbol", "si");

StateSequenceList* HmmListConverter::creatStateSequenceList(const std::string& hmmListFile) const {
    verify(stateSyms_ && hmmSyms_);
    Core::CompressedInputStream cis(hmmListFile);
    if (!cis) {
        error("cannot read %s", hmmListFile.c_str());
        return 0;
    }
    std::string        line;
    u32                nLine = 0;
    StateSequenceList* list  = new StateSequenceList();
    while (cis) {
        ++nLine;
        std::getline(cis, line);
        Core::stripWhitespace(line);
        if (line.empty())
            continue;
        if (line == ".eps" || line == ".wb")
            continue;
        std::vector<std::string> fields = Core::split(line, " ");
        if (fields.size() <= 1) {
            error("wrong format in line %d: '%s'", nLine, line.c_str());
        }
        else {
            if (!addHmm(list, fields)) {
                error("parse error in line %d", nLine);
            }
        }
    }
    log("created %zd state sequences", list->size());
    return list;
}

bool HmmListConverter::addHmm(StateSequenceList* list, const std::vector<std::string>& fields) const {
    s32 hmm = hmmSyms_->Find(fields[0]);
    if (hmm < 0) {
        error("unknown hmm symbol: '%s'", fields[0].c_str());
        return false;
    }
    hmm += hmmOffset;
    if (hmm >= list->size()) {
        list->resize(hmm + 1);
    }
    DBG(1) << VAR(fields[0]) << " " << VAR(hmm) << ENDDBG;
    std::string phoneSym  = Core::split(fields[0], "_")[0];
    const bool  isSilence = (phoneSym == silencePhone_);
    if (isSilence) {
        log("using silence transition model for hmm '%s'",
            fields[0].c_str());
    }
    StateSequence& states = (*list)[hmm];
    verify(states.nStates() == 0);
    for (u32 i = 1; i < fields.size(); ++i) {
        u32 s = stateSyms_->Find(fields[i]);
        if (s < 0) {
            error("unknown state symbol: '%s'",
                  fields[i].c_str());
            return false;
        }
        Am::AcousticModel::StateTransitionIndex trans =
                isSilence ? Am::TransitionModel::silence
                          : Am::TransitionModel::phone0;
        states.appendState(s + hmmStateOffset, trans);
    }
    return true;
}

// ============================================================================

const StateSequence* StateSequenceResolver::find(const std::string& phone, u8 boundary) const {
    Bliss::PhonemeInventoryRef pi = am_->phonology()->getPhonemeInventory();
    const Bliss::Phoneme*      p  = pi->phoneme(phone);
    verify(p);
    return find(p, boundary);
}

const StateSequence* StateSequenceResolver::find(const Bliss::Phoneme* phone, u8 boundary) const {
    return find(phone->id(), boundary);
}

const StateSequence* StateSequenceResolver::find(const Bliss::Phoneme::Id phone, u8 boundary) const {
    Am::Allophone allophone(phone, boundary);
    return find(allophone);
}

const StateSequence* StateSequenceResolver::find(const Am::Allophone& allophone) const {
    Core::Ref<const Am::AllophoneAlphabet> allophones = am_->allophoneAlphabet();
    Am::AllophoneIndex                     index      = allophones->index(allophone);
    return find(index);
}

const StateSequence* StateSequenceResolver::find(const Am::AllophoneIndex index) const {
    Core::Ref<const Am::AllophoneAlphabet> allophones = am_->allophoneAlphabet();
    const Am::Allophone*                   allophone  = allophones->allophone(index);
    verify(allophone);
    return find(allophone);
}

const StateSequence* StateSequenceResolver::find(const Am::Allophone* allophone) const {
    StateSequence states;
    states.createFromAllophone(am_, allophone);
    StateSequenceList::const_iterator s = std::find(states_.begin(), states_.end(), states);
    if (s != states_.end())
        return &(*s);
    else
        return 0;
}

const StateSequence* StateSequenceResolver::findSilence(Core::Ref<const Bliss::Lexicon> lexicon) const {
    const Bliss::Lemma*                 lemma = lexicon->specialLemma("silence");
    Bliss::Lemma::PronunciationIterator lpBegin, lpEnd;
    Core::tie(lpBegin, lpEnd) = lemma->pronunciations();
    verify(lpBegin != lpEnd);
    verify_eq(lpBegin->pronunciation()->length(), 1);
    Bliss::Phoneme::Id phone = lpBegin->pronunciation()->phonemes()[0];
    return find(phone, Am::Allophone::isInitialPhone | Am::Allophone::isFinalPhone);
}

}  // namespace Wfst
}  // namespace Search
