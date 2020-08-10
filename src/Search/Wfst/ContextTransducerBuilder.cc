/** Copyright 2020 RWTH Aachen University. All rights reserved.
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
#include <Core/Debug.hh>
#include <Core/Hash.hh>
#include <Fsa/Basic.hh>
#include <OpenFst/SymbolTable.hh>
#include <OpenFst/Types.hh>
#include <Search/Wfst/ContextTransducerBuilder.hh>
#include <Search/Wfst/LexiconBuilder.hh>
#include <Search/Wfst/NonWordTokens.hh>
#include <fst/connect.h>

namespace Search {
namespace Wfst {

using Am::Allophone;
using Am::AllophoneAlphabet;

class ContextTransducerBuilder::Builder {
public:
    Builder(Core::Ref<const Am::AcousticModel>, Core::Ref<const Bliss::Lexicon>);
    virtual ~Builder();
    OpenFst::VectorFst* build();
    void                initialize();

    void setDisambiguators(u32 nDisambiguators, u32 disambiguatorOffset) {
        nDisambiguators_     = nDisambiguators;
        disambiguatorOffset_ = disambiguatorOffset;
    }

    void setWordDisambiguators(u32 nWordDisambiguators) {
        nWordDisambiguators_ = nWordDisambiguators;
    }

    void setInitialPhoneOffset(u32 initialPhoneOffset) {
        initialPhoneOffset_ = initialPhoneOffset;
    }
    void setPhoneSymbols(const OpenFst::SymbolTable* symbols) {
        phoneSymbols_ = symbols;
    }
    void setAddNonWordPhones(NonWordTokens* nonWordTokens) {
        nonWordTokens_ = nonWordTokens;
    }
    void setSequenceEndSymbol(const std::string& symbol) {
        sequenceEndSymbol_ = symbol;
    }

    u32 getWordLabelOffset() const;
    u32 getDisambiguatorOffset() const;

protected:
    struct TriphoneContextAndBoundary {
        bool               boundary_;
        u8                 disambiguator_;
        Bliss::Phoneme::Id history_, central_;

        struct Hash {
            size_t operator()(const TriphoneContextAndBoundary& p) const {
                size_t h = 0;
                h |= p.central_;
                h |= (p.history_ << 8);
                h |= (p.disambiguator_ << 16);
                h |= (p.boundary_ << 31);
                return h;
            }
        };

        struct Equal {
            bool operator()(const TriphoneContextAndBoundary& a,
                            const TriphoneContextAndBoundary& b) const {
                return (a.history_ == b.history_ &&
                        a.central_ == b.central_ &&
                        a.boundary_ == b.boundary_);
            }
        };

        TriphoneContextAndBoundary(Bliss::Phoneme::Id h, Bliss::Phoneme::Id c,
                                   u8 disambiguator, bool boundary)
                : boundary_(boundary), disambiguator_(disambiguator), history_(h), central_(c) {}
    };

protected:
    typedef std::vector<Bliss::Phoneme::Id>                              PhoneList;
    typedef std::unordered_map<Bliss::Phoneme::Id, const Am::Allophone*> PhoneMap;
    typedef std::unordered_map<TriphoneContextAndBoundary, OpenFst::StateId,
                               TriphoneContextAndBoundary::Hash,
                               TriphoneContextAndBoundary::Equal>
            StateSet;

    OpenFst::StateId getStateId(Bliss::Phoneme::Id history, Bliss::Phoneme::Id center,
                                u8 disambiguator, bool boundary);
    OpenFst::Label   getAllophoneLabel(const Allophone* allophone);
    OpenFst::Label   getAllophoneDisambiguator(u32 disambiguator);
    OpenFst::Label   getPhoneLabel(Bliss::Phoneme::Id phone, bool initialPhone) const;
    void             addArc(OpenFst::StateId from, OpenFst::StateId to,
                            const Allophone* input, Bliss::Phoneme::Id output,
                            bool initialPhone);
    void             addArc(OpenFst::StateId from, OpenFst::StateId to,
                            Fsa::LabelId input, Fsa::LabelId output);
    void             addOutputDisambiguatorArc(OpenFst::StateId from, OpenFst::StateId to,
                                               Fsa::LabelId input, u32 disambiguator);
    void             addWordDisambiguatorArc(OpenFst::StateId from, OpenFst::StateId to,
                                             Fsa::LabelId input, u32 disambiguator);
    void             addInputDisambiguatorArc(OpenFst::StateId from, OpenFst::StateId to,
                                              u32 disambiguator, Fsa::LabelId output);
    void             addDisambiguatorArcs(OpenFst::StateId from, OpenFst::StateId to, const Allophone* input = 0);
    bool             isCiPhone(Bliss::Phoneme::Id phone) const {
        return ciPhones_.find(phone) != ciPhones_.end();
    }
    OpenFst::Label getSequenceEndSymbol() const {
        if (sequenceEndSymbol_.empty()) {
            return OpenFst::Epsilon;
        }
        else {
            OpenFst::Label l = phoneSymbols_->Find(sequenceEndSymbol_);
            verify(l > 0);
            return l;
        }
    }

    virtual void prepare()                          = 0;
    virtual void buildAllophone(const Allophone* a) = 0;
    virtual void finalize() {}
    template<class T>
    void removeDuplicates(std::vector<T>& list) const;

private:
    void printState(OpenFst::StateId s) const;

protected:
    OpenFst::VectorFst*                         c_;
    const OpenFst::SymbolTable*                 phoneSymbols_;
    OpenFst::SymbolTable*                       allophoneSymbols_;
    Core::Ref<const Am::AcousticModel>          model_;
    Core::Ref<Fsa::StaticAutomaton>             product_;
    Core::Ref<const Am::AllophoneAlphabet>      allophones_;
    Core::Ref<const Bliss::PhonemeAlphabet>     phonemes_;
    const Am::AllophoneAlphabet::AllophoneList* allophoneList_;
    NonWordTokens*                              nonWordTokens_;
    std::string                                 sequenceEndSymbol_;
    PhoneMap                                    ciPhones_;
    PhoneList                                   initialCoartPhones_, initialNonCoartPhones_,
            innerNonCoartPhones_, contextIndependentInnerPhones_;
    bool     isAcrossWordModel_;
    u32      nDisambiguators_;
    s32      disambiguatorOffset_;
    s32      nWordDisambiguators_;
    s32      initialPhoneOffset_;
    StateSet stateMap_;
};

ContextTransducerBuilder::Builder::Builder(Core::Ref<const Am::AcousticModel> model,
                                           Core::Ref<const Bliss::Lexicon>    lexicon)
        : model_(model), nonWordTokens_(0), disambiguatorOffset_(-1), nWordDisambiguators_(-1), initialPhoneOffset_(-1) {
    allophones_        = model_->allophoneAlphabet();
    allophoneList_     = &model_->allophoneAlphabet()->allophones();
    phonemes_          = lexicon->phonemeInventory()->phonemeAlphabet();
    isAcrossWordModel_ = model->isAcrossWordModelEnabled();
    allophoneSymbols_  = new OpenFst::SymbolTable("allophones");
    allophoneSymbols_->AddSymbol("eps", 0);
    initialize();
}

ContextTransducerBuilder::Builder::~Builder() {
    delete allophoneSymbols_;
}

u32 ContextTransducerBuilder::Builder::getWordLabelOffset() const {
    return allophones_->disambiguator(0);
}

u32 ContextTransducerBuilder::Builder::getDisambiguatorOffset() const {
    return disambiguatorOffset_;
}

OpenFst::StateId ContextTransducerBuilder::Builder::getStateId(Bliss::Phoneme::Id history, Bliss::Phoneme::Id center,
                                                               u8 disambiguator, bool boundary) {
    TriphoneContextAndBoundary tcb(history, center, disambiguator, boundary);
    StateSet::const_iterator   i = stateMap_.find(tcb);
    OpenFst::StateId           s;
    if (i != stateMap_.end()) {
        s = i->second;
    }
    else {
        s = c_->AddState();
        stateMap_.insert(StateSet::value_type(tcb, s));
        printState(s);
    }
    return s;
}

void ContextTransducerBuilder::Builder::addArc(OpenFst::StateId from, OpenFst::StateId to, OpenFst::Label input, OpenFst::Label output) {
    DBG(1) << from << " -> " << to << " i=" << input << " o=" << output << std::endl;
    c_->AddArc(from, OpenFst::Arc(input, output, OpenFst::Weight::One(), to));
}

OpenFst::Label ContextTransducerBuilder::Builder::getAllophoneLabel(const Allophone* allophone) {
    if (!allophone) {
        return OpenFst::Epsilon;
    }
    else {
        Fsa::LabelId ai     = Fsa::InvalidLabelId;
        std::string  symbol = "";
        if (nonWordTokens_ && nonWordTokens_->isNonWordAllophone(allophone)) {
            ai     = nonWordTokens_->allophoneId(allophone);
            symbol = allophones_->toString(*allophone);
        }
        else {
            ai     = allophones_->index(allophone);
            symbol = allophones_->symbol(ai);
        }
        OpenFst::Label l = allophoneSymbols_->Find(symbol);
        if (l < 0)
            l = allophoneSymbols_->AddSymbol(symbol, OpenFst::convertLabelFromFsa(ai));
        DBG(1) << "allophone: " << allophones_->index(allophone) << " " << symbol << " label=" << l << std::endl;
        return l;
    }
}

OpenFst::Label ContextTransducerBuilder::Builder::getAllophoneDisambiguator(u32 disambiguator) {
    Fsa::LabelId   ai     = allophones_->disambiguator(disambiguator);
    std::string    symbol = allophones_->symbol(ai);
    OpenFst::Label l      = allophoneSymbols_->Find(symbol);
    if (l < 0) {
        l = allophoneSymbols_->AddSymbol(symbol, OpenFst::convertLabelFromFsa(ai));
    }
    DBG(1) << " disambiguator=" << disambiguator << " -> " << symbol << " " << l << std::endl;
    return l;
}

OpenFst::Label ContextTransducerBuilder::Builder::getPhoneLabel(Bliss::Phoneme::Id phone, bool initialPhone) const {
    if (phone == Bliss::Phoneme::Id(-1)) {
        return OpenFst::Epsilon;
    }
    else if (nonWordTokens_ && nonWordTokens_->isNonWordPhone(phone)) {
        OpenFst::Label l = phoneSymbols_->Find(nonWordTokens_->phoneSymbol(phone));
        verify(l > 0);
        return l;
    }
    else {
        std::string phoneSymbol = phonemes_->symbol(phone);
        if (initialPhone)
            phoneSymbol += LexiconBuilder::initialSuffix;
        OpenFst::Label l = phoneSymbols_->Find(phoneSymbol);
        verify(l > 0);
        return l;
    }
}

void ContextTransducerBuilder::Builder::addArc(OpenFst::StateId from, OpenFst::StateId to, const Allophone* input,
                                               Bliss::Phoneme::Id output, bool initialPhone) {
    OpenFst::Label inputLabel  = getAllophoneLabel(input);
    OpenFst::Label outputLabel = getPhoneLabel(output, initialPhone);
    addArc(from, to, inputLabel, outputLabel);
    printState(from);
    printState(to);
    DBG(1) << " "
           << (inputLabel != OpenFst::Epsilon ? allophoneSymbols_->Find(inputLabel) : "eps")
           << " "
           << (outputLabel != OpenFst::Epsilon ? phoneSymbols_->Find(outputLabel) : "eps")
           << std::endl;
}

void ContextTransducerBuilder::Builder::addOutputDisambiguatorArc(OpenFst::StateId from, OpenFst::StateId to, OpenFst::Label input, u32 disambiguator) {
    std::string    disambiguatorSymbol = LexiconBuilder::phoneDisambiguatorSymbol(disambiguator);
    OpenFst::Label output              = phoneSymbols_->Find(disambiguatorSymbol);
    verify(output > 0);
    addArc(from, to, input, output);
    printState(from);
    printState(to);
    DBG(1) << " "
           << (input != OpenFst::Epsilon ? allophoneSymbols_->Find(input) : "eps")
           << " "
           << disambiguatorSymbol << "=" << output
           << std::endl;
}

void ContextTransducerBuilder::Builder::addInputDisambiguatorArc(
        OpenFst::StateId from, OpenFst::StateId to, u32 disambiguator, OpenFst::Label output) {
    OpenFst::Label input = getAllophoneDisambiguator(disambiguator);
    addArc(from, to, input, output);
    printState(from);
    printState(to);
    DBG(1) << " "
           << "#" << disambiguator << "=" << input
           << " "
           << (output != OpenFst::Epsilon ? phoneSymbols_->Find(output) : "eps")
           << std::endl;
}

void ContextTransducerBuilder::Builder::addDisambiguatorArcs(
        OpenFst::StateId from, OpenFst::StateId to, const Allophone* input) {
    for (u32 d = 0; d < nDisambiguators_; ++d) {
        OpenFst::Label inputLabel = input ? getAllophoneLabel(input) : getAllophoneDisambiguator(d);
        addOutputDisambiguatorArc(from, to, inputLabel, d);
    }
}

template<class T>
void ContextTransducerBuilder::Builder::removeDuplicates(std::vector<T>& list) const {
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
}

void ContextTransducerBuilder::Builder::initialize() {
    size_t maxHistory = 0, maxFuture = 0;
    for (Am::AllophoneAlphabet::AllophoneList::const_iterator ai = allophoneList_->begin();
         ai != allophoneList_->end(); ++ai) {
        const Allophone* a            = *ai;
        maxHistory                    = std::max(maxHistory, a->history().size());
        maxFuture                     = std::max(maxFuture, a->future().size());
        const Bliss::Phoneme* phoneme = model_->phonemeInventory()->phoneme(a->phoneme());
        if (!phoneme->isContextDependent()) {
            if ((*ai)->boundary & (Allophone::isInitialPhone | Allophone::isFinalPhone))
                ciPhones_[phoneme->id()] = a;
            else
                contextIndependentInnerPhones_.push_back(phoneme->id());
        }
        if (a->boundary & Allophone::isInitialPhone) {
            if (a->history().size() == 0)
                initialNonCoartPhones_.push_back(phoneme->id());
            else
                initialCoartPhones_.push_back(phoneme->id());
        }
        if (a->history().size() == 0 && !(a->boundary & Allophone::isInitialPhone))
            innerNonCoartPhones_.push_back(phoneme->id());
    }
    removeDuplicates(initialCoartPhones_);
    removeDuplicates(initialNonCoartPhones_);
    removeDuplicates(innerNonCoartPhones_);
    removeDuplicates(contextIndependentInnerPhones_);

    if (nonWordTokens_) {
        nonWordTokens_->createAllophones(allophones_);
        const NonWordTokens::AllophoneMap& nonWordAllophones = nonWordTokens_->allophones();
        for (NonWordTokens::AllophoneMap::const_iterator nw = nonWordAllophones.begin();
             nw != nonWordAllophones.end(); ++nw) {
            verify(!phonemes_->phonemeInventory()->isValidPhonemeId(nw->first));
            ciPhones_[nw->first] = nw->second;
            initialNonCoartPhones_.push_back(nw->first);
        }
    }

    /* require triphone context */
    require(maxHistory <= 1);
    require(maxFuture <= 1);
}

OpenFst::VectorFst* ContextTransducerBuilder::Builder::build() {
    verify(phoneSymbols_);
    verify(allophoneSymbols_);
    stateMap_.clear();
    c_ = new OpenFst::VectorFst();
    prepare();
    for (AllophoneAlphabet::AllophoneList::const_iterator ai = allophoneList_->begin();
         ai != allophoneList_->end(); ++ai) {
        buildAllophone(*ai);
    }
    if (nonWordTokens_) {
        const NonWordTokens::AllophoneMap& nonWordAllophones = nonWordTokens_->allophones();
        for (NonWordTokens::AllophoneMap::const_iterator ai = nonWordAllophones.begin();
             ai != nonWordAllophones.end(); ++ai) {
            buildAllophone(ai->second);
        }
    }
    finalize();
    c_->SetInputSymbols(allophoneSymbols_);
    c_->SetOutputSymbols(phoneSymbols_);
    FstLib::Connect(c_);
    return c_;
}

void ContextTransducerBuilder::Builder::printState(OpenFst::StateId s) const {
    if (DBG_LEVEL > 0) {
        StateSet::const_iterator i = stateMap_.begin();
        while (i != stateMap_.end() && i->second != s)
            ++i;
        verify(i != stateMap_.end());
        const TriphoneContextAndBoundary& t = i->first;
        DBG(1) << s << "=("
               << (t.history_ == Bliss::Phoneme::term ? "#" : phonemes_->symbol(t.history_))
               << ","
               << (t.central_ == Bliss::Phoneme::term ? "#" : phonemes_->symbol(t.central_))
               << (t.boundary_ ? " *, " : ", ")
               << (int)t.disambiguator_
               << ")" << std::endl;
    }
}

/**
 * builds a deterministic context dependency transducer for triphones.
 *
 * the states store information about the two previously read phones
 * and if a final allophone was the output of the incoming arc
 *
 * the first phone P of a word in the lexicon transducer is assumed to have
 * a special index:  index(P) + initialPhoneOffset_
 *
 * in general a transition looks like this:
 *   (A,B) -- C : B{A+C} --> (B,C)
 *
 * allophones with the final tag:
 *   (A,B) -- C! : B{A+C}@f --> (B,C,boundary=true)
 * where C! is the word start phone C (see above)
 *
 * allophones with the initial tag:
 *   (B,C,boundary=true) -- D : C{B+D}@i --> (C,D)
 *
 * in the resulting transducer, the allophones are on the input and
 * phones on the output (inverse of the above)
 *
 * the resulting transducer is deterministic on the output side (phones)
 * if allowNonCrossWordTransitions == false
 *
 * phone disambiguators are converted to allophone disambiguators using
 * loop transitions on all states with an incoming final allophone
 *
 * if sequenceEndSymbol_ == eps, a epsilon transition is created from
 * every boundary state to a final state
 *
 * if sequenceEndSymbol_ != eps, the state (#,sequenceEndSymbol) is final.
 * thereby the last phone symbol does not produce an allophone symbol.
 *
 * if exploitDisambiguators_ == true, an unshifted loop is created on
 * the initial state for all CI phones. The disambiguation symbols occuring
 * directly after a CI phone is deleted.
 */
class ContextTransducerBuilder::AcrossWordBuilder : public ContextTransducerBuilder::Builder {
public:
    AcrossWordBuilder(Core::Ref<const Am::AcousticModel> m, Core::Ref<const Bliss::Lexicon> l)
            : Builder(m, l),
              allowNonCrossWord_(false),
              addSuperFinal_(false),
              exploitDisambiguators_(false),
              unshiftCiPhones_(false),
              nonPhoneSequenceEnd_(false),
              finalCiLoop_(false) {}

protected:
    virtual void prepare() {
        disambiguatorStates_.clear();
        stateMap_.clear();
        iInitial_ = getStateId(Bliss::Phoneme::term, Bliss::Phoneme::term, 0, true);
        c_->SetStart(iInitial_);
        c_->SetFinal(iInitial_, OpenFst::Weight::One());
        disambiguatorStates_.insert(iInitial_);
        OpenFst::Label sequenceEnd = getSequenceEndSymbol();
        if (sequenceEnd != OpenFst::Epsilon)
            nonPhoneSequenceEnd_ = !model_->phonemeInventory()->isValidPhonemeId(sequenceEnd);
        if (sequenceEnd == OpenFst::Epsilon || addSuperFinal_ || nonPhoneSequenceEnd_) {
            iFinal_ = getStateId(Bliss::Phoneme::term, Bliss::Phoneme::term, 1, true);
            c_->SetFinal(iFinal_, OpenFst::Weight::One());
        }
        else {
            iFinal_ = OpenFst::InvalidStateId;
        }

        OpenFst::StateId iCiState = OpenFst::InvalidStateId;
        if (exploitDisambiguators_) {
            iCiState = getStateId(Bliss::Phoneme::term, Bliss::Phoneme::term, 2, true);
            addDisambiguatorArcs(iCiState, iInitial_);
        }
        if (nonPhoneSequenceEnd_ && (exploitDisambiguators_ || unshiftCiPhones_)) {
            addArc(iInitial_, iFinal_, OpenFst::Epsilon, sequenceEnd);
        }

        // for all initial non-coarticulated allophones A{#+X} build transition
        // (#,#) -- EPS : A --> (#,A)
        // if exploitDisambiguators_ build for all CI phones P
        // (#,#) -- P{#,#} : P --> (CI)
        // if unshiftCiPhones_ build for all CI phones P
        // (#,#) -- P{#,#} : P --> (#,#)
        for (PhoneList::const_iterator i = initialNonCoartPhones_.begin();
             i != initialNonCoartPhones_.end(); ++i) {
            PhoneMap::const_iterator ci        = ciPhones_.find(*i);
            const bool               isCiPhone = ci != ciPhones_.end();
            if (isCiPhone) {
                DBG(1) << "initial CI allophone=" << allophones_->symbol(allophones_->index(ci->second)) << " phone=" << phonemes_->symbol(ci->first) << std::endl;
            }
            else {
                DBG(1) << "initial phone: " << phonemes_->symbol(*i) << std::endl;
            }
            if (isCiPhone) {
                if (exploitDisambiguators_) {
                    DBG(1) << "arc to CI-state" << std::endl;
                    addArc(iInitial_, iCiState, ci->second, ci->first, true);
                }
                else if (unshiftCiPhones_) {
                    DBG(1) << "CI loop" << std::endl;
                    addArc(iInitial_, iInitial_, ci->second, ci->first, true);
                }
                else {
                    DBG(1) << "CI initial" << std::endl;
                    addArc(iInitial_, getStateId(Bliss::Phoneme::term, *i, 0, true), 0, *i, true);
                }
                if (iFinal_ != OpenFst::InvalidStateId && finalCiLoop_) {
                    addArc(iFinal_, iFinal_, ci->second, ci->first, true);
                }
            }
            else {
                DBG(1) << "initial arc" << std::endl;
                OpenFst::StateId iTo = getStateId(Bliss::Phoneme::term, *i, 0, true);
                addArc(iInitial_, iTo, 0, *i, true);
            }
        }
    }

private:
    /**
     * Build transitions for a final allophone with empty right context A{B,#}@f
     */
    void buildFinalRightCiAllophone(OpenFst::StateId iFrom, const Allophone* a, bool ciPhone) {
        DBG(1) << "final right-ci" << std::endl;
        if (!ciPhone) {
            DBG(1) << "cd phone" << std::endl;
            if (allowNonCrossWord_) {
                // for all initial non-coarticulated phones X
                // (A,B) -- X : B{A+#} --> (#,X)
                for (PhoneList::const_iterator p = initialNonCoartPhones_.begin();
                     p != initialNonCoartPhones_.end(); ++p) {
                    OpenFst::StateId iTo = getStateId(Bliss::Phoneme::term, *p, 0, true);
                    addArc(iFrom, iTo, a, *p, true);
                }
            }
            else {
                // for all non-coarticulated phones X
                // (A,B) -- X : B{A+#} --> (#,X)
                DBG(1) << "for all ci phones" << std::endl;
                for (PhoneMap::const_iterator p = ciPhones_.begin(); p != ciPhones_.end(); ++p) {
                    OpenFst::StateId iTo = getStateId(Bliss::Phoneme::term, p->first, 0, true);
                    addArc(iFrom, iTo, a, p->first, true);
                }
            }
        }
        else {
            DBG(1) << "ci phone" << std::endl;
            // non-coarticulated phone
            verify(a->history().size() == 0);

            if (exploitDisambiguators_) {
                // (#,A) -- #_ : A{#,#} --> (#,#)
                addDisambiguatorArcs(iFrom, iInitial_, a);
            }
            else if (unshiftCiPhones_) {
                // (#,A) -- eps : A{#,#} --> (#,#)
                DBG(1) << "unshifting arc" << std::endl;
                addArc(iFrom, iInitial_, a, Bliss::Phoneme::Id(-1), false);
            }
            else {
                // (#,A) -- B : A{#,#} --> (#,B)
                for (PhoneList::const_iterator p = initialNonCoartPhones_.begin();
                     p != initialNonCoartPhones_.end(); ++p) {
                    OpenFst::StateId iTo = getStateId(Bliss::Phoneme::term, *p, 0, true);
                    addArc(iFrom, iTo, a, *p, true);
                }
            }
        }
        if (getSequenceEndSymbol() == OpenFst::Epsilon) {
            // (A,B) -- EPS : B{A+#} --> ((final))
            addArc(iFrom, iFinal_, a, Bliss::Phoneme::Id(-1), false);
        }
        else if (nonPhoneSequenceEnd_ && !(unshiftCiPhones_ && ciPhone)) {
            // (A,B) -- $ : B{A+#} --> ((final))
            DBG(1) << "arc to final" << std::endl;
            addArc(iFrom, iFinal_, getAllophoneLabel(a), getSequenceEndSymbol());
        }
    }

    /**
     * Build transitions for a non-final allophone with empty right context A{B,#}
     */
    void buildRightCiAllophone(OpenFst::StateId iFrom, const Allophone* a, bool ciPhone) {
        if (!ciPhone) {
            // empty future, but not a final allophone
            //   -> must be a phone before a CI phone inside a word
            // add transitions for all CI phones X
            // (A,B) --> X : B{A,#} --> (#,X)
            for (PhoneList::const_iterator p = contextIndependentInnerPhones_.begin();
                 p != contextIndependentInnerPhones_.end(); ++p) {
                OpenFst::StateId iTo = getStateId(Bliss::Phoneme::term, *p, 0, false);
                addArc(iFrom, iTo, a, *p, false);
            }
        }
        else {
            // context-independent phone without the final tag.
            // add transitions for intra-word non-coarticulated phones X
            // (#,P) -- X : P{#,#} --> (#,X)
            for (PhoneList::const_iterator p = innerNonCoartPhones_.begin();
                 p != innerNonCoartPhones_.end(); ++p) {
                OpenFst::StateId iTo = getStateId(Bliss::Phoneme::term, *p, 0, false);
                addArc(iFrom, iTo, a, *p, false);
            }
        }
    }

protected:
    virtual void buildAllophone(const Allophone* a) {
        DBG(1) << allophones_->symbol(allophones_->index(a)) << std::endl;
        Bliss::Phoneme::Id history =
                (a->history().size() ? a->history()[0] : Bliss::Phoneme::term);
        Bliss::Phoneme::Id central = a->central();
        Bliss::Phoneme::Id future =
                (a->future().size() ? a->future()[0] : Bliss::Phoneme::term);
        const bool       ciPhone    = isCiPhone(central);
        const bool       finalPhone = a->boundary & Allophone::isFinalPhone;
        OpenFst::StateId iFrom      = getStateId(history, central, 0,
                                            a->boundary & Allophone::isInitialPhone);

        if (finalPhone && !(exploitDisambiguators_ && ciPhone))
            disambiguatorStates_.insert(iFrom);
        if (future == Bliss::Phoneme::term) {
            if (finalPhone) {
                // empty future and final allophone
                //  -> last phone of a word without across-word context
                buildFinalRightCiAllophone(iFrom, a, ciPhone);
            }
            else {
                // empty future but not final
                // phone inside a word having context independent inner phones
                // (for example silence inside a phrase)
                buildRightCiAllophone(iFrom, a, ciPhone);
            }
        }
        else {
            // (A,B) -- C : B{A,C} --> (B,C)
            OpenFst::StateId iTo = getStateId(central, future, 0,
                                              finalPhone);
            addArc(iFrom, iTo, a, future, finalPhone);
        }
    }

    virtual void finalize() {
        // create disambiguator loops
        for (std::set<OpenFst::StateId>::const_iterator s = disambiguatorStates_.begin();
             s != disambiguatorStates_.end(); ++s) {
            addDisambiguatorArcs(*s, *s);
        }
        OpenFst::Label seqEnd = getSequenceEndSymbol();
        if (seqEnd != OpenFst::Epsilon && !nonPhoneSequenceEnd_) {
            // create final state on (#,si),
            // i.e. use last silence symbol as sequence end symbol
            // verify(isCiPhone(sequenceEndSymbol_));
            TriphoneContextAndBoundary siState(Bliss::Phoneme::term, seqEnd, 0, true);
            StateSet::const_iterator   si = stateMap_.find(siState);
            verify(si != stateMap_.end());
            if (addSuperFinal_) {
                verify(iFinal_ != Fsa::InvalidStateId);
                const Allophone* a = allophones_->allophone(
                        Allophone(seqEnd, Allophone::isInitialPhone | Allophone::isFinalPhone));
                addArc(si->second, iFinal_, a, Bliss::Phoneme::Id(-1), false);
            }
            else {
                c_->SetFinal(si->second, OpenFst::Weight::One());
            }
        }
    }

public:
    void setSuperFinalState(bool addSuperFinal) {
        addSuperFinal_ = addSuperFinal;
    }

    void setAllowNonCrossword(bool allow) {
        allowNonCrossWord_ = allow;
    }

    void setExploitDisambiguators(bool exploit) {
        exploitDisambiguators_ = exploit;
    }
    void setUnshiftCiPhones(bool unshift) {
        unshiftCiPhones_ = unshift;
    }
    void setFinalCiLoop(bool loop) {
        finalCiLoop_ = loop;
    }

private:
    bool                       allowNonCrossWord_;
    bool                       addSuperFinal_, exploitDisambiguators_, unshiftCiPhones_;
    bool                       nonPhoneSequenceEnd_;
    bool                       finalCiLoop_;
    OpenFst::StateId           iInitial_, iFinal_;
    std::set<OpenFst::StateId> disambiguatorStates_;
};

// builds a non-deterministic C transducer
// transitions:
//  (A,B) -- B{A+C} : A --> (B,C)
//  (A,B) -- B{A+C}@f : B --> (B,C,@i)
//  (A,B,@i) -- B{A+C}@i : B@i --> (B,C)
//  (#,#) -- B{#+C} : B --> (B,C)
//  (A,B) -- B{A+#} : B --> (#,#,1)
//  (#,#,1) -- CI{#+#} : CI --> (#,#,0)
//  to enforce at least one CI phone between allophones A{B+#} C{#+D}
class ContextTransducerBuilder::NonDeterministicBuilder : public ContextTransducerBuilder::Builder {
public:
    NonDeterministicBuilder(Core::Ref<const Am::AcousticModel> m, Core::Ref<const Bliss::Lexicon> l)
            : Builder(m, l), iInitial_(OpenFst::InvalidStateId) {}

    virtual void prepare() {
        stateMap_.clear();
        iInitial_ = getStateId(Bliss::Phoneme::term, Bliss::Phoneme::term, 0, true);
        c_->SetStart(iInitial_);
        c_->SetFinal(iInitial_, OpenFst::Weight::One());
        disambiguatorStates_.insert(iInitial_);
    }

    virtual void buildAllophone(const Allophone* a) {
        DBG(1) << allophones_->symbol(allophones_->index(a)) << std::endl;
        Bliss::Phoneme::Id history =
                (a->history().size() ? a->history()[0] : Bliss::Phoneme::term);
        Bliss::Phoneme::Id center = a->central();
        Bliss::Phoneme::Id future =
                (a->future().size() ? a->future()[0] : Bliss::Phoneme::term);
        const bool       ciPhone      = isCiPhone(center);
        const bool       finalPhone   = a->boundary & Allophone::isFinalPhone;
        const bool       initialPhone = a->boundary & Allophone::isInitialPhone;
        OpenFst::StateId iFrom = OpenFst::InvalidStateId, iTo = OpenFst::InvalidStateId;

        if (ciPhone) {
            // for initial CI phones:
            //   (#,#,@i,1) -- si{#+#}@i : si@i --> (#,#,@i,0)
            //   (#,#,@i,0) -- si{#+#}@i : si@i --> (#,#,@i,0)
            // for in-word CI phones:
            //   (#,#,1) -- si{#+#} : si --> (#,#,0)
            //   (#,#,0) -- si{#+#} : si --> (#,#,0)
            iFrom = getStateId(Bliss::Phoneme::term, Bliss::Phoneme::term, 1, initialPhone);
            iTo   = getStateId(Bliss::Phoneme::term, Bliss::Phoneme::term, 0, initialPhone);
            addArc(iFrom, iTo, a, center, initialPhone);
            addArc(iTo, iTo, a, center, initialPhone);
        }
        else {
            if (history == Bliss::Phoneme::term)
                iFrom = getStateId(Bliss::Phoneme::term, Bliss::Phoneme::term, 0, initialPhone);
            else
                iFrom = getStateId(history, center, 0, initialPhone);

            if (future == Bliss::Phoneme::term)
                iTo = getStateId(Bliss::Phoneme::term, Bliss::Phoneme::term, 1, finalPhone);
            else
                iTo = getStateId(center, future, 0, finalPhone);
            addArc(iFrom, iTo, a, center, initialPhone);
        }
        if (finalPhone)
            disambiguatorStates_.insert(iTo);
    }

    virtual void finalize() {
        // disambiguator loops at word boundary states
        for (std::set<OpenFst::StateId>::const_iterator s = disambiguatorStates_.begin();
             s != disambiguatorStates_.end(); ++s) {
            addDisambiguatorArcs(*s, *s);
        }
        OpenFst::Label seqEnd = getSequenceEndSymbol();
        // all boundary CI-states (#,#,@i) are final
        for (StateSet::const_iterator s = stateMap_.begin(); s != stateMap_.end(); ++s) {
            const TriphoneContextAndBoundary& tcb = s->first;
            if (tcb.boundary_ && tcb.central_ == Bliss::Phoneme::term) {
                // no "read at least one CI phone state"
                c_->SetFinal(s->second, OpenFst::Weight::One());
                if (seqEnd != OpenFst::Epsilon) {
                    addArc(s->second, s->second, OpenFst::Epsilon, seqEnd);
                }
            }
        }
    }

private:
    OpenFst::StateId           iInitial_;
    std::set<OpenFst::StateId> disambiguatorStates_;
};

class ContextTransducerBuilder::WithinWordBuilder : public ContextTransducerBuilder::Builder {
public:
    WithinWordBuilder(Core::Ref<const Am::AcousticModel> m, Core::Ref<const Bliss::Lexicon> l)
            : Builder(m, l), exploitDisambiguators_(false) {}

    void setExploitDisambiguators(bool exploit) {
        exploitDisambiguators_ = exploit;
    }

protected:
    virtual void prepare() {
        iInitial_ = getStateId(Bliss::Phoneme::term, Bliss::Phoneme::term, 0, false);
        c_->SetStart(iInitial_);

        if (exploitDisambiguators_) {
            for (u32 d = 0; d < nDisambiguators_; ++d) {
                OpenFst::StateId iDisambiguatorState =
                        getStateId(Bliss::Phoneme::term, Bliss::Phoneme::term, d + 1, false);
                OpenFst::StateId iFinalDisambiguatorState =
                        getStateId(Bliss::Phoneme::term, Bliss::Phoneme::term, d + 1, true);
                c_->SetFinal(iFinalDisambiguatorState, OpenFst::Weight::One());
                // (#d) -- #d : eps --> (#d,*)
                addInputDisambiguatorArc(iDisambiguatorState, iFinalDisambiguatorState,
                                         d, OpenFst::Epsilon);
            }
        }
        else {
            c_->SetFinal(iInitial_, OpenFst::Weight::One());
            c_->SetFinal(getStateId(Bliss::Phoneme::term, Bliss::Phoneme::term, 0, true), OpenFst::Weight::One());
        }

        for (PhoneList::const_iterator i = initialNonCoartPhones_.begin(); i != initialNonCoartPhones_.end(); ++i) {
            if (exploitDisambiguators_) {
                OpenFst::StateId iTo = getStateId(Bliss::Phoneme::term, *i, 0, false);
                // (#,#) -- a : eps --> (#,a)
                addArc(iInitial_, iTo, 0, *i, false);
                for (u32 d = 0; d < nDisambiguators_; ++d) {
                    // (#d) -- #d : A --> (#,A)
                    OpenFst::StateId iFrom = getStateId(Bliss::Phoneme::term, Bliss::Phoneme::term, d + 1, false);
                    addInputDisambiguatorArc(iFrom, iTo, d, *i);
                }
            }
            else {
                OpenFst::StateId iTo = getStateId(Bliss::Phoneme::term, *i, 0, true);
                // (#,#) -- a : eps --> (#,a,*)
                addArc(iInitial_, iTo, 0, *i, false);
                // (#,#) -- a@i : eps --> (#,a,*)
                addArc(iInitial_, iTo, 0, *i, true);
                OpenFst::StateId iFrom = getStateId(Bliss::Phoneme::term, Bliss::Phoneme::term, 0, true);
                // (#,#,*) -- a@i : eps --> (#,a,*)
                addArc(iFrom, iTo, 0, *i, true);
            }
        }
    }

    virtual void buildAllophone(const Allophone* a) {
        DBG(1) << allophones_->symbol(allophones_->index(a)) << std::endl;
        Bliss::Phoneme::Id future =
                (a->future().size() ? a->future()[0] : Bliss::Phoneme::term);
        Bliss::Phoneme::Id history =
                (a->history().size() ? a->history()[0] : Bliss::Phoneme::term);
        Bliss::Phoneme::Id central = a->central();
        OpenFst::StateId   iFrom   = OpenFst::InvalidStateId;
        if (exploitDisambiguators_) {
            iFrom = getStateId(history, a->central(), 0, false);
        }
        else {
            iFrom = getStateId(history, a->central(), 0, a->boundary & Allophone::isInitialPhone);
        }

        if (a->boundary & Allophone::isFinalPhone) {
            if (exploitDisambiguators_) {
                for (u32 d = 0; d < nDisambiguators_; ++d) {
                    OpenFst::StateId iTo   = getStateId(Bliss::Phoneme::term, Bliss::Phoneme::term, d + 1, false);
                    OpenFst::Label   input = getAllophoneLabel(a);
                    addOutputDisambiguatorArc(iFrom, iTo, input, d);
                }
            }
            else {
                if (isCiPhone(central)) {
                    addArc(iFrom, iInitial_, a, Bliss::Phoneme::Id(-1), false);
                }
                else {
                    OpenFst::StateId iTo = getStateId(Bliss::Phoneme::term, Bliss::Phoneme::term, 0, true);
                    addArc(iFrom, iTo, a, Bliss::Phoneme::Id(-1), false);
                }
            }
        }
        else {
            if (future != Bliss::Phoneme::term) {
                OpenFst::StateId iTo = getStateId(a->central(), future, 0, false);
                addArc(iFrom, iTo, a, future, false);
            }
            else {
                for (PhoneList::const_iterator p = contextIndependentInnerPhones_.begin();
                     p != contextIndependentInnerPhones_.end(); ++p) {
                    OpenFst::StateId iTo = getStateId(Bliss::Phoneme::term, *p, 0, false);
                    addArc(iFrom, iTo, a, *p, false);
                }
            }
        }
    }

private:
    OpenFst::StateId iInitial_;
    bool             exploitDisambiguators_;
};

class ContextTransducerBuilder::MonophoneBuilder : public ContextTransducerBuilder::Builder {
public:
    MonophoneBuilder(Core::Ref<const Am::AcousticModel> m, Core::Ref<const Bliss::Lexicon> l)
            : Builder(m, l) {}

protected:
    virtual void prepare() {
        state_ = getStateId(Bliss::Phoneme::term, Bliss::Phoneme::term, 0, false);
        c_->SetStart(state_);
        c_->SetFinal(state_, OpenFst::Weight::One());
        addDisambiguatorArcs(state_, state_);
    }

    virtual void buildAllophone(const Allophone* a) {
        Bliss::Phoneme::Id phone       = a->central();
        std::string        phoneSymbol = phonemes_->symbol(phone);
        if (!isCiPhone(phone)) {
            if (a->boundary & Allophone::isInitialPhone)
                phoneSymbol += LexiconBuilder::initialSuffix;
            if (a->boundary & Allophone::isFinalPhone)
                phoneSymbol += LexiconBuilder::finalSuffix;
        }
        OpenFst::Label l = phoneSymbols_->Find(phoneSymbol);
        if (l <= 0) {
            Core::Application::us()->criticalError("unknown phoneme symbols '%s'", phoneSymbol.c_str());
        }
        addArc(state_, state_, getAllophoneLabel(a), l);
    }

protected:
    OpenFst::StateId state_;
};

const Core::ParameterString ContextTransducerBuilder::paramSequenceEndSymbol(
        "sequence-end-symbol",
        "symbol to determine end of phone sequence. empty string means epsilon.",
        "");

const Core::ParameterBool ContextTransducerBuilder::paramUseSentenceEndSymbol(
        "use-sentence-end",
        "use the sentence end marker from the lexicon as sequence end symbol",
        false);

const Core::ParameterBool ContextTransducerBuilder::paramAllowNonCrossWordTransitions(
        "allow-non-crossword-transitions",
        "allow non-across-word transitions between words",
        false);

const Core::ParameterBool ContextTransducerBuilder::paramAddWordDisambiguatorLoops(
        "add-word-disambiguators",
        "add loop transitions for word disambiguators",
        false);

const Core::ParameterBool ContextTransducerBuilder::paramAddSuperFinalState(
        "add-super-final",
        "add a final state connect with an output epsilon transition",
        false);

const Core::ParameterBool ContextTransducerBuilder::paramExploitDisambiguators(
        "exploit-disambiguators",
        "exploits the phone disambiguators which are assumed after the last phone at word ends",
        false);

const Core::ParameterBool ContextTransducerBuilder::paramUnshiftCiPhones(
        "unshift-ci-phones",
        "creates un-shifted loop transitions for CI phones",
        false);

const Core::ParameterBool ContextTransducerBuilder::paramAddNonWords(
        "add-non-words",
        "add symbols and arcs for non word phones used in the lexicon transducer",
        false);

const Core::ParameterBool ContextTransducerBuilder::paramNonDeterministic(
        "non-deterministic",
        "build non-deterministic C transducer",
        false);

const Core::ParameterBool ContextTransducerBuilder::paramFinalCiLoop(
        "final-ci-loop",
        "add loop transitions for CI phones on final state",
        false);

const Core::ParameterBool ContextTransducerBuilder::paramMonophones(
        "monophones",
        "build monophonic model",
        false);

OpenFst::VectorFst* ContextTransducerBuilder::build() {
    verify(disambiguatorOffset_ >= 0);
    verify(initialPhoneOffset_ >= 0);
    verify(nDisambiguators_ >= 0);

    Builder* builder = 0;
    if (paramNonDeterministic(config)) {
        log("building non-deterministic C");
        builder = new NonDeterministicBuilder(model_, lexicon_);
    }
    else if (model_->isAcrossWordModelEnabled()) {
        if (paramAllowNonCrossWordTransitions(config))
            log("allowing non-across-word transitions");
        if (paramAddSuperFinalState(config))
            log("adding super final state");
        AcrossWordBuilder* b = new AcrossWordBuilder(model_, lexicon_);
        b->setAllowNonCrossword(paramAllowNonCrossWordTransitions(config));

        b->setSuperFinalState(paramAddSuperFinalState(config));
        if (paramExploitDisambiguators(config)) {
            log("exploiting phone disambiguators");
            b->setExploitDisambiguators(true);
        }
        else if (paramUnshiftCiPhones(config)) {
            log("using un-shifted CI transitions");
            b->setUnshiftCiPhones(true);
        }
        if (paramFinalCiLoop(config)) {
            log("adding final CI loop");
            b->setFinalCiLoop(true);
        }
        builder = b;
    }
    else if (paramMonophones(config)) {
        log("building monophone model");
        builder = new MonophoneBuilder(model_, lexicon_);
    }
    else {
        WithinWordBuilder* b = new WithinWordBuilder(model_, lexicon_);
        if (paramExploitDisambiguators(config)) {
            log("exploiting phone disambiguators");
            b->setExploitDisambiguators(true);
        }
        builder = b;
    }
    log("disambiguator offset: %d", disambiguatorOffset_);
    log("disambiguators: %d", nDisambiguators_);
    builder->setDisambiguators(nDisambiguators_, disambiguatorOffset_);
    builder->setInitialPhoneOffset(initialPhoneOffset_);
    if (paramAddWordDisambiguatorLoops(config)) {
        log("adding word disambiguator loops for %d disambiguators", nWordDisambiguators_);
        builder->setWordDisambiguators(nWordDisambiguators_);
    }
    NonWordTokens* nonWordTokens = 0;
    if (paramAddNonWords(config)) {
        log("adding non word phones");
        nonWordTokens = new NonWordTokens(select("non-word-tokens"), *lexicon_);
        nonWordTokens->init();
        builder->setAddNonWordPhones(nonWordTokens);
    }
    std::string sequenceEndSymbol = paramSequenceEndSymbol(config);
    if (paramUseSentenceEndSymbol(config)) {
        log("using sentence end symbol");
        sequenceEndSymbol = LexiconBuilder::sentenceEndSymbol;
    }
    log("sequence end symbol: '%s'", sequenceEndSymbol.c_str());
    builder->setSequenceEndSymbol(sequenceEndSymbol);

    if (!phoneSymbols_) {
        log("creating phone symbols");
        LexiconBuilder lb(select("lexicon-builder"), *lexicon_);
        lb.createSymbolTables();
        phoneSymbols_ = lb.inputSymbols()->Copy();
    }
    builder->setPhoneSymbols(phoneSymbols_);
    builder->initialize();
    OpenFst::VectorFst* c   = builder->build();
    newWordLabelOffset_     = builder->getWordLabelOffset();
    newDisambiguatorOffset_ = builder->getDisambiguatorOffset();
    delete nonWordTokens;
    delete builder;
    return c;
}

}  // namespace Wfst
}  // namespace Search
