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
#ifndef _SEARCH_STATE_SEQUENCE_HH
#define _SEARCH_STATE_SEQUENCE_HH

#include <Am/AcousticModel.hh>
#include <Bliss/Lexicon.hh>
#include <OpenFst/Types.hh>
#include <Fsa/Hash.hh>
#include <fst/arc-map.h>

namespace fst {
    template <class A> class MutableFst;
}

namespace Search { namespace Wfst {

class NonWordTokens;

/**
 * abstraction of an allophone HMM.
 *
 * a state sequence consists of
 *  - a sequence of states (state: emission model index, transition model index)
 *  - information about the word boundary of the allphone (initial, final)
 */
class StateSequence
{
private:
    struct State
    {
        Am::AcousticModel::EmissionIndex emission_;
        Am::AcousticModel::StateTransitionIndex transition_;

        State(Am::AcousticModel::EmissionIndex e, Am::AcousticModel::StateTransitionIndex t) :
            emission_(e), transition_(t)
        {
        }
        State() :
            emission_(0), transition_(0)
        {
        }
        bool operator==(const State &o) const
        {
            return (emission_ == o.emission_ && transition_ == o.transition_);
        }
    };
    std::vector<State> states_;
    u8 flags_;
public:
    StateSequence(u8 flags = 0) :
        flags_(flags)
    {
    }

    void appendState(Am::AcousticModel::EmissionIndex emission, Am::AcousticModel::StateTransitionIndex transition)
    {
        states_.push_back(State(emission, transition));
    }

    size_t nStates() const
    {
        return states_.size();
    }
    const State& state(u32 s) const
    {
        return states_[s];
    }

    u8 flags() const
    {
        return flags_;
    }
    void setFlags(u8 flags)
    {
        flags_ = flags;
    }
    void addFlag(u8 flag)
    {
        flags_ |= flag;
    }
    void setFinal()
    {
        flags_ |= Am::Allophone::isFinalPhone;
    }
    void setInitial()
    {
        flags_ |= Am::Allophone::isInitialPhone;
    }
    bool isInitial() const
    {
        return flags_ & Am::Allophone::isInitialPhone;
    }
    bool isFinal() const
    {
        return flags_ & Am::Allophone::isFinalPhone;
    }

    bool operator==(const StateSequence &o) const
    {
        return states_ == o.states_ && flags_ == o.flags_;
    }
    bool read(Core::BinaryInputStream &i);
    bool write(Core::BinaryOutputStream &o) const;

    struct Hash
    {
        size_t operator()(const StateSequence &ss) const
        {
            size_t key = 0;
            for (std::vector<State>::const_iterator s = ss.states_.begin(); s != ss.states_.end(); ++s)
                key = ((key << 7) | (key >> 25)) ^ (u32(s->emission_) | (u32(s->transition_) << 14));
            return key;
        }
    };

    struct IgnoreFlagsEqual
    {
        bool operator()(const StateSequence &a, const StateSequence &b) const {
            return a.states_ == b.states_;
        }
    };

    size_t memoryUsage() const
    {
        return states_.capacity() * sizeof(State);
    }

    void createFromAllophone(Core::Ref<const Am::AcousticModel>, const Am::Allophone *allophone);
};

/**
 * set of all occurring state sequences
 */
class StateSequenceList: public std::vector<StateSequence>
{
    typedef std::vector<StateSequence> Precursor;
    static const char *MAGIC;
public:
    bool read(const std::string &filename);
    bool write(const std::string &filename) const;
    size_t memoryUsage() const;

    void dump(Core::Ref<const Am::AcousticModel> am, const Bliss::LexiconRef lexicon, Core::Channel &output) const;
};

/*
 * mapping from a StateSequence to an index
 */
class StateSequenceMap
{
public:
    virtual ~StateSequenceMap() {}
    virtual Fsa::LabelId index(const StateSequence&) = 0;
    virtual const StateSequence& get(Fsa::LabelId id) const = 0;
    virtual void createStateSequenceList(StateSequenceList&) const = 0;
    virtual size_t size() const = 0;
};

/**
 * do not apply any state tying, i.e. each state sequence
 * is mapped to a unique index
 */
class UniqueStateSequenceMap : public StateSequenceMap
{
public:
    virtual Fsa::LabelId index(const StateSequence &s) {
        Fsa::LabelId index = sequences_.size();
        sequences_.push_back(s);
        return index;
    }
    virtual const StateSequence& get(Fsa::LabelId id) const {
        return sequences_[id];
    }
    virtual void createStateSequenceList(StateSequenceList &list) const {
        list.resize(size());
        std::copy(sequences_.begin(), sequences_.end(), list.begin());
    }
    virtual size_t size() const {
        return sequences_.size();
    }
private:
    std::vector<StateSequence> sequences_;
};

/**
 * tie state sequences, i.e. two state sequences having
 * the same sequence of emission models, sequence of transition
 * models and the same word boundary are assigned to the same
 * index.
 */
class TiedStateSequenceMap : public StateSequenceMap
{
public:
    virtual Fsa::LabelId index(const StateSequence &s) {
        return sequences_.insert(s);
    }
    virtual const StateSequence& get(Fsa::LabelId id) const {
        return sequences_[id];
    }
    virtual void createStateSequenceList(StateSequenceList &list) const {
        list.resize(size());
        std::copy(sequences_.begin(), sequences_.end(), list.begin());
    }
    virtual size_t size() const {
        return sequences_.size();
    }
private:
    Fsa::Hash<StateSequence, StateSequence::Hash> sequences_;
};

class FullyTiedStateSequenceMap : public StateSequenceMap
{
public:
    virtual Fsa::LabelId index(const StateSequence &s) {
        return sequences_.insert(s);
    }
    virtual const StateSequence& get(Fsa::LabelId id) const {
        return sequences_[id];
    }
    virtual void createStateSequenceList(StateSequenceList &list) const {
        list.resize(size());
        std::copy(sequences_.begin(), sequences_.end(), list.begin());
    }
    virtual size_t size() const {
        return sequences_.size();
    }
private:
    Fsa::Hash<StateSequence, StateSequence::Hash, StateSequence::IgnoreFlagsEqual> sequences_;
};


/**
 * maps an allophone to a state sequence index.
 * the list of state sequences is constructed on the fly.
 */
class AllophoneToAlloponeStateSequenceMap
{
private:
    Core::Ref<const Am::AcousticModel> model_;
    Am::ConstAllophoneAlphabetRef allophoneAlphabet_;
    Am::ConstAllophoneStateAlphabetRef allophoneStateAlphabet_;
    typedef std::unordered_map<Fsa::LabelId, Fsa::LabelId> LabelTranslationMap;
    LabelTranslationMap labelMapping_;
    StateSequenceMap *stateSequences_;
    bool removeDisambiguators_;
    static const Fsa::LabelId DisambiguatorMask = 0x40000000;
    u32 nDisambiguators_;
public:
    AllophoneToAlloponeStateSequenceMap(Core::Ref<const Am::AcousticModel> model, bool removeDisambiguators,
                                        bool tieAllophones, bool ignoreFlags = false);
    ~AllophoneToAlloponeStateSequenceMap();
    Fsa::LabelId stateSequenceIndex(Fsa::LabelId allophoneIndex);
    const StateSequenceMap& stateSequences() const { return *stateSequences_; }
    u32 size() const { return stateSequences_->size(); }
    static bool isDisambiguator(Fsa::LabelId l) { return l & DisambiguatorMask; }
    static Fsa::LabelId getDisambiguator(u32 disambiguator) { return disambiguator | DisambiguatorMask; }
};


/**
 * Create state sequences for the list of allophones
 */
class StateSequenceBuilder : public Core::Component
{
private:
    static const Core::ParameterBool paramRemoveDisambiguators;
    static const Core::ParameterBool paramTiedAllophones;
    static const Core::ParameterBool paramIgnoreFlags;
    static const Core::ParameterBool paramAddNonWords;
public:
    typedef std::vector< std::list<Fsa::LabelId> > LabelToLabelsMap;

    StateSequenceBuilder(const Core::Configuration &c,
                         Core::Ref<const Am::AcousticModel> model,
                         Bliss::LexiconRef lexicon);
    ~StateSequenceBuilder();
    void setNumDisambiguators(u32 disambiguators) { nDisambiguators_ = disambiguators; }

    void build();
    StateSequenceList* createStateSequenceList() const;

    /**
     * relabels the input symbols of f from allophone indexes to state sequence indexes.
     * also changes the input symbol table.
     */
    void relabelTransducer(FstLib::MutableFst<OpenFst::Arc> *f) const;

    /**
     * creates a symbol table with a textual representation of the tied allophones
     */
    OpenFst::SymbolTable* createSymbols() const;

    /*
     * returns a mapping from state-sequence index to list of allophone indexes
     */
    const LabelToLabelsMap& tiedAllophones() const { return labelToAllophones_; }

    /**
     * creates a transducer which rewrites sequences of emission indexes to
     * a sequence of state sequence indexes.
     * input: state sequence index
     * output: emission index (mixture set id)
     */
    OpenFst::VectorFst* createStateSequenceToEmissionTransducer() const;

    static bool isFsaDisambiguator(Fsa::LabelId label);
    static bool isDisambiguator(OpenFst::Label label);
protected:
    void addToMap(Fsa::LabelId allophone, Fsa::LabelId label);
    void addNonWordsToList(StateSequenceList *list) const;
    Core::Ref<const Am::AcousticModel> am_;
    Bliss::LexiconRef lexicon_;
    u32 nDisambiguators_;
    AllophoneToAlloponeStateSequenceMap *map_;
    LabelToLabelsMap labelToAllophones_;
    std::vector<Fsa::LabelId> allophoneToLabel_;
    bool addNonWords_;
    NonWordTokens *nonWordTokens_;
};


/**
 * converts a hmm list to a StateSequenceList.
 * the hmm list is in textual format:
 *   <hmm-symbol> <hmm-state-1-symbol> <hmm-state-2-symbol> ...
 */
class HmmListConverter : public Core::Component
{
    static const Core::ParameterString paramSilencePhone;
public:
    HmmListConverter(const Core::Configuration &c) :
        Core::Component(c), hmmSyms_(0), stateSyms_(0),
        silencePhone_(paramSilencePhone(config)) {}

    void setHmmSymbols(const OpenFst::SymbolTable *hmmSyms) {
        hmmSyms_ = hmmSyms;
    }
    void setHmmStateSymbols(const OpenFst::SymbolTable *stateSyms) {
        stateSyms_ = stateSyms;
    }

    StateSequenceList* creatStateSequenceList(const std::string &hmmListFile) const;
private:
    bool addHmm(StateSequenceList *list, const std::vector<std::string> &fields) const;
    const OpenFst::SymbolTable *hmmSyms_, *stateSyms_;
    static const s32 hmmStateOffset = -2;
    static const s32 hmmOffset = -1;
    const std::string silencePhone_;
};



/**
 * Replace disambiguators (generated by StateSequenceBuilder) by epsilon
 */
template<class A>
class HmmDisambiguatorRemoveMapper
{
public:
    HmmDisambiguatorRemoveMapper(typename A::Label replacement = OpenFst::Epsilon)
        : replacement_(replacement) {}
    A operator()(const A &arc) const {
        if (arc.ilabel != OpenFst::Epsilon && StateSequenceBuilder::isDisambiguator(arc.ilabel)) {
            A newArc = arc;
            newArc.ilabel = replacement_;
            return newArc;
        } else {
            return arc;
        }
    }
    FstLib::MapFinalAction FinalAction() const { return FstLib::MAP_NO_SUPERFINAL; }
    FstLib::MapSymbolsAction InputSymbolsAction() const { return FstLib::MAP_COPY_SYMBOLS; }
    FstLib::MapSymbolsAction OutputSymbolsAction() const { return FstLib::MAP_COPY_SYMBOLS; }
    u64 Properties(u64 props) const { return props; }

private:
    typename A::Label replacement_;
};

/**
 *
 */
class StateSequenceResolver
{
public:
    StateSequenceResolver(Core::Ref<const Am::AcousticModel> am,
                          const StateSequenceList &states) : am_(am), states_(states) {}
    const StateSequence* find(const std::string &phone, u8 boundary) const;
    const StateSequence* find(const Bliss::Phoneme::Id phone, u8 boundary) const;
    const StateSequence* find(const Bliss::Phoneme *phone, u8 boundary) const;
    const StateSequence* find(const Am::Allophone &allophone) const;
    const StateSequence* find(const Am::AllophoneIndex index) const;
    const StateSequence* find(const Am::Allophone *allophone) const;

    const StateSequence* findSilence(Core::Ref<const Bliss::Lexicon> lexicon) const;
private:
    Core::Ref<const Am::AcousticModel> am_;
    const StateSequenceList &states_;
};

} // namespace Wfst
} // namespace Search

#endif /* _SEARCH_STATE_SEQUENCE_HH */
