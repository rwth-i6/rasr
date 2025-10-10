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
#ifndef _SEARCH_SEARCH_SPACE_ANALYZER_HH
#define _SEARCH_SEARCH_SPACE_ANALYZER_HH

#include <unordered_map>

#include <Core/Channel.hh>
#include <Core/Statistics.hh>
#include <Search/Wfst/SearchSpace.hh>

namespace Search {
namespace Wfst {
namespace Statistics {

/**
 * Provides access to the private members of SearchSpaceBase
 * using a friend relationship.
 */
class SearchSpaceData {
public:
    typedef SearchSpaceBase::ArcHyp             ArcHyp;
    typedef SearchSpaceBase::ArcHypotheses      ArcHypotheses;
    typedef SearchSpaceBase::HmmStateHyp        HmmStateHyp;
    typedef SearchSpaceBase::HmmStateHypotheses HmmStateHypotheses;
    typedef SearchSpaceBase::StateIndex         StateIndex;

    SearchSpaceData(const SearchSpaceBase& ss)
            : ss_(ss) {}

    const u32 nArcs(bool newArcs) const {
        return newArcs ? ss_.currentArcHypSize_ : ss_.activeArcs_.size();
    }
    const u32 nHyps(bool newHyps) const {
        return newHyps ? ss_.currentHmmStateHypSize_ : ss_.hmmStateHypotheses_.size();
    }
    const ArcHypotheses& arcs(bool newArcs) const {
        return newArcs ? ss_.newActiveArcs_ : ss_.activeArcs_;
    }
    const HmmStateHypotheses& hyps(bool newHyps) const {
        return newHyps ? ss_.newHmmStateHypotheses_ : ss_.hmmStateHypotheses_;
    }
    const StateSequenceList* stateSequences() const {
        return ss_.stateSequences_;
    }

protected:
    const SearchSpaceBase& ss_;
};

/**
 * Interface for statistic classes.
 */
class AbstractStatistic {
public:
    virtual ~AbstractStatistic() {}
    virtual void process(const SearchSpaceBase& ss) {}
    virtual void log(Core::XmlChannel& channel) const = 0;
    virtual void reset()                              = 0;
    virtual void add(u32 value)                       = 0;
    virtual void add(f32 value)                       = 0;
};

/**
 * Adds a Core::Statistics member to the statistics interface.
 */
template<class T>
class SearchSpaceStatistic : public AbstractStatistic {
public:
    SearchSpaceStatistic(const std::string& name)
            : stat_(name) {}
    void log(Core::XmlChannel& channel) const {
        channel << stat_;
    }
    void reset() {
        stat_.clear();
    }
    void add(u32 value) {
        stat_ += static_cast<T>(value);
    }
    void add(f32 value) {
        stat_ += static_cast<T>(value);
    }

protected:
    Core::Statistics<T> stat_;
};

/**
 * Interface for statistics classes which process individual arcs.
 * Applies the decorator pattern.
 */
class DetailedStatistic : public AbstractStatistic {
protected:
    typedef SearchSpaceData::ArcHyp      ArcHyp;
    typedef SearchSpaceData::HmmStateHyp HmmStateHyp;

public:
    DetailedStatistic(AbstractStatistic* base)
            : base_(base) {}
    virtual ~DetailedStatistic() {
        delete base_;
    }

    void log(Core::XmlChannel& channel) const {
        base_->log(channel);
    }

    void reset() {
        base_->reset();
    }

    virtual void startProcessing() {}
    virtual void endProcessing() {}
    virtual void processArc(const ArcHyp& arc) {}
    virtual void processHmmStateHyp(const ArcHyp& arc, const HmmStateHyp& hyp, bool isActive, u32 hmmState) {}

    void add(u32 value) {
        base_->add(value);
    }

    void add(f32 value) {
        base_->add(value);
    }

protected:
    AbstractStatistic* base_;
};

template<class T>
class DetailedSearchSpaceStatistic : public DetailedStatistic {
public:
    DetailedSearchSpaceStatistic(const std::string& name)
            : DetailedStatistic(new SearchSpaceStatistic<T>(name)) {}
};

/**
 * Interface and generic members for the collection of search space statistics.
 */
class AbstractCollector {
public:
    enum EventType {
        beforePruning,
        afterPruning,
        afterArcExpansion,
        nEvents
    };
    static const char* EventNames[nEvents];
    AbstractCollector() {
        eventNames_[beforePruning]     = "before pruning";
        eventNames_[afterPruning]      = "after pruning";
        eventNames_[afterArcExpansion] = "after arc expansion";
    }
    virtual ~AbstractCollector() {
        for (std::vector<StatisticList>::iterator list = statistics_.begin();
             list != statistics_.end(); ++list) {
            for (StatisticList::iterator s = list->begin(); s != list->end(); ++s)
                delete *s;
        }
    }
    virtual void log(Core::XmlChannel& channel) const {
        for (std::vector<StatisticList>::const_iterator list = statistics_.begin();
             list != statistics_.end(); ++list) {
            for (StatisticList::const_iterator s = list->begin(); s != list->end(); ++s)
                (*s)->log(channel);
        }
    }
    virtual void reset() {
        for (std::vector<StatisticList>::iterator list = statistics_.begin();
             list != statistics_.end(); ++list) {
            for (StatisticList::iterator s = list->begin(); s != list->end(); ++s)
                (*s)->reset();
        }
    }
    virtual void process(EventType event) = 0;

protected:
    template<class T>
    T* registerStatistics(EventType event) {
        if (event >= statistics_.size())
            statistics_.resize(event + 1);
        T* result = new T(event, eventNames_[event]);
        statistics_[event].push_back(result);
        return result;
    }
    typedef std::vector<AbstractStatistic*> StatisticList;
    std::vector<StatisticList>              statistics_;
    const char*                             eventNames_[nEvents];
};

class ActiveArcsStatistic : public SearchSpaceStatistic<u32> {
public:
    ActiveArcsStatistic(AbstractCollector::EventType eventType, const std::string& eventName)
            : SearchSpaceStatistic<u32>("active network arcs " + eventName),
              event_(eventType) {}
    void process(const SearchSpaceBase& ss) {
        u32 nArcs = SearchSpaceData(ss).nArcs(event_ == AbstractCollector::beforePruning);
        add(nArcs);
    }

protected:
    const AbstractCollector::EventType event_;
};

class ActiveHypsStatistic : public SearchSpaceStatistic<u32> {
public:
    ActiveHypsStatistic(AbstractCollector::EventType eventType, const std::string& eventName)
            : SearchSpaceStatistic<u32>("hmm states " + eventName) {}
    void process(const SearchSpaceBase& ss) {
        add(ss.nActiveHyps());
    }
};

class ActiveStatesStatistic : public SearchSpaceStatistic<u32> {
public:
    ActiveStatesStatistic(AbstractCollector::EventType, const std::string& eventName)
            : SearchSpaceStatistic<u32>("active network states " + eventName) {}
    void process(const SearchSpaceBase& ss) {
        add(ss.nActiveStates());
    }
};

class NumHypsStatistic : public SearchSpaceStatistic<u32> {
public:
    NumHypsStatistic(AbstractCollector::EventType eventType, const std::string& eventName)
            : SearchSpaceStatistic<u32>("total hmm states" + eventName),
              event_(eventType) {}
    void process(const SearchSpaceBase& ss) {
        u32 nStates = SearchSpaceData(ss).nHyps(event_ == AbstractCollector::beforePruning);
        add(nStates);
    }

protected:
    const AbstractCollector::EventType event_;
};

class DefaultCollector : public AbstractCollector {
    typedef SearchSpaceBase SearchSpace;
    using AbstractCollector::EventType;
    typedef AbstractCollector::StatisticList StatisticList;

public:
    DefaultCollector(const SearchSpace* ss)
            : searchSpace_(ss) {
        registerStatistics<ActiveArcsStatistic>(beforePruning);
        registerStatistics<ActiveArcsStatistic>(afterPruning);
        registerStatistics<ActiveHypsStatistic>(beforePruning);
        registerStatistics<ActiveHypsStatistic>(afterPruning);
        registerStatistics<NumHypsStatistic>(beforePruning);
        registerStatistics<NumHypsStatistic>(afterPruning);
        registerStatistics<ActiveStatesStatistic>(afterPruning);
        registerStatistics<ActiveStatesStatistic>(afterArcExpansion);
    }
    virtual ~DefaultCollector() {}

    void process(EventType event) {
        StatisticList& list = statistics_[event];
        for (StatisticList::iterator stat = list.begin(); stat != list.end(); ++stat)
            (*stat)->process(*searchSpace_);
    }

protected:
    const SearchSpace* searchSpace_;
};

class UniqueHmmStatistic : public DetailedSearchSpaceStatistic<u32> {
public:
    UniqueHmmStatistic(AbstractCollector::EventType eventType, const std::string& eventName)
            : DetailedSearchSpaceStatistic<u32>("unique hmms " + eventName) {}
    void setNumHmms(u32 nHmms) {
        currentHmms_.reserve(nHmms);
    }
    void startProcessing() {
        currentHmms_.clear();
    }
    void endProcessing() {
        add(static_cast<u32>(currentHmms_.size()));
    }
    void processArc(const DetailedStatistic::ArcHyp& arc) {
        currentHmms_.insert(arc.hmm);
    }

protected:
    std::unordered_set<const StateSequence*, Core::PointerHash<StateSequence>> currentHmms_;
};

class UniqueMixtureStatistic : public DetailedSearchSpaceStatistic<u32> {
public:
    UniqueMixtureStatistic(AbstractCollector::EventType eventType, const std::string& eventName)
            : DetailedSearchSpaceStatistic<u32>("unique mixtures " + eventName) {}
    void setNumMixtures(u32 nHmms) {
        currentMixtures_.reserve(nHmms);
    }
    void startProcessing() {
        currentMixtures_.clear();
    }
    void endProcessing() {
        add(static_cast<u32>(currentMixtures_.size()));
    }
    void processHmmStateHyp(const DetailedStatistic::ArcHyp&      arc,
                            const DetailedStatistic::HmmStateHyp& hyp,
                            bool isActive, u32 hmmState) {
        if (isActive)
            currentMixtures_.insert(arc.hmm->state(hmmState).emission_);
    }

protected:
    std::unordered_set<u32> currentMixtures_;
};

class InactiveHypsStatistic : public DetailedSearchSpaceStatistic<u32> {
public:
    InactiveHypsStatistic(AbstractCollector::EventType eventType, const std::string& eventName)
            : DetailedSearchSpaceStatistic<u32>("inactive hmm states " + eventName) {}
    void startProcessing() {
        count_ = 0;
    }
    void endProcessing() {
        add(count_);
    }
    void processHmmStateHyp(const DetailedStatistic::ArcHyp& arc, const DetailedStatistic::HmmStateHyp& hyp, bool isActive, u32 hmmState) {
        count_ += !isActive;
    }

protected:
    u32 count_;
};

class StateStatistic : public DetailedSearchSpaceStatistic<u32> {
protected:
    typedef SearchSpaceData::StateIndex StateIndex;

public:
    StateStatistic(const std::string& name)
            : DetailedSearchSpaceStatistic<u32>(name) {}
    void startProcessing() {
        states_.clear();
    }
    void endProcessing() {
        add(static_cast<u32>(states_.size()));
    }
    void processArc(const DetailedStatistic::ArcHyp& arc) {
        states_.insert(getState(arc));
    }

protected:
    virtual StateIndex                   getState(const DetailedStatistic::ArcHyp& arc) const = 0;
    std::unordered_set<OpenFst::StateId> states_;
};

class SourceStateStatistic : public StateStatistic {
public:
    SourceStateStatistic(AbstractCollector::EventType eventType, const std::string& eventName)
            : StateStatistic("source states " + eventName) {}

protected:
    StateIndex getState(const DetailedStatistic::ArcHyp& arc) const {
        return arc.state;
    }
};

class TargetStateStatistic : public StateStatistic {
public:
    TargetStateStatistic(AbstractCollector::EventType eventType, const std::string& eventName)
            : StateStatistic("target states " + eventName) {}

protected:
    StateIndex getState(const DetailedStatistic::ArcHyp& arc) const {
        return arc.target;
    }
};

template<class S>
class GrammarStateStatistic : public StateStatistic {
protected:
    typedef typename S::Network Network;

public:
    GrammarStateStatistic(const std::string& name)
            : StateStatistic(name),
              network_(0) {}
    void process(const SearchSpaceBase& ss) {
        const S& searchSpace = dynamic_cast<const S&>(ss);
        network_             = searchSpace.network_;
    }

protected:
    const Network* network_;
};

template<class S>
class TargetGrammarStateStatistic : public GrammarStateStatistic<S> {
    typedef typename GrammarStateStatistic<S>::StateIndex StateIndex;
    using GrammarStateStatistic<S>::network_;

public:
    TargetGrammarStateStatistic(AbstractCollector::EventType eventType, const std::string& eventName)
            : GrammarStateStatistic<S>("target grammar states " + eventName) {}

protected:
    StateIndex getState(const DetailedStatistic::ArcHyp& arc) const {
        return network_->grammarState(arc.target);
    }
};

template<class S>
class SourceGrammarStateStatistic : public GrammarStateStatistic<S> {
    typedef typename GrammarStateStatistic<S>::StateIndex StateIndex;
    using GrammarStateStatistic<S>::network_;

public:
    SourceGrammarStateStatistic(AbstractCollector::EventType eventType, const std::string& eventName)
            : GrammarStateStatistic<S>("source grammar states " + eventName) {}

protected:
    StateIndex getState(const DetailedStatistic::ArcHyp& arc) const {
        return network_->grammarState(arc.state);
    }
};

template<class S>
class DetailedCollector : public DefaultCollector {
    typedef S SearchSpace;

public:
    DetailedCollector(const SearchSpace* ss)
            : DefaultCollector(ss) {
        countModels();
        UniqueHmmStatistic* uniqueHmm = registerDetailedStatistics<UniqueHmmStatistic>(AbstractCollector::beforePruning);
        uniqueHmm->setNumHmms(nStateSequences_);
        uniqueHmm = registerDetailedStatistics<UniqueHmmStatistic>(AbstractCollector::afterPruning);
        uniqueHmm->setNumHmms(nStateSequences_);
        UniqueMixtureStatistic* uniqueMixture = registerDetailedStatistics<UniqueMixtureStatistic>(AbstractCollector::beforePruning);
        uniqueMixture->setNumMixtures(nMixtures_);
        uniqueMixture = registerDetailedStatistics<UniqueMixtureStatistic>(AbstractCollector::afterPruning);
        uniqueMixture->setNumMixtures(nMixtures_);
        registerDetailedStatistics<InactiveHypsStatistic>(AbstractCollector::beforePruning);
        registerDetailedStatistics<InactiveHypsStatistic>(AbstractCollector::afterPruning);
        registerDetailedStatistics<TargetStateStatistic>(AbstractCollector::beforePruning);
        registerDetailedStatistics<TargetStateStatistic>(AbstractCollector::afterPruning);
        registerDetailedStatistics<SourceStateStatistic>(AbstractCollector::beforePruning);
        registerDetailedStatistics<SourceStateStatistic>(AbstractCollector::afterPruning);
        if (SearchSpace::Network::hasGrammarState()) {
            registerDetailedStatistics<TargetGrammarStateStatistic<SearchSpace>>(AbstractCollector::beforePruning);
            registerDetailedStatistics<TargetGrammarStateStatistic<SearchSpace>>(AbstractCollector::afterPruning);
            registerDetailedStatistics<SourceGrammarStateStatistic<SearchSpace>>(AbstractCollector::beforePruning);
            registerDetailedStatistics<SourceGrammarStateStatistic<SearchSpace>>(AbstractCollector::afterPruning);
        }
    }
    void countModels() {
        const StateSequenceList* stateSeqs = SearchSpaceData(*searchSpace_).stateSequences();
        require(stateSeqs);
        std::unordered_set<u32> mixtures;
        for (StateSequenceList::const_iterator hmm = stateSeqs->begin(); hmm != stateSeqs->end(); ++hmm) {
            for (u32 s = 0; s < hmm->nStates(); ++s) {
                mixtures.insert(hmm->state(s).emission_);
            }
        }
        nStateSequences_ = stateSeqs->size();
        nMixtures_       = mixtures.size();
    }

    void process(AbstractCollector::EventType event) {
        DefaultCollector::process(event);
        if (event >= detailedStats_.size() || detailedStats_[event].empty())
            return;
        DetailedStatList&                           stats = detailedStats_[event];
        typedef SearchSpaceData::ArcHypotheses      ArcHyps;
        typedef SearchSpaceData::ArcHyp             Arc;
        typedef SearchSpaceData::HmmStateHyp        HmmStateHyp;
        typedef SearchSpaceData::HmmStateHypotheses Hyps;
        const bool                                  useNew = (event == AbstractCollector::beforePruning);
        const u32                                   nArcs  = SearchSpaceData(*searchSpace_).nArcs(useNew);
        const ArcHyps&                              arcs   = SearchSpaceData(*searchSpace_).arcs(useNew);
        const Hyps&                                 hyps   = SearchSpaceData(*searchSpace_).hyps(useNew);
        for (DetailedStatList::iterator stat = stats.begin(); stat != stats.end(); ++stat)
            (*stat)->startProcessing();
        u32 hypIndex = 0;
        for (u32 a = 0; a < nArcs; ++a) {
            const Arc& arc = arcs[a];
            for (DetailedStatList::iterator stat = stats.begin(); stat != stats.end(); ++stat)
                (*stat)->processArc(arc);
            u32 hmmState = 0;
            for (; hypIndex < arc.end; ++hypIndex, ++hmmState) {
                const HmmStateHyp& hyp      = hyps[hypIndex];
                const bool         isActive = SearchSpaceBase::isActiveHyp(hyp);
                for (DetailedStatList::iterator stat = stats.begin(); stat != stats.end(); ++stat)
                    (*stat)->processHmmStateHyp(arc, hyp, isActive, hmmState);
            }
        }
        for (DetailedStatList::iterator stat = stats.begin(); stat != stats.end(); ++stat)
            (*stat)->endProcessing();
    }

protected:
    template<class T>
    T* registerDetailedStatistics(AbstractCollector::EventType event) {
        T* s = AbstractCollector::registerStatistics<T>(event);
        if (event >= detailedStats_.size())
            detailedStats_.resize(event + 1);
        detailedStats_[event].push_back(s);
        return s;
    }

    typedef std::vector<DetailedStatistic*> DetailedStatList;
    std::vector<DetailedStatList>           detailedStats_;
    u32                                     nStateSequences_;
    u32                                     nMixtures_;
};

}  // namespace Statistics
}  // namespace Wfst
}  // namespace Search

#endif  // _SEARCH_SEARCH_SPACE_ANALYZER_HH
