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
#ifndef _SEARCH_WFST_SEARCHSPACE_HH
#define _SEARCH_WFST_SEARCHSPACE_HH

#include <Search/Histogram.hh>
#include <Search/Wfst/BookKeeping.hh>
#include <Search/Wfst/LatticeGenerator.hh>
#include <Search/Wfst/Network.hh>
#include <Search/Wfst/Types.hh>
#include <Search/Wfst/WordEnd.hh>

namespace Search {
namespace Wfst {

namespace Statistics {
class AbstractCollector;
class SearchSpaceData;
}  // namespace Statistics

class BestPath;

/**
 * base class for all SearchSpace implementations defining the interface
 * used by ExpandingFsaSearch
 */
class SearchSpaceBase {
public:
    typedef Wfst::Lattice Lattice;

    virtual ~SearchSpaceBase();

    virtual void     feed(const Mm::FeatureScorer::Scorer& scorer) = 0;
    virtual void     reset()                                       = 0;
    virtual void     setSegment(const std::string& name)           = 0;
    virtual void     getTraceback(BestPath*)                       = 0;
    virtual Lattice* createLattice(OutputType)                     = 0;

    void setPruningThreshold(Score threshold) {
        pruningThreshold_ = threshold;
    }

    void setPruningLimit(u32 limit) {
        pruningLimit_ = limit;
    }

    void setPurgeInterval(u32 interval) {
        purgeInterval_ = interval;
    }

    void setPruningBins(u32 bins) {
        stateHistogram_.setBins(bins);
    }

    void setInitialEpsilonPruning(bool prune) {
        initialEpsPruning_ = prune;
    }

    void setEpsilonPruning(bool prune) {
        epsilonArcPruning_ = prune;
    }

    void setProspectivePruning(bool prune) {
        prospectivePruning_ = prune;
    }

    void setLatticePruning(Score threshold) {
        latticePruning_ = threshold;
    }

    void setMergeEpsilonPaths(bool merge) {
        mergeEpsPaths_ = merge;
    }

    void setCreateLattice(bool create, LatticeTraceRecorder::LatticeType type) {
        createLattice_ = create;
        latticeType_   = type;
    }

    void setMergeSilenceLatticeArcs(bool merge) {
        mergeSilenceArcs_ = merge;
    }

    void setWordEndPruning(bool prune, Score threshold) {
        wordEndPruning_   = prune;
        wordEndThreshold_ = threshold;
    }

    void setWordEndType(WordEndDetector::WordEndType type) {
        wordEnds_.setType(type);
    }

    void setIgnoreLastOutput(bool ignore) {
        ignoreLastOutput_ = ignore;
    }

    void setStateSequences(const StateSequenceList* list) {
        stateSequences_ = list;
    }

    void setWeightScale(f32 scale) {
        weightScale_ = scale;
    }

    void setTransitionModel(Core::Ref<const Am::AcousticModel>);
    bool setNonWordPhones(Core::Ref<const Am::AcousticModel> am, const std::vector<std::string>& phones);
    void setUseNonWordModels(u32 nNonWordModels);

    void setSilence(const StateSequence* hmm, OpenFst::Label olabel) {
        silence_       = hmm;
        silenceOutput_ = olabel;
    }

    void setLexicon(Bliss::LexiconRef lexicon) {
        lexicon_ = lexicon;
    }

    virtual void setStatistics(bool detailed) = 0;
    virtual u32  nActiveStates() const        = 0;

    u32 nActiveHyps() const {
        return nActiveHmmStateHyps_;
    }

    virtual bool init(std::string& msg);

    static SearchSpaceBase* create(NetworkType networkType, bool allowSkips, const Core::Configuration& c);

protected:
    SearchSpaceBase();

private:
    template<class N, class C>
    static SearchSpaceBase* createSearchSpace(bool allowSkips, const Core::Configuration& c);

public:
    struct MemoryUsage {
        u32 bookkeeping, stateSequences, states, arcs, epsilonArcs,
                stateHyps, arcHyps, hmmStateHyps;
        u32 sum() const {
            return bookkeeping + stateSequences + states + arcs + epsilonArcs +
                   stateHyps + arcHyps + hmmStateHyps;
        }
    };
    virtual MemoryUsage memoryUsage() = 0;

    void resetStatistics();
    void logStatistics(Core::XmlChannel& channel) const;
    friend class Statistics::SearchSpaceData;

protected:
    typedef u32                     IndexType;
    typedef IndexType               StateIndex;
    typedef IndexType               ArcIndex;
    typedef IndexType               StateHypIndex;
    typedef IndexType               ArcHypIndex;
    typedef TraceRecorder::TraceRef TraceRef;
    typedef OpenFst::Label          Label;
    typedef u8                      TransitionModelIndex;

    /**
     * hypothesis of an hmm state ("inside" of on arc).
     * inactive hmm state hypotheses have trace == InvalidTraceRef
     */
    struct HmmStateHyp {
        Score    score;
        TraceRef trace;
    };
    typedef std::vector<HmmStateHyp> HmmStateHypotheses;
    /**
     * active arc
     */
    struct ArcHyp {
        ArcIndex             arc;    /*! arc id, to associate arc hyp with arc */
        StateHypIndex        end;    /*! index of last hyp in hmmStateHypotheses_ + 1, begin is determined by end of previous arcHyp */
        StateIndex           state;  /*! state id of source state */
        StateIndex           target; /*! state id of target state */
        Label                output; /*! output label */
        Score                score;
        const StateSequence* hmm; /*! input label mapped to the StateSequence (HMM) */
    };
    typedef std::vector<ArcHyp> ArcHypotheses;
    /**
     * a state hyp may contain up to two incoming hypothesis,
     * which activated the state hyp.
     * inactive hypotheses have trace == InvalidTraceRef
     */
    struct IncomingHyp {
        TraceRef trace;
        Score    score;
    };

    typedef std::pair<StateIndex, StateHypIndex> StateToHypElement;
    struct StateToHypHash {
        size_t operator()(const StateToHypElement& e) const {
            return e.first;
        }
    };
    struct StateToHypEqal {
        bool operator()(const StateToHypElement& a, const StateToHypElement& b) const {
            return a.first == b.first;
        }
    };
    /*! @todo replace by another container */
    typedef Fsa::Hash<StateToHypElement, StateToHypHash, StateToHypEqal> StateToHypMap;
    typedef std::unordered_map<StateIndex, Score>                        StateToScoreMap;

    typedef std::unordered_map<StateIndex, TransitionModelIndex> TransitionModelMap;
    TransitionModelMap                                           stateTransitionModels_;

    std::vector<bool> wordEndHyp_;

    Score unscaledScore(Score score) const {
        return score + currentScale_;
    }

    void setWordEndHyp(StateIndex state, bool isWordEnd);

public:
    template<class H>
    static bool isActiveHyp(const H& h) {
        return (h.trace != InvalidTraceRef);
    }

protected:
    ArcHypotheses      activeArcs_, newActiveArcs_;
    StateToHypMap      stateToHyp_;
    HmmStateHypotheses hmmStateHypotheses_, newHmmStateHypotheses_;
    StateHypIndex      currentHmmStateHypBase_;
    /**! number of used elements in newHmmStateHypotheses_ */
    StateHypIndex currentHmmStateHypSize_;
    /**! number of active HMM state hypotheses */
    u32 nActiveHmmStateHyps_;
    /**! number of used elements in newActiveArcs_ */
    ArcHypIndex currentArcHypSize_;

    /**! factor for estimating the size of newHmmStateHypotheses_ relative to hmmStateHypotheses_ */
    static const size_t HmmStateSizeIncreaseFactor = 4;
    /**! size increment used when newHmmStateHypotheses_ needs to be resized */
    static const size_t HmmStateSizeIncrement = 512;
    /**! factor for estimating the size of newActiveArcs_ relative to activeArcs_ */
    static const size_t ArcSizeIncreaseFactor = 6;
    /**! size increment used when newActiveArcs_ needs to be resized */
    static const size_t ArcSizeIncrement = 512;

    Score          currentBestScore_, currentMaxScore_, currentTreshold_, currentScale_;
    TraceRef       currentSentenceEnd_;
    TimeframeIndex time_;

    Statistics::AbstractCollector*               statisticsCollector_;
    TraceRecorder*                               book_;
    const StateSequenceList*                     stateSequences_;
    std::vector<const Am::StateTransitionModel*> transitionModels_;
    Bliss::LexiconRef                            lexicon_;
    Score                                        entryForwardScore;
    Score                                        entrySkipScore;

    Score                             pruningThreshold_, latticePruning_;
    u32                               pruningLimit_;
    u32                               purgeInterval_;
    bool                              createLattice_;
    bool                              ignoreLastOutput_;
    bool                              initialEpsPruning_, epsilonArcPruning_, prospectivePruning_, mergeEpsPaths_;
    bool                              twoPassPruning_;
    bool                              mergeSilenceArcs_;
    bool                              wordEndPruning_;
    bool                              outputIsWordEnd_;
    Score                             wordEndThreshold_;
    f32                               weightScale_;
    LatticeTraceRecorder::LatticeType latticeType_;
    const StateSequence*              silence_;
    Label                             silenceOutput_;

    Histogram       stateHistogram_;
    WordEndDetector wordEnds_;

    static const Score     InvalidScore;
    static const TraceRef  InvalidTraceRef;
    static const IndexType InvalidIndex;
};

}  // namespace Wfst
}  // namespace Search

#endif  // _SEARCH_WFST_SEARCHSPACE_HH
