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
#ifndef _SEARCH_LATTICE_GENERATOR_HH
#define _SEARCH_LATTICE_GENERATOR_HH

#include <Core/Hash.hh>
#include <Lattice/Lattice.hh>
#include <OpenFst/Types.hh>
#include <Search/Wfst/BookKeeping.hh>
#include <Search/Wfst/Lattice.hh>

namespace Search {
namespace Wfst {

class StateSequenceList;

/**
 * Base class for lattice constructing TraceRecorders
 */
class LatticeTraceRecorder : public TraceRecorder {
protected:
    LatticeTraceRecorder(const StateSequenceList& hmms);

public:
    enum LatticeType { HmmLattice,
                       DetermisticHmmLattice,
                       SimpleWordLattice,
                       SimpleNonDetWordLattice,
                       WordLattice };

    /**
     * Set input and output label of silence arcs
     */
    void setSilence(const StateSequence* hmm, OpenFst::Label output);

    /**
     * Enable/disable merging of silence arcs
     */
    void setMergeSilence(bool merge) {
        mergeSilence_ = merge;
    }
    /**
     * Set threshold used for lattice pruning.
     * Derived classes have to override pruneBegin(), pruneNotify(),
     * pruneEnd(), as declared in TraceRecoder.
     */
    void setPruningThreshold(Score threshold) {
        pruningThreshold_ = threshold;
    }

    virtual ~LatticeTraceRecorder();
    virtual void clear();
    void         purgeBegin();
    void         purgeEnd();
    void         purgeNotify(TraceRef trace);

    size_t memoryUsage() const;
    /**
     * Factory function
     */
    static LatticeTraceRecorder* create(LatticeType type, const StateSequenceList& s);

protected:
    typedef Search::Wfst::Lattice               Lattice;
    typedef Lattice::Arc                        Arc;
    typedef Arc::StateId                        StateId;
    typedef Arc::Label                          Label;
    typedef Arc::Weight                         Weight;
    typedef FstLib::StateIterator<Lattice>      StateIterator;
    typedef FstLib::ArcIterator<Lattice>        ArcIterator;
    typedef FstLib::MutableArcIterator<Lattice> MutableArcIterator;

    /**
     * Convert Hmm Pointer to input label.
     */
    Label getInputLabel(const StateSequence* hmm) const;
    /**
     * Convert input label to Hmm Pointer.
     */
    const StateSequence* getHmm(Label label) const;
    /**
     * Add a state to the lattice.
     * Tries to re-use deleted states.
     */
    StateId newState();
    /**
     * Called if a state is added to lattice_.
     * Allows to maintain data structures with additional state information
     * in derived classes.
     */
    virtual void enlarge(StateId s) {}

    /**
     * Finish the lattice construction with the given trace as final state.
     * Executes following sequence of calls to virtual functions:
     *   endLattice()
     *   finalizeReverseLattice()
     *   trimLattice()
     *   reverseLattice()
     *   optimizeLattice()
     */
    virtual void finalize(TraceRef end);
    /**
     * Set end as final state.
     */
    virtual void endLattice(TraceRef end);
    /**
     * Perform optimization of lattice with reversed arcs.
     */
    virtual void finalizeReverseLattice() {}
    /**
     * Remove non-accessible states.
     */
    virtual void trimLattice() {}
    /**
     * Reverse the lattice transducer.
     */
    virtual void reverseLattice();
    /**
     * Perform final optimizations on the lattice.
     */
    virtual void optimizeLattice() {}

    /**
     * Remove epsilon arcs.
     */
    void removeEpsilon(bool connect);
    /**
     * Generate the shortest path (using OpenFst's ShortestPath)
     */
    void shortestPath(BestPath* path) const;

    const StateSequence* const hmmsBegin_;
    Lattice*                   lattice_;
    Label                      silence_, silenceOutput_;
    bool                       mergeSilence_;
    Score                      pruningThreshold_;
    std::vector<bool>          active_;

private:
    std::vector<StateId> unusedStates_;
};

/**
 * Constructs lattices with HMM input labels and word output labels.
 * Performs intermediate and final pruning.
 */
class HmmLatticeTraceRecorder : public LatticeTraceRecorder {
public:
    HmmLatticeTraceRecorder(const StateSequenceList& hmms);
    virtual ~HmmLatticeTraceRecorder() {}
    void     clear();
    TraceRef addTrace(TraceRef sibling, TraceRef predecessor,
                      OpenFst::Label output, const StateSequence* hmm,
                      TimeframeIndex time, Score score, Score arcScore, bool wordEnd);

    void updateTime(TraceRef t, TimeframeIndex time);

    bool     hasWordEndTime(const WordEndDetector& wordEnds, TraceRef end);
    void     createBestPath(const WordEndDetector& wordEnds, bool ignoreLast, TraceRef end, BestPath* path);
    Lattice* createLattice(TraceRef);

    void pruneBegin();
    void pruneEnd();
    void pruneNotify(TraceRef trace);

    size_t memoryUsage() const;

protected:
    struct StateInfo {
        Score          score, diff;
        TimeframeIndex time;
        u16            bestArc;
    };

    StateId getState(TimeframeIndex time);
    void    addArc(StateId state, const Arc& arc, Score totalScore);
    void    enlarge(StateId s);
    void    prune(const std::vector<TraceRef>& finalStates);
    void    calculatePruningScores(const std::vector<TraceRef>& finalStates);

    void mergeArc(StateId state, Arc* arc);
    void mergeEpsilonArc(Arc* arc) const;
    bool mergePredecessorArcs(StateId state, const Arc& newArc, Score score);

    void reviseSilenceLabels();
    void finalizeReverseLattice();
    void reverseLattice();
    void trimLattice();
    void optimizeLattice();
    void finalize(TraceRef end);
    void invalidateTimestamps();

    void addArc(StateId state, const Arc& arc);

    std::vector<TraceRef>  curTraces_;
    std::vector<StateInfo> states_;
    std::vector<bool>      hasEps_;
    bool                   finished_;
};

/**
 * Creates lattices like HmmLatticeTraceRecorder, but applies
 * transducer determinization afterwards.
 */
class DetermisticHmmLatticeTraceRecorder : public HmmLatticeTraceRecorder {
public:
    DetermisticHmmLatticeTraceRecorder(const StateSequenceList& hmms)
            : HmmLatticeTraceRecorder(hmms) {}
    virtual ~DetermisticHmmLatticeTraceRecorder() {}
    void createBestPath(const WordEndDetector& wordEnds, bool ignoreLast, TraceRef end, BestPath* path);

protected:
    void optimizeLattice();
    void finalizeReverseLattice();
    void trimLattice();
};

/**
 * Creates a word lattice.
 * Construction as described in A. Ljolje, F. Pereira, M. Riley
 * "Efficient General Lattice Generation and Rescoring".
 * Generates an HMM lattice first, then projects to output labels,
 * removes epsilon arcs, and determinizes.
 */
class SimpleWordLatticeRecorder : public HmmLatticeTraceRecorder {
public:
    SimpleWordLatticeRecorder(const StateSequenceList& hmms)
            : HmmLatticeTraceRecorder(hmms) {}
    virtual ~SimpleWordLatticeRecorder() {}

protected:
    void finalizeReverseLattice() {}
    void trimLattice();
    void optimizeLattice();
};

/**
 * Creates a word lattice as SimpleWordLatticeRecorder, but without
 * final determinization.
 */
class SimpleNonDetWordLatticeRecorder : public SimpleWordLatticeRecorder {
public:
    SimpleNonDetWordLatticeRecorder(const StateSequenceList& hmms)
            : SimpleWordLatticeRecorder(hmms) {}
    virtual ~SimpleNonDetWordLatticeRecorder() {}

protected:
    void optimizeLattice();
};

}  // namespace Wfst
}  // namespace Search

#endif /* _SEARCH_LATTICE_GENERATOR_HH */
