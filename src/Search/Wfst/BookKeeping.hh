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
#ifndef _SEARCH_BOOK_KEEPING_HH
#define _SEARCH_BOOK_KEEPING_HH
#include <Core/Assertions.hh>
#include <Search/Types.hh>
#include <OpenFst/Types.hh>

namespace Search { namespace Wfst {

class StateSequence;
class BestPath;
class Lattice;
class LatticeGenerator;
class WordEndDetector;

/**
 * Interface for book keeping classes.
 */
class TraceRecorder
{
public:
    typedef u32 TraceRef;
    static const TraceRef InvalidTraceRef = -1;

    virtual ~TraceRecorder() {}
    /**
     * remove all trace objects
     */
    virtual void clear() = 0;
    /**
     * add a new trace
     */
    virtual TraceRef addTrace(TraceRef sibling, TraceRef predecessor, OpenFst::Label output,
            const StateSequence* input, TimeframeIndex time,
            Score score, Score arcScore, bool wordEnd) = 0;

    /**
     * Update the timestamp of the given trace object
     */
    virtual void updateTime(TraceRef t, TimeframeIndex time) = 0;
    /**
     * start purging process (reset all active flags)
     */
    virtual void purgeBegin() {}
    /**
     * remove all trace objects not labeled as active
     */
    virtual void purgeEnd() {}
    /**
     * label the trace object and its predecessors as active (set active flag)
     */
    virtual void purgeNotify(TraceRef trace) {}
    /**
     * start pruning process (for lattices only)
     */
    virtual void pruneBegin() {}
    /**
     * finish pruning (for lattices only)
     */
    virtual void pruneEnd() {}
    /**
     * label the trace as currently active (for lattices only)
     */
    virtual void pruneNotify(TraceRef trace) {}

    /**
     * memory usage in bytes.
     */
    virtual size_t memoryUsage() const { return 0; }

    /**
     * Check if the number of word boundary time stamps matches the number
     * of output labels on the path ending in @c end.
     */
    virtual bool hasWordEndTime(const WordEndDetector& wordEnds, TraceRef end) = 0;

    /**
     * Find the first best path ending in @c end.
     */
    virtual void createBestPath(const WordEndDetector &wordEnds, bool ignoreLast,
                                TraceRef end, BestPath *path) = 0;

    /**
     * Create a lattice.
     */
    virtual Lattice* createLattice(TraceRef) = 0;
};

/**
 * storage of Trace objects used by ExpandingFsaSearch
 *
 * free storage is organized using a linked list.
 * trace.predecessor is used as pointer for list items
 */
class FirstBestTraceRecorder : public TraceRecorder
{
protected:
    typedef OpenFst::Label Label;
    struct Trace
    {
        TraceRef predecessor, sibling;
        Label output;
        const StateSequence *input;
        TimeframeIndex time;
        Score score;
        bool wordEnd;
        bool active;
        bool used;
        Trace() : active(false), used(false) {}
        Trace(const TraceRef _predecessor, Label _output, TimeframeIndex _time,
              Score _score, bool _wordEnd) :
                predecessor(_predecessor), sibling(InvalidTraceRef), output(_output),
                input(0), time(_time), score(_score), wordEnd(_wordEnd),
                active(true), used(true) {}
    };

public:
    FirstBestTraceRecorder(bool createLattice = false) : next_(0), createLattice_(createLattice) {}
    virtual ~FirstBestTraceRecorder() {}

    void setCreateLattice(bool create) {
        verify(next_ == 0); // called before adding elements
        createLattice_ = create;
    }
    void clear();

    TraceRef addTrace(TraceRef sibling, TraceRef predecessor, OpenFst::Label output,
                      const StateSequence* input, TimeframeIndex time,
                      Score score, Score arcScore, bool wordEnd);
    virtual void updateTime(TraceRef t, TimeframeIndex time);
    void purgeBegin();
    void purgeEnd();
    void purgeNotify(TraceRef trace);

    size_t memoryUsage() const { return data_.capacity() * sizeof(Trace); }
    bool hasWordEndTime(const WordEndDetector& wordEnds, TraceRef end);
    void createBestPath(const WordEndDetector &wordEnds, bool ignoreLast,
                        TraceRef end, BestPath *path);
    Lattice* createLattice(TraceRef end);

private:
    static const size_t incrementSize = 512;

    void updateTrace(TraceRef sibling, TraceRef predecessor, Score score);

    void purgeNotifyDfs(TraceRef trace);
    void purgeNotifyLinear(TraceRef trace);

    void enlarge();
    typedef std::vector<Trace> TraceArray;
    TraceArray data_;
    TraceRef next_;
    bool createLattice_;
};

} // namespace Wfst
} // namespace Search


#endif /* _SEARCH_BOOK_KEEPING_HH */
