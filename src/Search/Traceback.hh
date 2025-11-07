/** Copyright 2025 RWTH Aachen University. All rights reserved.
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
#ifndef TRACEBACK_HH
#define TRACEBACK_HH

#include <Bliss/Lexicon.hh>
#include <Core/ReferenceCounting.hh>
#include <Lattice/Lattice.hh>
#include <Lattice/LatticeAdaptor.hh>
#include <Speech/Types.hh>

namespace Search {

/*
 * Struct to join AM and LM score and allow element-wise operations.
 */
struct ScoreVector {
    Speech::Score acoustic, lm;
    ScoreVector(Speech::Score a, Speech::Score l)
            : acoustic(a), lm(l) {}
    operator Speech::Score() const {
        return acoustic + lm;
    };
    ScoreVector operator+(ScoreVector const& other) const {
        return ScoreVector(acoustic + other.acoustic, lm + other.lm);
    }
    ScoreVector operator-(ScoreVector const& other) const {
        return ScoreVector(acoustic - other.acoustic, lm - other.lm);
    }
    ScoreVector& operator+=(ScoreVector const& other) {
        acoustic += other.acoustic;
        lm += other.lm;
        return *this;
    }
    ScoreVector& operator-=(ScoreVector const& other) {
        acoustic -= other.acoustic;
        lm -= other.lm;
        return *this;
    }
};

/*
 * Data associated with a single traceback node
 */
struct TracebackItem {
public:
    typedef Lattice::WordBoundary::Transit Transit;

public:
    const Bliss::LemmaPronunciation* pronunciation;  // Pronunciation of lexicon lemma for lattice creation
    Speech::TimeframeIndex           time;           // Ending time
    ScoreVector                      score;          // Absolute score
    Transit                          transit;        // Final transition description
    TracebackItem(const Bliss::LemmaPronunciation* p, Speech::TimeframeIndex t, ScoreVector s, Transit te)
            : pronunciation(p), time(t), score(s), transit(te) {}
};

/*
 * Vector of TracebackItems together with some functions for conversions and IO
 */
class Traceback : public std::vector<TracebackItem>,
                  public Core::ReferenceCounted {
public:
    void                    write(std::ostream& os, Core::Ref<const Bliss::PhonemeInventory>) const;
    Fsa::ConstAutomatonRef  lemmaAcceptor(Core::Ref<const Bliss::Lexicon>) const;
    Fsa::ConstAutomatonRef  lemmaPronunciationAcceptor(Core::Ref<const Bliss::Lexicon>) const;
    Lattice::WordLatticeRef wordLattice(Core::Ref<const Bliss::Lexicon>) const;
};

/*
 * TracebackItem together with predecessor and sibling pointers.
 * Used to build the lattice after recognition.
 * Siblings are traces which will share the same lattice state (i.e. hypotheses with the same ScoringContext) but
 * have different predecessors.
 * So a trace structure like this (where "<-" indicates a predecessor and "v" indicates a sibling)
 * A  <- B
 *       v
 * A' <- B'
 * will lead to a lattice like this:
 * O - O
 *   /
 * O
 *
 * Siblings form a chain so that the last sibling in the chain only has an empty Ref as its sibling.
 * An empty Ref predecessor means that this trace will be connected to the initial lattice state.
 *
 * Note: Don't connect traces as siblings or predecessor of each other in a circular way as this may result
 * in infinite loops during traversal.
 */
class LatticeTrace : public Core::ReferenceCounted,
                     public TracebackItem {
public:
    Core::Ref<LatticeTrace> predecessor;
    Core::Ref<LatticeTrace> sibling;

    LatticeTrace(Core::Ref<LatticeTrace> const&   predecessor,
                 Bliss::LemmaPronunciation const* pronunciation,
                 Speech::TimeframeIndex           timeframe,
                 ScoreVector                      scores,
                 Transit const&                   transit);

    LatticeTrace(Speech::TimeframeIndex timeframe, ScoreVector scores, const Transit& transit);

    LatticeTrace(LatticeTrace const& other) = delete;

    /*
     * Append sibling chain to the end of the own sibling chain
     * Example: If we have sibling chains
     *
     * A -> B -> C and D -> E
     *
     * then after A.appendSibling(D) it will be
     *
     * A -> B -> C -> D -> E
     */
    void appendSiblingToChain(Core::Ref<LatticeTrace> newSibling);

    /*
     * Perform best-predecessor traceback.
     * Ordered by increasing timestep.
     */
    Core::Ref<Traceback> performTraceback() const;

    /*
     * Build a word lattice from the traces. The given trace will represent the final lattice
     * state and it is traced back along predecessors and siblings until ending up at a trace with empty predecessor
     * which represents the initial state.
     *
     * This requires that all paths lead back to a single initial trace with empty predecessor.
     * It's also required that this trace itself does have a predecessor, i.e. initial and final
     * state in the lattice are different.
     */
    Core::Ref<const LatticeAdaptor> buildWordLattice(Core::Ref<const Bliss::Lexicon> lexicon) const;

    /*
     * Write valid pronunciations of associated traceback to output stream.
     */
    void write(std::ostream& os, Core::Ref<const Bliss::PhonemeInventory> phi) const;

    /*
     * Collect lemmas of valid pronunciations of associated traceback into `lemmaSequence`.
     */
    void getLemmaSequence(std::vector<Bliss::Lemma*>& lemmaSequence) const;

    /*
     * Count number of items with valid pronunciations along associated traceback.
     */
    u32 wordCount() const;
};

/*
 * Vector of Refs to LatticeTrace
 */
class LatticeTraceback : public Core::ReferenceCounted, public std::vector<Core::Ref<LatticeTrace>> {
};

/*
 * Perform traceback on the given LatticeTrace reference. Returns a vector reference containing the best path
 * ending in the given trace.
 */
inline Core::Ref<const LatticeTraceback> performLatticeTraceback(Core::Ref<LatticeTrace> trace) {
    LatticeTraceback* traceback = new LatticeTraceback();

    while (trace) {
        traceback->push_back(trace);
        trace = trace->predecessor;
    }

    std::reverse(traceback->begin(), traceback->end());
    return Core::Ref<const LatticeTraceback>(traceback);
}

}  // namespace Search

#endif  // TRACEBACK_HH
