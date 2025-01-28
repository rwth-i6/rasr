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
#include <Lattice/Lattice.hh>
#include <Search/LatticeAdaptor.hh>
#include <Speech/Types.hh>

namespace Search {

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

class Traceback : public std::vector<TracebackItem>, public Core::ReferenceCounted {
public:
    void                    write(std::ostream& os, Core::Ref<const Bliss::PhonemeInventory>) const;
    Fsa::ConstAutomatonRef  lemmaAcceptor(Core::Ref<const Bliss::Lexicon>) const;
    Fsa::ConstAutomatonRef  lemmaPronunciationAcceptor(Core::Ref<const Bliss::Lexicon>) const;
    Lattice::WordLatticeRef wordLattice(Core::Ref<const Bliss::Lexicon>) const;
};

}  // namespace Search

#endif  // TRACEBACK_HH
