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

#include "Traceback.hh"
#include <Lattice/LatticeAdaptor.hh>
#include <Speech/Types.hh>
#include <stack>

namespace Search {

void Traceback::write(std::ostream& os, Core::Ref<const Bliss::PhonemeInventory> phi) const {
    for (const_iterator tbi = begin(); tbi != end(); ++tbi) {
        os << "t=" << std::setw(5) << tbi->time << "    s=" << std::setw(8) << tbi->score;
        if (tbi->pronunciation) {
            os << "    "
               << std::setw(20) << std::setiosflags(std::ios::left)
               << tbi->pronunciation->lemma()->preferredOrthographicForm()
               << "    "
               << "/" << tbi->pronunciation->pronunciation()->format(phi) << "/";
        }
        os << "    "
           << ((tbi->transit.final == Bliss::Phoneme::term) ? "#" : phi->phoneme(tbi->transit.final)->symbol().str())
           << "|"
           << ((tbi->transit.initial == Bliss::Phoneme::term) ? "#" : phi->phoneme(tbi->transit.initial)->symbol().str())
           << std::endl;
    }
}

Fsa::ConstAutomatonRef Traceback::lemmaAcceptor(Core::Ref<const Bliss::Lexicon> lexicon) const {
    Bliss::LemmaAcceptor* result = new Bliss::LemmaAcceptor(lexicon);
    Fsa::State *          s1, *s2;
    s1 = result->newState();
    result->setInitialStateId(s1->id());
    for (u32 i = 0; i < size(); ++i)
        if ((*this)[i].pronunciation) {
            s2 = result->newState();
            s1->newArc(s2->id(), result->semiring()->one(), (*this)[i].pronunciation->lemma()->id());
            s1 = s2;
        }
    result->setStateFinal(s1);
    return Fsa::ConstAutomatonRef(result);
}

Fsa::ConstAutomatonRef Traceback::lemmaPronunciationAcceptor(Core::Ref<const Bliss::Lexicon> lexicon) const {
    Bliss::LemmaPronunciationAcceptor*       result = new Bliss::LemmaPronunciationAcceptor(lexicon);
    const Bliss::LemmaPronunciationAlphabet* abet   = result->lemmaPronunciationAlphabet();
    Fsa::State *                             s1, *s2;
    s1 = result->newState();
    result->setInitialStateId(s1->id());
    for (u32 i = 0; i < size(); ++i) {
        if (!(*this)[i].pronunciation)
            continue;
        s2 = result->newState();
        s1->newArc(s2->id(), result->semiring()->one(), abet->index((*this)[i].pronunciation));
        s1 = s2;
    }
    result->setStateFinal(s1);
    return Fsa::ConstAutomatonRef(result);
}

Lattice::WordLatticeRef Traceback::wordLattice(Core::Ref<const Bliss::Lexicon> lexicon) const {
    // NOTE: This function returns a word lattice with dummy word boundaries
    Lattice::WordLatticeRef result(new Lattice::WordLattice());
    result->setFsa(lemmaPronunciationAcceptor(lexicon), Lattice::WordLattice::acousticFsa);
    result->setWordBoundaries(Core::ref(new Lattice::WordBoundaries));
    return result;
}

LatticeTrace::LatticeTrace(
        Core::Ref<LatticeTrace> const&        pre,
        const Bliss::LemmaPronunciation*      p,
        Speech::TimeframeIndex                t,
        ScoreVector                           s,
        Search::TracebackItem::Transit const& transit)
        : TracebackItem(p, t, s, transit), predecessor(pre), sibling() {}

void LatticeTrace::appendSiblingToChain(Core::Ref<LatticeTrace> sibling) {
    if (sibling_) {
        sibling_.appendSiblingToChain(sibling);
    }
    else {
        sibling_ = sibling;
    }
}

Core::Ref<Traceback> LatticeTrace::getTraceback() const {
    Core::Ref<Traceback> traceback;

    if (predecessor) {
        traceback = predecessor->getTraceback();
    }
    else {
        traceback = Core::ref(new Traceback());
        traceback->push_back(TracebackItem(0, 0, {0, 0}, {}));
    }
    traceback->push_back(*this);

    return traceback;
}

Core::Ref<const LatticeAdaptor> LatticeTrace::buildWordLattice(Core::Ref<const Bliss::Lexicon> lexicon) {
    // use default LemmaAlphabet mode of StandardWordLattice
    Core::Ref<Lattice::StandardWordLattice> result(new Lattice::StandardWordLattice(lexicon));
    Core::Ref<Lattice::WordBoundaries>      wordBoundaries(new Lattice::WordBoundaries);

    // Map traces to lattice states
    std::unordered_map<const LatticeTrace*, Fsa::State*> stateMap;

    // Create an initial State at time 0 which represents empty predecessors
    Fsa::State* initialState = result->initialState();
    wordBoundaries->set(initialState->id(), Lattice::WordBoundary(0));

    // Stack for depth-first search through traces of all hypotheses in the beam
    std::stack<const LatticeTrace*> traceStack;

    // Create a final state which represents this trace itself
    Fsa::State* finalState = result->finalState();
    stateMap[this]         = finalState;
    traceStack.push(this);
    wordBoundaries->set(finalState->id(), Lattice::WordBoundary(this->time));

    // Perform depth-first search
    Fsa::State *preState, currentState;
    while (not traceStack.empty()) {
        auto* trace = traceStack.top();
        traceStack.pop();

        // A trace on the stack already has an associated state
        currentState = stateMap[trace];
        wordBoundaries->set(currentState->id(), Lattice::WordBoundary(trace->time));

        // Iterate through siblings of current trace
        // All siblings share the same lattice state
        for (auto arcTrace = trace; arcTrace; arcTrace = arcTrace->sibling) {
            // For current sibling, get its predecessor, create a state for that predecessor
            // and connect it to the current state.
            auto*       preTrace = arcTrace->predecessor.get();
            ScoreVector scores   = trace->score;
            if (preTrace == nullptr) {
                // If trace has no predecessor, it gets connected to the initial state
                preState = initialState;
            }
            else {
                // If trace has a predecessor, get or create a state for it. Arc score
                // is difference between trace scores from predecessor to current.
                scores -= preTrace->score;
                if (stateMap.find(preTrace) == stateMap.end()) {
                    preState           = result->newState();
                    stateMap[preTrace] = preState;
                    traceStack.push(preTrace);
                }
                else {
                    preState = stateMap[preTrace.get()];
                }
            }

            // Create arc from predecessor state to current state
            result->newArc(preState, currentState, arcTrace->pronunciation, scores.acoustic, scores.lm);
        }
    }

    result->setWordBoundaries(wordBoundaries);
    result->addAcyclicProperty();

    return Core::ref(new Lattice::WordLatticeAdaptor(result));
}

}  // namespace Search
