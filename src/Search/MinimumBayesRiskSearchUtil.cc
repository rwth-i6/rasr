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
#include <Fsa/Arithmetic.hh>
#include <Fsa/Basic.hh>
#include <Fsa/Best.hh>
#include <Fsa/Compose.hh>
#include <Fsa/Dfs.hh>
#include <Fsa/Levenshtein.hh>
#include <Fsa/Linear.hh>
#include <Fsa/Rational.hh>
#include <Fsa/Sort.hh>
#include <Fsa/Sssp.hh>
#include <Fsa/Stack.hh>
#include <Fsa/Static.hh>
#include <set>

using Fsa::ConstAlphabetRef;
using Fsa::ConstAutomatonRef;
using Fsa::ConstSemiringRef;
using Fsa::LabelId;
using Fsa::State;
using Fsa::StateId;
using Fsa::StaticAutomaton;
using Fsa::Weight;

namespace Search {

u32 levenshteinDistance(const std::vector<LabelId>& A, const std::vector<LabelId>& B) {
    std::size_t M = A.size() + 1;
    std::size_t N = B.size() + 1;
    std::size_t m, n;
    std::size_t a, b, c;

    u32** D;

    D = new u32*[M];

    for (m = 0; m < M; ++m) {
        D[m]    = new u32[N];
        D[m][0] = m;
    }  //end for m
    for (n = 0; n < N; ++n) {
        D[0][n] = n;
    }  //end for n

    for (n = 1; n < N; ++n) {
        for (m = 1; m < M; ++m) {
            a       = (D[m - 1][n] + 1);
            b       = (D[m][n - 1] + 1);
            c       = (D[m - 1][n - 1] + (A.at(m - 1) == B.at(n - 1) ? 0 : 1));
            D[m][n] = (a <= b ? (a <= c ? a : (b <= c ? b : c)) : (b <= c ? b : c));
        }  //end for m
    }      //end for n

    u32 result = D[M - 1][N - 1];

    for (m = 0; m < M; ++m) {
        delete[] D[m];
    }
    delete[] D;

    return result;
}  //end levensthein

std::set<StateId> getContour(std::set<StateId> oldContour, ConstAutomatonRef fsa) {
    std::set<StateId> contour;
    for (std::set<StateId>::const_iterator stateId = oldContour.begin(); stateId != oldContour.end(); ++stateId) {
        Fsa::ConstStateRef         state = fsa->getState(*stateId);
        Fsa::State::const_iterator arc;
        for (arc = state->begin(); arc != state->end(); ++arc) {
            if (arc->target() != *stateId) {
                contour.insert(arc->target());
            }  //end if
        }      //end for arc
    }
    return contour;
}  //end getContour

std::vector<Fsa::StateId> getDistances(ConstAutomatonRef fsa) {
    std::vector<StateId> distances;

    std::set<StateId>                 contour;
    std::set<StateId>::const_iterator state;
    contour.insert(fsa->initialStateId());

    u32 dist = 0;
    while (!contour.empty()) {
        for (state = contour.begin(); state != contour.end(); ++state) {
            if (distances.size() <= *state) {
                distances.resize(*state + 1);
            }  //end if
            distances.at(*state) = dist;
        }  //end for state
        contour = getContour(contour, fsa);
        ++dist;
    }  //end while
    return distances;
}  //end getDistances

ConstAutomatonRef createLinearAutomatonFromVector(const std::vector<LabelId>& sequence,
                                                  const Weight&               score,
                                                  ConstAlphabetRef            inputAlphabet,
                                                  ConstAlphabetRef            outputAlphabet,
                                                  ConstSemiringRef            semiring) {
    StaticAutomaton* automaton = new StaticAutomaton(Fsa::TypeAcceptor);
    automaton->addProperties(Fsa::PropertySorted);
    automaton->addProperties(Fsa::PropertyLinear | Fsa::PropertyAcyclic);
    automaton->setInputAlphabet(inputAlphabet);
    //automaton->setOutputAlphabet ( outputAlphabet );
    automaton->setSemiring(semiring);

    State* state    = automaton->newState();
    State* newState = state;
    automaton->setInitialStateId(state->id());
    for (std::vector<LabelId>::const_iterator label = sequence.begin(); label != sequence.end(); ++label) {
        newState = automaton->newState();
        state->newArc(newState->id(), Weight(0.0), *label);
        state = newState;
    }  //end for label
    automaton->setStateFinal(state, score);
    return ConstAutomatonRef(automaton);
}  //end getMbrAutomaton

Weight getNbestNormalizationConstant(ConstAutomatonRef nbestlist) {
    Weight             normalizationConstant = Fsa::LogSemiring->zero();
    Fsa::ConstStateRef state                 = nbestlist->getState(nbestlist->initialStateId());
    for (State::const_iterator arc = state->begin(); arc != state->end(); ++arc) {
        Fsa::LogSemiring->collect(normalizationConstant, arc->weight());
    }
    return normalizationConstant;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*                  NormalizedNbestAutomaton                             */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
ConstAutomatonRef normalizeNbestlist(ConstAutomatonRef nbest) {
    Core::Ref<StaticAutomaton> nbestlist = staticCopy(nbest);

    Fsa::Accumulator* collector    = Fsa::LogSemiring->getCollector(Fsa::LogSemiring->zero());
    State*            initialState = nbestlist->fastState(nbestlist->initialStateId());
    for (State::iterator arc = initialState->begin(); arc != initialState->end(); ++arc) {
        Weight score = arc->weight_;
        // follow trace
        State* s = nbestlist->fastState(arc->target());
        while (!s->isFinal()) {
            verify(s->nArcs() == 1);
            Fsa::Arc& a = (*s->begin());
            score       = Fsa::LogSemiring->extend(score, a.weight());
            a.weight_   = Fsa::LogSemiring->one();
            s           = nbestlist->fastState(a.target_);
        }
        score        = Fsa::LogSemiring->extend(score, s->weight_);
        arc->weight_ = score;
        collector->feed(score);
    }
    Weight inverseNormalizationConstant = Fsa::LogSemiring->invert(collector->get());
    delete collector;

    for (State::iterator arc = initialState->begin(); arc != initialState->end(); ++arc) {
        arc->weight_ = Fsa::LogSemiring->extend(arc->weight(), inverseNormalizationConstant);
    }

    return ConstAutomatonRef(nbestlist);
}

class PartialNbestlist : public Fsa::ModifyAutomaton {
private:
    u32 size_;

public:
    PartialNbestlist(ConstAutomatonRef fsa, u32 size)
            : Fsa::ModifyAutomaton(fsa),
              size_(size) {}  //end PartialNbestlist

    virtual std::string describe() const {
        return "partialNbest(" + fsa_->describe() + ")";
    }

    void modifyState(State* sp) const {
        if (sp->id() == fsa_->initialStateId()) {
            sp->truncate(sp->begin() + size_);
        }
    }
};

ConstAutomatonRef partialNbestlist(ConstAutomatonRef nbestlist, u32 size) {
    return ConstAutomatonRef(new PartialNbestlist(nbestlist, size));
}  //end partialNbestlist

Weight posteriorExpectedRisk(ConstAutomatonRef center, ConstAutomatonRef hypotheses) {
    //require ( center->hasProperty ( PropertyLinear ) );
    hypotheses                             = changeSemiring(hypotheses, Fsa::TropicalSemiring);
    ConstAutomatonRef levenshteinAutomaton = Fsa::levenshtein(center, hypotheses);

    ConstAutomatonRef m1 = Fsa::composeMatching(multiply(levenshteinAutomaton, Weight(0.0)), hypotheses);
    ConstAutomatonRef m2 = Fsa::composeMatching(levenshteinAutomaton, multiply(hypotheses, Weight(0.0)));
    m2                   = Fsa::staticCopy(m2);
    m1                   = Fsa::changeSemiring(m1, Fsa::LogSemiring);
    Weight result;  // = expectation ( m1, m2 );

    return result;
}

class SentenceEndAutomaton : public Fsa::SlaveAutomaton {
public:
    SentenceEndAutomaton(ConstAutomatonRef fsa)
            : Fsa::SlaveAutomaton(fsa) {
        ConstAutomatonRef trans = transpose(fsa_);
        return;
    }  //end SlaveAutomaton

    virtual std::string describe() const {
        return "partialNbest(" + fsa_->describe() + ")";
    }  //end describe
};

Weight collectWeights(ConstSemiringRef sr, const Core::Vector<Weight>& weights) {
    Weight            result;
    Fsa::Accumulator* collector = sr->getCollector();
    for (Core::Vector<Weight>::const_iterator w = weights.begin(); w != weights.end(); ++w) {
        collector->feed(*w);
    }  //end for arc
    result = collector->get();
    delete collector;
    return result;
}  //end collectWeights

}  //end namespace Search
