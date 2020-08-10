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
#ifndef _OPENFST_BEAMS_EARCH_HH
#define _OPENFST_BEAMS_EARCH_HH

#include <OpenFst/Types.hh>
#include <unordered_map>
#include <vector>
#include <fst/connect.h>
#include <fst/fst.h>
#include <fst/reverse.h>

namespace OpenFst {

namespace {
template<class A>
struct ShortestPathHyp {
    typename A::Weight  weight;
    typename A::StateId state, trace;
};
}  // namespace

/**
 * Arc-synchronous breadth-first beam search
 */
template<class A>
void shortestPaths(const FstLib::Fst<A>& fst, typename A::Weight beam, FstLib::VectorFst<A>* lattice) {
    typedef FstLib::Fst<A>      Fst;
    typedef typename A::StateId StateId;
    typedef typename A::Weight  Weight;
    typedef typename A::Label   Label;
    typedef ShortestPathHyp<A>  Hyp;
    std::vector<Hyp>            active, newActive;
    verify(Weight::Properties() & FstLib::kPath);
    FstLib::VectorFst<A>                        traceback;
    typedef std::unordered_map<StateId, size_t> StateToHypMap;
    StateToHypMap                               stateToHyp;
    lattice->DeleteStates();
    Hyp start;
    start.state  = fst.Start();
    start.weight = Weight::One();
    start.trace  = traceback.AddState();
    traceback.SetStart(start.trace);
    active.push_back(start);
    Hyp bestFinal;
    bestFinal.state  = FstLib::kNoStateId;
    bestFinal.weight = Weight::Zero();
    bestFinal.trace  = traceback.AddState();
    while (!active.empty() && bestFinal.state == FstLib::kNoStateId) {
        // expand active hypotheses
        Weight best = Weight::Zero();
        for (typename std::vector<Hyp>::const_iterator hyp = active.begin(); hyp != active.end(); ++hyp) {
            // determine best active final state if any
            if (fst.Final(hyp->state) != Weight::Zero()) {
                Weight finalWeight = FstLib::Times(hyp->weight, fst.Final(hyp->state));
                if (bestFinal.weight != FstLib::Plus(bestFinal.weight, finalWeight)) {
                    bestFinal.weight = FstLib::Plus(bestFinal.weight, finalWeight);
                    bestFinal.state  = hyp->state;
                }
                traceback.AddArc(bestFinal.trace, A(0, 0, fst.Final(hyp->state), hyp->trace));
            }
            // expand hypotheses
            for (FstLib::ArcIterator<Fst> aiter(fst, hyp->state); !aiter.Done(); aiter.Next()) {
                Hyp newHyp;
                A   arc                                  = aiter.Value();
                newHyp.state                             = arc.nextstate;
                newHyp.weight                            = FstLib::Times(hyp->weight, arc.weight);
                typename StateToHypMap::const_iterator i = stateToHyp.find(newHyp.state);
                if (i != stateToHyp.end()) {
                    // recombine hypotheses
                    Hyp& h       = newActive[i->second];
                    newHyp.trace = h.trace;
                    if (h.weight != FstLib::Plus(h.weight, newHyp.weight)) {
                        h = newHyp;
                    }
                }
                else {
                    stateToHyp.insert(typename StateToHypMap::value_type(newHyp.state, newActive.size()));
                    newHyp.trace = traceback.AddState();
                    newActive.push_back(newHyp);
                }
                // add backpointers
                arc.nextstate = hyp->trace;
                traceback.AddArc(newHyp.trace, arc);

                // keep track of the best score (for pruning)
                if (best != FstLib::Plus(best, newHyp.weight))
                    best = FstLib::Plus(best, newHyp.weight);
            }  // for aiter
        }      // for hyp
        // prune hypotheses
        active.clear();
        Weight threshold = FstLib::Times(best, beam);
        for (typename std::vector<Hyp>::const_iterator hyp = newActive.begin(); hyp != newActive.end(); ++hyp) {
            if (threshold != FstLib::Plus(hyp->weight, threshold))
                active.push_back(*hyp);
        }
        /*! @todo garbage collection in the traceback: remove non-reachable states */
        newActive.clear();
        stateToHyp.clear();
    }  // while
    if (bestFinal.state != FstLib::kNoStateId) {
        // create lattice
        traceback.SetFinal(traceback.Start(), Weight::One());
        traceback.SetStart(bestFinal.trace);
        FstLib::Connect(&traceback);
        FstLib::Reverse(traceback, lattice);
    }
    else {
        // no final state found
    }
}

}  // namespace OpenFst

#endif  // _OPENFST_BEAMS_EARCH_HH
