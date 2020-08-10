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
#ifndef _SEARCH_LABEL_MAPPER_HH
#define _SEARCH_LABEL_MAPPER_HH

#include <OpenFst/Types.hh>
#include <fst/arc-map.h>

namespace Search {
namespace Wfst {

/**
 * Transforms an acceptor to a transducer by mapping input labels
 * which represent word (or lemma pronunciation) labels to output labels.
 * The pseudo input labels l to be transformed are assumed to have
 *   wordLabelOffset <= l < wordLabelOffset + disambiguatorOffset
 * The output labels are created as (l - wordLabelOffset).
 */
template<class A>
class RestoreOutputLabelMapper {
    typedef A Arc;

public:
    RestoreOutputLabelMapper(int wordLabelOffset, int disambiguatorOffset)
            : wordLabelOffset_(wordLabelOffset), disambiguatorStart_(disambiguatorOffset) {}

    Arc operator()(const Arc& arc) const {
        // transform only acceptors
        verify(arc.ilabel == arc.olabel);
        Arc newArc = arc;
        if (arc.ilabel >= wordLabelOffset_ && arc.ilabel < disambiguatorStart_) {
            newArc.ilabel = OpenFst::Epsilon;
            newArc.olabel -= wordLabelOffset_;
        }
        else {
            newArc.olabel = OpenFst::Epsilon;
        }
        return newArc;
    }
    FstLib::MapFinalAction FinalAction() const {
        return FstLib::MAP_NO_SUPERFINAL;
    }
    FstLib::MapSymbolsAction InputSymbolsAction() const {
        return FstLib::MAP_COPY_SYMBOLS;
    }
    FstLib::MapSymbolsAction OutputSymbolsAction() const {
        return FstLib::MAP_CLEAR_SYMBOLS;
    }
    uint64 Properties(u64 props) const {
        return props & FstLib::kILabelInvariantProperties & FstLib::kOLabelInvariantProperties;
    }

private:
    int wordLabelOffset_;
    int disambiguatorStart_;
};

/**
 * Removes disambiguator labels by replacing them with Epsilon.
 * Relabels all arcs with
 * disambiguatorMin <= ilabel <= disambiguatorMax.
 */
template<class A>
class RemoveDisambiguatorMapper {
    typedef A Arc;

public:
    RemoveDisambiguatorMapper(int disambiguatorMin, int disambiguatorMax)
            : disambiguatorMin_(disambiguatorMin), disambiguatorMax_(disambiguatorMax) {}

    Arc operator()(const Arc& arc) const {
        if (arc.ilabel >= disambiguatorMin_ && arc.ilabel <= disambiguatorMax_) {
            return Arc(OpenFst::Epsilon, arc.olabel, arc.weight, arc.nextstate);
        }
        else {
            return arc;
        }
    }
    FstLib::MapFinalAction FinalAction() const {
        return FstLib::MAP_NO_SUPERFINAL;
    }
    FstLib::MapSymbolsAction InputSymbolsAction() const {
        return FstLib::MAP_COPY_SYMBOLS;
    }
    FstLib::MapSymbolsAction OutputSymbolsAction() const {
        return FstLib::MAP_COPY_SYMBOLS;
    }
    uint64 Properties(u64 props) const {
        return props & FstLib::kILabelInvariantProperties;
    }

private:
    int disambiguatorMin_, disambiguatorMax_;
};

template<class A>
void pushOutputLabels(FstLib::MutableFst<A>* f) {
    typedef typename A::StateId StateId;
    std::stack<StateId>         state_queue;
    std::vector<bool>           visited;
    StateId                     initial = f->Start();
    visited.resize(initial + 1, false);
    state_queue.push(initial);
    std::vector<A> newArcs;
    while (!state_queue.empty()) {
        StateId s = state_queue.top();
        state_queue.pop();
        while (visited.size() <= s)
            visited.push_back(false);
        if (visited[s])
            continue;
        visited[s] = true;
        typedef FstLib::ArcIterator<FstLib::Fst<A>> ConstArcIter;
        newArcs.clear();
        bool changedArcs = false;
        for (ConstArcIter aiter(*f, s); !aiter.Done(); aiter.Next()) {
            const A& arc = aiter.Value();
            if (arc.olabel != OpenFst::Epsilon) {
                verify(arc.ilabel == OpenFst::Epsilon);
                for (ConstArcIter niter(*f, arc.nextstate); !niter.Done(); niter.Next()) {
                    const A& nextArc = niter.Value();
                    A        newArc  = nextArc;
                    newArc.weight    = FstLib::Times(arc.weight, nextArc.weight);
                    verify(newArc.olabel == OpenFst::Epsilon);
                    newArc.olabel = arc.olabel;
                    newArcs.push_back(newArc);
                    changedArcs = true;
                    if (newArc.nextstate >= visited.size() || !visited[newArc.nextstate])
                        state_queue.push(newArc.nextstate);
                }
            }
            else {
                if (arc.nextstate >= visited.size() || !visited[arc.nextstate]) {
                    state_queue.push(arc.nextstate);
                }
                newArcs.push_back(arc);
            }
        }
        if (changedArcs) {
            f->DeleteArcs(s);
            for (typename std::vector<A>::const_iterator a = newArcs.begin();
                 a != newArcs.end(); ++a) {
                f->AddArc(s, *a);
            }
        }
    }
    FstLib::Connect(f);
}

}  // namespace Wfst
}  // namespace Search

#endif /* _SEARCH_LABEL_MAPPER_HH */
