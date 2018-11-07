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
#ifndef _OPENFST_COUNT_HH
#define _OPENFST_COUNT_HH

#include "Types.hh"
#include <Fsa/hInfo.hh>

namespace OpenFst
{

template<class A>
Fsa::AutomatonCounts count(const FstLib::Fst<A> &f)
{
    Fsa::AutomatonCounts counts;
    for (FstLib::StateIterator< FstLib::Fst<A> > s(f); !s.Done(); s.Next()) {
        ++counts.nStates_;
        StateId state = s.Value();
        if (f.Final(state) != A::Weight::Zero())
            ++counts.nFinals_;
        for (FstLib::ArcIterator< FstLib::Fst<A> > a(f, state); !a.Done(); a.Next()) {
            const A &arc = a.Value();
            ++counts.nArcs_;
            if (arc.ilabel == Epsilon && arc.olabel == Epsilon)
                ++counts.nIoEps_;
            if (arc.ilabel == Epsilon)
                ++counts.nIEps_;
            if (arc.olabel == Epsilon)
                ++counts.nOEps_;
        }
    }
    return counts;
}

template<class A>
u32 maxLabelId(const FstLib::Fst<A> &f, bool inputLabel)
{
    u32 maxLabel = 0;
    for (FstLib::StateIterator< FstLib::Fst<A> > s(f); !s.Done(); s.Next()) {
        for (FstLib::ArcIterator< FstLib::Fst<A> > a(f, s.Value()); !a.Done(); a.Next()) {
            const A &arc = a.Value();
            const typename A::Label l = (inputLabel ? arc.ilabel : arc.olabel);
            if (l > maxLabel) maxLabel = l;
        }
    }
    return maxLabel;
}

template<class A>
class InDegree
{
    typedef A Arc;
public:
    InDegree(const FstLib::Fst<A> &f) { computeInDegree(f); }

    u32 operator[](u32 state) const {
        verify(state < inDegree_.size());
        return inDegree_[state];
    }

private:
    void computeInDegree(const FstLib::Fst<A> &f) {
        for (FstLib::StateIterator< FstLib::Fst<A> > si(f); !si.Done(); si.Next()) {
            for (FstLib::ArcIterator< FstLib::Fst<A> > ai(f, si.Value()); !ai.Done(); ai.Next()) {
                const Arc &arc = ai.Value();
                if (arc.nextstate >= inDegree_.size())
                    inDegree_.resize(arc.nextstate + 1, 0);
                ++inDegree_[arc.nextstate];
            }
        }
    }

    std::vector<u32> inDegree_;
};

}
#endif // _OPENFST_COUNT_HH
