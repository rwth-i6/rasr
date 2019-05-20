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
#ifndef _OPENFST_UTILITY_HH
#define _OPENFST_UTILITY_HH

#include <OpenFst/Types.hh>
#include <fst/expanded-fst.h>

namespace OpenFst {

template<class F>
OpenFst::StateId findFinalState(const F& fst, bool* moreThanOne) {
    *moreThanOne           = false;
    OpenFst::StateId state = OpenFst::InvalidStateId;
    for (FstLib::StateIterator<F> siter(fst); !siter.Done(); siter.Next()) {
        if (OpenFst::isFinalState(fst, siter.Value())) {
            if (state != OpenFst::InvalidStateId) {
                *moreThanOne = true;
                break;
            }
            else {
                state = siter.Value();
            }
        }
    }
    return state;
}

template<class A>
void addArcs(FstLib::VectorFst<A>* fst, OpenFst::StateId s, const std::vector<A>& arcs) {
    for (typename std::vector<A>::const_iterator a = arcs.begin(); a != arcs.end(); ++a)
        fst->AddArc(s, *a);
}

}  // namespace OpenFst

#endif  // _OPENFST_UTILITY_HH
