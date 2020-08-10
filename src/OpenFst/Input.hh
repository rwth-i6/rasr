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
#ifndef _OPENFST_INPUT_HH
#define INPUT_HH_

#include <Fsa/Basic.hh>
#include <Fsa/Resources.hh>
#include <Fsa/Static.hh>
#include <Fsa/Storage.hh>
#include <OpenFst/FstMapper.hh>
#include <OpenFst/Types.hh>

namespace OpenFst {

template<class A>
Core::Ref<Fsa::StaticAutomaton> convertToFsa(const FstLib::Fst<A>& f, Fsa::ConstSemiringRef s = Fsa::TropicalSemiring) {
    FstMapperAutomaton<Fsa::Semiring, A>* mapper = new FstMapperAutomaton<Fsa::Semiring, A>(&f, s);
    Fsa::ConstAutomatonRef                mapperRef(mapper);
    return Fsa::staticCopy(mapperRef);
}

bool readOpenFst(const Fsa::Resources& resources, Fsa::StorageAutomaton* f, std::istream& i);

}  // namespace OpenFst

#endif /* _OPENFST_INPUT_HH */
