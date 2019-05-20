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
#include "Input.hh"
#include <Fsa/tCopy.hh>

namespace OpenFst {

bool readOpenFst(const Fsa::Resources& resources, Fsa::StorageAutomaton* f, std::istream& i) {
    if (i) {
        FstLib::FstReadOptions                                    options;
        VectorFst*                                                fst = VectorFst::Read(i, options);
        typedef FstMapperAutomaton<Fsa::Semiring, VectorFst::Arc> FstToFsaMapper;
        Core::Ref<FstToFsaMapper>                                 mapper(new FstToFsaMapper(fst, Fsa::TropicalSemiring));
        Ftl::copy(f, mapper);
        delete fst;
        return true;
    }
    return false;
}

}  // namespace OpenFst
