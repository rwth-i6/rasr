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
#ifndef _OPENFST_TYPES_HH
#define _OPENFST_TYPES_HH

#include <fst/vector-fst.h>

// rename namespace to meet Sprint conventions
namespace FstLib = fst;

/**
 * The namespace OpenFst includes all entities required
 * to access and to use functionality of the OpenFst library.
 * The classes and functions of the library itself are defined
 * in the namespace FstLib
 */
namespace OpenFst {
typedef fst::StdVectorFst                  VectorFst;
typedef fst::StdArc                        Arc;
typedef fst::StdArc::StateId               StateId;
typedef fst::StdArc::Weight                Weight;
typedef fst::StdArc::Label                 Label;
typedef fst::SymbolTable                   SymbolTable;
typedef fst::StateIterator<VectorFst>      StateIterator;
typedef fst::ArcIterator<VectorFst>        ArcIterator;
typedef fst::MutableArcIterator<VectorFst> MutableArcIterator;
typedef fst::LogArc                        LogArc;
typedef fst::VectorFst<LogArc>             LogVectorFst;

static const Label   Epsilon        = 0;
static const StateId InvalidStateId = FstLib::kNoStateId;
static const Label   InvalidLabelId = FstLib::kNoLabel;

template<class A>
bool isFinalState(const FstLib::Fst<A>& fst, StateId s) {
    return fst.Final(s) != A::Weight::Zero();
}
}  // namespace OpenFst

#endif  // _OPENFST_TYPES_HH
