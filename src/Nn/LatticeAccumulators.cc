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
#include "LatticeAccumulators.hh"

#include <iterator>
#include <Bliss/Fsa.hh>
#include <Bliss/Lexicon.hh>
#include <Core/Assertions.hh>
#include <Core/Types.hh>
#include <Fsa/Automaton.hh>
#include <Fsa/tAutomaton.hh>
#include <Math/CudaMatrix.hh>
#include <Math/CudaVector.hh>
#include <Mm/Types.hh>
#include <Speech/Types.hh>

#include "ClassLabelWrapper.hh"

namespace Nn {


/*
 *
 * ErrorSignalAccumulator
 *
 */

template<typename T>
ErrorSignalAccumulator<T>::ErrorSignalAccumulator(NnMatrix *errorSignal, const ClassLabelWrapper *labelWrapper) :
    errorSignal_(errorSignal),
    labelWrapper_(labelWrapper)
    {}

template<typename T>
void ErrorSignalAccumulator<T>::accumulate(Speech::TimeframeIndex t, Mm::MixtureIndex m, Mm::Weight w){
    u32 colIndex = t;
    u32 rowIndex = labelWrapper_->getOutputIndexFromClassIndex(m);
    errorSignal_->at(rowIndex, colIndex) += w;
}

/*
 *
 * AlignmentAccumulator
 *
 */

AlignmentAccumulator::AlignmentAccumulator(Math::CudaVector<u32> *alignment, const ClassLabelWrapper *labelWrapper) :
    alignment_(alignment),
    labelWrapper_(labelWrapper)
    {}

void AlignmentAccumulator::accumulate(Speech::TimeframeIndex t, Mm::MixtureIndex m, Mm::Weight w){
    u32 colIndex = t;
    u32 outputIndex = labelWrapper_->getOutputIndexFromClassIndex(m);
    // assume that alignment is not set yet
    require_eq(alignment_->at(colIndex), Core::Type<u32>::max);
    alignment_->at(colIndex) = outputIndex;
}
template class CachedAcousticAccumulator<AlignmentAccumulator>;
template class ErrorSignalAccumulator<f32>;
template class CachedAcousticAccumulator<ErrorSignalAccumulator<f32> >;
}
