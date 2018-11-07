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
#include "MmiSegmentwiseNnTrainer.hh"

#include <Core/Assertions.hh>
#include <Core/ReferenceCounting.hh>
#include <Core/Types.hh>
#include <Fsa/Arithmetic.hh>
#include <Fsa/Basic.hh>
#include <Fsa/Cache.hh>
#include <Fsa/Semiring.hh>
#include <Fsa/Sssp.hh>
#include <Lattice/Lattice.hh>
#include <Speech/AuxiliarySegmentwiseTrainer.hh>

namespace Nn {


template<typename T>
MmiSegmentwiseNnTrainer<T>::MmiSegmentwiseNnTrainer(const Core::Configuration &c) :
    Core::Component(c),
    Precursor(c)
{ }

template<typename T>
Speech::PosteriorFsa MmiSegmentwiseNnTrainer<T>::getDenominatorPosterior(Lattice::ConstWordLatticeRef lattice){
    Speech::PosteriorFsa result;
    result.fsa = Fsa::posterior64(
            Fsa::changeSemiring(lattice->part(this->part_), Fsa::LogSemiring),
            result.totalInv,
            this->posteriorTolerance());
    if (Core::isAlmostEqualUlp(f32(result.totalInv), Core::Type<f32>::min, this->posteriorTolerance())) {
        this->log("discard segment because it has vanishing total flow");
        return Speech::PosteriorFsa();
    }
    result.fsa = Fsa::expm(result.fsa);
    return result;
}


// TODO why caching ? just because the FSA is small? on the other hand, the FSA is processed only once
template<typename T>
Speech::PosteriorFsa MmiSegmentwiseNnTrainer<T>::getNumeratorPosterior(Lattice::ConstWordLatticeRef lattice)
{
    Speech::PosteriorFsa result;
    result.fsa = Fsa::posterior64(
            Fsa::changeSemiring(lattice->part(this->part_), Fsa::LogSemiring),
            result.totalInv,
            this->posteriorTolerance());

    result.fsa = Fsa::cache(Fsa::expm(result.fsa));

    return result;
}



template<typename T>
bool MmiSegmentwiseNnTrainer<T>::computeInitialErrorSignal(Lattice::ConstWordLatticeRef lattice, Lattice::ConstWordLatticeRef numeratorLattice,
        Bliss::SpeechSegment *segment, T &objectiveFunction, bool objectiveFunctionOnly){
    require(numeratorLattice);
    Speech::PosteriorFsa numeratorPosterior;
    Speech::PosteriorFsa denominatorPosterior;
    denominatorPosterior = getDenominatorPosterior(lattice);
    if (!denominatorPosterior) {
        this->log("failed to compute denominator posterior FSA, skipping segment");
        return false;
    }
    objectiveFunction = f32(denominatorPosterior.totalInv);
    u32 nRejectedObsInSeq = 0;
    if (!objectiveFunctionOnly){
        this->accumulateStatisticsOnLattice(denominatorPosterior.fsa, lattice->wordBoundaries(), 1.0);
        // frame rejection heuristic described in Vesely et al: Sequence-discriminative training of DNNs, in Interspeech 2013
        if (this->frameRejectionThreshold_ > 0){
            for (u32 t = 0; t < alignment_.size(); ++t){
                verify_ge(errorSignal_.back().at(alignment_.at(t), t), 0);
                if (errorSignal_.back().at(alignment_.at(t), t) < this->frameRejectionThreshold_ ){
                    weights_.at(t) = 0.0;
                    ++nRejectedObsInSeq;
                }
            }
        }
    }
    this->numberOfRejectedObservations_ += nRejectedObsInSeq;
    this->log("denominator-lattice-objective-function: ") << objectiveFunction;

    numeratorPosterior = getNumeratorPosterior(numeratorLattice);
    if (!numeratorPosterior) {
        this->log("failed to compute numerator posterior FSA, skipping segment");
        return false;
    }
    objectiveFunction -= f32(numeratorPosterior.totalInv);
    if (!objectiveFunctionOnly)
        this->accumulateStatisticsOnLattice(numeratorPosterior.fsa, numeratorLattice->wordBoundaries(), -1.0);

    if (!objectiveFunctionOnly && this->frameRejectionThreshold_ > 0)
        this->log("rejected ") << nRejectedObsInSeq << " out of " << alignment_.size() << " observations (" << 100.0 * nRejectedObsInSeq / alignment_.size() << "%)";
    this->log("numerator-lattice-objective-function: ") << f32(numeratorPosterior.totalInv);
    this->log("MMI-objective-function: ") << objectiveFunction;
    return true;
}


template class MmiSegmentwiseNnTrainer<f32>;
//template class MmiSegmentwiseNnTrainer<f64>;

}
