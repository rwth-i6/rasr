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
#include "MeSegmentwiseNnTrainer.hh"

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

/*
 *
 * Minimum Error Training
 *
 *
 */

template<typename T>
Core::ParameterString MinimumErrorSegmentwiseNnTrainer<T>::paramAccuracyName("accuracy-name", "name of lattice with accuracies", Lattice::WordLattice::accuracyFsa);

template<typename T>
MinimumErrorSegmentwiseNnTrainer<T>::MinimumErrorSegmentwiseNnTrainer(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c),
          accuracyPart_(paramAccuracyName(c)) {}

template<typename T>
Speech::PosteriorFsa MinimumErrorSegmentwiseNnTrainer<T>::getDenominatorPosterior(Lattice::ConstWordLatticeRef lattice) {
    Speech::PosteriorFsa result;

    result.fsa = Fsa::posteriorE(Fsa::changeSemiring(lattice->part(this->part_), Fsa::LogSemiring),
                                 lattice->part(accuracyPart_),
                                 result.totalInv,
                                 true,
                                 this->posteriorTolerance());

    if (Core::isAlmostEqualUlp(f32(result.totalInv), Core::Type<f32>::min, this->posteriorTolerance())) {
        this->log("discard segment because it has vanishing total flow");
        return Speech::PosteriorFsa();
    }
    return result;
}

template<typename T>
bool MinimumErrorSegmentwiseNnTrainer<T>::computeInitialErrorSignal(Lattice::ConstWordLatticeRef lattice, Lattice::ConstWordLatticeRef numeratorLattice,
                                                                    Bliss::SpeechSegment* segment, T& objectiveFunction, bool objectiveFunctionOnly) {
    require(numeratorLattice);
    // frame rejection heuristic
    // requires accumulation of MMI error signal
    if (this->frameRejectionThreshold_ > 0) {
        Speech::PosteriorFsa mmiDenominatorPosterior;
        mmiDenominatorPosterior.fsa = Fsa::posterior64(
                Fsa::changeSemiring(lattice->part(this->part_), Fsa::LogSemiring),
                mmiDenominatorPosterior.totalInv,
                this->posteriorTolerance());
        if (Core::isAlmostEqualUlp(f32(mmiDenominatorPosterior.totalInv), Core::Type<f32>::min, this->posteriorTolerance())) {
            this->log("discard segment because it has vanishing total flow");
            return false;
        }
        mmiDenominatorPosterior.fsa = Fsa::expm(mmiDenominatorPosterior.fsa);
        if (!mmiDenominatorPosterior) {
            this->log("failed to compute MMI-denominator posterior FSA, skipping segment");
            return false;
        }
        u32 nRejectedObsInSeq = 0;
        if (!objectiveFunctionOnly) {
            this->accumulateStatisticsOnLattice(mmiDenominatorPosterior.fsa, lattice->wordBoundaries(), 1.0);
            // frame rejection heuristic described in Vesely et al: Sequence-discriminative training of DNNs, in Interspeech 2013
            if (this->frameRejectionThreshold_ > 0) {
                for (u32 t = 0; t < alignment_.size(); ++t) {
                    verify_ge(errorSignal_.back().at(alignment_.at(t), t), 0);
                    if (errorSignal_.back().at(alignment_.at(t), t) < this->frameRejectionThreshold_) {
                        weights_.at(t) = 0.0;
                        ++nRejectedObsInSeq;
                    }
                }
            }
            errorSignal_.back().setToZero();
        }
        this->numberOfRejectedObservations_ += nRejectedObsInSeq;
    }
    Speech::PosteriorFsa numeratorPosterior;
    Speech::PosteriorFsa denominatorPosterior;
    denominatorPosterior = getDenominatorPosterior(lattice);
    if (!denominatorPosterior) {
        this->log("failed to compute denominator posterior FSA, skipping segment");
        return false;
    }
    if (!objectiveFunctionOnly) {
        this->accumulateStatisticsOnLattice(denominatorPosterior.fsa, lattice->wordBoundaries(), -1.0);
        this->accumulateStatisticsOnLattice(Fsa::multiply(denominatorPosterior.fsa, Fsa::Weight(-1.0)), lattice->wordBoundaries(), 1.0);
    }
    objectiveFunction -= f32(denominatorPosterior.totalInv);
    this->log("denominator-lattice-objective-function: ") << -f32(denominatorPosterior.totalInv);
    return true;
}

template class MinimumErrorSegmentwiseNnTrainer<f32>;

}  // namespace Nn
