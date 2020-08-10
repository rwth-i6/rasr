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
#ifndef MMISEGMENTWISENNTRAINER_HH_
#define MMISEGMENTWISENNTRAINER_HH_

#include "SegmentwiseNnTrainer.hh"

namespace Nn {

/**
 *  MMI Trainer
 */
template<typename T>
class MmiSegmentwiseNnTrainer : public SegmentwiseNnTrainer<T> {
    typedef SegmentwiseNnTrainer<T> Precursor;

protected:
    using Precursor::alignment_;
    using Precursor::errorSignal_;
    using Precursor::weights_;

public:
    MmiSegmentwiseNnTrainer(const Core::Configuration& config);
    virtual ~MmiSegmentwiseNnTrainer() {}

protected:
    virtual bool computeInitialErrorSignal(Lattice::ConstWordLatticeRef lattice, Lattice::ConstWordLatticeRef numeratorLattice,
                                           Bliss::SpeechSegment* segment, T& objectiveFunction, bool objectiveFunctionOnly);

protected:
    Speech::PosteriorFsa getDenominatorPosterior(Lattice::ConstWordLatticeRef lattice);
    Speech::PosteriorFsa getNumeratorPosterior(Lattice::ConstWordLatticeRef lattice);
};

} /* namespace Nn */

#endif /* MMISEGMENTWISENNTRAINER_HH_ */
