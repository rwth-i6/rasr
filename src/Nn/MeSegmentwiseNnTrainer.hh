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
#ifndef MESEGMENTWISENNTRAINER_HH_
#define MESEGMENTWISENNTRAINER_HH_

#include "SegmentwiseNnTrainer.hh"

namespace Nn {

/**
 *  Minimum Error Trainer
 *
 *  depending on the definition of the accuracies, this trainer optimizes MPE, sMBR, etc
 *
 */

template<typename T>
class MinimumErrorSegmentwiseNnTrainer : public SegmentwiseNnTrainer<T> {
    typedef SegmentwiseNnTrainer<T> Precursor;
protected:
    using Precursor::errorSignal_;
    using Precursor::alignment_;
    using Precursor::weights_;
protected:
    static Core::ParameterString paramAccuracyName;

    std::string accuracyPart_;
public:
    MinimumErrorSegmentwiseNnTrainer(const Core::Configuration &config);
    virtual ~MinimumErrorSegmentwiseNnTrainer() {}
protected:
    virtual bool computeInitialErrorSignal(Lattice::ConstWordLatticeRef lattice, Lattice::ConstWordLatticeRef numeratorLattice,
            Bliss::SpeechSegment *segment, T &objectiveFunction, bool objectiveFunctionOnly);
protected:
    Speech::PosteriorFsa getDenominatorPosterior(Lattice::ConstWordLatticeRef lattice);
};



} /* namespace Nn */

#endif /* MESEGMENTWISENNTRAINER_HH_ */
