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
#ifndef LATTICEACCUMULATORS_HH_
#define LATTICEACCUMULATORS_HH_

#include <utility>

#include <Am/AcousticModel.hh>
#include <Core/ReferenceCounting.hh>
#include <Core/Types.hh>
#include <Fsa/Automaton.hh>
#include <Lattice/Accumulator.hh>
#include <Math/CudaVector.hh>
#include <Mm/Types.hh>
#include <Speech/Types.hh>
#include "Types.hh"

namespace Nn {
class ClassLabelWrapper;
}

namespace Nn {

/*
 *
 * CachedAcousticAccumulator
 * similar to Lattice::CachedAcousticAccumulator, but accumulates only the weight instead of weight * feature
 */

template<class Trainer>
class CachedAcousticAccumulator : public Lattice::AcousticAccumulator<Trainer> {
    typedef Lattice::AcousticAccumulator<Trainer> Precursor;

protected:
    Lattice::Collector collector_;
    Mm::Weight         factor_;  // can be set to -1 for denominator accumulation
protected:
    virtual void process(typename Precursor::TimeframeIndex, Mm::MixtureIndex, Mm::Weight);
    virtual void reset() {
        collector_.clear();
    }

public:
    CachedAcousticAccumulator(typename Precursor::ConstSegmentwiseFeaturesRef features,
                              typename Precursor::AlignmentGeneratorRef       alignmentGenerator,
                              Trainer*                                        trainer,
                              Mm::Weight                                      weightThreshold,
                              Core::Ref<const Am::AcousticModel>              acousticModel,
                              Mm::Weight                                      factor);
    virtual ~CachedAcousticAccumulator() {}
    virtual void finish();
    void         accumulate(Speech::TimeframeIndex t, Mm::MixtureIndex m, Mm::Weight w) {
        this->trainer_->accumulate(t, m, w);
    }
    void discoverState(Fsa::ConstStateRef sp);
};

template<typename Trainer>
CachedAcousticAccumulator<Trainer>::CachedAcousticAccumulator(
        typename Precursor::ConstSegmentwiseFeaturesRef features,
        typename Precursor::AlignmentGeneratorRef       alignmentGenerator,
        Trainer*                                        trainer,
        Mm::Weight                                      weightThreshold,
        Core::Ref<const Am::AcousticModel>              acousticModel,
        Mm::Weight                                      factor)
        : Precursor(features, alignmentGenerator, trainer, weightThreshold, acousticModel),
          factor_(factor) {}

template<class Trainer>
void CachedAcousticAccumulator<Trainer>::process(typename Precursor::TimeframeIndex t,
                                                 Mm::MixtureIndex                   m,
                                                 Mm::Weight                         w) {
    collector_.collect(Lattice::Key(t, m), w);
}

template<class Trainer>
void CachedAcousticAccumulator<Trainer>::finish() {
    for (Lattice::Collector::const_iterator c = collector_.begin(); c != collector_.end(); ++c) {
        this->accumulate(c->first.t, c->first.m, factor_ * c->second);
    }
}

template<class Trainer>
void CachedAcousticAccumulator<Trainer>::discoverState(Fsa::ConstStateRef sp) {
    Precursor::discoverState(sp);
    this->trainer_->processState(sp);
}

/**
 *
 * ErrorSignalAccumulator
 *
 * adds the error signal collected from the lattice to the error signal matrix
 *
 */
template<typename T>
class ErrorSignalAccumulator {
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;

protected:
    NnMatrix*                errorSignal_;
    const ClassLabelWrapper* labelWrapper_;

public:
    ErrorSignalAccumulator(NnMatrix* errorSignal, const ClassLabelWrapper* labelWrapper);
    virtual ~ErrorSignalAccumulator() {}

public:
    void accumulate(Speech::TimeframeIndex t, Mm::MixtureIndex m, Mm::Weight w);
    void processState(Fsa::ConstStateRef sp) {}
};

typedef CachedAcousticAccumulator<ErrorSignalAccumulator<f32>> NnAccumulator;

/**
 *
 * AlignmentAccumulator
 *
 * determines the state sequence from a lattice
 * assumption: lattice contains only a single path
 *
 */

class AlignmentAccumulator {
protected:
    Math::CudaVector<u32>*   alignment_;
    const ClassLabelWrapper* labelWrapper_;

public:
    AlignmentAccumulator(Math::CudaVector<u32>* alignment, const ClassLabelWrapper* labelWrapper);
    virtual ~AlignmentAccumulator() {}

public:
    void accumulate(Speech::TimeframeIndex t, Mm::MixtureIndex m, Mm::Weight w);
    void processState(Fsa::ConstStateRef sp) {}
};

} /* namespace Nn */

#endif /* LATTICEACCUMULATORS_HH_ */
