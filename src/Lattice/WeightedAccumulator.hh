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
#ifndef _LATTICE_WEIGHTED_ACCUMULATOR_HH
#define _LATTICE_WEIGHTED_ACCUMULATOR_HH

#include <Me/FeaturesAccumulator.hh>
#include <Mm/StatePosteriorFeatureScorer.hh>
#include <Sparse/Feature.hh>
#include <Sparse/SpeechFeature.hh>
#include <Speech/AuxiliarySegmentwiseTrainer.hh>
#include <Speech/Confidences.hh>
#include "Accumulator.hh"
#include "MgramFeatures.hh"
#include "TransitionFeatures.hh"

namespace Mm {
class AssigningFeatureScorer;
}

namespace Lattice {

/**
 * WeightedCachedAcousticAccumulator
 */
template<class Trainer>
class WeightedCachedAcousticAccumulator : public CachedAcousticAccumulator<Trainer> {
    typedef CachedAcousticAccumulator<Trainer> Precursor;
    typedef Speech::Confidences                Confidences;

private:
    const Confidences& confidences_;

private:
    virtual void process(typename Precursor::TimeframeIndex, Mm::MixtureIndex, Mm::Weight);

public:
    WeightedCachedAcousticAccumulator(typename Precursor::ConstSegmentwiseFeaturesRef,
                                      typename Precursor::AlignmentGeneratorRef,
                                      Trainer*, Mm::Weight, Core::Ref<const Am::AcousticModel>,
                                      const Confidences&);
    virtual ~WeightedCachedAcousticAccumulator() {}
};

/**
 * CachedAcousticSparseAccumulator.
 * adds accumulate() for sparse features to the interface.
 */
template<class Trainer>
class CachedAcousticSparseAccumulator : public CachedAcousticAccumulator<Trainer> {
    typedef CachedAcousticAccumulator<Trainer> Precursor;

protected:
    virtual void accumulate(Core::Ref<const Sparse::Feature::SparseVector> sf, Mm::MixtureIndex m, Mm::Weight w) {
        defect();
    }

public:
    CachedAcousticSparseAccumulator(typename Precursor::ConstSegmentwiseFeaturesRef,
                                    typename Precursor::AlignmentGeneratorRef,
                                    Trainer*, Mm::Weight, Core::Ref<const Am::AcousticModel>);
    virtual ~CachedAcousticSparseAccumulator() {}
    virtual void finish();
};

/**
 * DensityCachedAcousticAccumulator
 */
template<class Trainer>
class DensityCachedAcousticAccumulator : public CachedAcousticSparseAccumulator<Trainer> {
    typedef CachedAcousticSparseAccumulator<Trainer> Precursor;

protected:
    typedef const Mm::StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer PosteriorScorer;
    typedef Mm::StatePosteriorFeatureScorer::Filter                                  Filter;
    typedef Mm::StatePosteriorFeatureScorer::ConstFilterRef                          ConstFilterRef;

public:
    typedef Mm::StatePosteriorFeatureScorer::PosteriorsAndDensities PosteriorsAndDensities;

protected:
    Core::Ref<Mm::StatePosteriorFeatureScorer> posteriorFeatureScorer_;

protected:
    virtual void accumulate(Mm::Feature::VectorRef f, Mm::MixtureIndex m, Mm::Weight w) {}
    virtual void accumulate(Mm::Feature::VectorRef f, const PosteriorsAndDensities& p) {}
    virtual void accumulate(Core::Ref<const Sparse::Feature::SparseVector> sf, Mm::MixtureIndex m, Mm::Weight w) {}

public:
    DensityCachedAcousticAccumulator(typename Precursor::ConstSegmentwiseFeaturesRef,
                                     typename Precursor::AlignmentGeneratorRef,
                                     Trainer*, Mm::Weight, Core::Ref<const Am::AcousticModel>);
    virtual ~DensityCachedAcousticAccumulator() {}

    virtual void finish();

    void setFeatureScorer(Core::Ref<Mm::StatePosteriorFeatureScorer> fs) {
        posteriorFeatureScorer_ = fs;
    }
};

/**
 * TdpAccumulator
 */
template<class Trainer>
class TdpAccumulator : public AcousticAccumulator<Trainer> {
    typedef AcousticAccumulator<Trainer> Precursor;

private:
    Core::Ref<TransitionFeatures> transitions_;

protected:
    virtual void process(typename Precursor::TimeframeIndex, Mm::MixtureIndex, Mm::Weight) {
        defect();
    }
    virtual void accumulate(Core::Ref<const Sparse::Feature::SparseVector> sf, Mm::MixtureIndex m, Mm::Weight w) {}

public:
    TdpAccumulator(typename Precursor::ConstSegmentwiseFeaturesRef,
                   typename Precursor::AlignmentGeneratorRef,
                   Trainer*, Mm::Weight,
                   Core::Ref<const Am::AcousticModel>);
    virtual ~TdpAccumulator() {}

    virtual void discoverState(Fsa::ConstStateRef sp);

    void setTransitionFeatures(Core::Ref<TransitionFeatures> transitions) {
        transitions_ = transitions;
    }
};

/**
 * LmAccumulator
 */
template<class Trainer>
class LmAccumulator : public BaseAccumulator<Trainer> {
    typedef BaseAccumulator<Trainer>  Precursor;
    typedef Core::Vector<Lm::History> Histories;

private:
    Core::Ref<MgramFeatures>                 mgrams_;
    Core::Ref<const Lm::LanguageModel>       languageModel_;
    const Bliss::LemmaPronunciationAlphabet* alphabet_;
    Histories                                histories_;

protected:
    virtual void accumulate(Core::Ref<const Sparse::Feature::SparseVector> sf, Mm::MixtureIndex m, Mm::Weight w) {}

public:
    LmAccumulator(Trainer*, Mm::Weight, Core::Ref<const Lm::LanguageModel>);
    virtual ~LmAccumulator() {}

    virtual void discoverState(Fsa::ConstStateRef sp);

    void setMgramFeatures(Core::Ref<MgramFeatures> mgrams) {
        mgrams_ = mgrams;
    }

    virtual void setFsa(Fsa::ConstAutomatonRef);
};

/**
 * WeightedDensityCachedAcousticAccumulator
 */
template<class Trainer>
class WeightedDensityCachedAcousticAccumulator : public DensityCachedAcousticAccumulator<Trainer> {
    typedef DensityCachedAcousticAccumulator<Trainer> Precursor;
    typedef Speech::Confidences                       Confidences;

private:
    const Confidences& confidences_;

protected:
    virtual void process(typename Precursor::TimeframeIndex, Mm::MixtureIndex, Mm::Weight);

public:
    WeightedDensityCachedAcousticAccumulator(typename Precursor::ConstSegmentwiseFeaturesRef,
                                             typename Precursor::AlignmentGeneratorRef,
                                             Trainer*, Mm::Weight, Core::Ref<const Am::AcousticModel>,
                                             const Confidences& confidences);
    virtual ~WeightedDensityCachedAcousticAccumulator() {}
};

#include "WeightedAccumulator.tcc"

}  // namespace Lattice

#endif
