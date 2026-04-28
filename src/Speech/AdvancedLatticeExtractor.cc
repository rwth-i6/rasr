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
#include "AdvancedLatticeExtractor.hh"
#include <Am/Module.hh>
#include <Am/Utilities.hh>
#include <Bliss/Orthography.hh>
#include <Core/Hash.hh>
#include <Core/Vector.hh>
#include <Fsa/Arithmetic.hh>
#include <Fsa/Best.hh>
#include <Fsa/Cache.hh>
#include <Fsa/Compose.hh>
#include <Fsa/Determinize.hh>
#include <Fsa/Dfs.hh>
#include <Fsa/Minimize.hh>
#include <Fsa/Project.hh>
#include <Fsa/Rational.hh>
#include <Fsa/RemoveEpsilons.hh>
#include <Lattice/Archive.hh>
#include <Lattice/Basic.hh>
#include <Lattice/Lattice.hh>
#include <Lattice/Static.hh>
#include <Lm/FsaLm.hh>
#ifndef CMAKE_DISABLE_MODULES_HH
#include <Modules.hh>
#endif
#include <Search/AdvancedTreeSearch/AdvancedTreeSearch.hh>
#include <Search/Module.hh>
#include "AdvancedAccuracyFsaBuilder.hh"
#include "Alignment.hh"
#include "AllophoneStateGraphBuilder.hh"
#include "DataExtractor.hh"
#include "LatticeExtractorAutomaton.hh"
#include "ModelCombination.hh"
#include "Module.hh"

using namespace Speech;
using namespace LatticeExtratorInternal;

/*
 * EmissionLatticeRescorerAutomaton
 */
class EmissionLatticeRescorerAutomaton : public CachedLatticeRescorerAutomaton {
    typedef CachedLatticeRescorerAutomaton               Precursor;
    typedef Core::Ref<PhonemeSequenceAlignmentGenerator> AlignmentGeneratorRef;

private:
    AlignmentGeneratorRef        alignmentGenerator_;
    ConstSegmentwiseFeaturesRef  features_;
    Core::Ref<Am::AcousticModel> acousticModel_;

protected:
    virtual Fsa::Weight score(Fsa::StateId s, const Fsa::Arc& a) const;

public:
    EmissionLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef,
                                     AlignmentGeneratorRef, ConstSegmentwiseFeaturesRef,
                                     Core::Ref<Am::AcousticModel>);
    virtual ~EmissionLatticeRescorerAutomaton() {}
    virtual std::string describe() const {
        return Core::form("emission-rescore(%s)", fsa_->describe().c_str());
    }
    Fsa::Weight _score(const Bliss::Coarticulated<Bliss::LemmaPronunciation>&,
                       TimeframeIndex begtime, TimeframeIndex endtime) const;
};

EmissionLatticeRescorerAutomaton::EmissionLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef lattice,
                                                                   AlignmentGeneratorRef        alignmentGenerator,
                                                                   ConstSegmentwiseFeaturesRef  features,
                                                                   Core::Ref<Am::AcousticModel> acousticModel)
        : Precursor(lattice),
          alignmentGenerator_(alignmentGenerator),
          features_(features),
          acousticModel_(acousticModel) {
    require(alignmentGenerator_ && acousticModel_);
}

Fsa::Weight EmissionLatticeRescorerAutomaton::score(Fsa::StateId s, const Fsa::Arc& a) const {
    const Bliss::LemmaPronunciationAlphabet* alphabet      = required_cast(const Bliss::LemmaPronunciationAlphabet*, fsa_->getInputAlphabet().get());
    const Bliss::LemmaPronunciation*         pronunciation = alphabet->lemmaPronunciation(a.input());
    const TimeframeIndex                     begtime       = wordBoundaries_->time(s);
    if (pronunciation && begtime != InvalidTimeframeIndex) {
        Bliss::Coarticulated<Bliss::LemmaPronunciation> coarticulatedPronunciation(
                *pronunciation, wordBoundaries_->transit(s).final,
                wordBoundaries_->transit(fsa_->getState(a.target())->id()).initial);
        const TimeframeIndex endtime = wordBoundaries_->time(fsa_->getState(a.target())->id());
        return _score(coarticulatedPronunciation, begtime, endtime);
    }
    else {
        return fsa_->semiring()->one();
    }
}

Fsa::Weight EmissionLatticeRescorerAutomaton::_score(const Bliss::Coarticulated<Bliss::LemmaPronunciation>& coarticulatedPronunciation,
                                                     TimeframeIndex begtime, TimeframeIndex endtime) const {
    if (begtime < endtime) {
        f32              score = fsa_->semiring()->one();
        const Alignment* alignment =
                alignmentGenerator_->getAlignment(coarticulatedPronunciation, begtime, endtime);
        const_cast<Alignment*>(alignment)->setAlphabet(acousticModel_->allophoneStateAlphabet());
        for (std::vector<AlignmentItem>::const_iterator al = alignment->begin();
             al != alignment->end(); ++al) {
            Mm::FeatureScorer::Scorer scorer =
                    acousticModel_->featureScorer()->getScorer((*features_)[al->time]);
            score += scorer->score(acousticModel_->emissionIndex(al->emission));
        }
        return Fsa::Weight(score);
    }
    else {
        Core::Application::us()->warning("score 0 assigned to arc with begin time ")
                << begtime << " , end time " << endtime << " and label id " << coarticulatedPronunciation.object().id();
        return fsa_->semiring()->one();
    }
}

/*
 * LatticeRescorer: emission
 */
const Core::ParameterString EmissionLatticeRescorer::paramPortName(
        "port-name",
        "port name of features",
        "features");

const Core::ParameterString EmissionLatticeRescorer::paramSparsePortName(
        "sparse-port-name",
        "sparse port name of features",
        "");

EmissionLatticeRescorer::EmissionLatticeRescorer(const Core::Configuration& c, bool initialize)
        : Precursor(c),
          portId_(Flow::IllegalPortId),
          sparsePortId_(Flow::IllegalPortId) {
    if (initialize) {
        ModelCombination modelCombination(select("model-combination"),
                                          ModelCombination::useAcousticModel,
                                          Am::AcousticModel::noStateTransition);
        modelCombination.load();
        acousticModel_ = modelCombination.acousticModel();
    }
}

EmissionLatticeRescorer::EmissionLatticeRescorer(const Core::Configuration&   c,
                                                 Core::Ref<Am::AcousticModel> acousticModel)
        : Precursor(c),
          portId_(Flow::IllegalPortId),
          sparsePortId_(Flow::IllegalPortId) {
    acousticModel_ = acousticModel;
}

void EmissionLatticeRescorer::setSegmentwiseFeatureExtractor(Core::Ref<SegmentwiseFeatureExtractor> segmentwiseFeatureExtractor) {
    segmentwiseFeatureExtractor_ = segmentwiseFeatureExtractor;
    portId_                      = segmentwiseFeatureExtractor_->addPort(paramPortName(config));
    const std::string sparsePortName(paramSparsePortName(config));
    if (!sparsePortName.empty()) {
        sparsePortId_ = segmentwiseFeatureExtractor_->addPort(sparsePortName);
    }
}

Lattice::ConstWordLatticeRef EmissionLatticeRescorer::work(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* segment) {
    alignmentGenerator_->setSpeechSegment(segment);

    if (segmentwiseFeatureExtractor_) {
        segmentwiseFeatureExtractor_->checkCompatibility(portId_, acousticModel_);
    }
    ConstSegmentwiseFeaturesRef features;
    if (segmentwiseFeatureExtractor_) {
        features = segmentwiseFeatureExtractor_->features(portId_);
    }
    else {
        features = alignmentGenerator_->features();
    }
    EmissionLatticeRescorerAutomaton* f      = new EmissionLatticeRescorerAutomaton(lattice, alignmentGenerator_, features, acousticModel_);
    Lattice::WordLattice*             result = new Lattice::WordLattice;
    result->setWordBoundaries(lattice->wordBoundaries());
    result->setFsa(Fsa::ConstAutomatonRef(f), Lattice::WordLattice::acousticFsa);
    return Lattice::ConstWordLatticeRef(result);
}

/*
 * TdpLatticeRescorerAutomaton
 */
class TdpLatticeRescorerAutomaton : public CachedLatticeRescorerAutomaton {
    typedef CachedLatticeRescorerAutomaton               Precursor;
    typedef Core::Ref<PhonemeSequenceAlignmentGenerator> AlignmentGeneratorRef;

private:
    AlignmentGeneratorRef                    alignmentGenerator_;
    AllophoneStateGraphBuilder*              allophoneStateGraphBuilder_;
    Core::Ref<const Am::AcousticModel>       acousticModel_;
    const Bliss::LemmaPronunciationAlphabet* alphabet_;

private:
    virtual Fsa::Weight score(Fsa::StateId s, const Fsa::Arc& a) const;

public:
    TdpLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef, AlignmentGeneratorRef,
                                AllophoneStateGraphBuilder*,
                                Core::Ref<const Am::AcousticModel>);
    virtual ~TdpLatticeRescorerAutomaton() {}
    virtual std::string describe() const {
        return Core::form("tdp-rescore(%s)", fsa_->describe().c_str());
    }
    Fsa::Weight _score(const Bliss::Coarticulated<Bliss::LemmaPronunciation>&,
                       TimeframeIndex begtime, TimeframeIndex endtime) const;
};

TdpLatticeRescorerAutomaton::TdpLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef       lattice,
                                                         AlignmentGeneratorRef              alignmentGenerator,
                                                         AllophoneStateGraphBuilder*        allophoneStateGraphBuilder,
                                                         Core::Ref<const Am::AcousticModel> acousticModel)
        : Precursor(lattice),
          alignmentGenerator_(alignmentGenerator),
          allophoneStateGraphBuilder_(allophoneStateGraphBuilder),
          acousticModel_(acousticModel),
          alphabet_(required_cast(const Bliss::LemmaPronunciationAlphabet*, fsa_->getInputAlphabet().get())) {
    require(alignmentGenerator_ && allophoneStateGraphBuilder_ && acousticModel_);
    require(alphabet_);
}

Fsa::Weight TdpLatticeRescorerAutomaton::score(Fsa::StateId s, const Fsa::Arc& a) const {
    const Bliss::LemmaPronunciation* pronunciation =
            alphabet_->lemmaPronunciation(a.input());
    const TimeframeIndex begtime = wordBoundaries_->time(s);
    if (pronunciation && begtime != InvalidTimeframeIndex) {
        const TimeframeIndex endtime =
                wordBoundaries_->time(fsa_->getState(a.target())->id());
        Bliss::Coarticulated<Bliss::LemmaPronunciation> coarticulatedPronunciation(
                *pronunciation, wordBoundaries_->transit(s).final,
                wordBoundaries_->transit(fsa_->getState(a.target())->id()).initial);
        return _score(coarticulatedPronunciation, begtime, endtime);
    }
    else {
        return fsa_->semiring()->one();
    }
}

Fsa::Weight TdpLatticeRescorerAutomaton::_score(
        const Bliss::Coarticulated<Bliss::LemmaPronunciation>& coarticulatedPronunciation,
        TimeframeIndex begtime, TimeframeIndex endtime) const {
    if (begtime < endtime) {
        const Alignment* alignment = alignmentGenerator_->getAlignment(
                coarticulatedPronunciation, begtime, endtime);
        Fsa::Weight score = Fsa::bestscore(
                Fsa::staticCopy(
                        allophoneStateGraphBuilder_->build(*alignment,
                                                           Bliss::Coarticulated<Bliss::Pronunciation>(*coarticulatedPronunciation.object().pronunciation(),
                                                                                                      coarticulatedPronunciation.leftContext(),
                                                                                                      coarticulatedPronunciation.rightContext()))));
        if (fsa_->semiring()->compare(score, fsa_->semiring()->invalid()) == 0) {
            score = Fsa::Weight(1e9);  // fsa_->semiring()->zero();
        }
        return score;
    }
    else {
        Core::Application::us()->warning("score 0 assigned to arc with begin time ")
                << begtime << " , end time " << endtime << " and label id " << coarticulatedPronunciation.object().id();
        return fsa_->semiring()->one();
    }
}

/*
 * LatticeRescorer: tdp
 */
const Core::ParameterStringVector TdpLatticeRescorer::paramSilencesAndNoises(
        "silences-and-noises",
        "list of silence and noise lemmata (strings)",
        ",");

TdpLatticeRescorer::TdpLatticeRescorer(const Core::Configuration& c, bool initialize)
        : Precursor(c) {
    if (initialize) {
        ModelCombination modelCombination(select("model-combination"),
                                          ModelCombination::useAcousticModel,
                                          Am::AcousticModel::noEmissions);
        modelCombination.load();
        allophoneStateGraphBuilder_ = Module::instance().createAllophoneStateGraphBuilder(
                select("allophone-state-graph-builder"),
                modelCombination.lexicon(),
                modelCombination.acousticModel());
        std::vector<std::string> silencesAndNoises = paramSilencesAndNoises(config);
        allophoneStateGraphBuilder_->setSilencesAndNoises(silencesAndNoises);
        acousticModel_ = modelCombination.acousticModel();
    }
}

TdpLatticeRescorer::~TdpLatticeRescorer() {
    delete allophoneStateGraphBuilder_;
}

Lattice::ConstWordLatticeRef TdpLatticeRescorer::work(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* segment) {
    alignmentGenerator_->setSpeechSegment(segment);
    TdpLatticeRescorerAutomaton* f      = new TdpLatticeRescorerAutomaton(lattice, alignmentGenerator_,
                                                                          allophoneStateGraphBuilder_,
                                                                          acousticModel_);
    Lattice::WordLattice*        result = new Lattice::WordLattice;
    result->setWordBoundaries(lattice->wordBoundaries());
    result->setFsa(Fsa::ConstAutomatonRef(f), Lattice::WordLattice::acousticFsa);
    return Lattice::ConstWordLatticeRef(result);
}

/*
 * CombinedAcousticLatticeRescorerAutomaton
 */
class CombinedAcousticLatticeRescorerAutomaton : public LatticeRescorerAutomaton {
    typedef LatticeRescorerAutomaton                     Precursor;
    typedef Core::Ref<PhonemeSequenceAlignmentGenerator> AlignmentGeneratorRef;

protected:
    Core::Ref<const EmissionLatticeRescorerAutomaton> emissionRescorer_;
    Core::Ref<const TdpLatticeRescorerAutomaton>      tdpRescorer_;

protected:
    virtual Fsa::Weight score(Fsa::StateId s, const Fsa::Arc& a) const;

public:
    CombinedAcousticLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef, AlignmentGeneratorRef,
                                             Core::Ref<Am::AcousticModel>,
                                             ConstSegmentwiseFeaturesRef, AllophoneStateGraphBuilder*);
    virtual ~CombinedAcousticLatticeRescorerAutomaton() {}
    virtual std::string describe() const {
        return Core::form("combined-acoustic-rescore(%s)", fsa_->describe().c_str());
    }
    Fsa::Weight _score(const Bliss::Coarticulated<Bliss::LemmaPronunciation>&,
                       TimeframeIndex begtime, TimeframeIndex endtime) const;
};

CombinedAcousticLatticeRescorerAutomaton::CombinedAcousticLatticeRescorerAutomaton(
        Lattice::ConstWordLatticeRef lattice,
        AlignmentGeneratorRef        alignmentGenerator,
        Core::Ref<Am::AcousticModel> acousticModel,
        ConstSegmentwiseFeaturesRef  features,
        AllophoneStateGraphBuilder*  allophoneStateGraphBuilder)
        : Precursor(lattice),
          emissionRescorer_(new EmissionLatticeRescorerAutomaton(lattice, alignmentGenerator, features, acousticModel)),
          tdpRescorer_(new TdpLatticeRescorerAutomaton(lattice, alignmentGenerator, allophoneStateGraphBuilder, acousticModel)) {}

Fsa::Weight CombinedAcousticLatticeRescorerAutomaton::score(Fsa::StateId s, const Fsa::Arc& a) const {
    const Bliss::LemmaPronunciationAlphabet* alphabet      = required_cast(const Bliss::LemmaPronunciationAlphabet*, fsa_->getInputAlphabet().get());
    const Bliss::LemmaPronunciation*         pronunciation = alphabet->lemmaPronunciation(a.input());
    const TimeframeIndex                     begtime       = wordBoundaries_->time(s);
    if (pronunciation && begtime != InvalidTimeframeIndex) {
        Bliss::Coarticulated<Bliss::LemmaPronunciation> coarticulatedPronunciation(
                *pronunciation, wordBoundaries_->transit(s).final,
                wordBoundaries_->transit(fsa_->getState(a.target())->id()).initial);
        const TimeframeIndex endtime = wordBoundaries_->time(fsa_->getState(a.target())->id());
        return _score(coarticulatedPronunciation, begtime, endtime);
    }
    else {
        return fsa_->semiring()->one();
    }
}

Fsa::Weight CombinedAcousticLatticeRescorerAutomaton::_score(const Bliss::Coarticulated<Bliss::LemmaPronunciation>& coarticulatedPronunciation,
                                                             TimeframeIndex begtime, TimeframeIndex endtime) const {
    return fsa_->semiring()->extend(
            emissionRescorer_->_score(coarticulatedPronunciation, begtime, endtime),
            tdpRescorer_->_score(coarticulatedPronunciation, begtime, endtime));
}

/*
 * CombinedAcousticSummedPronunciationLatticeRescorerAutomaton
 */
class CombinedAcousticSummedPronunciationLatticeRescorerAutomaton : public CombinedAcousticLatticeRescorerAutomaton {
    typedef CombinedAcousticLatticeRescorerAutomaton     Precursor;
    typedef Core::Ref<PhonemeSequenceAlignmentGenerator> AlignmentGeneratorRef;

protected:
    virtual Fsa::Weight score(Fsa::StateId s, const Fsa::Arc& a) const;

public:
    CombinedAcousticSummedPronunciationLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef,
                                                                AlignmentGeneratorRef,
                                                                Core::Ref<Am::AcousticModel>,
                                                                ConstSegmentwiseFeaturesRef,
                                                                AllophoneStateGraphBuilder*);
    virtual ~CombinedAcousticSummedPronunciationLatticeRescorerAutomaton() {}
    virtual std::string describe() const {
        return Core::form("combined-acoustic-summed-pronunciation-rescore(%s)", fsa_->describe().c_str());
    }
};

CombinedAcousticSummedPronunciationLatticeRescorerAutomaton::CombinedAcousticSummedPronunciationLatticeRescorerAutomaton(
        Lattice::ConstWordLatticeRef lattice,
        AlignmentGeneratorRef        alignmentGenerator,
        Core::Ref<Am::AcousticModel> acousticModel,
        ConstSegmentwiseFeaturesRef  features,
        AllophoneStateGraphBuilder*  allophoneStateGraphBuilder)
        : Precursor(lattice, alignmentGenerator, acousticModel, features, allophoneStateGraphBuilder) {}

Fsa::Weight CombinedAcousticSummedPronunciationLatticeRescorerAutomaton::score(Fsa::StateId s, const Fsa::Arc& a) const {
    Fsa::Weight                              resultScore   = fsa_->semiring()->one();
    const Bliss::LemmaPronunciationAlphabet* alphabet      = required_cast(const Bliss::LemmaPronunciationAlphabet*, fsa_->getInputAlphabet().get());
    const Bliss::LemmaPronunciation*         pronunciation = alphabet->lemmaPronunciation(a.input());
    const TimeframeIndex                     begtime       = wordBoundaries_->time(s);
    if (pronunciation && begtime != InvalidTimeframeIndex) {
        const TimeframeIndex                endtime = wordBoundaries_->time(fsa_->getState(a.target())->id());
        Bliss::Lemma::PronunciationIterator p, p_end;
        for (Core::tie(p, p_end) = pronunciation->lemma()->pronunciations(); p != p_end; ++p) {
            Bliss::Coarticulated<Bliss::LemmaPronunciation> coarticulatedPronunciation(*p, wordBoundaries_->transit(s).final,
                                                                                       wordBoundaries_->transit(fsa_->getState(a.target())->id()).initial);

            resultScore = fsa_->semiring()->collect(Precursor::_score(coarticulatedPronunciation, begtime, endtime), resultScore);
        }
    }
    return resultScore;
}

/*
 * CombinedAcousticLatticeRescorer
 */
const Core::ParameterBool CombinedAcousticLatticeRescorer::paramShouldSumOverPronunciations(
        "should-sum-over-pronunciations",
        "sum over different pronunciations",
        false);

CombinedAcousticLatticeRescorer::CombinedAcousticLatticeRescorer(const Core::Configuration& c)
        : Precursor(c),
          EmissionLatticeRescorer(c, false),
          TdpLatticeRescorer(c, false),
          shouldSumOverPronunciations_(paramShouldSumOverPronunciations(config)) {
    ModelCombination modelCombination(select("model-combination"),
                                      ModelCombination::useAcousticModel,
                                      Am::AcousticModel::complete);
    modelCombination.load();
    allophoneStateGraphBuilder_ = Module::instance().createAllophoneStateGraphBuilder(
            config, modelCombination.lexicon(), modelCombination.acousticModel());

    std::vector<std::string> silencesAndNoises = paramSilencesAndNoises(config);
    allophoneStateGraphBuilder_->setSilencesAndNoises(silencesAndNoises);
    acousticModel_ = modelCombination.acousticModel();
}

CombinedAcousticLatticeRescorer::CombinedAcousticLatticeRescorer(const Core::Configuration&   c,
                                                                 Core::Ref<Am::AcousticModel> acousticModel)
        : Precursor(c),
          EmissionLatticeRescorer(c, false),
          TdpLatticeRescorer(c, false),
          shouldSumOverPronunciations_(paramShouldSumOverPronunciations(config)) {
    ModelCombination modelCombination(select("model-combination"), ModelCombination::useLexicon);
    modelCombination.load();
    allophoneStateGraphBuilder_ = Module::instance().createAllophoneStateGraphBuilder(
            config, modelCombination.lexicon(), acousticModel);

    std::vector<std::string> silencesAndNoises = paramSilencesAndNoises(config);
    allophoneStateGraphBuilder_->setSilencesAndNoises(silencesAndNoises);
    acousticModel_ = acousticModel;
}

Lattice::ConstWordLatticeRef CombinedAcousticLatticeRescorer::work(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* segment) {
    alignmentGenerator_->setSpeechSegment(segment);
    Fsa::ConstAutomatonRef      acoustic;
    ConstSegmentwiseFeaturesRef features = segmentwiseFeatureExtractor_ ? segmentwiseFeatureExtractor_->features(portId_) : alignmentGenerator_->features();
    if (!shouldSumOverPronunciations_) {
        acoustic = Fsa::ConstAutomatonRef(
                new CombinedAcousticLatticeRescorerAutomaton(
                        lattice, alignmentGenerator_, acousticModel_,
                        features, allophoneStateGraphBuilder_));
    }
    else {
        acoustic = Fsa::ConstAutomatonRef(new CombinedAcousticSummedPronunciationLatticeRescorerAutomaton(
                lattice, alignmentGenerator_, acousticModel_,
                features, allophoneStateGraphBuilder_));
    }
    Lattice::WordLattice* result = new Lattice::WordLattice;
    result->setWordBoundaries(lattice->wordBoundaries());
    result->setFsa(acoustic, Lattice::WordLattice::acousticFsa);
    return Lattice::ConstWordLatticeRef(result);
}

/*
 * PronunciationLatticeRescorerAutomaton
 */
class PronunciationLatticeRescorerAutomaton : public LatticeRescorerAutomaton {
private:
    const Bliss::LemmaPronunciationAlphabet* alphabet_;
    f32                                      pronunciationScale_;

protected:
    virtual Fsa::Weight score(Fsa::StateId s, const Fsa::Arc& a) const;

public:
    PronunciationLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef, f32);
    virtual ~PronunciationLatticeRescorerAutomaton() {}

    virtual std::string describe() const {
        return Core::form("pronunciation-rescore(%s)", fsa_->describe().c_str());
    }
};

PronunciationLatticeRescorerAutomaton::PronunciationLatticeRescorerAutomaton(
        Lattice::ConstWordLatticeRef lattice, f32 pronunciationScale)
        : LatticeRescorerAutomaton(lattice),
          alphabet_(required_cast(const Bliss::LemmaPronunciationAlphabet*, fsa_->getInputAlphabet().get())),
          pronunciationScale_(pronunciationScale) {}

Fsa::Weight PronunciationLatticeRescorerAutomaton::score(Fsa::StateId s, const Fsa::Arc& a) const {
    const Bliss::LemmaPronunciation* pronunciation = alphabet_->lemmaPronunciation(a.input());
    if (pronunciation) {
        return Fsa::Weight(pronunciationScale_ * pronunciation->pronunciationScore());
    }
    return fsa_->semiring()->one();
}

/*
 * LatticeRescorer: pronunciation
 */
PronunciationLatticeRescorer::PronunciationLatticeRescorer(const Core::Configuration& c)
        : Precursor(c) {
    ModelCombination modelCombination(select("model-combination"), ModelCombination::useLexicon);
    modelCombination.load();
    pronunciationScale_ = modelCombination.pronunciationScale();
}

Lattice::ConstWordLatticeRef PronunciationLatticeRescorer::work(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment*) {
    PronunciationLatticeRescorerAutomaton* f      = new PronunciationLatticeRescorerAutomaton(lattice, pronunciationScale_);
    Lattice::WordLattice*                  result = new Lattice::WordLattice;
    result->setWordBoundaries(f->wordBoundaries());
    result->setFsa(Fsa::ConstAutomatonRef(f), Lattice::WordLattice::acousticFsa);
    return Lattice::ConstWordLatticeRef(result);
}

/*
 * LatticeRescorerAutomaton:
 */
class RestoreScoresLatticeRescorerAutomaton : public LatticeRescorerAutomaton {
    typedef LatticeRescorerAutomaton                 Precursor;
    typedef Lm::History                              History;
    typedef Core::Vector<History>                    Histories;
    typedef Core::Ref<const Lm::ScaledLanguageModel> ConstLanguageModelRef;

private:
    struct Context {
        TimeframeIndex begtim, endtim;
        Fsa::LabelId   label;
        History        history;
        Context(TimeframeIndex _begtim, TimeframeIndex _endtim,
                Fsa::LabelId _label, const History& _history)
                : begtim(_begtim),
                  endtim(_endtim),
                  label(_label),
                  history(_history) {}
    };
    struct ContextHash {
        size_t operator()(const Context& c) const {
            return ((c.begtim & 0x000000ff) |
                    ((c.endtim & 0x000000ff) << 8) |
                    ((c.label & 0x000000ff) << 16) |
                    ((c.history.hashKey() & 0x000000ff) << 24));
        }
    };
    struct ContextEquality {
        bool operator()(const Context& lhs, const Context& rhs) const {
            return lhs.begtim == rhs.begtim && lhs.endtim == rhs.endtim && lhs.label == rhs.label && lhs.history == rhs.history;
        }
    };
    typedef std::unordered_map<Context, Fsa::Weight, ContextHash, ContextEquality> Scores;
    class SetScoresDfsState : public Lattice::DfsState {
        typedef Lattice::DfsState Precursor;

    private:
        ConstLanguageModelRef                              languageModel_;
        Core::Ref<const Bliss::LemmaPronunciationAlphabet> alphabet_;
        Histories                                          histories_;
        Scores&                                            scores_;

    public:
        SetScoresDfsState(Lattice::ConstWordLatticeRef lattice,
                          Scores&                      scores,
                          ConstLanguageModelRef        languageModel)
                : Precursor(lattice),
                  languageModel_(languageModel),
                  alphabet_(required_cast(const Bliss::LemmaPronunciationAlphabet*, fsa_->getInputAlphabet().get())),
                  scores_(scores) {
            histories_.grow(fsa_->initialStateId());
            histories_[fsa_->initialStateId()] = languageModel_->startHistory();
        }
        virtual ~SetScoresDfsState() {}
        virtual void discoverState(Fsa::ConstStateRef sp) {
            TimeframeIndex begtim = wordBoundaries_->time(sp->id());
            require(histories_[sp->id()].isValid());
            for (Fsa::State::const_iterator a = sp->begin(); a != sp->end(); ++a) {
                TimeframeIndex endtim = wordBoundaries_->time(fsa_->getState(a->target())->id());
                Context        context(begtim, endtim, a->input(), histories_[sp->id()]);
                if (scores_.find(context) != scores_.end()) {
                    scores_[context] = fsa_->semiring()->collect(scores_[context], a->weight());
                }
                else {
                    scores_[context] = a->weight();
                }

                histories_.grow(a->target());
                History                          hist = histories_[sp->id()];
                const Bliss::LemmaPronunciation* lp   = alphabet_->lemmaPronunciation(a->input());
                if (lp) {
                    f32 dummy = 0;
                    Lm::addLemmaPronunciationScore(languageModel_, lp, dummy, dummy, hist, dummy);
                }
                else if (fsa_->getState(a->target())->isFinal()) {
                    hist = languageModel_->startHistory();
                }
                if (!histories_[a->target()].isValid()) {
                    histories_[a->target()] = hist;
                }
                else {
                    require(histories_[a->target()] == hist);
                }
            }
        }
    };

protected:
    virtual Fsa::Weight score(Fsa::StateId s, const Fsa::Arc& a) const;

private:
    ConstLanguageModelRef                              languageModel_;
    Core::Ref<const Bliss::LemmaPronunciationAlphabet> alphabet_;
    mutable Histories                                  histories_;
    Scores                                             scores_;

public:
    RestoreScoresLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef lattice,
                                          Lattice::ConstWordLatticeRef latticeWithScores,
                                          ConstLanguageModelRef        languageModel)
            : Precursor(lattice),
              languageModel_(languageModel),
              alphabet_(required_cast(const Bliss::LemmaPronunciationAlphabet*, fsa_->getInputAlphabet().get())) {
        SetScoresDfsState setter(latticeWithScores, scores_, languageModel_);
        setter.dfs();

        histories_.grow(fsa_->initialStateId());
        histories_[fsa_->initialStateId()] = languageModel_->startHistory();
    }
    virtual ~RestoreScoresLatticeRescorerAutomaton() {}
    virtual std::string describe() const {
        return Core::form("restore-scores-rescore(%s)", fsa_->describe().c_str());
    }
};

Fsa::Weight RestoreScoresLatticeRescorerAutomaton::score(Fsa::StateId s, const Fsa::Arc& a) const {
    TimeframeIndex begtim = wordBoundaries_->time(s);
    TimeframeIndex endtim = wordBoundaries_->time(fsa_->getState(a.target())->id());
    require(histories_[s].isValid());
    Context context(begtim, endtim, a.input(), histories_[s]);
    require(scores_.find(context) != scores_.end());
    histories_.grow(a.target());
    History                          hist = histories_[s];
    const Bliss::LemmaPronunciation* lp   = alphabet_->lemmaPronunciation(a.input());
    if (lp) {
        f32 dummy = 0;
        Lm::addLemmaPronunciationScore(languageModel_, lp, dummy, dummy, hist, dummy);
    }
    else if (fsa_->getState(a.target())->isFinal()) {
        hist = languageModel_->startHistory();
    }
    if (!histories_[a.target()].isValid()) {
        histories_[a.target()] = hist;
    }
    else {
        require(histories_[a.target()] == hist);
    }
    return scores_.find(context)->second;
}

/*
 * RestoreScoresLatticeRescorer
 */
const Core::ParameterString RestoreScoresLatticeRescorer::paramFsaPrefix(
        "fsa-prefix",
        "prefix of automaton in archive");

RestoreScoresLatticeRescorer::RestoreScoresLatticeRescorer(const Core::Configuration& c, Bliss::LexiconRef lexicon)
        : Precursor(c),
          archiveReader_(0) {
    fsaPrefix_ = paramFsaPrefix(c);
    if (fsaPrefix_.empty()) {
        fsaPrefix_ = Lattice::WordLattice::mainFsa;
    }

    archiveReader_ = Lattice::Archive::openForReading(select("lattice-archive"), lexicon);
    if (!archiveReader_ || archiveReader_->hasFatalErrors()) {
        delete archiveReader_;
        archiveReader_ = 0;
        error("failed to open lattice archive");
        return;
    }

    ModelCombination modelCombination(select("model-combination"),
                                      ModelCombination::useLanguageModel);
    modelCombination.load();
    languageModel_ = Core::Ref<const Lm::ScaledLanguageModel>(modelCombination.languageModel().get());
}

Lattice::ConstWordLatticeRef RestoreScoresLatticeRescorer::work(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* segment) {
    verify(archiveReader_);
    Lattice::ConstWordLatticeRef latticeWithScores = archiveReader_->get(segment->fullName(), fsaPrefix_);
    if (latticeWithScores && latticeWithScores->nParts() == 1) {
        RestoreScoresLatticeRescorerAutomaton* f      = new RestoreScoresLatticeRescorerAutomaton(lattice,
                                                                                                  latticeWithScores,
                                                                                                  languageModel_);
        Lattice::WordLattice*                  result = new Lattice::WordLattice;
        result->setWordBoundaries(lattice->wordBoundaries());
        result->setFsa(Fsa::ConstAutomatonRef(f), Lattice::WordLattice::acousticFsa);
        return Lattice::ConstWordLatticeRef(result);
    }
    else {
        criticalError("Failed to retrieve lattice '%s' for segment '%s'.",
                      fsaPrefix_.c_str(), segment->fullName().c_str());
        return lattice;
    }
}

OrthographyApproximatePhoneAccuracyMaskLatticeRescorer::OrthographyApproximatePhoneAccuracyMaskLatticeRescorer(const Core::Configuration& c,
                                                                                                               Bliss::LexiconRef          lexicon)
        : Precursor(c, lexicon),
          builder_(0) {
    builder_ = new OrthographyApproximatePhoneAccuracyMaskLatticeBuilder(select("approximate-phone-accuracy-lattice-builder"), lexicon);
}

OrthographyApproximatePhoneAccuracyMaskLatticeRescorer::~OrthographyApproximatePhoneAccuracyMaskLatticeRescorer() {
    delete builder_;
}

Fsa::ConstAutomatonRef OrthographyApproximatePhoneAccuracyMaskLatticeRescorer::getDistanceFsa(Lattice::ConstWordLatticeRef lattice,
                                                                                              Bliss::SpeechSegment*        segment) {
    verify(builder_);
    alignmentGenerator_->setSpeechSegment(segment);
    return builder_->createFunctor(segment->fullName(),
                                   segment->orth(),
                                   lattice,
                                   alignmentGenerator_)
            .build();
}

/*
 * DistanceLatticeRescorer: frame state accuracy
 */
FrameStateAccuracyLatticeRescorer::FrameStateAccuracyLatticeRescorer(const Core::Configuration& c, Bliss::LexiconRef lexicon)
        : Precursor(c, lexicon) {}

FrameStateAccuracyLatticeRescorer::~FrameStateAccuracyLatticeRescorer() {}

ArchiveFrameStateAccuracyLatticeRescorer::ArchiveFrameStateAccuracyLatticeRescorer(const Core::Configuration& c, Bliss::LexiconRef lexicon)
        : Precursor(c, lexicon),
          builder_(0) {
    builder_ = new ArchiveFrameStateAccuracyLatticeBuilder(select("frame-state-accuracy-lattice-builder"), lexicon);
}

ArchiveFrameStateAccuracyLatticeRescorer::~ArchiveFrameStateAccuracyLatticeRescorer() {
    delete builder_;
}

Fsa::ConstAutomatonRef ArchiveFrameStateAccuracyLatticeRescorer::getDistanceFsa(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* segment) {
    verify(builder_);
    alignmentGenerator_->setSpeechSegment(segment);
    return builder_->createFunctor(segment->fullName(),
                                   segment->fullName(),
                                   lattice,
                                   alignmentGenerator_)
            .build();
}

OrthographyFrameStateAccuracyLatticeRescorer::OrthographyFrameStateAccuracyLatticeRescorer(const Core::Configuration& c, Bliss::LexiconRef lexicon)
        : Precursor(c, lexicon),
          builder_(0) {
    builder_ = new OrthographyFrameStateAccuracyLatticeBuilder(select("frame-state-accuracy-lattice-builder"), lexicon);
}

OrthographyFrameStateAccuracyLatticeRescorer::~OrthographyFrameStateAccuracyLatticeRescorer() {
    delete builder_;
}

Fsa::ConstAutomatonRef OrthographyFrameStateAccuracyLatticeRescorer::getDistanceFsa(Lattice::ConstWordLatticeRef lattice,
                                                                                    Bliss::SpeechSegment*        segment) {
    verify(builder_);
    alignmentGenerator_->setSpeechSegment(segment);
    return builder_->createFunctor(segment->fullName(),
                                   segment->orth(),
                                   lattice,
                                   alignmentGenerator_)
            .build();
}

OrthographySmoothedFrameStateAccuracyLatticeRescorer::OrthographySmoothedFrameStateAccuracyLatticeRescorer(const Core::Configuration& c,
                                                                                                           Bliss::LexiconRef          lexicon)
        : Precursor(c, lexicon),
          builder_(0) {
    builder_ = new OrthographySmoothedFrameStateAccuracyLatticeBuilder(select("smoothed-frame-state-accuracy-lattice-builder"), lexicon);
}

OrthographySmoothedFrameStateAccuracyLatticeRescorer::~OrthographySmoothedFrameStateAccuracyLatticeRescorer() {
    delete builder_;
}

/*
 * assumption: lattice contains total scores
 */
Fsa::ConstAutomatonRef OrthographySmoothedFrameStateAccuracyLatticeRescorer::getDistanceFsa(Lattice::ConstWordLatticeRef lattice,
                                                                                            Bliss::SpeechSegment*        segment) {
    verify(builder_);
    alignmentGenerator_->setSpeechSegment(segment);
    return builder_->createFunctor(segment->fullName(),
                                   segment->orth(),
                                   lattice,
                                   alignmentGenerator_)
            .build();
}

/*
 * DistanceLatticeRescorer: word accuracy
 */
WordAccuracyLatticeRescorer::WordAccuracyLatticeRescorer(const Core::Configuration& c, Bliss::LexiconRef lexicon)
        : Precursor(c),
          orthToLemma_(0) {
    lemmaToEval_      = lexicon->createLemmaToEvaluationTokenTransducer();
    lemmaPronToLemma_ = lexicon->createLemmaPronunciationToLemmaTransducer();

    verify(!orthToLemma_);
    orthToLemma_ = new Bliss::OrthographicParser(select("orthographic-parser"), lexicon);
}

WordAccuracyLatticeRescorer::~WordAccuracyLatticeRescorer() {
    delete orthToLemma_;
}

Lattice::ConstWordLatticeRef WordAccuracyLatticeRescorer::work(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* segment) {
    return Lattice::getExactWordAccuracy(
            lattice, segment->orth(), orthToLemma_,
            lemmaPronToLemma_, lemmaToEval_);
}

/*
 * DistanceLatticeRescorer: phoneme accuracy
 */
PhonemeAccuracyLatticeRescorer::PhonemeAccuracyLatticeRescorer(
        const Core::Configuration& c, Bliss::LexiconRef lexicon)
        : Precursor(c),
          orthToLemma_(0) {
    lemmaPronToPhoneme_ = Fsa::cache(Fsa::invert(
            Fsa::replaceInputDisambiguationSymbols(lexicon->createPhonemeToLemmaPronunciationTransducer(),
                                                   Fsa::Epsilon)));

    lemmaToPhoneme_ = Fsa::cache(Fsa::composeSequencing(
            Fsa::invert(lexicon->createLemmaPronunciationToLemmaTransducer()),
            lemmaPronToPhoneme_));

    verify(!orthToLemma_);
    orthToLemma_ = new Bliss::OrthographicParser(select("orthographic-parser"), lexicon);
}

PhonemeAccuracyLatticeRescorer::~PhonemeAccuracyLatticeRescorer() {
    delete orthToLemma_;
}

Lattice::ConstWordLatticeRef PhonemeAccuracyLatticeRescorer::work(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* segment) {
    return Lattice::getExactPhonemeAccuracy(
            lattice, segment->orth(), orthToLemma_,
            lemmaPronToPhoneme_, lemmaToPhoneme_);
}

/*
 * DistanceLatticeRescorer: levenshtein on n-best lists
 */
LevenshteinListRescorer::LevenshteinListRescorer(const Core::Configuration& c, Bliss::LexiconRef lexicon)
        : Precursor(c),
          builder_(0) {
    builder_ = new LevenshteinNBestListBuilder(select("levenshtein-distance-list-builder"), lexicon);
}

LevenshteinListRescorer::~LevenshteinListRescorer() {
    delete builder_;
}

Lattice::ConstWordLatticeRef LevenshteinListRescorer::work(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* segment) {
    verify(builder_);
    Lattice::WordLattice* result = new Lattice::WordLattice;
    result->setWordBoundaries(lattice->wordBoundaries());
    result->setFsa(builder_->createFunctor(segment->fullName(), segment->orth(), lattice->part(0)).build(),
                   Lattice::WordLattice::acousticFsa);
    return Lattice::ConstWordLatticeRef(result);
}

/*
 * DistanceLatticeRescorer: frame word accuracy
 */
OrthographyFrameWordAccuracyLatticeRescorer::OrthographyFrameWordAccuracyLatticeRescorer(const Core::Configuration& c,
                                                                                         Bliss::LexiconRef          lexicon)
        : Precursor(c, lexicon),
          builder_(0) {
    builder_ = new OrthographyFrameWordAccuracyLatticeBuilder(select("frame-word-accuracy-lattice-builder"), lexicon);
}

OrthographyFrameWordAccuracyLatticeRescorer::~OrthographyFrameWordAccuracyLatticeRescorer() {
    delete builder_;
}

Fsa::ConstAutomatonRef OrthographyFrameWordAccuracyLatticeRescorer::getDistanceFsa(
        Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* segment) {
    verify(builder_);
    return builder_->createFunctor(segment->fullName(),
                                   segment->orth(),
                                   lattice)
            .build();
}

/*
 * DistanceLatticeRescorer: frame phone accuracy
 */
OrthographyFramePhoneAccuracyLatticeRescorer::OrthographyFramePhoneAccuracyLatticeRescorer(const Core::Configuration& c,
                                                                                           Bliss::LexiconRef          lexicon)
        : Precursor(c, lexicon),
          builder_(0) {
    builder_ = new OrthographyFramePhoneAccuracyLatticeBuilder(select("frame-phone-accuracy-lattice-builder"), lexicon);
}

OrthographyFramePhoneAccuracyLatticeRescorer::~OrthographyFramePhoneAccuracyLatticeRescorer() {
    delete builder_;
}

Fsa::ConstAutomatonRef OrthographyFramePhoneAccuracyLatticeRescorer::getDistanceFsa(Lattice::ConstWordLatticeRef lattice,
                                                                                    Bliss::SpeechSegment*        segment) {
    verify(builder_);
    alignmentGenerator_->setSpeechSegment(segment);
    return builder_->createFunctor(segment->fullName(),
                                   segment->orth(),
                                   lattice,
                                   alignmentGenerator_)
            .build();
}

/*
 * LatticeRescorer: posterior
 */
Core::Choice PosteriorLatticeRescorer::choicePosteriorType(
        "probability", probability,
        "expectation", expectation,
        "combined-probability", combinedProbability,
        Core::Choice::endMark());

Core::ParameterChoice PosteriorLatticeRescorer::paramPosteriorType(
        "posterior-type",
        &choicePosteriorType,
        "type of posterior to apply",
        probability);

const Core::ParameterInt PosteriorLatticeRescorer::paramTolerance(
        "tolerance",
        "tolerance in posterior computation, i.e., error of forward and backward flows w.r.t. least significant bits",
        100,
        0,
        Core::Type<s32>::max);

const Core::ParameterBool PosteriorLatticeRescorer::paramPNormalized(
        "p-normalized",
        "posteriors include normalization",
        true);

PosteriorLatticeRescorer::PosteriorLatticeRescorer(const Core::Configuration& c)
        : Precursor(c),
          tolerance_(paramTolerance(c)),
          pNormalized_(paramPNormalized(c)),
          accumulator_(0) {}

PosteriorLatticeRescorer::~PosteriorLatticeRescorer() {
    log("objective-function: ") << accumulator_;
}

Lattice::ConstWordLatticeRef PosteriorLatticeRescorer::work(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* segment) {
    Fsa::Weight            totalInv;
    Fsa::ConstAutomatonRef fsa = Fsa::posterior64(Fsa::changeSemiring(lattice->part(Lattice::WordLattice::totalFsa), Fsa::LogSemiring),
                                                  totalInv,
                                                  tolerance_);
    accumulate(f32(totalInv));
    if (!pNormalized_) {
        fsa = Fsa::extend(fsa, totalInv);
    }
    Core::Ref<Lattice::WordLattice> result(new Lattice::WordLattice);
    result->setWordBoundaries(lattice->wordBoundaries());
    result->setFsa(fsa, Lattice::WordLattice::posteriorFsa);
    return result;
}

void PosteriorLatticeRescorer::accumulate(f32 toAcc) {
    log("objective-function: ") << toAcc;
    accumulator_ += toAcc;
}

LatticeRescorer* PosteriorLatticeRescorer::createPosteriorLatticeRescorer(const Core::Configuration& config,
                                                                          Bliss::LexiconRef          lexicon) {
    PosteriorType    type     = static_cast<PosteriorType>(paramPosteriorType(config));
    LatticeRescorer* rescorer = 0;
    switch (type) {
        case probability:
            rescorer = new PosteriorLatticeRescorer(config);
            break;
        case expectation:
            rescorer = new ExpectationPosteriorLatticeRescorer(config);
            break;
        case combinedProbability:
            rescorer = new CombinedPosteriorLatticeRescorer(config, lexicon);
            break;
        default:
            defect();
            break;
    }
    return rescorer;
}

/*
 * PosteriorLatticeRescorer: expectation
 */
const Core::ParameterBool ExpectationPosteriorLatticeRescorer::paramVNormalized(
        "v-normalized",
        "posteriors include v-normalization",
        true);

ExpectationPosteriorLatticeRescorer::ExpectationPosteriorLatticeRescorer(
        const Core::Configuration& c)
        : Precursor(c),
          vNormalized_(paramVNormalized(c)) {}

Lattice::ConstWordLatticeRef ExpectationPosteriorLatticeRescorer::work(Lattice::ConstWordLatticeRef lattice,
                                                                       Bliss::SpeechSegment*        segment) {
    Fsa::Weight            expectation;
    Fsa::ConstAutomatonRef fsa = Fsa::posteriorE(
            Fsa::changeSemiring(lattice->part(Lattice::WordLattice::totalFsa), Fsa::LogSemiring),
            lattice->part(Lattice::WordLattice::accuracyFsa),
            expectation,
            vNormalized_,
            tolerance_);
    accumulate(f32(expectation));
    if (!pNormalized_) {
        Fsa::Weight totalInv;
        Fsa::posterior64(fsa, totalInv, tolerance_);
        fsa = Fsa::extend(fsa, totalInv);
    }
    Core::Ref<Lattice::WordLattice> result(new Lattice::WordLattice);
    result->setWordBoundaries(lattice->wordBoundaries());
    result->setFsa(fsa, Lattice::WordLattice::posteriorFsa);
    return result;
}

/*
 * PosteriorLatticeRescorer: combined probability
 */
CombinedPosteriorLatticeRescorer::CombinedPosteriorLatticeRescorer(const Core::Configuration& c, Bliss::LexiconRef lexicon)
        : Precursor(c),
          archiveToCombine_(0) {
    require(!pNormalized_);

    archiveToCombine_ = Lattice::Archive::openForReading(select("lattice-archive-to-combine"), lexicon);
    if (!archiveToCombine_ || archiveToCombine_->hasFatalErrors()) {
        delete archiveToCombine_;
        archiveToCombine_ = 0;
        error("failed to open lattice archive to combine");
    }
}

CombinedPosteriorLatticeRescorer::~CombinedPosteriorLatticeRescorer() {
    delete archiveToCombine_;
}

Lattice::ConstWordLatticeRef CombinedPosteriorLatticeRescorer::work(Lattice::ConstWordLatticeRef lattice,
                                                                    Bliss::SpeechSegment*        segment) {
    const std::string      part = lattice->name(0);
    Fsa::ConstAutomatonRef fsa  = lattice->part(part);

    verify(archiveToCombine_);
    Lattice::ConstWordLatticeRef toCombine = archiveToCombine_->get(segment->fullName(), part);

    Fsa::ConstAutomatonRef united = Fsa::unite(lattice->part(part), toCombine->part(part));
    Fsa::Weight            totalInvToCombine;
    Fsa::posterior64(Fsa::changeSemiring(fsa, Fsa::LogSemiring),
                     totalInvToCombine,
                     tolerance_);

    Fsa::Weight totalInv;
    fsa                          = Fsa::posterior64(Fsa::changeSemiring(fsa, Fsa::LogSemiring), totalInv, tolerance_);
    Fsa::Weight combinedTotalInv = fsa->semiring()->extend(totalInvToCombine, fsa->semiring()->invert(totalInv));
    fsa                          = Fsa::extend(fsa, combinedTotalInv);
    accumulate(f32(combinedTotalInv));

    Core::Ref<Lattice::WordLattice> result(new Lattice::WordLattice);
    result->setWordBoundaries(lattice->wordBoundaries());
    result->setFsa(fsa, Lattice::WordLattice::posteriorFsa);
    return result;
}

/*
 * RecognizerWithConstrainedLanguageModel
 */
const Core::ParameterString RecognizerWithConstrainedLanguageModel::paramPortName(
        "port-name",
        "port name of features",
        "features");

RecognizerWithConstrainedLanguageModel::RecognizerWithConstrainedLanguageModel(const Core::Configuration& c, Bliss::LexiconRef lexicon)
        : Precursor(c),
          portId_(Flow::IllegalPortId),
          recognizer_(0) {
    /*
     * Make sure that there is a single instance of the
     * lexicon because the lexicon is compared by pointers/references.
     * This is why the acoustic model and the language
     * model are created here.
     */
    Core::Ref<Am::AcousticModel> acousticModel = Am::Module::instance().createAcousticModel(select("acoustic-model"), lexicon);
    if (!acousticModel) {
        criticalError("failed to initialize the acoustic model");
    }
    acousticModel_ = acousticModel;

    Core::Ref<Lm::FsaLm> fsaLm(new Lm::FsaLm(select("fsa-lm"), lexicon));
    fsaLm->init();
    Core::Ref<Lm::ScaledLanguageModel> languageModel(
            new Lm::LanguageModelScaling(select("fsa-lm"), fsaLm));
    if (!languageModel) {
        criticalError("failed to initialize language model");
    }
    ModelCombination modelCombination(select("model-combination"), lexicon, acousticModel, languageModel);
    modelCombination.load();

    recognizer_ = new Search::AdvancedTreeSearchManager(select("recognizer"));
    recognizer_->setModelCombination(modelCombination);
    recognizer_->init();
    lemmaPronunciationToLemmaTransducer_ = lexicon->createLemmaPronunciationToLemmaTransducer();
    lemmaToSyntacticTokenTransducer_     = lexicon->createLemmaToSyntacticTokenTransducer();
}

RecognizerWithConstrainedLanguageModel::~RecognizerWithConstrainedLanguageModel() {
    delete recognizer_;
}

void RecognizerWithConstrainedLanguageModel::setGrammar(Fsa::ConstAutomatonRef g) {
    verify(recognizer_);
    recognizer_->setGrammar(g);
    recognizer_->resetStatistics();
    recognizer_->restart();
}

void RecognizerWithConstrainedLanguageModel::feed() {
    require(segmentwiseFeatureExtractor_);
    segmentwiseFeatureExtractor_->checkCompatibility(portId_, acousticModel_);
    ConstSegmentwiseFeaturesRef                     features = segmentwiseFeatureExtractor_->features(portId_);
    Core::Ref<const Mm::FeatureScorer>              featureScorer(acousticModel_->featureScorer());
    std::vector<Core::Ref<Feature>>::const_iterator f = features->begin();
    for (; f != features->end(); ++f) {
        recognizer_->feed(featureScorer->getScorer(*f));
    }
}

Lattice::ConstWordLatticeRef RecognizerWithConstrainedLanguageModel::extract(Lattice::ConstWordLatticeRef lattice,
                                                                             Bliss::SpeechSegment*        segment) {
    Fsa::ConstAutomatonRef f = lattice->mainPart();

    // restrict search space to word sequences in the automaton f
    if (f->getInputAlphabet() == lemmaPronunciationToLemmaTransducer_->getInputAlphabet()) {
        f = Fsa::composeMatching(f, lemmaPronunciationToLemmaTransducer_);
    }
    require(f->getOutputAlphabet() == lemmaToSyntacticTokenTransducer_->getInputAlphabet());
    Fsa::ConstAutomatonRef g = Fsa::multiply(Fsa::projectOutput(Fsa::composeMatching(f, lemmaToSyntacticTokenTransducer_)), Fsa::Weight(f32(0)));
    /*
     * not yet checked: is new implementation more efficient than the old miminize implementation,
     * i.e., Fsa::determinize(Fsa::transpose(Fsa::determinize(Fsa::transpose(g))))?
     */
    g = Fsa::minimize(Fsa::determinize(Fsa::removeEpsilons(g)));
    setGrammar(g);

    // initialize acoustic model
    feed();

    // search
    Search::LatticeHandler*                 handler  = Search::Module::instance().createLatticeHandler(config);
    Core::Ref<const Search::LatticeAdaptor> l        = recognizer_->getCurrentWordLattice();
    Lattice::ConstWordLatticeRef            rescored = l->wordLattice(handler);
    Core::Ref<Lattice::WordLattice>         result(new Lattice::WordLattice);
    result->setWordBoundaries(rescored->wordBoundaries());
    result->setFsa(rescored->part(Lattice::WordLattice::acousticFsa),
                   Lattice::WordLattice::acousticFsa);
    recognizer_->logStatistics();
    return result;
}

void RecognizerWithConstrainedLanguageModel::setSegmentwiseFeatureExtractor(
        Core::Ref<SegmentwiseFeatureExtractor> segmentwiseFeatureExtractor) {
    segmentwiseFeatureExtractor_ = segmentwiseFeatureExtractor;
    portId_                      = segmentwiseFeatureExtractor_->addPort(paramPortName(config));
}
