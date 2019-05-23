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
#include "EmissionLatticeRescorer.hh"

#include <sys/time.h>

#include <Core/Utility.hh>
#include <Speech/ModelCombination.hh>

#include "ClassLabelWrapper.hh"
#include "LinearAndActivationLayer.hh"
#include "NeuralNetwork.hh"

using namespace Nn;

/*
 * LatticeRescorer: emission
 */
const Core::ParameterString EmissionLatticeRescorer::paramPortName(
        "port-name",
        "port name of features",
        "features");

const Core::ParameterBool EmissionLatticeRescorer::paramMeasureTime(
        "measure-time", "Measures time for executing methods in FeedForwardTrainer", false);

const Core::ParameterBool EmissionLatticeRescorer::paramCheckValues(
        "check-values", "check output of network for finiteness", false);

EmissionLatticeRescorer::EmissionLatticeRescorer(const Core::Configuration& c, bool initialize)
        : Precursor(c),
          measureTime_(paramMeasureTime(c)),
          checkValues_(paramCheckValues(c)),
          timeMemoryAllocation_(0),
          timeForwarding_(0),
          timeIO_(0),
          portId_(Flow::IllegalPortId) {
    if (initialize) {
        Speech::ModelCombination modelCombination(select("model-combination"),
                                                  Speech::ModelCombination::useAcousticModel,
                                                  Am::AcousticModel::noStateTransition);
        modelCombination.load();
        acousticModel_ = modelCombination.acousticModel();
    }
    logProperties();
}

EmissionLatticeRescorer::EmissionLatticeRescorer(const Core::Configuration&   c,
                                                 Core::Ref<Am::AcousticModel> acousticModel)
        : Precursor(c),
          measureTime_(paramMeasureTime(c)),
          checkValues_(paramCheckValues(c)),
          timeMemoryAllocation_(0),
          timeForwarding_(0),
          timeIO_(0),
          portId_(Flow::IllegalPortId) {
    acousticModel_ = acousticModel;
    logProperties();
}

void EmissionLatticeRescorer::setSegmentwiseFeatureExtractor(Core::Ref<Speech::SegmentwiseFeatureExtractor> segmentwiseFeatureExtractor) {
    segmentwiseFeatureExtractor_ = segmentwiseFeatureExtractor;
    portId_                      = segmentwiseFeatureExtractor_->addPort(paramPortName(config));
}

Lattice::ConstWordLatticeRef EmissionLatticeRescorer::work(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* segment) {
    EmissionLatticeRescorerAutomaton* f = 0;
// read alignment and forward network in parallel threads
// TODO better read emission indices directly from a file here
#pragma omp parallel sections
    {
#pragma omp section  // read alignment
        {
            timeval start, end;
            TIMER_START(start)
            alignmentGenerator_->setSpeechSegment(segment);
            TIMER_STOP(start, end, timeIO_);
        }
#pragma omp section  // forward network
        {
            timeval                             start, end;
            Speech::ConstSegmentwiseFeaturesRef features;
            verify(segmentwiseFeatureExtractor_);
            features = segmentwiseFeatureExtractor_->features(portId_);

            // always resize the activations to the sequence length
            // TODO set a maximum in order to limit memory
            TIMER_START(start);
            network().resizeActivations(features->size());
            TIMER_GPU_STOP(start, end, measureTime_, timeMemoryAllocation_)

            TIMER_START(start);
            f = new EmissionLatticeRescorerAutomaton(
                    lattice, alignmentGenerator_, acousticModel_, features, checkValues_);
            TIMER_GPU_STOP(start, end, measureTime_, timeForwarding_);
        }
    }
    // implicit omp barrier
    verify(f);
    Lattice::WordLattice* result = new Lattice::WordLattice;
    result->setWordBoundaries(lattice->wordBoundaries());
    result->setFsa(Fsa::ConstAutomatonRef(f), Lattice::WordLattice::acousticFsa);
    return Lattice::ConstWordLatticeRef(result);
}

void EmissionLatticeRescorer::setAlignmentGenerator(AlignmentGeneratorRef alignmentGenerator) {
    Precursor::setAlignmentGenerator(alignmentGenerator);
    require(alignmentGenerator_);
    if (alignmentGenerator_->labelType() == Speech::Alignment::allophoneStateIds)
        log("alignment generator uses label type: allophone-state-ids");
    else
        log("alignment generator uses label type: emission-ids");
}

void EmissionLatticeRescorer::logProperties() const {
    log("using feature port with name ") << paramPortName(config);
}

void EmissionLatticeRescorer::finalize() {
    if (measureTime_) {
        this->log() << Core::XmlOpen("time-emission-lattice-rescorer")
                    << Core::XmlFull("IO", timeIO_)
                    << Core::XmlFull("memory-allocation", timeMemoryAllocation_)
                    << Core::XmlFull("forwarding", timeForwarding_)
                    << Core::XmlClose("time-emission-lattice-rescorer");
    }
}

/*
 * EmissionLatticeRescorerAutomaton
 */
EmissionLatticeRescorerAutomaton::EmissionLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef        lattice,
                                                                   AlignmentGeneratorRef               alignmentGenerator,
                                                                   Core::Ref<Am::AcousticModel>        acousticModel,
                                                                   Speech::ConstSegmentwiseFeaturesRef features,
                                                                   bool                                checkValues)
        : Precursor(lattice),
          alignmentGenerator_(alignmentGenerator),
          acousticModel_(acousticModel),
          features_(features) {
    require(alignmentGenerator_);
    labelType_ = alignmentGenerator_->labelType();

    bool forwardOk = forwardNetwork(checkValues);
    require(forwardOk);
}

EmissionLatticeRescorerAutomaton::~EmissionLatticeRescorerAutomaton() {}

bool EmissionLatticeRescorerAutomaton::forwardNetwork(bool checkValues) {
    if (features_->size() == 0) {
        Core::Application::us()->warning("no features in segment");
        return true;
    }
    u32 nStreams = features_->at(0)->nStreams();
    u32 nFrames  = features_->size();
    inputFeatures_.resize(nStreams, NnMatrix());

    for (u32 stream = 0; stream < nStreams; ++stream) {
        u32 dim = (*features_->at(0))[stream]->size();
        inputFeatures_.at(stream).resize(dim, nFrames);
        for (u32 t = 0; t < nFrames; t++) {
            Math::copy(dim, &((*features_->at(t))[stream]->at(0)), 1, &(inputFeatures_.at(stream).at(0, t)), 1);
        }
    }

    if (!network().isComputing()) {
        network().initComputation();
    }
    bool result = network().forward(inputFeatures_);
    network().getTopLayerOutput().finishComputation();
    if (checkValues && !network().getTopLayerOutput().isFinite()) {
        Core::Application::us()->error("non-finite scores, check whether model is valid (maybe learning rate too large in sequence training?)");
    }

    return result;
}

Fsa::Weight EmissionLatticeRescorerAutomaton::score(Fsa::StateId s, const Fsa::Arc& a) const {
    // TODO avoid all of this and load emission indices (or even neural network output indices) directly from file
    const Bliss::LemmaPronunciationAlphabet* alphabet      = required_cast(const Bliss::LemmaPronunciationAlphabet*, fsa_->getInputAlphabet().get());
    const Bliss::LemmaPronunciation*         pronunciation = alphabet->lemmaPronunciation(a.input());
    const TimeframeIndex                     begtime       = wordBoundaries_->time(s);
    if (pronunciation && begtime != Speech::InvalidTimeframeIndex) {
        Bliss::Coarticulated<Bliss::LemmaPronunciation> coarticulatedPronunciation(*pronunciation,
                                                                                   wordBoundaries_->transit(s).final,
                                                                                   wordBoundaries_->transit(fsa_->getState(a.target())->id()).initial);
        const TimeframeIndex                            endtime = wordBoundaries_->time(fsa_->getState(a.target())->id());
        return _score(coarticulatedPronunciation, begtime, endtime);
    }
    else {
        return fsa_->semiring()->one();
    }
}

Fsa::Weight EmissionLatticeRescorerAutomaton::_score(const Bliss::Coarticulated<Bliss::LemmaPronunciation>& coarticulatedPronunciation,
                                                     TimeframeIndex begtime, TimeframeIndex endtime) const {
    verify(acousticModel_);
    f32 score = fsa_->semiring()->one();
    if (begtime < endtime) {
        const Speech::Alignment* alignment = alignmentGenerator_->getAlignment(coarticulatedPronunciation, begtime, endtime);
        if (labelType_ == Speech::Alignment::allophoneStateIds) {
            for (std::vector<Speech::AlignmentItem>::const_iterator al = alignment->begin();
                 al != alignment->end(); ++al) {
                score -= network().getTopLayerOutput()(labelWrapper().getOutputIndexFromClassIndex(acousticModel_->emissionIndex(al->emission)), al->time);
            }
        }
        else {
            for (std::vector<Speech::AlignmentItem>::const_iterator al = alignment->begin(); al != alignment->end(); ++al) {
                score -= network().getTopLayerOutput()(labelWrapper().getOutputIndexFromClassIndex(al->emission), al->time);
            }
        }
    }
    else {
        Core::Application::us()->warning("score 0 assigned to arc with begin time ")
                << begtime << " , end time " << endtime << " and label id " << coarticulatedPronunciation.object().id();
    }
    return Fsa::Weight(score);
}

/*
 *
 *  CachedNeuralNetworkFeatureScorer
 *
 */

inline Mm::Score CachedNeuralNetworkFeatureScorer::score(u32 time, Mm::EmissionIndex e) const {
    return -network().getTopLayerOutput().at(labelWrapper().getOutputIndexFromClassIndex(e), time);
}

Mm::ComponentIndex CachedNeuralNetworkFeatureScorer::dimension() const {
    return network().getLayer(0).getInputDimension(0);
}

Core::Ref<const Mm::FeatureScorer::ContextScorer> CachedNeuralNetworkFeatureScorer::getScorer(Core::Ref<const Mm::Feature>) const {
    criticalError("getScorer(Mm::Feature) not available");
    return Scorer(new ActivationLookupScorer(this, 0));
}

Core::Ref<const Mm::FeatureScorer::ContextScorer> CachedNeuralNetworkFeatureScorer::getScorer(const Mm::FeatureVector& f) const {
    criticalError("getScorer(Mm::FeatureVector) not available");
    return Scorer(new ActivationLookupScorer(this, 0));
}

Core::Ref<const Mm::FeatureScorer::ContextScorer> CachedNeuralNetworkFeatureScorer::getTimeIndexedScorer(u32 time) const {
    return Scorer(new ActivationLookupScorer(this, time));
}
