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
#include "AcousticSegmentwiseTrainer.hh"

#ifndef CMAKE_DISABLE_MODULE_HH
#include <Modules.hh>
#endif
#include "SegmentwiseGmmTrainer.hh"
#ifdef MODULE_NN_SEQUENCE_TRAINING
#include <Nn/SegmentwiseNnTrainer.hh>
#endif

using namespace Speech;

/**
 *  AbstractSegmentwiseTrainer: base class
 */
Core::ParameterString AbstractAcousticSegmentwiseTrainer::paramPortName(
        "port-name",
        "port name for posteriors",
        "features");

Core::ParameterString AbstractAcousticSegmentwiseTrainer::paramSparsePortName(
        "sparse-port-name",
        "sparse port name for posteriors",
        "");

Core::ParameterString AbstractAcousticSegmentwiseTrainer::paramAccumulationPortName(
        "accumulation-port-name",
        "port name for accumulation",
        "features");

Core::ParameterString AbstractAcousticSegmentwiseTrainer::paramAccumulationSparsePortName(
        "accumulation-sparse-port-name",
        "sparse port name for accumulation",
        "");

AbstractAcousticSegmentwiseTrainer::AbstractAcousticSegmentwiseTrainer(
        const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c),
          portId_(Flow::IllegalPortId),
          sparsePortId_(Flow::IllegalPortId),
          accumulationPortId_(Flow::IllegalPortId),
          accumulationSparsePortId_(Flow::IllegalPortId) {}

AbstractAcousticSegmentwiseTrainer::~AbstractAcousticSegmentwiseTrainer() {}

ConstSegmentwiseFeaturesRef AbstractAcousticSegmentwiseTrainer::features(Flow::PortId portId, Flow::PortId sparsePortId) const {
    if (segmentwiseFeatureExtractor()) {
        return segmentwiseFeatureExtractor()->features(portId);
    }
    else {
        return ConstSegmentwiseFeaturesRef();
    }
}

void AbstractAcousticSegmentwiseTrainer::processWordLattice(Lattice::ConstWordLatticeRef lattice,
                                                            Bliss::SpeechSegment*        s) {
    Mm::FeatureDescription description(*this);
    if (features() and !features()->empty()) {
        description = Mm::FeatureDescription(*this, *features()->front());
    }
    setFeatureDescription(description);
}

void AbstractAcousticSegmentwiseTrainer::setSegmentwiseFeatureExtractor(Core::Ref<Speech::SegmentwiseFeatureExtractor> segmentwiseFeatureExtractor) {
    require(segmentwiseFeatureExtractor);
    segmentwiseFeatureExtractor_ = segmentwiseFeatureExtractor;
    const std::string portName   = paramPortName(config);
    if (!portName.empty()) {
        portId_ = segmentwiseFeatureExtractor_->addPort(portName);
        if (portId_ == Flow::IllegalPortId) {
            criticalError("Failed to retrieve output from flow network.");
        }
    }
    const std::string accumulationPortName = paramAccumulationPortName(config);
    if (!accumulationPortName.empty()) {
        accumulationPortId_ = segmentwiseFeatureExtractor_->addPort(accumulationPortName);
        if (accumulationPortId_ == Flow::IllegalPortId) {
            criticalError("Failed to retrieve accumulation output from flow network.");
        }
    }
    Precursor::setSegmentwiseFeatureExtractor(segmentwiseFeatureExtractor_);
}

void AbstractAcousticSegmentwiseTrainer::setAlignmentGenerator(Core::Ref<Speech::PhonemeSequenceAlignmentGenerator> alignmentGenerator) {
    require(alignmentGenerator);
    alignmentGenerator_ = alignmentGenerator;
    Precursor::setAlignmentGenerator(alignmentGenerator);
}

/**
 * factory
 */
Core::Choice AbstractAcousticSegmentwiseTrainer::choiceModelType(
        "gaussian-mixture", gaussianMixture,
        "neural-network", neuralNetwork,
        Core::Choice::endMark());

Core::ParameterChoice AbstractAcousticSegmentwiseTrainer::paramModelType(
        "model-type",
        &choiceModelType,
        "type of model",
        gaussianMixture);

Core::ParameterBool AbstractAcousticSegmentwiseTrainer::paramSinglePrecision(
        "single-precision", "use single precision NN sequence trainer", true);

AbstractAcousticSegmentwiseTrainer* AbstractAcousticSegmentwiseTrainer::createAbstractAcousticSegmentwiseTrainer(const Core::Configuration& config) {
    switch (paramModelType(config)) {
        case gaussianMixture:
            return SegmentwiseGmmTrainer::createSegmentwiseGmmTrainer(config);
            break;
#ifdef MODULE_NN_SEQUENCE_TRAINING
        case neuralNetwork:
            if (paramSinglePrecision(config))
                return Nn::SegmentwiseNnTrainer<f32>::createSegmentwiseNnTrainer(config);
            else
                Core::Application::us()->criticalError("double precision sequence training not implemented yet");
            break;
#endif
        default:
            defect();
    }
    return 0;
}
