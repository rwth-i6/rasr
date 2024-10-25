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
#include <Core/Configuration.hh>
#include <Core/FormatSet.hh>
#include <Flow/Registry.hh>
#include "LabelScorer/Encoder.hh"
#include "LabelScorer/EncoderDecoderLabelScorer.hh"
#include "LabelScorer/LabelScorer.hh"
#ifdef MODULE_ONNX
#include "LabelScorer/LimitCtxTimesyncOnnxDecoder.hh"
#include "LabelScorer/OnnxEncoder.hh"
#include "LabelScorer/StatefulFullEncOnnxDecoder.hh"
#endif

#ifndef CMAKE_DISABLE_MODULES_HH
#include <Modules.hh>
#endif
#include "Module.hh"
#include "Statistics.hh"

#ifdef MODULE_NN
#include <Mm/FeatureScorerFactory.hh>
#include <Mm/Module.hh>
#ifdef MODULE_NN_SEQUENCE_TRAINING
#include "EmissionLatticeRescorer.hh"
#endif
#include "BatchFeatureScorer.hh"
#include "FeatureScorer.hh"
#include "NeuralNetworkForwardNode.hh"
#include "TrainerFeatureScorer.hh"
#endif
#ifdef MODULE_PYTHON
#include "PythonFeatureScorer.hh"
#endif

using namespace Nn;

Module_::Module_()
        : formats_(0) {
    Flow::Registry::Instance& registry = Flow::Registry::instance();

#ifdef MODULE_NN
    /* neural network forward node */
    registry.registerFilter<NeuralNetworkForwardNode>();

    Mm::Module::instance().featureScorerFactory()->registerFeatureScorer<OnDemandFeatureScorer, Mm::MixtureSet, Mm::AbstractMixtureSetLoader>(
            nnOnDemanHybrid, "nn-on-demand-hybrid");
    Mm::Module::instance().featureScorerFactory()->registerFeatureScorer<FullFeatureScorer, Mm::MixtureSet, Mm::AbstractMixtureSetLoader>(
            nnFullHybrid, "nn-full-hybrid");
    Mm::Module::instance().featureScorerFactory()->registerFeatureScorer<PrecomputedFeatureScorer, Mm::MixtureSet, Mm::AbstractMixtureSetLoader>(
            nnPrecomputedHybrid, "nn-precomputed-hybrid");
    Mm::Module::instance().featureScorerFactory()->registerFeatureScorer<BatchFeatureScorer, Mm::MixtureSet, Mm::AbstractMixtureSetLoader>(
            nnBatchFeatureScorer, "nn-batch-feature-scorer");
    Mm::Module::instance().featureScorerFactory()->registerFeatureScorer<TrainerFeatureScorer, Mm::MixtureSet, Mm::AbstractMixtureSetLoader>(
            nnTrainerFeatureScorer, "nn-trainer-feature-scorer");
#endif
#ifdef MODULE_NN_SEQUENCE_TRAINING
    Mm::Module::instance().featureScorerFactory()->registerFeatureScorer<CachedNeuralNetworkFeatureScorer, Mm::MixtureSet, Mm::AbstractMixtureSetLoader>(
            nnCached, "nn-cached");

#endif
#ifdef MODULE_PYTHON
    Mm::Module::instance().featureScorerFactory()->registerFeatureScorer<PythonFeatureScorer, Mm::MixtureSet, Mm::AbstractMixtureSetLoader>(
            pythonFeatureScorer, "python-feature-scorer");
#endif
};

Module_::~Module_() {
    if (formats_)
        delete formats_;
}

Core::FormatSet& Module_::formats() {
    Core::Configuration c = Core::Configuration(Core::Application::us()->getConfiguration(), "file-format-set");
    if (!formats_) {
        formats_ = new Core::FormatSet(c);
        formats_->registerFormat("bin", new Core::CompressedBinaryFormat<Statistics<f32>>(), true);
        formats_->registerFormat("bin", new Core::CompressedBinaryFormat<Statistics<f64>>(), true);
    }
    return *formats_;
}

const Core::Choice Module_::encoderTypeChoice(
        // Assume encoder inputs are already finished states and just pass them on without transformations
        "no-op", EncoderType::NoOpEncoderType,
        // Forward encoder inputs through an onnx network
        "onnx-encoder", EncoderType::OnnxEncoderType,
        Core::Choice::endMark());

const Core::Choice Module_::decoderTypeChoice(
        // Assume encoder states are already finished scores and just pass them on without transformations
        "no-op", DecoderType::NoOpDecoderType,
        // Wrapper around legacy Mm::FeatureScorer
        "legacy-feature-scorer", DecoderType::LegacyFeatureScorerDecoderType,
        // Forward a single encoder state and (fixed-size) history through an onnx network each step
        "limited-ctx-timesync-onnx-decoder", DecoderType::LimitCtxTimesyncOnnxDecoderType,
        // Forward all encoder states and a hidden state through an onnx network each step
        "stateful-full-enc-onnx-decoder", DecoderType::StatefulFullEncOnnxDecoderType,
        Core::Choice::endMark());

const Core::Choice Module_::labelScorerTypeChoice(
        "no-op", LabelScorerType::NoOpLabelScorerType,
        "encoder-decoder", LabelScorerType::EncoderDecoderLabelScorerType,
        "encoder-only", LabelScorerType::EncoderOnlyLabelScorerType,
        "decoder-only", LabelScorerType::DecoderOnlyLabelScorerType,
        Core::Choice::endMark());

const Core::ParameterChoice Module_::paramEncoderType(
        "encoder-type",
        &Module_::encoderTypeChoice,
        "Choice from a set of encoder types.",
        EncoderType::NoOpEncoderType);

const Core::ParameterChoice Module_::paramDecoderType(
        "decoder-type",
        &decoderTypeChoice,
        "Choice from a set of decoder types.",
        DecoderType::NoOpDecoderType);

const Core::ParameterChoice Module_::paramLabelScorerType(
        "type", &Module_::labelScorerTypeChoice, "Choice from a set of label scorer types.", LabelScorerType::NoOpLabelScorerType);

Core::Ref<Encoder> Module_::createEncoder(const Core::Configuration& config) const {
    auto encoderConfig = Core::Configuration(config, "encoder");

    Core::Ref<Encoder> result;
    switch (paramEncoderType(config)) {
        case EncoderType::NoOpEncoderType:
            result = Core::ref(new NoOpEncoder(encoderConfig));
            break;
#ifdef MODULE_ONNX
        case EncoderType::OnnxEncoderType:
            result = Core::ref(new OnnxEncoder(encoderConfig));
            break;
#endif
        default:
            Core::Application::us()->criticalError("unknown encoder type: %d", paramEncoderType(config));
    }

    return result;
}

Core::Ref<Decoder> Module_::createDecoder(const Core::Configuration& config) const {
    auto decoderConfig = Core::Configuration(config, "decoder");

    Core::Ref<Decoder> result;
    switch (paramDecoderType(config)) {
        case DecoderType::NoOpDecoderType:
            result = Core::ref(new Nn::NoOpDecoder(decoderConfig));
            break;
        case DecoderType::LegacyFeatureScorerDecoderType:
            result = Core::ref(new Nn::LegacyFeatureScorerDecoder(decoderConfig));
            break;
        case DecoderType::LimitCtxTimesyncOnnxDecoderType:
            result = Core::ref(new Nn::LimitedCtxOnnxDecoder(decoderConfig));
            break;
        case DecoderType::StatefulFullEncOnnxDecoderType:
            result = Core::ref(new Nn::StatefulFullEncOnnxDecoder(decoderConfig));
            break;
        default:
            Core::Application::us()->criticalError("unknown decoder type: %d", paramDecoderType(config));
    }

    return result;
}

Core::Ref<LabelScorer> Module_::createLabelScorer(const Core::Configuration& config) const {
    Core::Ref<LabelScorer> result;
    switch (paramLabelScorerType(config)) {
        case LabelScorerType::NoOpLabelScorerType:
            result = Core::ref(new EncoderDecoderLabelScorer(config, Core::ref(new Nn::NoOpEncoder(config)), Core::ref(new Nn::NoOpDecoder(config))));
        case LabelScorerType::EncoderOnlyLabelScorerType:
            result = Core::ref(new EncoderDecoderLabelScorer(config, createEncoder(config), Core::ref(new Nn::NoOpDecoder(config))));
            break;
        case LabelScorerType::DecoderOnlyLabelScorerType:
            result = Core::ref(new EncoderDecoderLabelScorer(config, Core::ref(new Nn::NoOpEncoder(config)), createDecoder(config)));
            break;
        case LabelScorerType::EncoderDecoderLabelScorerType:
            result = Core::ref(new EncoderDecoderLabelScorer(config, createEncoder(config), createDecoder(config)));
            break;
        default:
            Core::Application::us()->criticalError("unknown label scorer type: %d", paramLabelScorerType(config));
    }
    return result;
}
