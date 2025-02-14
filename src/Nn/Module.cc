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
#include "LabelScorer/CombineLabelScorer.hh"
#include "LabelScorer/Encoder.hh"
#include "LabelScorer/EncoderDecoderLabelScorer.hh"
#include "LabelScorer/LabelScorer.hh"
#include "LabelScorer/LegacyFeatureScorerLabelScorer.hh"
#include "LabelScorer/NoOpLabelScorer.hh"
#include "Nn/LabelScorer/CTCPrefixLabelScorer.hh"
#include "Nn/LabelScorer/EncoderFactory.hh"
#ifdef MODULE_ONNX
#include "LabelScorer/LimitedCtxOnnxLabelScorer.hh"
#include "LabelScorer/NoCtxOnnxLabelScorer.hh"
#include "LabelScorer/OnnxEncoder.hh"
#include "LabelScorer/StatefulOnnxLabelScorer.hh"
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
        : formats_(0),
          encoderFactory_(),
          labelScorerFactory_() {
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

    // Doesn't apply any transformation to the inputs; just pass them on
    encoderFactory_.registerEncoder(
            "no-op",
            [](Core::Configuration const& config) {
                return Core::ref(new NoOpEncoder(config));
            });

    // Forward encoder inputs through an onnx network
    encoderFactory_.registerEncoder(
            "onnx",
            [](Core::Configuration const& config) {
                return Core::ref(new OnnxEncoder(config));
            });

    // Forward  encoder inputs chunkwise through an onnx network
    encoderFactory_.registerEncoder(
            "chunked-onnx",
            [](Core::Configuration const& config) {
                return Core::ref(new ChunkedOnnxEncoder(config));
            });

    // Assume inputs are already finished scores and just passes on the score at the current step
    labelScorerFactory_.registerLabelScorer(
            "no-op",
            [](Core::Configuration const& config) {
                return Core::ref(new StepwiseNoOpLabelScorer(config));
            });

    // Wrapper around legacy Mm::FeatureScorer
    labelScorerFactory_.registerLabelScorer(
            "legacy-feature-scorer",
            [](Core::Configuration const& config) {
                return Core::ref(new LegacyFeatureScorerLabelScorer(config));
            });

    // Forward the feature at the current step through an onnx network
    labelScorerFactory_.registerLabelScorer(
            "no-ctx-onnx",
            [](Core::Configuration const& config) {
                return Core::ref(new NoCtxOnnxLabelScorer(config));
            });

    // Forward the feature at the current step together with a (fixed-size) history through an onnx network
    labelScorerFactory_.registerLabelScorer(
            "limited-ctx-onnx",
            [](Core::Configuration const& config) {
                return Core::ref(new LimitedCtxOnnxLabelScorer(config));
            });

    // A label scorer consisting of an encoder that pre-processes the features and another label scorer acting as decoder
    labelScorerFactory_.registerLabelScorer(
            "encoder-decoder",
            [this](Core::Configuration const& config) {
                return Core::ref(new EncoderDecoderLabelScorer(
                        config,
                        createEncoder(Core::Configuration(config, "encoder")),
                        createLabelScorer(Core::Configuration(config, "decoder"))));
            });

    // A label scorer consisting of an encoder that produces scores based on the features
    labelScorerFactory_.registerLabelScorer(
            "encoder-only",
            [this](Core::Configuration const& config) {
                return Core::ref(new EncoderDecoderLabelScorer(
                        config,
                        createEncoder(Core::Configuration(config, "encoder")),
                        Core::ref(new StepwiseNoOpLabelScorer(config))));
            });

    // Scoring based on hidden states that are initialized and updated based on features and history tokens
    labelScorerFactory_.registerLabelScorer(
            "stateful-onnx",
            [](Core::Configuration const& config) {
                return Core::ref(new StatefulOnnxLabelScorer(config));
            });

    // Adds the scores of multiple scaled sub-label-scorers
    labelScorerFactory_.registerLabelScorer(
            "combine",
            [](Core::Configuration const& config) {
                return Core::ref(new CombineLabelScorer(config));
            });

    // Performs label-synchronous prefix scoring using an underlying CTC model
    labelScorerFactory_.registerLabelScorer(
            "ctc-prefix",
            [](Core::Configuration const& config) {
                return Core::ref(new CTCPrefixLabelScorer(config));
            });
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

EncoderFactory& Module_::encoderFactory() {
    return encoderFactory_;
}

LabelScorerFactory& Module_::labelScorerFactory() {
    return labelScorerFactory_;
}

Core::Ref<Encoder> Module_::createEncoder(Core::Configuration const& config) const {
    return encoderFactory_.createEncoder(config);
}

Core::Ref<LabelScorer> Module_::createLabelScorer(Core::Configuration const& config) const {
    return labelScorerFactory_.createLabelScorer(config);
}
