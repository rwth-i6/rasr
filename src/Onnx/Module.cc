/*
 * Copyright 2022 AppTek LLC. All rights reserved.
 */
#include "Module.hh"

#include <Flow/Registry.hh>
#include <Mm/FeatureScorerFactory.hh>
#include <Nn/Module.hh>

#include "OnnxEncoder.hh"
#include "OnnxFeatureScorer.hh"
#include "OnnxForwardNode.hh"

namespace Onnx {

Module_::Module_() {
    Mm::Module::instance().featureScorerFactory()->registerFeatureScorer<OnnxFeatureScorer, Mm::MixtureSet, Mm::AbstractMixtureSetLoader>(
            0x400 + 0, "onnx-feature-scorer");  // TODO enum value
    Mm::Module::instance().featureScorerFactory()->registerFeatureScorer<OnnxFeatureScorer, Mm::MixtureSet, Mm::EmptyMixtureSetLoader>(
            0x400 + 1, "onnx-feature-scorer-no-mixture");  // TODO enum value

    Flow::Registry::instance().registerFilter<OnnxForwardNode>();

    // Forward encoder inputs through an onnx model
    Nn::Module::instance().encoderFactory().registerEncoder("onnx", [](Core::Configuration const& config) { return Core::ref(new OnnxEncoder(config)); });
}

}  // namespace Onnx
