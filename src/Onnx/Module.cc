/*
 * Copyright 2022 AppTek LLC. All rights reserved.
 */
#include "Module.hh"

#include <Flow/Registry.hh>
#include <Mm/FeatureScorerFactory.hh>

#include "OnnxFeatureScorer.hh"
#include "OnnxForwardNode.hh"

namespace Onnx {

Module_::Module_() {
    Mm::Module::instance().featureScorerFactory()->registerFeatureScorer<OnnxFeatureScorer, Mm::MixtureSet, Mm::AbstractMixtureSetLoader>(
            0x400 + 0, "onnx-feature-scorer");  // TODO enum value

    Flow::Registry::instance().registerFilter<OnnxForwardNode>();
}

}  // namespace Onnx
