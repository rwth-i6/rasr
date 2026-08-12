/** Copyright 2026 RWTH Aachen University. All rights reserved.
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

#include "Module.hh"

#include <Flow/Registry.hh>
#include <Mm/FeatureScorerFactory.hh>
#include <Nn/Module.hh>

#include "TorchEncoder.hh"
#include "TorchFeatureScorer.hh"
#include "TorchForwardNode.hh"

namespace Torch {

Module_::Module_() {
    Mm::Module::instance().featureScorerFactory()->registerFeatureScorer<TorchFeatureScorer, Mm::MixtureSet, Mm::AbstractMixtureSetLoader>(
            0x500 + 0, "torch-feature-scorer");
    Mm::Module::instance().featureScorerFactory()->registerFeatureScorer<TorchFeatureScorer, Mm::MixtureSet, Mm::EmptyMixtureSetLoader>(
            0x500 + 1, "torch-feature-scorer-no-mixture");

    Flow::Registry::instance().registerFilter<TorchForwardNode>();

    // Forward encoder inputs through a Torch model
    Nn::Module::instance().encoderFactory().registerEncoder(
            "torch",
            [](Core::Configuration const& config, Nn::ModelCache& modelCache) {
                return Core::ref(new TorchEncoder(config, modelCache));
            });
}

}  // namespace Torch
