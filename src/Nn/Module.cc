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
#include <Core/FormatSet.hh>
#include <Flow/Registry.hh>
#include <Modules.hh>

#include "LabelScorer/LabelScorerFactory.hh"
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

LabelScorerFactory& Module_::labelScorerFactory() {
    return labelScorerFactory_;
}
