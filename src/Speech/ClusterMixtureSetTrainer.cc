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
#include "ClusterMixtureSetTrainer.hh"
#include <typeinfo>

using namespace Speech;

/**
 * Cluster mixture set trainer
 */
ClusterMixtureSetTrainer::ClusterMixtureSetTrainer(const Core::Configuration &c) :
    Core::Component(c),
    Precursor(c)
{
    estimator_ = 0;
}

ClusterMixtureSetTrainer::~ClusterMixtureSetTrainer()
{}

void ClusterMixtureSetTrainer::cluster()
{
    Core::Ref<Mm::MixtureSet> originalMixtureSet  = estimate();
    Core::Ref<Mm::MixtureSet> clusteredMixtureSet =
        Core::ref(originalMixtureSet->createOneMixtureClusterCopy(select("clustering")));
    if (estimator_ and (typeid(estimator_) == typeid(Mm::ConvertMixtureSetEstimator*))) {
        Mm::ConvertMixtureSetEstimator& estim = *(dynamic_cast<Mm::ConvertMixtureSetEstimator*>(estimator_));
        estim.setMixtureSet(clusteredMixtureSet);
    }
    else
        defect();
}
