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
#include <Core/MapParser.hh>
#include <Math/Vector.hh>
#include <Math/Lapack/Lapack.hh>
#include "MeanEstimator.hh"

using namespace Speech;


const Core::ParameterString MeanEstimator::paramFile(
    "file", "output filename for mean");

MeanEstimator::MeanEstimator(const Core::Configuration &c)
    : Component(c), Extractor(c), Estimator(c), needResize_(true)
{}

MeanEstimator::~MeanEstimator()
{}

void MeanEstimator::processFeature(Core::Ref<const Feature> feature)
{
    accumulate(*feature->mainStream());
}

void MeanEstimator::setFeatureDescription(const Mm::FeatureDescription &description)
{
    description.verifyNumberOfStreams(1);
    size_t d;
    description.mainStream().getValue(Mm::FeatureDescription::nameDimension, d);
    if(needResize_) {
        setDimension(d);
        needResize_ = false;
    }
}
