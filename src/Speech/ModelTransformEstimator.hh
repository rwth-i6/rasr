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
#ifndef _SPEECH_MODEL_TRANSFORM_ESTIMATOR_HH
#define _SPEECH_MODEL_TRANSFORM_ESTIMATOR_HH

#include <Am/AcousticModel.hh>
#include <Am/AdaptationTree.hh>
#include <Bliss/CorpusKey.hh>
#include <Core/IoRef.hh>
#include <Core/ObjectCache.hh>
#include <Math/Matrix.hh>
#include <Mm/AffineFeatureTransformAccumulator.hh>
#include <Mm/MllrAdaptation.hh>
#include "AcousticModelTrainer.hh"
#include "Feature.hh"
#include "KeyedEstimator.hh"

namespace Speech {

/**
 * ModelTransformEstimator
 */
class ModelTransformEstimator : public KeyedEstimator {
    typedef KeyedEstimator       Precursor;
    typedef Mm::AdaptorEstimator ConcreteAccumulator;

public:
    static const Core::Choice          mllrModelingChoice;
    static const Core::ParameterChoice paramMllrModeling;
    enum MllrModelingMode {
        fullMllr,
        semiTiedMllr,
        bandMllr,
        shiftMllr
    };

private:
    Core::Ref<Am::AdaptationTree> adaptationTree_;
    MllrModelingMode              mllrModeling_;

    typedef Core::ObjectCache<Core::MruObjectCacheList<
            std::string,
            Core::IoRef<Mm::Adaptor>,
            Core::StringHash,
            Core::StringEquality>>
            AdaptorCache;

protected:
    void createAccumulator(std::string key);

public:
    ModelTransformEstimator(const Core::Configuration& c, Operation op = estimate);
    virtual ~ModelTransformEstimator();

    void postProcess();
};

}  // namespace Speech

#endif  // _SPEECH_MODEL_TRANSFORM_ESTIMATOR_HH
