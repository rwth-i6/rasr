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
#ifndef _MM_ISMOOTHING_MIXTURE_ESTIMATOR_HH
#define _MM_ISMOOTHING_MIXTURE_ESTIMATOR_HH

#include "MixtureSetEstimator.hh"

namespace Mm {

class DiscriminativeMixtureEstimator;

/**
 *  i-smoothing mixture estimator
 */
class ISmoothingMixtureEstimator {
    friend class MixtureSetEstimatorIndexMap;

private:
    DiscriminativeMixtureEstimator* parent_;
    std::vector<Weight>             iMixtureWeights_;
    Weight                          constant_;

protected:
    virtual void removeDensity(DensityIndex indexInMixture);

public:
    ISmoothingMixtureEstimator();
    virtual ~ISmoothingMixtureEstimator();

    void           set(DiscriminativeMixtureEstimator*);
    virtual void   clear();
    Sum            getObjectiveFunction() const;
    void           setIMixture(const Mixture*);
    virtual Weight iMixtureWeight(DensityIndex dns) const {
        return iMixtureWeights_[dns];
    }
    void setConstant(Weight constant) {
        constant_ = constant;
    }
    Weight constant() const {
        return constant_;
    }
    u32 nMixtureWeights() const {
        return iMixtureWeights_.size();
    }
};

}  // namespace Mm

#endif  //_MM_ISMOOTHING_MIXTURE_ESTIMATOR_HH
