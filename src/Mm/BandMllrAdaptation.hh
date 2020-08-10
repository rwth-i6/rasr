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
#ifndef _MM_BANDMLLRADAPTATION_HH
#define _MM_BANDMLLRADAPTATION_HH

#include "MllrAdaptation.hh"

namespace Mm {

/**
 * Estimator for MLLR using band matrices
 */
class BandMllrEstimator : public FullAdaptorViterbiEstimator {
public:
    typedef FullAdaptorViterbiEstimator Precursor;

private:
    static const Core::ParameterInt paramNBands_;
    u16                             nBands_;

    Math::Vector<Matrix::Type>
            solveRowEquation(const Matrix& g, const Matrix& z, u32 row);

protected:
    BandMllrEstimator(const Core::Configuration&          c,
                      ComponentIndex                      dimension,
                      const Core::Ref<Am::AdaptationTree> adaptationTree);

    virtual void estimateWMatrices();

public:
    BandMllrEstimator(const Core::Configuration&          c,
                      const Core::Ref<Am::AdaptationTree> adaptationTree);
    BandMllrEstimator(const Core::Configuration&          c,
                      Core::Ref<const Mm::MixtureSet>     mixtureSet,
                      const Core::Ref<Am::AdaptationTree> adaptationTree);
    virtual ~BandMllrEstimator(){};

    // virtual AdaptorEstimator* clone() const;
    virtual std::string typeName() const {
        return "band-mllr-estimator";
    }
    virtual bool write(Core::BinaryOutputStream& o) const;
    virtual bool read(Core::BinaryInputStream& i);
};

}  //namespace Mm

#endif  // _MM_BANDMLLRADAPTATION_HH
