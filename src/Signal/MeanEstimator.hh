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
#ifndef _SIGNAL_MEAN_ESTIMATOR_HH
#define _SIGNAL_MEAN_ESTIMATOR_HH

#include <Core/Component.hh>
#include <Math/Vector.hh>
#include "Node.hh"

namespace Signal {

/**
 *  Estimator class for covariance matrix
 */
class MeanEstimator : virtual public Core::Component {
    typedef Core::Component Precursor;

public:
    typedef f32 Data;
    typedef f64 Sum;
    typedef u32 Count;

public:
    static const Core::ParameterString paramFilename;
    static const Core::ParameterInt    paramOutputPrecision;

private:
    size_t featureDimension_;
    /* Accumulates vector per class */
    Math::Vector<Sum> vectorSum_;
    /* Class frequency */
    Count count_;
    bool  needInit_;

private:
    void initialize();

public:
    MeanEstimator(const Core::Configuration& c);
    ~MeanEstimator();

    void setDimension(size_t dimension);

    void accumulate(const Math::Vector<Data>&);
    bool finalize(Math::Vector<Data>& mean) const;
    /**
     *  Saves mean.
     *  Calls finalize and saves mean vector
     */
    bool write() const;
    void reset();
};

class MeanEstimatorNode : public Flow::SleeveNode,
                          public MeanEstimator {
    typedef Flow::SleeveNode Precursor;

public:
    MeanEstimatorNode(const Core::Configuration& c)
            : Core::Component(c),
              Precursor(c),
              MeanEstimator(c) {}
    virtual ~MeanEstimatorNode() {}
    bool               configure();
    bool               work(Flow::PortId p);
    static std::string filterName() {
        return "signal-mean-estimator";
    }
};

}  // namespace Signal

#endif  // _SIGNAL_MEAN_ESTIMATOR_HH
