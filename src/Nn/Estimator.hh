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
#ifndef _NN_NEURAL_NETWORK_ESTIMATOR_HH
#define _NN_NEURAL_NETWORK_ESTIMATOR_HH

#include "NeuralNetwork.hh"
#include "Statistics.hh"
#include "Types.hh"

namespace Nn {

/*---------------------------------------------------------------------------*/

/* Base class for weight estimators */

template<typename T>
class Estimator : virtual public Core::Component {
    typedef Core::Component             Precursor;
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;

public:
    enum EstimatorType {
        dummy,
        dryRun,
    };

protected:
    Core::XmlChannel statisticsChannel_;

public:
    static const Core::Choice          choiceEstimatorType;
    static const Core::ParameterChoice paramEstimatorType;
    static const Core::ParameterBool   paramBatchMode;
    static const Core::ParameterInt    paramAccumulateMultipleBatches;
    static const Core::ParameterFloat  paramLearningRate;
    static const Core::ParameterFloat  paramBiasLearningRate;
    static const Core::ParameterBool   paramLogStepSize;

protected:
    bool fullBatchMode_;
    int  accumulateMultipleBatches_;
    T    initialLearningRate_;
    T    biasLearningRate_;
    bool logStepSize_;

public:
    Estimator(const Core::Configuration& config);
    virtual ~Estimator() {}
    // Note that the batch settings are not used by the estimator itself
    // (estimate() will not depend on it)
    // but rather the trainer should check for it and implement the
    // necessary behavior.
    // operate in full batch mode ( = pass over full training data)
    virtual bool fullBatchMode() const {
        return fullBatchMode_;
    }
    virtual void setFullBatchMode(bool fullBatchMode) {
        fullBatchMode_ = fullBatchMode;
    }
    // accumulate over multiple batches
    virtual int accumulateMultipleBatches() const {
        return accumulateMultipleBatches_;
    }
    virtual T learningRate() const {
        return initialLearningRate_;
    }
    virtual T biasLearningRateFactor() const {
        return biasLearningRate_;
    }
    virtual void setLearningRate(T rate) {
        initialLearningRate_ = rate;
    }
    // estimate new model based on previous model and statistics
    virtual void estimate(NeuralNetwork<T>& network, Statistics<T>& statistics) {};
    // name of estimator
    virtual std::string type() const {
        return "dummy";
    }
    // id of required statistics
    virtual u32 requiredStatistics() const {
        return Statistics<T>::NONE;
    }
    // creation method
    static Estimator<T>* createEstimator(const Core::Configuration& config);
};

/*---------------------------------------------------------------------------*/

// Another dummy, which is useful for dry-runs where you want to have the gradient calculated.
template<typename T>
class DryRunEstimator : public Estimator<T> {
    typedef Estimator<T> Precursor;

public:
    DryRunEstimator(const Core::Configuration& config)
            : Core::Component(config),
              Precursor(config) {}
    virtual std::string type() const {
        return "dry-run";
    }
    virtual u32 requiredStatistics() const {
        return Statistics<T>::GRADIENT;
    }
};

}  // namespace Nn

#endif
