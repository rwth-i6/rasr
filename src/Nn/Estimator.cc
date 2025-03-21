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
#include "Estimator.hh"
#include "LinearLayer.hh"
#include "MeanNormalizedSgdEstimator.hh"
#include "RpropEstimator.hh"

using namespace Nn;

template<class T>
const Core::Choice Estimator<T>::choiceEstimatorType(
        "dummy", dummy,
        "dry-run", dryRun,
        "steepest-descent", steepestDescentEstimator,
        "steepest-descent-l1-clipping", steepestDescentL1Clipping,
        "mean-normalized-steepest-descent", meanNormalizedSgd,
        "mean-normalized-steepest-descent-l1-clipping", meanNormalizedSgdL1Clipping,
        "rprop", rprop,
        "prior-estimator", priorEstimator,
        "adam", adam,
        "adagrad", adagrad,
        "adadelta", adadelta,
        "rmsprop", rmsprop,
        Core::Choice::endMark());

template<class T>
const Core::ParameterChoice Estimator<T>::paramEstimatorType(
        "estimator", &choiceEstimatorType,
        "estimator for weights estimation in training", dummy);

template<typename T>
const Core::ParameterBool Estimator<T>::paramBatchMode(
        "batch-mode",
        "use batch estimator, i.e. do not update after each mini-batch, but accumulate statistics", false);

template<typename T>
const Core::ParameterInt Estimator<T>::paramAccumulateMultipleBatches(
        "accumulate-multiple-batches",
        "If greater than 1, will accumulate these number of batches. "
        "0 = not used. batch-mode=true is like accumulate-multiple-batches=<corpus-batch-number>. "
        "Note that this option only make sense if you have mini-batches with different sizes "
        "such as with BufferedSegmentFeatureProcessor -- otherwise, you could just change the "
        "mini-batch size.",
        0);

template<typename T>
const Core::ParameterFloat Estimator<T>::paramLearningRate("learning-rate", "(initial) learning-rate", 1.0);

template<typename T>
const Core::ParameterFloat Estimator<T>::paramBiasLearningRate(
        "bias-learning-rate", "bias is optimized with bias-learning-rate * learning-rate", 1.0);

template<typename T>
const Core::ParameterBool Estimator<T>::paramLogStepSize(
        "log-step-size", "log the step size, if true", false);

template<typename T>
Estimator<T>::Estimator(const Core::Configuration& config)
        : Precursor(config),
          statisticsChannel_(config, "statistics"),
          fullBatchMode_(paramBatchMode(config)),
          accumulateMultipleBatches_(paramAccumulateMultipleBatches(config)),
          initialLearningRate_(paramLearningRate(config)),
          biasLearningRate_(paramBiasLearningRate(config)),
          logStepSize_(paramLogStepSize(config)) {
    this->log("initial learning rate: ") << initialLearningRate_;
    if (fullBatchMode_)
        this->log("using full batch estimator");
    if (!fullBatchMode_ && accumulateMultipleBatches_ == 0)
        accumulateMultipleBatches_ = 1;
    if (accumulateMultipleBatches_ > 1)
        this->log("accumulate over %i batches", accumulateMultipleBatches_);
    if (accumulateMultipleBatches_ < 0)
        this->criticalError("%s cannot be negative", paramAccumulateMultipleBatches.name().c_str());
    if (fullBatchMode_ && accumulateMultipleBatches_ > 0)
        this->criticalError("full batch (%s=true) and %s does not make sense",
                            paramBatchMode.name().c_str(), paramAccumulateMultipleBatches.name().c_str());
    if (biasLearningRate_ != 1.0)
        this->log("bias learning rate: ") << biasLearningRate_;
    if (logStepSize_) {
        this->log("logging step size norm");
    }
}

template<class T>
Estimator<T>* Estimator<T>::createEstimator(const Core::Configuration& config) {
    Estimator<T>* estimator;

    EstimatorType type = (EstimatorType)paramEstimatorType(config);
    switch (type) {
        case steepestDescentEstimator:
            estimator = new SteepestDescentEstimator<T>(config);
            Core::Application::us()->log("Create Estimator: steepest-descent");
            break;
        case steepestDescentL1Clipping:
            estimator = new SteepestDescentL1ClippingEstimator<T>(config);
            Core::Application::us()->log("Create Estimator: steepest-descent-l1-clipping");
            break;
        case meanNormalizedSgd:
            estimator = new MeanNormalizedSgd<T>(config);
            Core::Application::us()->log("Create Estimator: mean-normalized-steepest-descent");
            break;
        case meanNormalizedSgdL1Clipping:
            estimator = new MeanNormalizedSgdL1Clipping<T>(config);
            Core::Application::us()->log("Create Estimator: mean-normalized-steepest-descent-l1-clipping-estimator");
            break;
        case rprop:
            estimator = new RpropEstimator<T>(config);
            Core::Application::us()->log("Create Estimator: Rprop");
            break;
        case priorEstimator:
            estimator = new PriorEstimator<T>(config);
            Core::Application::us()->log("Create Estimator: Prior estimator");
            break;
        case adam:
            estimator = new Adam<T>(config);
            Core::Application::us()->log("Create Estimator: Adam");
            break;
        case adagrad:
            estimator = new AdaGrad<T>(config);
            Core::Application::us()->log("Create Estimator: AdaGrad");
            break;
        case adadelta:
            estimator = new AdaDelta<T>(config);
            Core::Application::us()->log("Create Estimator: AdaDelta");
            break;
        case rmsprop:
            estimator = new RmsProp<T>(config);
            Core::Application::us()->log("Create Estimator: RMSProp");
            break;
        case dryRun:
            estimator = new DryRunEstimator<T>(config);
            Core::Application::us()->log("Create Estimator: dry-run (with gradient calculation)");
            break;
        default:
            verify_eq(type, EstimatorType::dummy);
            estimator = new Estimator<T>(config);
            Core::Application::us()->log("Create Estimator: dummy");
            break;
    };

    return estimator;
}

//=============================================================================

template<typename T>
const Core::ParameterBool SteepestDescentEstimator<T>::paramUsePredefinedLearningRateDecay(
        "use-predefined-learning-rate-decay", "use learning-rate * tau / (tau + numberOfUpdates) as learning-rate", false);

template<typename T>
const Core::ParameterFloat SteepestDescentEstimator<T>::paramLearningRateTau("learning-rate-tau", "", 1000.0);

template<typename T>
const Core::ParameterInt SteepestDescentEstimator<T>::paramNumberOfUpdates(
        "number-of-updates", "number of updates done so far", 0);

template<typename T>
const Core::ParameterFloat SteepestDescentEstimator<T>::paramClippingThreshold(
        "clipping-threshold", "clip updates if larger than learning-rate * clipping-threshold", Core::Type<T>::max);

template<typename T>
const Core::ParameterFloat SteepestDescentEstimator<T>::paramMomentumFactor(
        "momentum-factor", "momentum factor, suggested value: 0.9", 0.0);

template<typename T>
SteepestDescentEstimator<T>::SteepestDescentEstimator(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config),
          usePredefinedLearningRateDecay_(paramUsePredefinedLearningRateDecay(config)),
          tau_(paramLearningRateTau(config)),
          nUpdates_(paramNumberOfUpdates(config)),
          clippingThreshold_(paramClippingThreshold(config)),
          momentumFactor_(paramMomentumFactor(config)),
          momentum_(momentumFactor_ > 0),
          oldDeltas_(0) {
    if (usePredefinedLearningRateDecay_) {
        this->log("using predefined learning rate decay with parameter tau: ") << tau_;
        this->log("number of updates so far is ") << nUpdates_;
    }
    if (clippingThreshold_ < Core::Type<T>::max) {
        this->log("clipping updates if larger than ") << clippingThreshold_;
        if (momentum_)
            this->error("momentum with clipping not implemented yet ..");
    }
    if (momentum_)
        this->log("using momentum with momentum factor: ") << momentumFactor_;
}

template<typename T>
SteepestDescentEstimator<T>::~SteepestDescentEstimator() {
    if (momentum_) {
        delete oldDeltas_;
    }
}

template<typename T>
bool SteepestDescentEstimator<T>::isDefaultConfig() {
    if (usePredefinedLearningRateDecay_)
        return false;
    if (clippingThreshold_ < Core::Type<T>::max)
        return false;
    if (momentum_)
        return false;
    return true;
}

template<typename T>
void SteepestDescentEstimator<T>::estimate(NeuralNetwork<T>& network, Statistics<T>& statistics) {
    T learningRate = initialLearningRate_;
    if (usePredefinedLearningRateDecay_) {
        learningRate = initialLearningRate_ * tau_ / (tau_ + nUpdates_);
    }

    /* estimation of parameters */
    require(statistics.hasGradient());

    // if momentum is used and oldDeltas_ not yet initialized, just copy the statistics
    if (momentum_ && (oldDeltas_ == 0)) {
        oldDeltas_ = new Statistics<T>(statistics);
    }

    if (usePredefinedLearningRateDecay_ && statisticsChannel_.isOpen())
        statisticsChannel_ << "learningRate: " << learningRate;

    std::vector<T> stepSizes(network.nLayers());
    for (u32 layer = 0; layer < network.nLayers(); layer++) {
        if (network.getLayer(layer).isTrainable()) {
            /* estimation of weights */
            for (u32 stream = 0; stream < network.getLayer(layer).nInputActivations(); stream++) {
                NnMatrix* weights = network.getLayer(layer).getWeights(stream);
                require(weights);
                T localLearningRate = learningRate * network.getLayer(layer).learningRate();
                /* regular update, if no momentum used */
                if (!momentum_) {
                    if (clippingThreshold_ < Core::Type<T>::max) {
                        T localClippingThreshold = localLearningRate > 0 ? clippingThreshold_ / localLearningRate : 0;
                        statistics.gradientWeights(layer)[stream].clip(localClippingThreshold);
                    }
                    weights->add(statistics.gradientWeights(layer)[stream], (T)-localLearningRate);
                }
                /* momentum update */
                else {
                    // update old deltas (include new statistics)
                    oldDeltas_->gradientWeights(layer)[stream].scale(momentumFactor_);
                    oldDeltas_->gradientWeights(layer)[stream].add(statistics.gradientWeights(layer)[stream], (T)(1 - momentumFactor_));
                    // update weights
                    weights->add(oldDeltas_->gradientWeights(layer)[stream], (T)-localLearningRate);
                }
                /* log step size */
                if (logStepSize_) {
                    if (!momentum_)
                        stepSizes[layer] = statistics.gradientWeights(layer)[stream].l1norm() * localLearningRate;
                    else
                        stepSizes[layer] = oldDeltas_->gradientWeights(layer)[stream].l1norm() * localLearningRate;
                }
            }

            /* estimation of bias */
            NnVector* bias = network.getLayer(layer).getBias();
            require(bias);
            T localLearningRate = learningRate * biasLearningRate_ * network.getLayer(layer).learningRate();
            /* regular update, if no momentum used */
            if (!momentum_) {
                if (clippingThreshold_ < Core::Type<T>::max) {
                    T localClippingThreshold = localLearningRate > 0 ? clippingThreshold_ / localLearningRate : 0;
                    statistics.gradientBias(layer).clip(localClippingThreshold);
                }
                bias->add(statistics.gradientBias(layer), (T)-localLearningRate);
            }
            /* momentum update */
            else {
                // update old deltas (include new statistics)
                oldDeltas_->gradientBias(layer).scale(momentumFactor_);
                oldDeltas_->gradientBias(layer).add(statistics.gradientBias(layer), (T)(1 - momentumFactor_));
                // update bias
                bias->add(oldDeltas_->gradientBias(layer), (T)-localLearningRate);
            }
            /* log step size */
            if (logStepSize_) {
                if (!momentum_)
                    stepSizes[layer] += statistics.gradientBias(layer).l1norm() * localLearningRate;
                else
                    stepSizes[layer] += oldDeltas_->gradientBias(layer).l1norm() * localLearningRate;
            }
        }
    }

    if (logStepSize_ && statisticsChannel_.isOpen()) {
        T stepSize = Math::asum<T>(stepSizes.size(), &stepSizes.at(0), 1);
        statisticsChannel_ << "step-size: " << stepSize << " (" << Core::vector2str(stepSizes, ",") << ")";
    }

    nUpdates_++;
}

//=============================================================================

template<typename T>
SteepestDescentL1ClippingEstimator<T>::SteepestDescentL1ClippingEstimator(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config) {}

template<typename T>
void SteepestDescentL1ClippingEstimator<T>::estimate(NeuralNetwork<T>& network, Statistics<T>& statistics) {
    Precursor::estimate(network, statistics);

    T learningRate = this->initialLearningRate_;
    if (this->usePredefinedLearningRateDecay_) {
        learningRate = this->initialLearningRate_ * this->tau_ / (this->tau_ + this->nUpdates_ - 1);
    }

    for (u32 layer = 0; layer < network.nLayers(); layer++) {
        if (network.getLayer(layer).isTrainable()) {
            /* estimation of weights */
            for (u32 stream = 0; stream < network.getLayer(layer).nInputActivations(); stream++) {
                NnMatrix* weights = network.getLayer(layer).getWeights(stream);
                require(weights);
                weights->l1clipping(network.getLayer(layer).regularizationConstant() * learningRate * network.getLayer(layer).learningRate());
            }
            /* estimation of bias */
            NnVector* bias = network.getLayer(layer).getBias();
            require(bias);
            bias->l1clipping(network.getLayer(layer).regularizationConstant() * learningRate * network.getLayer(layer).learningRate());
        }
    }

    if (logStepSize_ && statisticsChannel_.isOpen()) {
        statisticsChannel_ << "step size does not include l1-regularization";
    }
}

//=============================================================================

template<typename T>
const Core::ParameterBool Adam<T>::paramUsePredefinedLearningRateDecay(
        "use-predefined-learning-rate-decay", "use learning-rate * tau / (tau + numberOfUpdates) as learning-rate", false);

template<typename T>
const Core::ParameterFloat Adam<T>::paramLearningRateTau("learning-rate-tau", "", 1000.0);

template<typename T>
const Core::ParameterInt Adam<T>::paramNumberOfUpdates(
        "number-of-updates", "number of updates done so far", 0);

template<typename T>
const Core::ParameterFloat Adam<T>::paramClippingThreshold(
        "clipping-threshold", "clip updates if larger than learning-rate * clipping-threshold", Core::Type<T>::max);

template<typename T>
const Core::ParameterFloat Adam<T>::paramMomentumFactor(
        "momentum-factor", "momentum factor, suggested value: 0.9", 0.0);

template<typename T>
const Core::ParameterFloat Adam<T>::paramBeta1("adam-beta1", "beta1", 0.9);

template<typename T>
const Core::ParameterFloat Adam<T>::paramBeta2("adam-beta2", "beta2", 0.999);

template<typename T>
const Core::ParameterFloat Adam<T>::paramEpsilon("adam-epsilon", "epsilon", 1e-8);

template<typename T>
Adam<T>::Adam(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config),
          usePredefinedLearningRateDecay_(paramUsePredefinedLearningRateDecay(config)),
          tau_(paramLearningRateTau(config)),
          nUpdates_(paramNumberOfUpdates(config)),
          clippingThreshold_(paramClippingThreshold(config)),
          momentumFactor_(paramMomentumFactor(config)),
          momentum_(momentumFactor_ > 0),
          oldDeltas_(0),
          m_(),
          v_(),
          M_(),
          V_(),
          g2_(),
          max_stream_(0),
          b1_(paramBeta1(config)),
          b2_(paramBeta2(config)),
          eps_(paramEpsilon(config)) {
    this->log("Initializing Adam with b1=") << b1_ << " b2=" << b2_ << " eps=" << eps_;

    if (usePredefinedLearningRateDecay_) {
        this->log("using predefined learning rate decay with parameter tau: ") << tau_;
        this->log("number of updates so far is ") << nUpdates_;
    }
    if (clippingThreshold_ < Core::Type<T>::max) {
        this->log("clipping updates if larger than ") << clippingThreshold_;
        if (momentum_)
            this->error("momentum with clipping not implemented yet ..");
    }
    if (momentum_)
        this->log("using momentum with momentum factor: ") << momentumFactor_;
}

template<typename T>
Adam<T>::~Adam() {
    if (momentum_) {
        delete oldDeltas_;
    }
}

template<typename T>
bool Adam<T>::isDefaultConfig() {
    if (usePredefinedLearningRateDecay_)
        return false;
    if (clippingThreshold_ < Core::Type<T>::max)
        return false;
    if (momentum_)
        return false;
    return true;
}

template<typename T>
void Adam<T>::estimate(NeuralNetwork<T>& network, Statistics<T>& statistics) {
    T learningRate = initialLearningRate_;
    T b1           = b1_;
    T b2           = b2_;
    T eps          = eps_;

    /* estimation of parameters */
    require(statistics.hasGradient());

    if (usePredefinedLearningRateDecay_ && statisticsChannel_.isOpen())
        statisticsChannel_ << "learningRate: " << learningRate;

    /**
     * data structures are allocated as follows:
     * max_stream_ = maximum number of streams per layer (over all layers), e.g. 2
     * pointer maps are indexed using the following scheme (layer l, stream s)
     * weights "w", bias "b"
     *
     *   0 -> l0s0w (0 * (2+1) + 0)
     *   1 -> l0s1w (0 * (2+1) + 1)
     *   2 -> l0b   (0 * (2+1) + 2)
     *
     *   3 -> l1s0w (1 * (2+1) + 0)
     *   5 -> l1b   (1 * (2+1) + 2)
     *
     *   6 -> l2s0w (2 * (2+1) + 0)
     *   8 -> l2b   (2 * (2+1) + 2)
     *  ...
     * for weight matrices:
     *  idx = layer * (max_stream_+1) + stream
     * for bias vectors:
     *  idx = layer * (max_stream_+1) + max_stream_
     */

    if (m_.size() == 0) {
        nUpdates_ = 1;
        u32 idx;
        for (u32 layer = 0; layer < network.nLayers(); layer++) {
            max_stream_ = std::max(network.getLayer(layer).nInputActivations(), max_stream_);
        }

        for (u32 layer = 0; layer < network.nLayers(); layer++) {
            if (!network.getLayer(layer).isTrainable())
                continue;
            for (u32 stream = 0; stream < network.getLayer(layer).nInputActivations(); stream++) {
                NnMatrix* w = network.getLayer(layer).getWeights(stream);
                idx         = layer * (max_stream_ + 1) + stream;
                m_[idx]     = new NnMatrix(w->nRows(), w->nColumns());
                v_[idx]     = new NnMatrix(w->nRows(), w->nColumns());
                V_[idx]     = new NnMatrix(w->nRows(), w->nColumns());
                g2_[idx]    = new NnMatrix(w->nRows(), w->nColumns());

                m_[idx]->setToZero();
                v_[idx]->setToZero();

                m_[idx]->initComputation();
                v_[idx]->initComputation();
                V_[idx]->initComputation();
                g2_[idx]->initComputation();
            }
            // biases
            idx         = layer * (max_stream_ + 1) + max_stream_;
            NnVector* b = network.getLayer(layer).getBias();
            m_[idx]     = new NnMatrix(b->nRows(), 1);
            v_[idx]     = new NnMatrix(b->nRows(), 1);
            V_[idx]     = new NnMatrix(b->nRows(), 1);
            g2_[idx]    = new NnMatrix(b->nRows(), 1);

            m_[idx]->setToZero();
            v_[idx]->setToZero();

            m_[idx]->initComputation();
            v_[idx]->initComputation();
            V_[idx]->initComputation();
            g2_[idx]->initComputation();
        }
    }

    std::vector<T> stepSizes(network.nLayers());
    for (u32 layer = 0; layer < network.nLayers(); layer++) {
        if (network.getLayer(layer).isTrainable()) {
            /* estimation of weights */
            for (u32 stream = 0; stream < network.getLayer(layer).nInputActivations(); stream++) {
                NnMatrix* weights = network.getLayer(layer).getWeights(stream);
                require(weights);
                T localLearningRate = learningRate * network.getLayer(layer).learningRate();
                /* regular update, if no momentum used */
                if (!momentum_) {
                    if (clippingThreshold_ < Core::Type<T>::max) {
                        T localClippingThreshold = localLearningRate > 0 ? clippingThreshold_ / localLearningRate : 0;
                        statistics.gradientWeights(layer)[stream].clip(localClippingThreshold);
                    }

                    // SGD:
                    // w = w + (-lr * g)
                    // weights->add(statistics.gradientWeights(layer)[stream],(T) -localLearningRate);
                    //
                    // Adam:
                    // init m=v=0
                    //
                    // m = b1*m + (1-b1)*g
                    // v = b2*v + (1-b2)*g^2 (elementwise square)
                    //
                    // lr_ = lr * sqrt(1 - b2^t) / (1 - b1^t)
                    // w = w + (-lr_ * m/(sqrt(v)+eps))

                    NnMatrix* w   = network.getLayer(layer).getWeights(stream);
                    NnMatrix* g   = &statistics.gradientWeights(layer)[stream];
                    u32       idx = layer * (max_stream_ + 1) + stream;

                    NnMatrix* m  = m_[idx];
                    NnMatrix* v  = v_[idx];
                    NnMatrix* g2 = g2_[idx];
                    NnMatrix* V  = V_[idx];

                    m->scale(b1);
                    m->add(*g, T(1 - b1));

                    g2->copy(*g);
                    g2->pow(T(2));
                    v->scale(b2);
                    v->add(*g2, T(1 - b2));

                    V->copy(*v);
                    V->pow(T(0.5));
                    V->addConstantElementwise(eps);
                    V->pow(T(-1));
                    V->elementwiseMultiplication(*m);

                    T t   = nUpdates_;
                    T lr_ = -localLearningRate * sqrt(1 - pow(b2, t)) / (1 - pow(b1, t));
                    w->add(*V, lr_);
                }
            }

            /* estimation of bias */
            NnVector* bias = network.getLayer(layer).getBias();
            require(bias);
            T localLearningRate = learningRate * biasLearningRate_ * network.getLayer(layer).learningRate();
            /* regular update, if no momentum used */
            if (!momentum_) {
                if (clippingThreshold_ < Core::Type<T>::max) {
                    T localClippingThreshold = localLearningRate > 0 ? clippingThreshold_ / localLearningRate : 0;
                    statistics.gradientBias(layer).clip(localClippingThreshold);
                }

                NnVector* w   = network.getLayer(layer).getBias();
                NnVector* g   = &statistics.gradientBias(layer);
                u32       idx = layer * (max_stream_ + 1) + max_stream_;

                NnMatrix* m  = m_[idx];
                NnMatrix* v  = v_[idx];
                NnMatrix* g2 = g2_[idx];
                NnMatrix* V  = V_[idx];

                m->scale(b1);
                m->addToAllColumns(*g, T(1 - b1));

                g2->setColumn(0, *g);
                g2->pow(T(2));
                v->scale(b2);
                v->add(*g2, T(1 - b2));

                V->copy(*v);
                V->pow(T(0.5));
                V->addConstantElementwise(eps);
                V->pow(T(-1));
                V->elementwiseMultiplication(*m);

                T t   = nUpdates_;
                T lr_ = -localLearningRate * sqrt(1 - pow(b2, t)) / (1 - pow(b1, t));
                w->addSummedColumns(*V, lr_);
            }
        }
    }
    if (logStepSize_ && statisticsChannel_.isOpen()) {
        T stepSize = Math::asum<T>(stepSizes.size(), &stepSizes.at(0), 1);
        statisticsChannel_ << "step-size: " << stepSize << " (" << Core::vector2str(stepSizes, ",") << ")";
    }

    nUpdates_++;
}

//=============================================================================

template<typename T>
const Core::ParameterBool AdaGrad<T>::paramUsePredefinedLearningRateDecay(
        "use-predefined-learning-rate-decay", "use learning-rate * tau / (tau + numberOfUpdates) as learning-rate", false);

template<typename T>
const Core::ParameterFloat AdaGrad<T>::paramLearningRateTau("learning-rate-tau", "", 1000.0);

template<typename T>
const Core::ParameterInt AdaGrad<T>::paramNumberOfUpdates(
        "number-of-updates", "number of updates done so far", 0);

template<typename T>
const Core::ParameterFloat AdaGrad<T>::paramClippingThreshold(
        "clipping-threshold", "clip updates if larger than learning-rate * clipping-threshold", Core::Type<T>::max);

template<typename T>
const Core::ParameterFloat AdaGrad<T>::paramMomentumFactor(
        "momentum-factor", "momentum factor, suggested value: 0.9", 0.0);

template<typename T>
const Core::ParameterFloat AdaGrad<T>::paramBeta1("adagrad-beta1", "beta1 (initial accumulator value)", 0.1);

template<typename T>
const Core::ParameterFloat AdaGrad<T>::paramEpsilon("adagrad-epsilon", "epsilon", 1e-8);

template<typename T>
AdaGrad<T>::AdaGrad(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config),
          usePredefinedLearningRateDecay_(paramUsePredefinedLearningRateDecay(config)),
          tau_(paramLearningRateTau(config)),
          nUpdates_(paramNumberOfUpdates(config)),
          clippingThreshold_(paramClippingThreshold(config)),
          momentumFactor_(paramMomentumFactor(config)),
          momentum_(momentumFactor_ > 0),
          oldDeltas_(0),
          v_(),
          V_(),
          g2_(),
          max_stream_(0),
          b1_(paramBeta1(config)),
          eps_(paramEpsilon(config)) {
    this->log("Initializing AdaGrad with b1=") << b1_ << " eps=" << eps_;

    if (usePredefinedLearningRateDecay_) {
        this->log("using predefined learning rate decay with parameter tau: ") << tau_;
        this->log("number of updates so far is ") << nUpdates_;
    }
    if (clippingThreshold_ < Core::Type<T>::max) {
        this->log("clipping updates if larger than ") << clippingThreshold_;
        if (momentum_)
            this->error("momentum with clipping not implemented yet ..");
    }
    if (momentum_)
        this->log("using momentum with momentum factor: ") << momentumFactor_;
}

template<typename T>
AdaGrad<T>::~AdaGrad() {
    if (momentum_) {
        delete oldDeltas_;
    }
}

template<typename T>
bool AdaGrad<T>::isDefaultConfig() {
    if (usePredefinedLearningRateDecay_)
        return false;
    if (clippingThreshold_ < Core::Type<T>::max)
        return false;
    if (momentum_)
        return false;
    return true;
}

template<typename T>
void AdaGrad<T>::estimate(NeuralNetwork<T>& network, Statistics<T>& statistics) {
    T learningRate = initialLearningRate_;
    T b1           = b1_;
    T eps          = eps_;

    /* estimation of parameters */
    require(statistics.hasGradient());

    if (usePredefinedLearningRateDecay_ && statisticsChannel_.isOpen())
        statisticsChannel_ << "learningRate: " << learningRate;

    /**
     * data structures are allocated as follows:
     * max_stream_ = maximum number of streams per layer (over all layers), e.g. 2
     * pointer maps are indexed using the following scheme (layer l, stream s)
     *
     *   0 -> l0s0w (0 * (2+1) + 0)
     *   1 -> l0s1w (0 * (2+1) + 1)
     *   2 -> l0b   (0 * (2+1) + 2)
     *
     *   3 -> l1s0w (1 * (2+1) + 0)
     *   5 -> l1b   (1 * (2+1) + 2)
     *
     *   6 -> l2s0w (2 * (2+1) + 0)
     *   8 -> l2b   (2 * (2+1) + 2)
     *  ...
     * for weight matrices:
     *  idx = layer * (max_stream_+1) + stream
     * for bias vectors:
     *  idx = layer * (max_stream_+1) + max_stream_
     */

    if (v_.size() == 0) {
        nUpdates_ = 1;
        u32 idx;
        for (u32 layer = 0; layer < network.nLayers(); layer++) {
            max_stream_ = std::max(network.getLayer(layer).nInputActivations(), max_stream_);
        }

        for (u32 layer = 0; layer < network.nLayers(); layer++) {
            if (!network.getLayer(layer).isTrainable())
                continue;
            for (u32 stream = 0; stream < network.getLayer(layer).nInputActivations(); stream++) {
                NnMatrix* w = network.getLayer(layer).getWeights(stream);
                idx         = layer * (max_stream_ + 1) + stream;
                v_[idx]     = new NnMatrix(w->nRows(), w->nColumns());
                V_[idx]     = new NnMatrix(w->nRows(), w->nColumns());
                g2_[idx]    = new NnMatrix(w->nRows(), w->nColumns());

                v_[idx]->initComputation();
                V_[idx]->initComputation();
                g2_[idx]->initComputation();

                v_[idx]->fill(b1);
            }
            // biases
            idx         = layer * (max_stream_ + 1) + max_stream_;
            NnVector* b = network.getLayer(layer).getBias();
            v_[idx]     = new NnMatrix(b->nRows(), 1);
            V_[idx]     = new NnMatrix(b->nRows(), 1);
            g2_[idx]    = new NnMatrix(b->nRows(), 1);

            v_[idx]->initComputation();
            V_[idx]->initComputation();
            g2_[idx]->initComputation();

            v_[idx]->fill(b1);
        }
    }

    std::vector<T> stepSizes(network.nLayers());
    for (u32 layer = 0; layer < network.nLayers(); layer++) {
        if (network.getLayer(layer).isTrainable()) {
            /* estimation of weights */
            for (u32 stream = 0; stream < network.getLayer(layer).nInputActivations(); stream++) {
                NnMatrix* weights = network.getLayer(layer).getWeights(stream);
                require(weights);
                T localLearningRate = learningRate * network.getLayer(layer).learningRate();
                /* regular update, if no momentum used */
                if (!momentum_) {
                    if (clippingThreshold_ < Core::Type<T>::max) {
                        T localClippingThreshold = localLearningRate > 0 ? clippingThreshold_ / localLearningRate : 0;
                        statistics.gradientWeights(layer)[stream].clip(localClippingThreshold);
                    }

                    // SGD:
                    // w = w + (-lr * g)
                    // weights->add(statistics.gradientWeights(layer)[stream],(T) -localLearningRate);
                    //
                    // AdaGrad:
                    // init v=0 (or 0.1?), eps=1e-8 (or 0?)
                    //
                    // v += g^2 (elementwise square)
                    //
                    // w = w + (-lr * g/(sqrt(v)+eps))

                    // printf("layer %d, stream %d\n", layer, stream);
                    NnMatrix* w   = network.getLayer(layer).getWeights(stream);
                    NnMatrix* g   = &statistics.gradientWeights(layer)[stream];
                    u32       idx = layer * (max_stream_ + 1) + stream;

                    NnMatrix* v  = v_[idx];
                    NnMatrix* g2 = g2_[idx];
                    NnMatrix* V  = V_[idx];

                    g2->copy(*g);
                    g2->pow(T(2));
                    v->add(*g2);

                    V->copy(*v);
                    if (eps != 0) {
                        V->pow(T(0.5));
                        V->addConstantElementwise(eps);
                        V->pow(T(-1));
                    }
                    else {
                        V->pow(T(-0.5));
                    }
                    V->elementwiseMultiplication(*g);

                    w->add(*V, -localLearningRate);
                }
            }

            /* estimation of bias */
            NnVector* bias = network.getLayer(layer).getBias();
            require(bias);
            T localLearningRate = learningRate * biasLearningRate_ * network.getLayer(layer).learningRate();
            /* regular update, if no momentum used */
            if (!momentum_) {
                if (clippingThreshold_ < Core::Type<T>::max) {
                    T localClippingThreshold = localLearningRate > 0 ? clippingThreshold_ / localLearningRate : 0;
                    statistics.gradientBias(layer).clip(localClippingThreshold);
                }

                NnVector* w   = network.getLayer(layer).getBias();
                NnVector* g   = &statistics.gradientBias(layer);
                u32       idx = layer * (max_stream_ + 1) + max_stream_;

                NnMatrix* v  = v_[idx];
                NnMatrix* g2 = g2_[idx];
                NnMatrix* V  = V_[idx];

                g2->setColumn(0, *g);
                g2->pow(T(2));
                v->add(*g2);

                V->copy(*v);
                if (eps != 0) {
                    V->pow(T(0.5));
                    V->addConstantElementwise(eps);
                    V->pow(T(-1));
                }
                else {
                    V->pow(T(-0.5));
                }
                g2->setColumn(0, *g);
                V->elementwiseMultiplication(*g2);

                w->addSummedColumns(*V, localLearningRate);
            }
        }
    }
    if (logStepSize_ && statisticsChannel_.isOpen()) {
        T stepSize = Math::asum<T>(stepSizes.size(), &stepSizes.at(0), 1);
        statisticsChannel_ << "step-size: " << stepSize << " (" << Core::vector2str(stepSizes, ",") << ")";
    }

    nUpdates_++;
}

//=============================================================================

template<typename T>
const Core::ParameterBool AdaDelta<T>::paramUsePredefinedLearningRateDecay(
        "use-predefined-learning-rate-decay", "use learning-rate * tau / (tau + numberOfUpdates) as learning-rate", false);

template<typename T>
const Core::ParameterFloat AdaDelta<T>::paramLearningRateTau("learning-rate-tau", "", 1000.0);

template<typename T>
const Core::ParameterInt AdaDelta<T>::paramNumberOfUpdates(
        "number-of-updates", "number of updates done so far", 0);

template<typename T>
const Core::ParameterFloat AdaDelta<T>::paramClippingThreshold(
        "clipping-threshold", "clip updates if larger than learning-rate * clipping-threshold", Core::Type<T>::max);

template<typename T>
const Core::ParameterFloat AdaDelta<T>::paramMomentumFactor(
        "momentum-factor", "momentum factor, suggested value: 0.9", 0.0);

template<typename T>
const Core::ParameterFloat AdaDelta<T>::paramBeta1("adadelta-beta1", "beta1", 0.1);

template<typename T>
const Core::ParameterFloat AdaDelta<T>::paramEpsilon("adadelta-epsilon", "epsilon", 1e-8);

template<typename T>
AdaDelta<T>::AdaDelta(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config),
          usePredefinedLearningRateDecay_(paramUsePredefinedLearningRateDecay(config)),
          tau_(paramLearningRateTau(config)),
          nUpdates_(paramNumberOfUpdates(config)),
          clippingThreshold_(paramClippingThreshold(config)),
          momentumFactor_(paramMomentumFactor(config)),
          momentum_(momentumFactor_ > 0),
          oldDeltas_(0),
          g2_(),
          Eg2_(),
          Edx2_(),
          RMSg_(),
          max_stream_(0),
          b1_(paramBeta1(config)),
          eps_(paramEpsilon(config)) {
    this->log("Initializing AdaDelta with b1=") << b1_ << " eps=" << eps_;

    if (usePredefinedLearningRateDecay_) {
        this->log("using predefined learning rate decay with parameter tau: ") << tau_;
        this->log("number of updates so far is ") << nUpdates_;
    }
    if (clippingThreshold_ < Core::Type<T>::max) {
        this->log("clipping updates if larger than ") << clippingThreshold_;
        if (momentum_)
            this->error("momentum with clipping not implemented yet ..");
    }
    if (momentum_)
        this->log("using momentum with momentum factor: ") << momentumFactor_;
}

template<typename T>
AdaDelta<T>::~AdaDelta() {
    if (momentum_) {
        delete oldDeltas_;
    }
}

template<typename T>
bool AdaDelta<T>::isDefaultConfig() {
    if (usePredefinedLearningRateDecay_)
        return false;
    if (clippingThreshold_ < Core::Type<T>::max)
        return false;
    if (momentum_)
        return false;
    return true;
}

template<typename T>
void AdaDelta<T>::estimate(NeuralNetwork<T>& network, Statistics<T>& statistics) {
    T learningRate = initialLearningRate_;
    T b1           = b1_;
    T eps          = eps_;

    /* estimation of parameters */
    require(statistics.hasGradient());

    if (usePredefinedLearningRateDecay_ && statisticsChannel_.isOpen())
        statisticsChannel_ << "learningRate: " << learningRate;

    /**
     * data structures are allocated as follows:
     * max_stream_ = maximum number of streams per layer (over all layers), e.g. 2
     * pointer maps are indexed using the following scheme (layer l, stream s)
     *
     *   0 -> l0s0w (0 * (2+1) + 0)
     *   1 -> l0s1w (0 * (2+1) + 1)
     *   2 -> l0b   (0 * (2+1) + 2)
     *
     *   3 -> l1s0w (1 * (2+1) + 0)
     *   5 -> l1b   (1 * (2+1) + 2)
     *
     *   6 -> l2s0w (2 * (2+1) + 0)
     *   8 -> l2b   (2 * (2+1) + 2)
     *  ...
     * for weight matrices:
     *  idx = layer * (max_stream_+1) + stream
     * for bias vectors:
     *  idx = layer * (max_stream_+1) + max_stream_
     */

    if (g2_.size() == 0) {
        nUpdates_ = 1;
        u32 idx;
        for (u32 layer = 0; layer < network.nLayers(); layer++) {
            max_stream_ = std::max(network.getLayer(layer).nInputActivations(), max_stream_);
        }

        for (u32 layer = 0; layer < network.nLayers(); layer++) {
            if (!network.getLayer(layer).isTrainable())
                continue;
            for (u32 stream = 0; stream < network.getLayer(layer).nInputActivations(); stream++) {
                NnMatrix* w = network.getLayer(layer).getWeights(stream);
                idx         = layer * (max_stream_ + 1) + stream;
                g2_[idx]    = new NnMatrix(w->nRows(), w->nColumns());
                Eg2_[idx]   = new NnMatrix(w->nRows(), w->nColumns());
                Edx2_[idx]  = new NnMatrix(w->nRows(), w->nColumns());
                RMSg_[idx]  = new NnMatrix(w->nRows(), w->nColumns());

                g2_[idx]->initComputation();
                Eg2_[idx]->initComputation();
                Edx2_[idx]->initComputation();
                RMSg_[idx]->initComputation();

                Eg2_[idx]->setToZero();
                Edx2_[idx]->setToZero();
            }
            // biases
            idx         = layer * (max_stream_ + 1) + max_stream_;
            NnVector* b = network.getLayer(layer).getBias();
            g2_[idx]    = new NnMatrix(b->nRows(), 1);
            Eg2_[idx]   = new NnMatrix(b->nRows(), 1);
            Edx2_[idx]  = new NnMatrix(b->nRows(), 1);
            RMSg_[idx]  = new NnMatrix(b->nRows(), 1);

            g2_[idx]->initComputation();
            Eg2_[idx]->initComputation();
            Edx2_[idx]->initComputation();
            RMSg_[idx]->initComputation();

            Eg2_[idx]->setToZero();
            Edx2_[idx]->setToZero();
        }
    }

    std::vector<T> stepSizes(network.nLayers());
    for (u32 layer = 0; layer < network.nLayers(); layer++) {
        if (network.getLayer(layer).isTrainable()) {
            /* estimation of weights */
            for (u32 stream = 0; stream < network.getLayer(layer).nInputActivations(); stream++) {
                NnMatrix* weights = network.getLayer(layer).getWeights(stream);
                require(weights);
                T localLearningRate = learningRate * network.getLayer(layer).learningRate();
                /* regular update, if no momentum used */
                if (!momentum_) {
                    if (clippingThreshold_ < Core::Type<T>::max) {
                        T localClippingThreshold = localLearningRate > 0 ? clippingThreshold_ / localLearningRate : 0;
                        statistics.gradientWeights(layer)[stream].clip(localClippingThreshold);
                    }

                    // SGD:
                    // w = w + (-lr * g)
                    // weights->add(statistics.gradientWeights(layer)[stream],(T) -localLearningRate);
                    //
                    // AdaDelta:
                    // see http://arxiv.org/pdf/1212.5701v1.pdf
                    // Matthew D. Zeiler "ADADELTA: An adaptive learning rate method."
                    //
                    // init Eg2 = 0, Edx2 = 0
                    //
                    // g2  = g^2
                    // Eg2 = r * Eg2 + (1-r) * g2
                    //
                    // RMSdx = sqrt(Edx2 + eps)
                    // RMSg  = sqrt(Eg2  + eps)
                    //
                    // dx    = -lr * g * (RMSdx/RMSg)
                    // dx2   = dx^2
                    //
                    // Edx2  = r * Edx2 + (1-r) * dx2
                    //
                    // w = w + dx

                    // printf("layer %d, stream %d\n", layer, stream);
                    NnMatrix* w   = network.getLayer(layer).getWeights(stream);
                    NnMatrix* g   = &statistics.gradientWeights(layer)[stream];
                    u32       idx = layer * (max_stream_ + 1) + stream;

                    NnMatrix* g2   = g2_[idx];
                    NnMatrix* Eg2  = Eg2_[idx];
                    NnMatrix* Edx2 = Edx2_[idx];
                    NnMatrix* RMSg = RMSg_[idx];

                    g2->copy(*g);
                    g2->pow(T(2));

                    Eg2->scale(b1);
                    Eg2->add(*g2, T(1 - b1));

                    // we don't need g2 anymore, so use it for storing RMSdx now
                    NnMatrix* RMSdx = g2;

                    RMSdx->copy(*Edx2);
                    RMSdx->addConstantElementwise(eps);
                    RMSdx->pow(T(0.5));

                    RMSg->copy(*Eg2);
                    RMSg->addConstantElementwise(eps);
                    // RMSg->pow(T(0.5)); // next use RMSg as accumulator 'dx' to save memory
                    RMSg->pow(T(-0.5));

                    NnMatrix* dx = RMSg;

                    dx->elementwiseMultiplication(*RMSdx);
                    dx->elementwiseMultiplication(*g);
                    dx->scale(-localLearningRate);

                    w->add(*dx);

                    dx->pow(T(2.0));
                    Edx2->scale(b1);
                    Edx2->add(*dx, T(1 - b1));
                }
            }

            /* estimation of bias */
            NnVector* bias = network.getLayer(layer).getBias();
            require(bias);
            T localLearningRate = learningRate * biasLearningRate_ * network.getLayer(layer).learningRate();
            /* regular update, if no momentum used */
            if (!momentum_) {
                if (clippingThreshold_ < Core::Type<T>::max) {
                    T localClippingThreshold = localLearningRate > 0 ? clippingThreshold_ / localLearningRate : 0;
                    statistics.gradientBias(layer).clip(localClippingThreshold);
                }

                NnVector* w   = network.getLayer(layer).getBias();
                NnVector* g   = &statistics.gradientBias(layer);
                u32       idx = layer * (max_stream_ + 1) + max_stream_;

                NnMatrix* g2   = g2_[idx];
                NnMatrix* Eg2  = Eg2_[idx];
                NnMatrix* Edx2 = Edx2_[idx];
                NnMatrix* RMSg = RMSg_[idx];

                // matrix representation of NnVector g
                NnMatrix g_copy(g->nRows(), 1);
                g_copy.initComputation();
                g_copy.setColumn(0, *g);

                g2->setColumn(0, *g);
                g2->pow(T(2));

                Eg2->scale(b1);
                Eg2->add(*g2, T(1 - b1));

                // we don't need g2 anymore, so use it for storing RMSdx now
                NnMatrix* RMSdx = g2;

                RMSdx->copy(*Edx2);
                RMSdx->addConstantElementwise(eps);
                RMSdx->pow(T(0.5));

                RMSg->copy(*Eg2);
                RMSg->addConstantElementwise(eps);
                // RMSg->pow(T(0.5)); // next use RMSg as accumulator 'dx' to save memory
                RMSg->pow(T(-0.5));

                NnMatrix* dx = RMSg;

                dx->elementwiseMultiplication(*RMSdx);
                dx->elementwiseMultiplication(g_copy);
                dx->scale(-localLearningRate);

                w->addSummedColumns(*dx);  // ==add(*dx), since dx has only 1 column

                dx->pow(T(2.0));
                Edx2->scale(b1);
                Edx2->add(*dx, T(1 - b1));
            }
        }
    }
    if (logStepSize_ && statisticsChannel_.isOpen()) {
        T stepSize = Math::asum<T>(stepSizes.size(), &stepSizes.at(0), 1);
        statisticsChannel_ << "step-size: " << stepSize << " (" << Core::vector2str(stepSizes, ",") << ")";
    }

    nUpdates_++;
}
//=============================================================================

template<typename T>
const Core::ParameterBool RmsProp<T>::paramUsePredefinedLearningRateDecay(
        "use-predefined-learning-rate-decay", "use learning-rate * tau / (tau + numberOfUpdates) as learning-rate", false);

template<typename T>
const Core::ParameterFloat RmsProp<T>::paramLearningRateTau("learning-rate-tau", "", 1000.0);

template<typename T>
const Core::ParameterInt RmsProp<T>::paramNumberOfUpdates(
        "number-of-updates", "number of updates done so far", 0);

template<typename T>
const Core::ParameterFloat RmsProp<T>::paramClippingThreshold(
        "clipping-threshold", "clip updates if larger than learning-rate * clipping-threshold", Core::Type<T>::max);

template<typename T>
const Core::ParameterFloat RmsProp<T>::paramMomentumFactor(
        "momentum-factor", "momentum factor, suggested value: 0.9", 0.0);

template<typename T>
const Core::ParameterFloat RmsProp<T>::paramBeta1("rmsprop-beta1", "beta1", 0.9);

template<typename T>
const Core::ParameterFloat RmsProp<T>::paramBeta2("rmsprop-beta2", "beta2 (momentum term)", 0.9);

template<typename T>
const Core::ParameterFloat RmsProp<T>::paramEpsilon("rmsprop-epsilon", "epsilon", 1e-10);

template<typename T>
RmsProp<T>::RmsProp(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config),
          usePredefinedLearningRateDecay_(paramUsePredefinedLearningRateDecay(config)),
          tau_(paramLearningRateTau(config)),
          nUpdates_(paramNumberOfUpdates(config)),
          clippingThreshold_(paramClippingThreshold(config)),
          momentumFactor_(paramMomentumFactor(config)),
          momentum_(momentumFactor_ > 0),
          oldDeltas_(0),
          m_(),
          v_(),
          M_(),
          V_(),
          g2_(),
          max_stream_(0),
          b1_(paramBeta1(config)),
          b2_(paramBeta2(config)),
          eps_(paramEpsilon(config)) {
    this->log("Initializing RMSProp with b1=") << b1_ << " b2=" << b2_ << " eps=" << eps_;

    if (usePredefinedLearningRateDecay_) {
        this->log("using predefined learning rate decay with parameter tau: ") << tau_;
        this->log("number of updates so far is ") << nUpdates_;
    }
    if (clippingThreshold_ < Core::Type<T>::max) {
        this->log("clipping updates if larger than ") << clippingThreshold_;
        if (momentum_)
            this->error("momentum with clipping not implemented yet ..");
    }
    if (momentum_)
        this->log("using momentum with momentum factor: ") << momentumFactor_;
}

template<typename T>
RmsProp<T>::~RmsProp() {
    if (momentum_) {
        delete oldDeltas_;
    }
}

template<typename T>
bool RmsProp<T>::isDefaultConfig() {
    if (usePredefinedLearningRateDecay_)
        return false;
    if (clippingThreshold_ < Core::Type<T>::max)
        return false;
    if (momentum_)
        return false;
    return true;
}

template<typename T>
void RmsProp<T>::estimate(NeuralNetwork<T>& network, Statistics<T>& statistics) {
    T learningRate = initialLearningRate_;
    T b1           = b1_;
    T b2           = b2_;
    T eps          = eps_;

    /* estimation of parameters */
    require(statistics.hasGradient());

    if (usePredefinedLearningRateDecay_ && statisticsChannel_.isOpen())
        statisticsChannel_ << "learningRate: " << learningRate;

    /**
     * data structures are allocated as follows:
     * max_stream_ = maximum number of streams per layer (over all layers), e.g. 2
     * pointer maps are indexed using the following scheme (layer l, stream s)
     *
     *   0 -> l0s0w (0 * (2+1) + 0)
     *   1 -> l0s1w (0 * (2+1) + 1)
     *   2 -> l0b   (0 * (2+1) + 2)
     *
     *   3 -> l1s0w (1 * (2+1) + 0)
     *   5 -> l1b   (1 * (2+1) + 2)
     *
     *   6 -> l2s0w (2 * (2+1) + 0)
     *   8 -> l2b   (2 * (2+1) + 2)
     *  ...
     * for weight matrices:
     *  idx = layer * (max_stream_+1) + stream
     * for bias vectors:
     *  idx = layer * (max_stream_+1) + max_stream_
     */

    if (m_.size() == 0) {
        nUpdates_ = 1;
        u32 idx;
        for (u32 layer = 0; layer < network.nLayers(); layer++) {
            max_stream_ = std::max(network.getLayer(layer).nInputActivations(), max_stream_);
        }

        for (u32 layer = 0; layer < network.nLayers(); layer++) {
            if (!network.getLayer(layer).isTrainable())
                continue;
            for (u32 stream = 0; stream < network.getLayer(layer).nInputActivations(); stream++) {
                NnMatrix* w = network.getLayer(layer).getWeights(stream);
                idx         = layer * (max_stream_ + 1) + stream;
                m_[idx]     = new NnMatrix(w->nRows(), w->nColumns());
                v_[idx]     = new NnMatrix(w->nRows(), w->nColumns());
                M_[idx]     = new NnMatrix(w->nRows(), w->nColumns());
                V_[idx]     = new NnMatrix(w->nRows(), w->nColumns());
                g2_[idx]    = new NnMatrix(w->nRows(), w->nColumns());

                m_[idx]->setToZero();
                v_[idx]->setToZero();
                M_[idx]->setToZero();

                m_[idx]->initComputation();
                v_[idx]->initComputation();
                M_[idx]->initComputation();
                V_[idx]->initComputation();
                g2_[idx]->initComputation();
            }
            // biases
            idx         = layer * (max_stream_ + 1) + max_stream_;
            NnVector* b = network.getLayer(layer).getBias();
            m_[idx]     = new NnMatrix(b->nRows(), 1);
            v_[idx]     = new NnMatrix(b->nRows(), 1);
            M_[idx]     = new NnMatrix(b->nRows(), 1);
            V_[idx]     = new NnMatrix(b->nRows(), 1);
            g2_[idx]    = new NnMatrix(b->nRows(), 1);

            m_[idx]->setToZero();
            v_[idx]->setToZero();
            M_[idx]->setToZero();

            m_[idx]->initComputation();
            v_[idx]->initComputation();
            V_[idx]->initComputation();
            M_[idx]->initComputation();
            g2_[idx]->initComputation();
        }
    }

    std::vector<T> stepSizes(network.nLayers());
    for (u32 layer = 0; layer < network.nLayers(); layer++) {
        if (network.getLayer(layer).isTrainable()) {
            /* estimation of weights */
            for (u32 stream = 0; stream < network.getLayer(layer).nInputActivations(); stream++) {
                NnMatrix* weights = network.getLayer(layer).getWeights(stream);
                require(weights);
                T localLearningRate = learningRate * network.getLayer(layer).learningRate();
                /* regular update, if no momentum used */
                if (!momentum_) {
                    if (clippingThreshold_ < Core::Type<T>::max) {
                        T localClippingThreshold = localLearningRate > 0 ? clippingThreshold_ / localLearningRate : 0;
                        statistics.gradientWeights(layer)[stream].clip(localClippingThreshold);
                    }

                    // SGD:
                    // w = w + (-lr * g)
                    // weights->add(statistics.gradientWeights(layer)[stream],(T) -localLearningRate);
                    //
                    // RMSProp:
                    // init m=v=0
                    //
                    // m = b1*m + (1-b1)*g
                    // v = b1*v + (1-b1)*g^2 (elementwise square)
                    //
                    // M = b2*M -lr * g/(sqrt(v-m^2+eps))
                    //
                    // w = w + M

                    NnMatrix* w   = network.getLayer(layer).getWeights(stream);
                    NnMatrix* g   = &statistics.gradientWeights(layer)[stream];
                    u32       idx = layer * (max_stream_ + 1) + stream;

                    NnMatrix* m  = m_[idx];
                    NnMatrix* v  = v_[idx];
                    NnMatrix* g2 = g2_[idx];
                    NnMatrix* V  = V_[idx];
                    NnMatrix* M  = M_[idx];

                    m->scale(b1);
                    m->add(*g, T(1 - b1));

                    g2->copy(*g);
                    g2->pow(T(2));
                    v->scale(b1);
                    v->add(*g2, T(1 - b1));

                    V->copy(*m);
                    V->pow(T(2));
                    V->scale(T(-1));
                    V->add(*v);
                    V->addConstantElementwise(eps);
                    V->pow(T(-0.5));
                    V->elementwiseMultiplication(*g);

                    M->scale(b2);
                    M->add(*V, T(-localLearningRate));

                    w->add(*M);
                }
            }

            /* estimation of bias */
            NnVector* bias = network.getLayer(layer).getBias();
            require(bias);
            T localLearningRate = learningRate * biasLearningRate_ * network.getLayer(layer).learningRate();
            /* regular update, if no momentum used */
            if (!momentum_) {
                if (clippingThreshold_ < Core::Type<T>::max) {
                    T localClippingThreshold = localLearningRate > 0 ? clippingThreshold_ / localLearningRate : 0;
                    statistics.gradientBias(layer).clip(localClippingThreshold);
                }

                NnVector* w   = network.getLayer(layer).getBias();
                NnVector* g   = &statistics.gradientBias(layer);
                u32       idx = layer * (max_stream_ + 1) + max_stream_;

                NnMatrix* m  = m_[idx];
                NnMatrix* v  = v_[idx];
                NnMatrix* g2 = g2_[idx];
                NnMatrix* V  = V_[idx];
                NnMatrix* M  = M_[idx];

                m->scale(b1);
                m->addToAllColumns(*g, T(1 - b1));

                g2->setColumn(0, *g);
                g2->pow(T(2));
                v->scale(b1);
                v->add(*g2, T(1 - b1));

                V->copy(*m);
                V->pow(T(2));
                V->scale(T(-1));
                V->add(*v);
                V->addConstantElementwise(eps);
                V->pow(T(-0.5));

                g2->setColumn(0, *g);
                V->elementwiseMultiplication(*g2);

                M->scale(b2);
                M->add(*V, T(-localLearningRate));

                w->addSummedColumns(*M);
            }
        }
    }
    if (logStepSize_ && statisticsChannel_.isOpen()) {
        T stepSize = Math::asum<T>(stepSizes.size(), &stepSizes.at(0), 1);
        statisticsChannel_ << "step-size: " << stepSize << " (" << Core::vector2str(stepSizes, ",") << ")";
    }

    nUpdates_++;
}

//=============================================================================

template<typename T>
PriorEstimator<T>::PriorEstimator(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config) {
    if (!this->fullBatchMode_) {
        this->fullBatchMode_ = true;
        this->log("using batch mode, because prior estimation only possible in batch mode");
    }
}

// explicit template instantiation
namespace Nn {

template class Estimator<f32>;
template class Estimator<f64>;

template class SteepestDescentEstimator<f32>;
template class SteepestDescentEstimator<f64>;

template class SteepestDescentL1ClippingEstimator<f32>;
template class SteepestDescentL1ClippingEstimator<f64>;

template class PriorEstimator<f32>;
template class PriorEstimator<f64>;

template class Adam<f32>;
template class Adam<f64>;

template class AdaGrad<f32>;
template class AdaGrad<f64>;

template class AdaDelta<f32>;
template class AdaDelta<f64>;

template class RmsProp<f32>;
template class RmsProp<f64>;
}  // namespace Nn
