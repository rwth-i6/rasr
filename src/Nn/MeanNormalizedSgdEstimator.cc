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
#include "MeanNormalizedSgdEstimator.hh"
#include <algorithm>

using namespace Nn;

template<typename T>
MeanNormalizedSgd<T>::MeanNormalizedSgd(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config),
          firstEstimation_(true) {}

/**
 * for all trainable layers:
 * run through all preceeding layers and ensure they collect activation statistics
 */
template<typename T>
void MeanNormalizedSgd<T>::checkForStatistics(NeuralNetwork<T>& network) {
    for (u32 layer = 0; layer < network.nLayers(); layer++) {
        if (network.getLayer(layer).isTrainable()) {
            for (u32 stream = 0; stream < network.getLayer(layer).nPredecessors(); stream++) {
                u32 predecessor = network.getLayer(layer).getPredecessor(stream);
                if (!network.getLayer(predecessor).hasActivationStatistics()) {
                    this->warning() << network.getLayer(predecessor).getName()
                                    << " is a predecessor of " << network.getLayer(layer).getName()
                                    << ", but has no activation statistics. Assume zero mean for this input stream.";
                }
            }
        }
    }
}

/**
 * estimation with mean-normalized SGD
 * see Wiesler, Richard, Schlüter & Ney: Mean-Normalized SGD, ICASSP 2014
 */
template<typename T>
void MeanNormalizedSgd<T>::estimate(NeuralNetwork<T>& network, Statistics<T>& statistics) {
    if (firstEstimation_) {
        checkForStatistics(network);
        firstEstimation_ = false;
    }

    T learningRate = initialLearningRate_;

    /*
     * estimation of parameters:
     *
     * W -= eta * delta_W  (update weight matrix)
     * a -= eta * delta_a  (update bias vector)
     *
     * with
     *
     * delta_W =  gradient_W  + b . gradient_a^T
     * delta_a = \gradient_W^T . b + (1 + b^T b) \gradient_a
     *
     * The shift b is the negative of the (smoothed) activation mean.
     * The update matrices are stored in place, i.e. the gradients are replaced by the update terms.
     *
     */
    require(statistics.hasGradient());
    std::vector<T> stepSizes(network.nLayers());

    for (u32 layer = 0; layer < network.nLayers(); layer++) {
        if (network.getLayer(layer).isTrainable()) {
            /* modify weights gradient and update weights */
            for (u32 stream = 0; stream < network.getLayer(layer).nPredecessors(); stream++) {
                // if layer has predecessors with activation statistics...
                u32 predecessor = network.getLayer(layer).getPredecessor(stream);
                if (network.getLayer(predecessor).hasActivationStatistics()) {
                    // ... modify gradient of weights for the current input stream
                    // delta_W =  gradient_W  + b . gradient_a^T
                    statistics.gradientWeights(layer)[stream].addOuterProduct(network.getLayer(predecessor).getActivationMean(),
                                                                              statistics.gradientBias(layer), -1.0);
                }
                // update weights
                NnMatrix* weights = network.getLayer(layer).getWeights(stream);
                require(weights);
                T localLearningRate = learningRate * network.getLayer(layer).learningRate();
                weights->add(statistics.gradientWeights(layer)[stream], (T)-localLearningRate);

                // log step size
                if (logStepSize_)
                    stepSizes[layer] += statistics.gradientWeights(layer)[stream].l1norm() * localLearningRate;
            }

            /* modify bias gradient and update bias */
            for (u32 stream = 0; stream < network.getLayer(layer).nPredecessors(); stream++) {
                // if layer has predecessors with activation statistics...
                u32 predecessor = network.getLayer(layer).getPredecessor(stream);
                if (network.getLayer(predecessor).hasActivationStatistics()) {
                    // ... modify gradient of bias
                    // delta_a = \gradient_W^T . b + (1 + b^T b) \gradient_a
                    //         = delta_a + \delta_W^T b
                    statistics.gradientWeights(layer)[stream].multiply(network.getLayer(predecessor).getActivationMean(),
                                                                       statistics.gradientBias(layer), true, -1.0, 1.0);
                }
            }
            // update bias
            NnVector* bias = network.getLayer(layer).getBias();
            require(bias);
            T localLearningRate = learningRate * biasLearningRate_ * network.getLayer(layer).learningRate();
            bias->add(statistics.gradientBias(layer), (T)-localLearningRate);
            /* log step size */
            if (logStepSize_)
                stepSizes[layer] += statistics.gradientBias(layer).l1norm() * localLearningRate;
        }
    }
    if (logStepSize_ && statisticsChannel_.isOpen()) {
        T stepSize = Math::asum<T>(stepSizes.size(), &stepSizes.at(0), 1);
        statisticsChannel_ << "step-size: " << stepSize << " (" << Core::vector2str(stepSizes, ",") << ")";
    }
}

//=============================================================================

template<typename T>
MeanNormalizedSgdL1Clipping<T>::MeanNormalizedSgdL1Clipping(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config) {}

template<typename T>
void MeanNormalizedSgdL1Clipping<T>::estimate(NeuralNetwork<T>& network, Statistics<T>& statistics) {
    Precursor::estimate(network, statistics);

    T learningRate = this->initialLearningRate_;

    for (u32 layer = 0; layer < network.nLayers(); layer++) {
        if (network.getLayer(layer).isTrainable()) {
            if (network.getLayer(layer).nInputActivations() != 1) {
                Core::Application::us()->criticalError("Estimation for multiple streams not yet implemented.");
            }
            NnMatrix* weights = network.getLayer(layer).getWeights(0);
            NnVector* bias    = network.getLayer(layer).getBias();
            require(weights);
            require(bias);
            weights->l1clipping(network.getLayer(layer).regularizationConstant() * learningRate * network.getLayer(layer).learningRate());
            bias->l1clipping(network.getLayer(layer).regularizationConstant() * learningRate * network.getLayer(layer).learningRate());
        }
    }

    if (logStepSize_ && statisticsChannel_.isOpen()) {
        statisticsChannel_ << "step size does not include l1-regularization";
    }
}

//=============================================================================

// explicit template instantiation
namespace Nn {

template class MeanNormalizedSgd<f32>;
template class MeanNormalizedSgd<f64>;

template class MeanNormalizedSgdL1Clipping<f32>;
template class MeanNormalizedSgdL1Clipping<f64>;

}  // namespace Nn
