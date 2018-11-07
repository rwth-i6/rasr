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
#include "Regularizer.hh"
#include "LinearLayer.hh"

using namespace Nn;

template<class T>
const Core::Choice Regularizer<T>::choiceRegularizerType(
        "none", none,
        "l1-regularizer", l1Regularizer,
        "l2-regularizer", l2Regularizer,
        "centered-l2-regularizer", centeredRegularizer,
        Core::Choice::endMark());

template<class T>
const Core::ParameterChoice Regularizer<T>::paramRegularizerType(
        "regularizer", &choiceRegularizerType,
        "regularizer (adds regularization term to objective function)", none);

template<typename T>
Regularizer<T>::Regularizer(const Core::Configuration& config) :
Core::Component(config)
{}

template<class T>
Regularizer<T>* Regularizer<T>::createRegularizer(const Core::Configuration& config) {
    Regularizer<T>* regularizer;

    switch ( (RegularizerType) paramRegularizerType(config) ) {
    case l1Regularizer:
        regularizer = new L1Regularizer<T>(config);
        Core::Application::us()->log("Create regularizer: l1-regularizer");
        break;
    case l2Regularizer:
        regularizer = new L2Regularizer<T>(config);
        Core::Application::us()->log("Create regularizer: l2-regularizer");
        break;
    case centeredRegularizer:
        regularizer = new CenteredL2Regularizer<T>(config);
        Core::Application::us()->log("Create regularizer: centered-l2-regularizer");
        break;
    default:
        regularizer = new Regularizer<T>(config);
        Core::Application::us()->log("Create regularizer: none");
        break;
    };

    return regularizer;
}

//=============================================================================

template<typename T>
L1Regularizer<T>::L1Regularizer(const Core::Configuration& config):
Core::Component(config),
Precursor(config)
{
    signMatrix_.initComputation(false);
    signVector_.initComputation(false);
}

template<typename T>
T L1Regularizer<T>::objectiveFunction(NeuralNetwork<T>& network, T factor) {
    T objectiveFunction = 0;
    for (u32 layer = 0; layer < network.nLayers(); layer++) {
        if ((network.getLayer(layer).isTrainable()) &&
                (network.getLayer(layer).regularizationConstant() > 0)) {
            T tmpObjectiveFunction = 0;

            NnVector *bias = network.getLayer(layer).getBias();
            require(bias);
            tmpObjectiveFunction = bias->l1norm();
            for (u32 stream = 0; stream < network.getLayer(layer).nInputActivations(); stream++) {
                NnMatrix *matrix = network.getLayer(layer).getWeights(stream);
                require(matrix);
                tmpObjectiveFunction += matrix->l1norm();
            }
            tmpObjectiveFunction *= network.getLayer(layer).regularizationConstant();
            objectiveFunction += tmpObjectiveFunction;
        }
    }
    return factor * objectiveFunction;
}

template<typename T>
void L1Regularizer<T>::addGradient(NeuralNetwork<T>& network, Statistics<T>& statistics, T factor) {
    for (u32 layer = 0; layer < network.nLayers(); layer++) {
        if ((network.getLayer(layer).isTrainable()) &&
                (network.getLayer(layer).regularizationConstant() > 0)) {
            for (u32 stream = 0; stream < network.getLayer(layer).nInputActivations(); stream++) {
                NnMatrix *weights = network.getLayer(layer).getWeights(stream);
                require(weights);

                signMatrix_.resize(weights->nRows(), weights->nColumns());
                signMatrix_.sign(*weights);

                statistics.gradientWeights(layer)[stream].add(signMatrix_, network.getLayer(layer).regularizationConstant() * factor);
            }
            NnVector *bias = network.getLayer(layer).getBias();
            require(bias);

            signVector_.resize(bias->nRows());
            signVector_.sign(*bias);

            statistics.gradientBias(layer).add(signVector_, network.getLayer(layer).regularizationConstant() * factor);
        }
    }
}

//=============================================================================

template<typename T>
L2Regularizer<T>::L2Regularizer(const Core::Configuration& config):
Core::Component(config),
Precursor(config)
{}

template<typename T>
T L2Regularizer<T>::objectiveFunction(NeuralNetwork<T>& network, T factor) {
    T objectiveFunction = 0;
    for (u32 layer = 0; layer < network.nLayers(); layer++) {
        if ((network.getLayer(layer).isTrainable()) &&
                (network.getLayer(layer).regularizationConstant() > 0)) {
            T tmpObjectiveFunction = 0;

            NnVector *bias = network.getLayer(layer).getBias();
            require(bias);
            tmpObjectiveFunction = bias->sumOfSquares();
            for (u32 stream = 0; stream < network.getLayer(layer).nInputActivations(); stream++) {
                NnMatrix *matrix = network.getLayer(layer).getWeights(stream);
                require(matrix);
                tmpObjectiveFunction += matrix->sumOfSquares();
            }
            tmpObjectiveFunction *= network.getLayer(layer).regularizationConstant() / 2.0;
            objectiveFunction += tmpObjectiveFunction;
        }
    }
    return factor * objectiveFunction;
}

template<typename T>
void L2Regularizer<T>::addGradient(NeuralNetwork<T>& network, Statistics<T>& statistics, T factor) {
    for (u32 layer = 0; layer < network.nLayers(); layer++) {
        if ((network.getLayer(layer).isTrainable()) &&
                (network.getLayer(layer).regularizationConstant() > 0)) {
            for (u32 stream = 0; stream < network.getLayer(layer).nInputActivations(); stream++) {
                NnMatrix *weights = network.getLayer(layer).getWeights(stream);
                require(weights);
                statistics.gradientWeights(layer)[stream].add(*weights, network.getLayer(layer).regularizationConstant() * factor);
            }
            NnVector *bias = network.getLayer(layer).getBias();
            require(bias);
            statistics.gradientBias(layer).add(*bias, network.getLayer(layer).regularizationConstant() * factor);
        }
    }
}

//=============================================================================

template<typename T>
const Core::ParameterString CenteredL2Regularizer<T>::paramCenterParameters(
        "center-parameters", "parameters of regularization center");

template<typename T>
CenteredL2Regularizer<T>::CenteredL2Regularizer(const Core::Configuration& config):
Core::Component(config),
Precursor(config),
centerNetwork_(config)
{
    centerNetwork_.initializeNetwork(1, paramCenterParameters(config));
}

template<typename T>
T CenteredL2Regularizer<T>::objectiveFunction(NeuralNetwork<T>& network, T factor) {
    T objectiveFunction = 0;
    for (u32 layer = 0; layer < network.nLayers(); layer++) {
        if ((network.getLayer(layer).isTrainable()) &&
                (network.getLayer(layer).regularizationConstant() > 0)) {
            T tmpObjectiveFunction = 0;

            NnVector *bias = network.getLayer(layer).getBias();
            NnVector *centerBias = centerNetwork_.getLayer(layer).getBias();
            require(bias);
            require(centerBias);
            // TODO implement matrix functions to do this directly
            diffVector_.resize(bias->size());
            diffVector_.copy(*bias);
            diffVector_.add(*centerBias, T(-1.0));
            tmpObjectiveFunction = diffVector_.sumOfSquares();
            for (u32 stream = 0; stream < network.getLayer(layer).nInputActivations(); stream++) {
                NnMatrix *weightMatrix = network.getLayer(layer).getWeights(stream);
                NnMatrix *centerWeightMatrix = centerNetwork_.getLayer(layer).getWeights(stream);
                require(weightMatrix);
                require(centerWeightMatrix);
                diffMatrix_.resize(weightMatrix->nRows(), weightMatrix->nColumns());
                diffMatrix_.copy(*weightMatrix);
                diffMatrix_.add(*centerWeightMatrix, T(-1.0));
                tmpObjectiveFunction += diffMatrix_.sumOfSquares();
            }
            tmpObjectiveFunction *= network.getLayer(layer).regularizationConstant() / 2.0;
            objectiveFunction += tmpObjectiveFunction;
        }
    }
    return factor * objectiveFunction;
}

template<typename T>
void CenteredL2Regularizer<T>::addGradient(NeuralNetwork<T>& network, Statistics<T>& statistics, T factor) {
    for (u32 layer = 0; layer < network.nLayers(); layer++) {
        if ((network.getLayer(layer).isTrainable()) &&
                (network.getLayer(layer).regularizationConstant() > 0)) {
            for (u32 stream = 0; stream < network.getLayer(layer).nInputActivations(); stream++) {
                NnMatrix *weightMatrix = network.getLayer(layer).getWeights(stream);
                NnMatrix *centerWeightMatrix = centerNetwork_.getLayer(layer).getWeights(stream);
                require(weightMatrix);
                require(centerWeightMatrix);
                statistics.gradientWeights(layer)[stream].add(*weightMatrix, network.getLayer(layer).regularizationConstant() * factor);
                statistics.gradientWeights(layer)[stream].add(*centerWeightMatrix, -network.getLayer(layer).regularizationConstant() * factor);
            }
            NnVector *bias = network.getLayer(layer).getBias();
            NnVector *centerBias = centerNetwork_.getLayer(layer).getBias();
            require(bias);
            require(centerBias);
            statistics.gradientBias(layer).add(*bias, network.getLayer(layer).regularizationConstant() * factor);
            statistics.gradientBias(layer).add(*centerBias, -network.getLayer(layer).regularizationConstant() * factor);
        }
    }
}

//=============================================================================

// explicit template instantiation
namespace Nn {

template class Regularizer<f32>;
template class Regularizer<f64>;

template class L2Regularizer<f32>;
template class L2Regularizer<f64>;

template class CenteredL2Regularizer<f32>;
template class CenteredL2Regularizer<f64>;

}
