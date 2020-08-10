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
#include "FeedForwardTrainer.hh"

namespace Nn {

template<typename T>
T FeedForwardTrainer<T>::getNewError() {
    // Get error.
    criterion().reinputWithNewNnOutput(network().getTopLayerOutput());
    T newError = 0;
    criterion().getObjectiveFunction(newError);
    // apply regularization only when not in batch mode
    if (!estimator().fullBatchMode()) {
        u32 batchSize = network().getLayerInput(0)[0]->nColumns();
        newError += regularizer().objectiveFunction(network(), T(batchSize));
    }
    return newError;
}

template<typename T>
template<typename Params>
void FeedForwardTrainer<T>::gradientCheckComponent(
        T grad, T* paramPtr, Params& params, u32 layerIdx) {
    u32 paramIdx = paramPtr - params.begin();

    // First calculate the symmetric numeric gradient ((f(x + h) - f(x - h)) / (2 h)).

    const int precision    = gradientCheckPrecision_;
    const T   perturbation = gradientCheckPerturbation_;
    const T   diffs[]      = {perturbation, -perturbation};
    T         origParam    = 0;
    T         errors[]     = {0, 0};

    for (short i = 0; i <= 2; ++i) {
        // We always sync the whole vector/matrix. This is of course very inefficient,
        // but it would require some more complicated syncing code otherwise,
        // and is probably anyway not the bottleneck when doing the gradient check.
        // We expect to be not in computation mode here.
        if (i == 0)
            origParam = *paramPtr;
        if (i == 2)
            *paramPtr = origParam;
        else
            *paramPtr = origParam + diffs[i];
        params.initComputation(true);  // sync to GPU
        if (i < 2) {
            // Use already set features.
            network().forwardLayers(layerIdx);

            // Get error.
            errors[i] = getNewError();
            require(!criterion().discardCurrentInput());
        }
        params.finishComputation(false);  // we expect the CPU memory to be correct
    }

    T numericGrad = (errors[0] - errors[1]) / (2 * perturbation);

    // Now check with the gradient which we got before from the criterion.

    // Threshold based on minimum grad and precision.
    T threshold = (T)pow(T(10.0), std::max<T>(T(0.0), ceil(log10(std::min(fabs(grad), fabs(numericGrad))))) - precision);
    T diff      = fabs(grad - numericGrad);
    ((Math::isnan(diff) || diff > threshold) ? Core::Component::warning("Gradient check failed: ") : Core::Component::log("Gradient check succeeded: "))
            << "paramIdx: " << paramIdx
            << ", param: " << origParam
            << ", grad: " << grad
            << ", numericGrad: " << numericGrad
            << " (leftError: " << errors[0]
            << ", rightError: " << errors[1] << ")";
}

template<typename T>
void FeedForwardTrainer<T>::gradientCheck() {
    // Only check gradient components / parameters of last layer.
    // The gradient check is mostly to check the derivation of the criterion.
    s32                    layerIdx = (s32)network().nLayers() - 1;
    NeuralNetworkLayer<T>& layer    = network().getLayer(layerIdx);
    require(layer.isTrainable());

    if (typename Types<T>::NnVector* bias = layer.getBias()) {
        bias->finishComputation(true);  // sync to CPU

        typename Types<T>::NnVector& gradientBias = statistics().gradientBias(layerIdx);
        gradientBias.finishComputation(true);  // sync to CPU

        for (T* gradPtr = gradientBias.begin(); gradPtr != gradientBias.end(); ++gradPtr) {
            T* paramPtr = bias->begin() + (gradPtr - gradientBias.begin());

            gradientCheckComponent(*gradPtr, paramPtr, *bias, layerIdx);
        }

        gradientBias.initComputation(false);
        bias->initComputation(false);
    }

    for (u32 stream = 0; stream < statistics().gradientWeights(layerIdx).size(); stream++) {
        typename Types<T>::NnMatrix* weights = layer.getWeights(stream);
        if (!weights)
            continue;
        weights->finishComputation(true);  // sync to GPU

        typename Types<T>::NnMatrix& gradientWeights = statistics().gradientWeights(layerIdx)[stream];
        gradientWeights.finishComputation(true);  // sync to GPU

        for (T* gradPtr = gradientWeights.begin(); gradientWeights.end(); ++gradPtr) {
            T* paramPtr = weights->begin() + (gradPtr - gradientWeights.begin());

            gradientCheckComponent(*gradPtr, paramPtr, *weights, layerIdx);
        }

        gradientWeights.initComputation(false);
        weights->initComputation(false);
    }
}

// Calculates grad^T * grad * learningRate.
template<typename T>
T FeedForwardTrainer<T>::getDirectionalEstimate() {
    T learningRate           = estimator().learningRate();
    T biasLearningRateFactor = estimator().biasLearningRateFactor();

    T sum = 0;
    for (s32 layerIdx = 0; layerIdx < (s32)network().nLayers(); ++layerIdx) {
        NeuralNetworkLayer<T>& layer = network().getLayer(layerIdx);
        if (!layer.isTrainable())
            continue;

        T layerLearningRateFactor = layer.learningRate();

        if (layer.getBias()) {
            T                            localLearningRate = layerLearningRateFactor * biasLearningRateFactor;
            typename Types<T>::NnVector& gradientBias      = statistics().gradientBias(layerIdx);
            sum += gradientBias.sumOfSquares() * localLearningRate;
        }

        for (u32 stream = 0; stream < statistics().gradientWeights(layerIdx).size(); stream++) {
            if (!layer.getWeights(stream))
                continue;
            T                            localLearningRate = layerLearningRateFactor;
            typename Types<T>::NnMatrix& gradientWeights   = statistics().gradientWeights(layerIdx)[stream];
            sum += gradientWeights.sumOfSquares() * localLearningRate;
        }
    }

    return sum * learningRate;
}

template<typename T>
void FeedForwardTrainer<T>::simpleGradientCheck(T oldError) {
    const int precision = gradientCheckPrecision_;

    // We expect the Steepest-Descent-Estimator to know the estimator step.
    auto* est = dynamic_cast<SteepestDescentEstimator<T>*>(&estimator());
    if (!est) {
        Core::Component::error("simple gradient check: need steepest-descent-estimator");
        return;
    }
    if (!est->isDefaultConfig()) {
        Core::Component::error("simple gradient check: need steepest-descent-estimator with default config, ")
                << "i.e. no decay, no momentum";
        return;
    }

    // Get error.
    T newError = getNewError();
    require(!criterion().discardCurrentInput());

    // Now calculate grad^T * grad * learningRate, which is an estimation of oldError - newError.
    T numericStep = getDirectionalEstimate();
    T realStep    = oldError - newError;
    // Threshold based on minimum grad and precision.
    T threshold = (T)pow(T(10.0), std::max<T>(T(0.0), ceil(log10(std::min(fabs(realStep), fabs(numericStep))))) - precision);
    T diff      = fabs(realStep - numericStep);
    if (Math::isnan(diff) || diff > threshold) {
        Core::Component::warning("Simple gradient check failed: ")
                << "oldError: " << oldError
                << ", newError: " << newError
                << ", errStep: " << (oldError - newError)
                << ", numeric errStep: " << numericStep;
    }
    else
        Core::Component::log("Simple gradient check succeeded");
}

template<typename T>
bool FeedForwardTrainer<T>::convergenceCheckRepeat(T& error, NnMatrix& errorSignal) {
    // Get new error.
    T newError = getNewError();
    require(!criterion().discardCurrentInput());
    T errDiff      = error - newError;
    T learningRate = estimator().learningRate();
    if (errDiff < 0) {
        Core::Component::warning("Convergence check: error got worse: ")
                << "oldError: " << error
                << ", newError: " << newError;
        require_lt(convergenceCheckLearningRateFactor_, 1);
        require_gt(convergenceCheckLearningRateFactor_, 0);
        Core::Component::log("lowering learning rate, ")
                << "current: " << learningRate;
        learningRate *= convergenceCheckLearningRateFactor_;
        Core::Component::log("new learning rate: ") << learningRate;
        require_gt(learningRate, Core::Type<T>::delta * 2);
        estimator().setLearningRate(learningRate);
    }
    else {  // errDiff >= 0
        if (errDiff > 0)
            Core::Component::log("Convergence check: new lower error: ")
                    << newError
                    << " (oldError: " << error << ", diff: " << errDiff << ")";
        else
            Core::Component::log("Convergence check: no error diff")
                    << " error: " << newError;

        // Calculate the 2-norm^2 of the error signal.
        // getDirectionalEstimate() mostly does this, except the learning rate factor.
        // Note that we cannot check whether the error (objective function) is zero,
        // because the error function does not necessarily have the minima zero.
        T errorNorm = getDirectionalEstimate();
        if (errorNorm < convergenceCheckGradNormLimit_ || Core::isAlmostEqualUlp(errDiff, 0, 20)) {
            Core::Component::log("Convergence check: stopping with gradient norm: ")
                    << errorNorm;
            return false;
        }
        else {
            Core::Component::log("Convergence check: gradient norm: ")
                    << errorNorm;
            // Repeat. Fall through.
        }
    }

    // Get new error signal and error.
    // In getNewError(), we refeeded the current NN output to the criterion.
    criterion().getErrorSignal_naturalPairing(errorSignal, network().getTopLayer());
    error = 0;
    criterion().getObjectiveFunction(error);

    // Reset statistics (error + gradients).
    statistics().addToObjectiveFunction(-statistics().objectiveFunction());
    for (s32 layerIdx = lowestTrainableLayerIndex_; layerIdx < (s32)network().nLayers(); ++layerIdx) {
        if (!network().getLayer(layerIdx).isTrainable())
            continue;
        statistics().gradientBias(layerIdx).setToZero();
        for (u32 stream = 0; stream < statistics().gradientWeights(layerIdx).size(); stream++) {
            statistics().gradientWeights(layerIdx)[stream].setToZero();
        }
    }

    // Repeat.
    return true;
}

// explicit template instantiation
template class FeedForwardTrainer<f32>;
template class FeedForwardTrainer<f64>;

}  // namespace Nn
