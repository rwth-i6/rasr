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
#include "OperationLayer.hh"

#include <Math/Random.hh>
#include <Math/Module.hh>
#include <Math/Vector.hh>
#include <Math/Matrix.hh>

using namespace Nn;


template<typename T>
const Core::Choice OperationLayer<T>::choiceOperation(
        "streams-linear-interpolation", streamLinearCombine,
        "streams-posterior-combine-inverse-entropy", posteriorCombInvEntropy,
        "streams-posterior-combine-dempster-shafer", posteriorCombDS,
        Core::Choice::endMark());

template<typename T>
const Core::ParameterChoice OperationLayer<T>::paramOperation(
        "operation", &choiceOperation,
        "operation to perform (on streams)", streamLinearCombine);

template<typename T>
const Core::ParameterFloatVector OperationLayer<T>::paramInterpolationWeights(
        "interpolation-weights", "streams weights for interpolation (space separated floats)", " ");

template<typename T>
const Core::ParameterBool OperationLayer<T>::paramApplyLog(
        "apply-log", "apply log to the output (e.g. for posterior combination)", false);

template<typename T>
const Core::ParameterFloat OperationLayer<T>::paramGamma(
        "gamma", "scaling factor (e.g. for DS posterior combination)", 0);

template<typename T>
const Core::ParameterBool OperationLayer<T>::paramHasBias(
        "has-bias", "has bias", true);


template<typename T>
const Core::ParameterBool OperationLayer<T>::paramTrainable(
        "trainable", "Can the parameters of this layer be trained?", false);


template<typename T>
OperationLayer<T>::OperationLayer(const Core::Configuration &config) :
    Core::Component(config),
    NeuralNetworkLayer<T>(config),
    operation_((Operation)paramOperation(config)),
    applyLog_(paramApplyLog(config)),
    hasBias_(paramHasBias(config)),
    bias_(0),
    weights_(0),
    trainable_(paramTrainable(config)),
    gamma_(paramGamma(config)),
    timeForwardLinear_(0),
    timeForwardBias_(0),
    timeBackward_(0),
    interpolation_weights_(paramInterpolationWeights(config))
{
    std::string out = "Operation layer performs ";
    switch (operation_) {
        case streamLinearCombine:
            out += "linear combination of input streams using weights ";
            for (size_t i=0; i<interpolation_weights_.size(); ++i) out += Core::form("%f ", interpolation_weights_[i]);
            break;
        case posteriorCombInvEntropy:
            out += "inverse-entropy combination of input streams (need to be normalized posteriors!)";
            break;
        case posteriorCombDS:
            out += "Dempster-Shafer combination of input streams (need to be normalized posteriors!), ";
            out += Core::form("gamma=%f", gamma_);
            break;
    }
    this->log(out.c_str());
}

template<typename T>
OperationLayer<T>::~OperationLayer() {}

template<typename T>
void OperationLayer<T>::setInputDimension(u32 stream, u32 size) {
    Precursor::setInputDimension(stream, size);
    if ((weights_.size() <= stream) || (weights_[stream].nRows() != size)) {
        Precursor::needInit_ = true;
    }
    Precursor::needInit_ = false;
}

template<typename T>
void OperationLayer<T>::setOutputDimension(u32 size) {
    Precursor::outputDimension_ = size;
}

template<typename T>
void OperationLayer<T>::initializeNetworkParameters() {
}

/**	Initialize the weights with random values */
template<typename T>
void OperationLayer<T>::initializeParametersRandomly() {
}

/**	Initialize the weights with zero */
template<typename T>
void OperationLayer<T>::initializeParametersWithZero() {
}

/**	Initialize the weights with zero */
template<typename T>
void OperationLayer<T>::initializeParametersWithIdentityMatrix() {
}

/**	Initialize the weights from file */
template<typename T>
void OperationLayer<T>::loadNetworkParameters(const std::string &filename) {
    // Initialization done
    Precursor::needInit_ = false;
}

/**	Save weights to file */
template<typename T>
inline void OperationLayer<T>::saveNetworkParameters(const std::string &filename) const {
}

/**	Forward the input */
template<typename T>
void OperationLayer<T>::_forward(const std::vector<NnMatrix*>& input, NnMatrix& output, bool reset) {
    // boundary check
    require_eq(Precursor::inputDimensions_.size(), interpolation_weights_.size());
    timeval start, end;

    gettimeofday(&start, NULL);
    NnVector norm;
    NnVector *weight;
    T Hmax;
    output.fill(0);

    switch (operation_) {
        case streamLinearCombine:
            for (u32 stream = 0; stream < Precursor::inputDimensions_.size(); stream++) {
                output.add(*(input.at(stream)), (T)interpolation_weights_[stream]);
            }
            break;

        case posteriorCombInvEntropy:
            if (frame_weights_.size() != Precursor::inputDimensions_.size()) {
                frame_weights_.resize(Precursor::inputDimensions_.size());
                for (u32 stream = 0; stream < Precursor::inputDimensions_.size(); stream++) {
                    frame_weights_[stream] = new NnVector(input.at(stream)->nColumns());
                }
            }
            for (u32 stream = 0; stream < Precursor::inputDimensions_.size(); stream++) {
                weight = frame_weights_[stream];
                weight->initComputation(false);
                weight->columnEntropy(*(input.at(stream)));
                weight->pow(-1.0);
            }
            norm.resize(input.at(0)->nColumns());
            norm.initComputation(false);
            norm.setToZero();
            for (u32 stream = 0; stream < Precursor::inputDimensions_.size(); stream++) {
                norm.add(*(frame_weights_[stream]));
                output.addWithColumnWeights(*(input.at(stream)), *(frame_weights_[stream]));
            }
            output.divideColumnsByScalars(norm);
            if (applyLog_) output.log();

            break;

        case posteriorCombDS:
            require_eq(Precursor::inputDimensions_.size(), 2);
            if (frame_weights_.size() != Precursor::inputDimensions_.size()) {
                frame_weights_.resize(Precursor::inputDimensions_.size());
                for (u32 stream = 0; stream < Precursor::inputDimensions_.size(); stream++) {
                    frame_weights_[stream] = new NnVector(input.at(stream)->nColumns());
                }
            }
            Hmax = std::log(input.at(0)->nRows());
            for (u32 stream = 0; stream < Precursor::inputDimensions_.size(); stream++) {
                weight = frame_weights_[stream];
                weight->initComputation(false);
                weight->columnEntropy(*(input.at(stream)));
                weight->scale(-1.0/Hmax);
                weight->addConstantElementwise(1.0);
                weight->pow(gamma_);
            }
            input.at(0)->finishComputation(false);
            NnMatrix tmp(*input.at(0));
            input.at(0)->initComputation(false);
            tmp.initComputation(false);
            tmp.elementwiseMultiplication(*(input.at(1)));
            tmp.add(*(input.at(0)), T(-1.0));
            tmp.add(*(input.at(1)), T(-1.0));
            tmp.multiplyColumnsByScalars(*(frame_weights_[0]));
            tmp.multiplyColumnsByScalars(*(frame_weights_[1]));

            for (u32 stream = 0; stream < Precursor::inputDimensions_.size(); stream++) {
                output.addWithColumnWeights(*(input.at(stream)), *(frame_weights_[stream]));
            }
            output.add(tmp);
            if (applyLog_) {
                output.ensureMinimalValue(1e-20);
                output.log();
            }
            break;
    }

    Math::Cuda::deviceSync(Precursor::measureTime_ && Math::CudaDataStructure::hasGpu());
    gettimeofday(&end, NULL);
    timeForwardLinear_  += Core::timeDiff(start, end);
}

template<typename T>
void OperationLayer<T>::_backpropagateWeights(const NnMatrix& errorSignalIn, std::vector<NnMatrix*>& errorSignalOut) {
}

template<typename T>
void OperationLayer<T>::forward(const std::vector<NnMatrix*>& input, NnMatrix& output, bool reset) {
    require(!Precursor::needInit_);
    _forward(input, output, reset);
}

template<typename T>
void OperationLayer<T>::backpropagateWeights(const NnMatrix& errorSignalIn, std::vector<NnMatrix*>& errorSignalOut) {
    errorSignalOut[0]->copy(errorSignalIn);
}

template<typename T>
void OperationLayer<T>::addToWeightsGradient(const NnMatrix& layerInput,
        const NnMatrix& errorSignalIn, u32 stream, NnMatrix& gradientWeights) {
}

template<typename T>
void OperationLayer<T>::addToBiasGradient(const NnMatrix& layerInput,
        const NnMatrix& errorSignalIn, u32 stream, NnVector& gradientBias) {
}

template<typename T>
inline void OperationLayer<T>::forward(const std::vector<NnMatrix*>& input, NnMatrix& output) {
    forward(input, output, true);
}

template<typename T>
void OperationLayer<T>::setParameters(const Math::Matrix<T>& parameters) {
    Precursor::needInit_ = false;
}

template<typename T>
void OperationLayer<T>::initComputation(bool sync) const {
    isComputing_ = true;
}

template<typename T>
void OperationLayer<T>::finishComputation(bool sync) const {
    isComputing_ = false;
}

template<typename T>
void OperationLayer<T>::finalize() {
    if (this->measureTime_){
        this->log("Operation layer: Time for linear part of forward pass: ") << timeForwardLinear_;
        this->log("Operation layer: Time for bias part of forward pass: ") << timeForwardBias_;
        this->log("Operation layer: Time for backward pass: ") << timeBackward_;
    }
    Precursor::finalize();
}

template<typename T>
u32 OperationLayer<T>::getNumberOfFreeParameters() const {
    u32 params = 0;
    return params;
}

/*===========================================================================*/
// explicit template instantiation
namespace Nn {
template class OperationLayer<f32>;
template class OperationLayer<f64>;
}
