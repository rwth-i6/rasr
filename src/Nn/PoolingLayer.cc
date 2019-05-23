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
#include "PoolingLayer.hh"

#include <Math/Matrix.hh>
#include <Math/Module.hh>
#include <Math/Vector.hh>

using namespace Nn;

template<typename T>
const Core::ParameterBool PoolingLayer<T>::paramTrainable(
        "trainable", "Can the parameters of this layer be trained?", false);

template<typename T>
const Core::ParameterInt PoolingLayer<T>::paramPoolingSize(
        "pooling-size", "Pooling size", 2);

template<typename T>
const Core::ParameterBool PoolingLayer<T>::paramPoolingAbs(
        "pooling-abs", "Select max(abs(x_i)) instead of max(x_i)", false);

template<typename T>
const Core::ParameterInt PoolingLayer<T>::paramPoolingPnorm(
        "pooling-pnorm", "Pooling operation: L_p norm (active when p>0); default pooling operation: max", 0);

template<typename T>
PoolingLayer<T>::PoolingLayer(const Core::Configuration& config)
        : Core::Component(config),
          NeuralNetworkLayer<T>(config),
          trainable_(paramTrainable(config)),
          poolingSize_(paramPoolingSize(config)),
          poolingAbs_(paramPoolingAbs(config)),
          poolingPnorm_(paramPoolingPnorm(config)),
          argmax_(0),
          timeForwardLinear_(0),
          timeBackward_(0) {
    std::string mode = "max";
    if (poolingPnorm_ > 0)
        mode = "P-norm";
    if (poolingAbs_)
        mode = "max(abs)";

    Core::Component::log("Pooling layer mode '") << mode << "', pooling size = " << poolingSize_;
}

template<typename T>
PoolingLayer<T>::~PoolingLayer() {}

template<typename T>
void PoolingLayer<T>::setInputDimension(u32 stream, u32 size) {
    Precursor::setInputDimension(stream, size);
    Precursor::needInit_ = false;
}

template<typename T>
void PoolingLayer<T>::setOutputDimension(u32 size) {
    Precursor::outputDimension_ = size;
    Precursor::needInit_        = false;
    // TODO: sanity check outsize == insize/poolingsize
}

template<typename T>
void PoolingLayer<T>::_forward(const std::vector<NnMatrix*>& input, NnMatrix& output, bool reset) {
    timeval start, end;
    gettimeofday(&start, NULL);

    if (poolingPnorm_ == 0) {
        argmax_.resize(output.nRows(), output.nColumns());
        output.addPoolingMax(*(input.at(0)), argmax_, poolingSize_, poolingAbs_);
    }
    else {
        output.addPoolingPnorm(*(input.at(0)), poolingSize_, poolingPnorm_);
    }

    gettimeofday(&end, NULL);
    timeForwardLinear_ += Core::timeDiff(start, end);
}

template<typename T>
void PoolingLayer<T>::_backpropagateWeights(const NnMatrix& errorSignalIn, std::vector<NnMatrix*>& errorSignalOut) {
    timeval start, end;
    gettimeofday(&start, NULL);

    if (poolingPnorm_ == 0) {
        require_eq(argmax_.nRows(), errorSignalIn.nRows());
        errorSignalOut.at(0)->backpropPoolingMax(argmax_, errorSignalIn);
    }
    else {
        errorSignalOut.at(0)->backpropPoolingPnorm(errorSignalIn, poolingSize_, poolingPnorm_);
    }

    gettimeofday(&end, NULL);
    timeBackward_ += Core::timeDiff(start, end);
}

template<typename T>
void PoolingLayer<T>::forward(const std::vector<NnMatrix*>& input, NnMatrix& output, bool reset) {
    _forward(input, output, reset);
}

template<typename T>
void PoolingLayer<T>::backpropagateWeights(const NnMatrix& errorSignalIn, std::vector<NnMatrix*>& errorSignalOut) {
    _backpropagateWeights(errorSignalIn, errorSignalOut);
}

template<typename T>
inline void PoolingLayer<T>::forward(const std::vector<NnMatrix*>& input, NnMatrix& output) {
    forward(input, output, true);
}

template<typename T>
void PoolingLayer<T>::initComputation(bool sync) const {
    if (!isComputing_) {
        argmax_.initComputation(sync);
    }
    isComputing_ = true;
}

template<typename T>
void PoolingLayer<T>::finishComputation(bool sync) const {
    if (isComputing_) {
        argmax_.finishComputation(sync);
    }
    isComputing_ = false;
}

template<typename T>
void PoolingLayer<T>::finalize() {
    if (this->measureTime_) {
        this->log("Pooling layer: Time for linear part of forward pass: ") << timeForwardLinear_;
        this->log("Pooling layer: Time for backward pass: ") << timeBackward_;
    }
    Precursor::finalize();
}

template<typename T>
u32 PoolingLayer<T>::getNumberOfFreeParameters() const {
    u32 params = 0;
    if (trainable_) {
    }
    return params;
}

/*===========================================================================*/
// explicit template instantiation
namespace Nn {
template class PoolingLayer<f32>;
template class PoolingLayer<f64>;
}  // namespace Nn
