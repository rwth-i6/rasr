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
#include "LinearAndActivationLayer.hh"

#include <Math/Module.hh>
#include <Math/Vector.hh>

#include "ClassLabelWrapper.hh"

using namespace Nn;

template<typename T>
LinearAndSigmoidLayer<T>::LinearAndSigmoidLayer(const Core::Configuration& config)
        : Core::Component(config),
          NeuralNetworkLayer<T>(config),
          LinearLayer<T>(config),
          SigmoidLayer<T>(config) {}

template<typename T>
LinearAndSigmoidLayer<T>::~LinearAndSigmoidLayer() {}

template<typename T>
inline void LinearAndSigmoidLayer<T>::_forward(const std::vector<NnMatrix*>& input, NnMatrix& output, bool reset) {
    PrecursorLinear::_forward(input, output, reset);
    PrecursorSigmoid::_forward(output, output);
}

template<typename T>
inline void LinearAndSigmoidLayer<T>::forward(const std::vector<NnMatrix*>& input, NnMatrix& output) {
    PrecursorLinear::forward(input, output);
}

template<typename T>
inline void LinearAndSigmoidLayer<T>::backpropagateActivations(const NnMatrix& errorSignalIn,
                                                               NnMatrix& errorSignalOut, const NnMatrix& activations) {
    PrecursorSigmoid::backpropagateActivations(errorSignalIn, errorSignalOut, activations);
}

template<typename T>
inline void LinearAndSigmoidLayer<T>::backpropagateWeights(const NnMatrix&         errorSignalIn,
                                                           std::vector<NnMatrix*>& errorSignalOut) {
    PrecursorLinear::backpropagateWeights(errorSignalIn, errorSignalOut);
}

template<typename T>
void LinearAndSigmoidLayer<T>::addToWeightsGradient(const NnMatrix& layerInput, const NnMatrix& errorSignalIn, u32 stream, NnMatrix& gradientWeights) {
    PrecursorLinear::addToWeightsGradient(layerInput, errorSignalIn, stream, gradientWeights);
}

template<typename T>
void LinearAndSigmoidLayer<T>::addToBiasGradient(const NnMatrix& layerInput, const NnMatrix& errorSignalIn, u32 stream, NnVector& gradientBias) {
    PrecursorLinear::addToBiasGradient(layerInput, errorSignalIn, stream, gradientBias);
}

template<typename T>
inline void LinearAndSigmoidLayer<T>::initComputation(bool sync) const {
    PrecursorLinear::initComputation(sync);
}

template<typename T>
inline void LinearAndSigmoidLayer<T>::finalize() {
    PrecursorLinear::finalize();
    PrecursorSigmoid::finalize();
}

template<typename T>
u32 LinearAndSigmoidLayer<T>::getNumberOfFreeParameters() const {
    return PrecursorLinear::getNumberOfFreeParameters();
}

//=============================================================================

template<typename T>
const Core::ParameterBool LinearAndSoftmaxLayer<T>::paramEvaluateSoftmax(
        "evaluate-softmax", "apply softmax", true);

template<typename T>
LinearAndSoftmaxLayer<T>::LinearAndSoftmaxLayer(const Core::Configuration& config)
        : Core::Component(config),
          NeuralNetworkLayer<T>(config),
          LinearLayer<T>(config),
          SoftmaxLayer<T>(config),
          evaluateSoftmax_(paramEvaluateSoftmax(config)),
          logPriorIsRemovedFromBias_(false) {
    if (!evaluateSoftmax_)
        this->log("linear+softmax layer: do not evaluate softmax-nonlinearity");
}

template<typename T>
inline void LinearAndSoftmaxLayer<T>::_forward(const std::vector<NnMatrix*>& input, NnMatrix& output, bool reset) {
    PrecursorLinear::_forward(input, output, reset);
    if (evaluateSoftmax_)
        PrecursorSoftmax::_forward(output, output);

    if (PrecursorSoftmax::dataChannel_.isOpen()) {
        PrecursorSoftmax::dataChannel_ << Core::XmlOpen("layer-output-data");
        output.write(PrecursorSoftmax::dataChannel_);
        PrecursorSoftmax::dataChannel_ << Core::XmlClose("layer-output-data");
    }
}

template<typename T>
inline void LinearAndSoftmaxLayer<T>::forward(const std::vector<NnMatrix*>& input, NnMatrix& output) {
    PrecursorLinear::forward(input, output);
}

template<typename T>
inline void LinearAndSoftmaxLayer<T>::applySoftmax(NnMatrix& activations) {
    PrecursorSoftmax::_forward(activations, activations);
}

template<typename T>
inline void LinearAndSoftmaxLayer<T>::backpropagateActivations(const NnMatrix& errorSignalIn,
                                                               NnMatrix&       errorSignalOut,
                                                               const NnMatrix& activations) {
    PrecursorSoftmax::backpropagateActivations(errorSignalIn, errorSignalOut, activations);
}

template<typename T>
inline void LinearAndSoftmaxLayer<T>::backpropagateWeights(const NnMatrix&         errorSignalIn,
                                                           std::vector<NnMatrix*>& errorSignalOut) {
    PrecursorLinear::backpropagateWeights(errorSignalIn, errorSignalOut);
}

template<typename T>
void LinearAndSoftmaxLayer<T>::addToWeightsGradient(const NnMatrix& layerInput,
                                                    const NnMatrix& errorSignalIn,
                                                    u32             stream,
                                                    NnMatrix&       gradientWeights) {
    PrecursorLinear::addToWeightsGradient(layerInput, errorSignalIn, stream, gradientWeights);
}

template<typename T>
void LinearAndSoftmaxLayer<T>::addToBiasGradient(const NnMatrix& layerInput,
                                                 const NnMatrix& errorSignalIn,
                                                 u32             stream,
                                                 NnVector&       gradientBias) {
    PrecursorLinear::addToBiasGradient(layerInput, errorSignalIn, stream, gradientBias);
}

template<typename T>
inline T LinearAndSoftmaxLayer<T>::getScore(const NnMatrix& in, u32 columnIndex) {
    T result = -PrecursorLinear::bias_.at(columnIndex);
    for (u32 stream = 0; stream < this->nInputActivations(); stream++) {
        result -= PrecursorLinear::weights_[stream].dotWithColumn(in, columnIndex);
    }
    return result;
}

template<typename T>
inline void LinearAndSoftmaxLayer<T>::initComputation(bool sync) const {
    PrecursorLinear::initComputation(sync);
}

template<typename T>
inline void LinearAndSoftmaxLayer<T>::finalize() {
    PrecursorLinear::finalize();
    PrecursorSoftmax::finalize();
}

template<typename T>
u32 LinearAndSoftmaxLayer<T>::getNumberOfFreeParameters() const {
    return PrecursorLinear::getNumberOfFreeParameters();
}

//=============================================================================

namespace Nn {

template class LinearAndSigmoidLayer<f32>;
template class LinearAndSigmoidLayer<f64>;

template class LinearAndSoftmaxLayer<f32>;
template class LinearAndSoftmaxLayer<f64>;

}  // namespace Nn
