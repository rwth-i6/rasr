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
#include "LookupLayer.hh"

#include <Math/Module.hh>
#include <Math/Vector.hh>
#include <Math/Matrix.hh>

using namespace Nn;

template<typename T>
LookupLayer<T>::LookupLayer(Core::Configuration const& config) : Core::Component(config), NeuralNetworkLayer<T>(config), LinearLayer<T>(config) {
    Precursor::trainable_ = false;
}

template<typename T>
LookupLayer<T>::~LookupLayer() {}

template<typename T>
void LookupLayer<T>::setInputDimension(u32 stream, u32 size) {
    Precursor::setInputDimension(stream, size);
    require_eq(stream, 0u);
}

/**	Forward the input */
template<typename T>
void LookupLayer<T>::_forward(const std::vector<NnMatrix*>& input, NnMatrix& output, bool reset) {
    require_eq(bias_.size() * input[0]->nRows(), output.nRows());
    require_eq(weights_[0].nRows() * input[0]->nRows(), output.nRows());
    timeval start, end;

    gettimeofday(&start, NULL);
    output.copySelectedRowsOfMatrixIntoColumns(weights_[0], *(input.at(0)));

    Math::Cuda::deviceSync(Precursor::measureTime_ && Math::CudaDataStructure::hasGpu());
    gettimeofday(&end, NULL);
    Precursor::timeForwardLinear_ += Core::timeDiff(start, end);

    gettimeofday(&start, NULL);
    if (hasBias_) {
        for (u32 i = 0; i < input[0]->nRows(); i++) {
            output.addToAllColumnsWithOffset(bias_, i * bias_.size());
        }
    }
    Math::Cuda::deviceSync(Precursor::measureTime_ && Math::CudaDataStructure::hasGpu());
    gettimeofday(&end, NULL);
    Precursor::timeForwardBias_ += Core::timeDiff(start, end);
}

template<typename T>
void LookupLayer<T>::setParameters(const Math::Matrix<T>& parameters) {
    for (u32 stream = 0; stream < weights_.size(); stream++)
        require(!weights_[stream].isComputing());
    require(!bias_.isComputing());

    // resize bias/ weights
    u32 totalInputSize = 0;
    for (u32 stream = 0; stream < this->nInputActivations(); stream++) {
        totalInputSize += this->getInputDimension(stream);
    }
    require_eq(parameters.nRows() * totalInputSize, this->getOutputDimension());

    u32 inputSizeFromFile = hasBias_ ? parameters.nColumns() - 1 : parameters.nColumns();
    bias_.resize(parameters.nRows());
    weights_.resize(1);
    weights_[0].resize(parameters.nRows(), parameters.nColumns() - (hasBias_ ? 1 : 0));

    // Convert Flow::Matrix -> Flow::FastMatrix
    for (u32 row = 0; row < parameters.nRows(); row++) {
        // first: bias (first column)
        if (hasBias_) {
            bias_.at(row) = parameters[row][0];
        }

        // second: weights (all other elements)
        u32 column = hasBias_ ? 1 : 0;
        for (u32 r = 0; r < weights_[0].nColumns(); r++) {
            weights_[0].at(row, r) = parameters[row][column];
            column++;
        }
    }

    Precursor::needInit_ = false;
}

/*===========================================================================*/
// explicit template instantiation
namespace Nn {
template class LookupLayer<f32>;
template class LookupLayer<f64>;
}
