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
#ifndef _NN_NEURAL_NETWORK_POOLING_LAYER_HH
#define _NN_NEURAL_NETWORK_POOLING_LAYER_HH

#include <string>
#include <vector>

#include <Core/Types.hh>
#include <Math/Matrix.hh>
#include "NeuralNetworkLayer.hh"
#include "Types.hh"

namespace Nn {

/*
 * (Maximum) pooling layer
 *
 */
template<typename T>
class PoolingLayer : public NeuralNetworkLayer<T> {
    typedef NeuralNetworkLayer<T> Precursor;

protected:
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;

    using Precursor::isComputing_;

    static const Core::ParameterBool paramTrainable;
    static const Core::ParameterInt  paramPoolingSize;
    static const Core::ParameterBool paramPoolingAbs;
    static const Core::ParameterInt  paramPoolingPnorm;

    bool                          trainable_;
    u32                           poolingSize_;
    bool                          poolingAbs_;
    u32                           poolingPnorm_;
    Types<unsigned int>::NnMatrix argmax_;

private:
    double timeForwardLinear_, timeBackward_;

public:
    PoolingLayer(const Core::Configuration& config);
    virtual ~PoolingLayer();

    // initialization methods
    virtual void setInputDimension(u32 stream, u32 size);
    virtual void setOutputDimension(u32 size);

    // getter methods
    virtual bool isTrainable() const {
        return trainable_;
    }

    // forward
    void         forward(const std::vector<NnMatrix*>& input, NnMatrix& output, bool reset);
    virtual void forward(const std::vector<NnMatrix*>& input, NnMatrix& output);

    // backward
    virtual void backpropagateWeights(const NnMatrix& errorSignalIn, std::vector<NnMatrix*>& errorSignalOut);

    virtual void finalize();
    virtual u32  getNumberOfFreeParameters() const;

    virtual void initComputation(bool sync = true) const;
    virtual void finishComputation(bool sync = true) const;

protected:
    virtual void _forward(const std::vector<NnMatrix*>& input, NnMatrix& output, bool reset);
    virtual void _backpropagateWeights(const NnMatrix& errorSignalIn, std::vector<NnMatrix*>& errorSignalOut);
};

}  // namespace Nn

#endif  // _NN_NEURAL_NETWORK_POOLING_LAYER_HH
