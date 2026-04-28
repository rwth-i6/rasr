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
#ifndef _NN_NEURAL_NETWORK_LAYER_ACTIVATION_FUNCTION_HH
#define _NN_NEURAL_NETWORK_LAYER_ACTIVATION_FUNCTION_HH

// Neural Network Layer implementation
#include <cmath>
#include <numeric>

#include <Math/CudaMatrix.hh>
#include <Math/CudaVector.hh>
#include "NeuralNetworkLayer.hh"

namespace Nn {

//=============================================================================
/**	Apply identity activation to the input */
template<typename T>
class IdentityLayer : public virtual NeuralNetworkLayer<T> {
protected:
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;

public:
    IdentityLayer(const Core::Configuration& config);
    virtual ~IdentityLayer();

public:
    virtual void forward(const std::vector<NnMatrix*>& input, NnMatrix& output);
    virtual void backpropagateActivations(const NnMatrix& errorSignalIn, NnMatrix& errorSignalOut,
                                          const NnMatrix& activations);
};

//=============================================================================
/**	Apply tanh activation to the input */
template<typename T>
class TanhLayer : public virtual NeuralNetworkLayer<T> {
protected:
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;

public:
    TanhLayer(const Core::Configuration& config);
    virtual ~TanhLayer();

protected:
    /**	Apply the tanh function to the input features */
    void _forward(const NnMatrix& input, NnMatrix& output);
    void _backpropagateActivations(const NnMatrix& errorSignalIn, NnMatrix& errorSignalOut, const NnMatrix& activations);

public:
    /**	Apply the tanh function to the input features */
    virtual void forward(const std::vector<NnMatrix*>& input, NnMatrix& output);
    virtual void backpropagateActivations(const NnMatrix& errorSignalIn, NnMatrix& errorSignalOut,
                                          const NnMatrix& activations);
};

//=============================================================================
/**	Apply sigmoid activation to the input */
template<typename T>
class SigmoidLayer : public virtual NeuralNetworkLayer<T> {
protected:
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;

private:
    const T        gamma_;
    bool           logOutput_;
    mutable double timeForwardSigmoid_, timeBackwardSigmoid_;

protected:
    static const Core::ParameterFloat paramScaleGamma;
    static const Core::ParameterBool  paramLogOutput;

    T getGamma() const {
        return gamma_;
    }

public:
    SigmoidLayer(const Core::Configuration& config);
    virtual ~SigmoidLayer();

protected:
    void _forward(const NnMatrix& input, NnMatrix& output);
    void _backpropagateActivations(const NnMatrix& errorSignalIn, NnMatrix& errorSignalOut, const NnMatrix& activations);

public:
    virtual void forward(const std::vector<NnMatrix*>& input, NnMatrix& output);
    virtual void backpropagateActivations(const NnMatrix& errorSignalIn,
                                          NnMatrix&       errorSignalOut,
                                          const NnMatrix& activations);
    // log runtime statistics
    virtual void finalize();
};

//=============================================================================
/**	Apply softmax activation to the input */
template<typename T>
class SoftmaxLayer : public virtual NeuralNetworkLayer<T> {
    typedef NeuralNetworkLayer<T> Precursor;

protected:
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;

private:
    mutable double timeForwardSoftmax_, timeBackwardSoftmax_;

public:
    SoftmaxLayer(const Core::Configuration& config);
    virtual ~SoftmaxLayer();

protected:
    void _forward(const NnMatrix& input, NnMatrix& output);
    void _backpropagateActivations(const NnMatrix& errorSignalIn,
                                   NnMatrix&       errorSignalOut,
                                   const NnMatrix& activations);

public:
    virtual void forward(const std::vector<NnMatrix*>& input, NnMatrix& output);
    virtual void backpropagateActivations(const NnMatrix& errorSignalIn,
                                          NnMatrix&       errorSignalOut,
                                          const NnMatrix& activations);
    // log runtime statistics
    virtual void finalize();
};

//=============================================================================
/**     Apply non-overlapping maxout activation, different reduction size per maxout node is possible */
template<typename T>
class MaxoutVarLayer : public virtual NeuralNetworkLayer<T> {
    typedef NeuralNetworkLayer<T> Precursor;

protected:
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;
    static const Core::ParameterInt     paramMaxoutSize;     // either same input size for each maxout node
    static const Core::ParameterString  paramParameterFile;  // or     use different input size per maxout node
    Types<u32>::NnVector                mixture;             // mixture size per output
    Types<u32>::NnVector                offset;              // offset where the mixture starts
    Types<u32>::NnMatrix                maxindex;            // to store the index of the maximum, assumed forward and BP called properly...
    u32                                 avgmixture;          // average mixture size
private:
    mutable double   timeForwardMaxoutVar_, timeBackwardMaxoutVar_;
    mutable NnVector tmpVector_;
    mutable NnMatrix tmpMatrix_;

public:
    MaxoutVarLayer(const Core::Configuration& config);
    virtual void setInputDimension(u32 stream, u32 dim);
    virtual ~MaxoutVarLayer();

protected:
    void _forward(const NnMatrix& input, NnMatrix& output);
    void _backpropagateActivations(const NnMatrix& errorSignalIn,
                                   NnMatrix&       errorSignalOut,
                                   const NnMatrix& activations);

public:
    virtual void forward(const std::vector<NnMatrix*>& input, NnMatrix& output);
    virtual void backpropagateActivations(const NnMatrix& errorSignalIn,
                                          NnMatrix&       errorSignalOut,
                                          const NnMatrix& activations);

    const Types<u32>::NnVector& getMixture() const {
        return mixture;
    }
    const Types<u32>::NnVector& getOffset() const {
        return offset;
    }
    Types<u32>::NnMatrix& getMaxindex() {
        return maxindex;
    }

    // log runtime statistics
    virtual void finalize();
};

//=============================================================================
/**	Apply linear rectified activation to the input */
template<typename T>
class RectifiedLayer : public virtual NeuralNetworkLayer<T> {
protected:
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;

private:
    mutable double timeForwardRectified_, timeBackwardRectified_;

public:
    RectifiedLayer(const Core::Configuration& config);
    virtual ~RectifiedLayer();

protected:
    void _forward(const NnMatrix& input, NnMatrix& output);
    void _backpropagateActivations(const NnMatrix& errorSignalIn,
                                   NnMatrix&       errorSignalOut,
                                   const NnMatrix& activations);

public:
    virtual void forward(const std::vector<NnMatrix*>& input, NnMatrix& output);
    virtual void backpropagateActivations(const NnMatrix& errorSignalIn,
                                          NnMatrix&       errorSignalOut,
                                          const NnMatrix& activations);
    // log runtime statistics
    virtual void finalize();
};

//=============================================================================
/**	Apply exponential linear units to the input */
/* http://arxiv.org/pdf/1511.07289v1.pdf
 * Fast and accurate deep network learning by exponential linear units (ELUs)
 * Djork-Arne Clevert, Thomas Unterthiner, Sepp Hochreiter
 */
template<typename T>
class ExponentialLinearLayer : public virtual NeuralNetworkLayer<T> {
protected:
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;

private:
    T              alpha_;
    mutable double timeForwardExponentialLinear_, timeBackwardExponentialLinear_;

public:
    ExponentialLinearLayer(const Core::Configuration& config);
    virtual ~ExponentialLinearLayer();

protected:
    void _forward(const NnMatrix& input, NnMatrix& output);
    void _backpropagateActivations(const NnMatrix& errorSignalIn,
                                   NnMatrix&       errorSignalOut,
                                   const NnMatrix& activations);

public:
    virtual void forward(const std::vector<NnMatrix*>& input, NnMatrix& output);
    virtual void backpropagateActivations(const NnMatrix& errorSignalIn,
                                          NnMatrix&       errorSignalOut,
                                          const NnMatrix& activations);
    // log runtime statistics
    virtual void finalize();
};
}  // namespace Nn

#endif  // _NN_NEURAL_NETWORK_LAYER_ACTIVATION_FUNCTION_HH
