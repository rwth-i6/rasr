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
#ifndef NN_PYTHONLAYER_HH
#define NN_PYTHONLAYER_HH

#include "NeuralNetworkLayer.hh"
#include <Core/Component.hh>
#include <Nn/Types.hh>
#include <Python.h>
#include <string>
#include <Python/Numpy.hh>
#include <Python/Init.hh>
#include <Python/Utilities.hh>

namespace Nn {

template<typename T>
class PythonLayer : public virtual NeuralNetworkLayer<T> {
public:
    typedef NeuralNetworkLayer<T> Precursor;
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;

protected:
    Python::Initializer pythonInitializer_;
    Python::ObjRef pyObject_;
    Python::ObjRef backpropRes_;
    double timeForward_, timeBackward_;

public:
    PythonLayer(const Core::Configuration &config);
    virtual ~PythonLayer();

    Core::Component::Message pythonCriticalError(const char* msg = 0, ...) const;
    Python::CriticalErrorFunc getPythonCriticalErrorFunc() const;

    // trainer needs to access weights and bias
    virtual NnMatrix* getWeights(u32 stream) { return NULL; }
    virtual NnVector* getBias() { return NULL; }
    virtual const NnMatrix* getWeights(u32 stream) const { return NULL; }
    virtual const NnVector* getBias() const { return NULL; }

    // IO
    virtual void initializeNetworkParameters();
    virtual void loadNetworkParameters(const std::string &filename);
    virtual void saveNetworkParameters(const std::string &filename) const;

    // initialization methods
    virtual void setInputDimension(u32 stream, u32 size);
    virtual void setOutputDimension(u32 size);

    // getter methods
    virtual bool isTrainable() const;

    // forward
    virtual void forward(const std::vector<NnMatrix*>& input, NnMatrix& output);

    // backward
    virtual void backpropagateActivations(const NnMatrix& errorSignalIn, NnMatrix& errorSignalOut, const NnMatrix& activations);
    virtual void backpropagateWeights(const NnMatrix& errorSignalIn, std::vector<NnMatrix*>& errorSignalOut);
    virtual void addToWeightsGradient(const NnMatrix& layerInput, const NnMatrix& errorSignalIn, u32 stream, NnMatrix& gradientWeights) {}
    virtual void addToBiasGradient(const NnMatrix& layerInput, const NnMatrix& errorSignalIn, u32 stream, NnVector& gradientBias) {}

    // If the layer is trainable, the Statistics class will collect the gradients.
    // It will expect these trainable params and call these functions.
    virtual void resizeWeightsGradient(Types<f32>::NnMatrix &gradient, u32 stream) const { gradient.resize(0, 0); }
    virtual void resizeBiasGradient(Types<f32>::NnVector &gradient) const { gradient.resize(0); }
    virtual void resizeWeightsGradient(Types<f64>::NnMatrix &gradient, u32 stream) const { gradient.resize(0, 0); }
    virtual void resizeBiasGradient(Types<f64>::NnVector &gradient) const { gradient.resize(0); }

    virtual void finalize();
    virtual u32 getNumberOfFreeParameters() const;


};

}

#endif // PYTHONLAYER_HH
