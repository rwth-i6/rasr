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
#include <Nn/Types.hh>
#include <Math/CudaMatrix.hh>
#include <Math/CudaVector.hh>
#include <Core/Types.hh>
#include <Core/Utility.hh>
#include <Python/Init.hh>
#include <Python/Utilities.hh>
#include <Python/Numpy.hh>
#include "PythonLayer.hh"

namespace Nn {

static const Core::ParameterString paramPyModPath(
        "pymod-path", "the path containing the Python module", "");

static const Core::ParameterString paramPyModName(
        "pymod-name", "the module-name, such that 'import x' would work", "");

static const Core::ParameterString paramPyModConfig(
        "pymod-config",
        "config-string, passed to init()",
        "");

template<typename T>
PythonLayer<T>::PythonLayer(const Core::Configuration &config)
    : Core::Component(config), Precursor(config), timeForward_(0), timeBackward_(0)
{
    pythonInitializer_.init();

    // Get us the CPython GIL. However, when we return here,
    // it will get released and other Python threads can run.
    Python::ScopedGIL gil;

    std::string pyModPath(paramPyModPath(config));
    if(!pyModPath.empty())
        Python::addSysPath(pyModPath);

    std::string pyModName(paramPyModName(config));
    if(pyModName.empty()) {
        pythonCriticalError("PythonLayer: need Python module name (pymod-name)");
        return;
    }

    Python::ObjRef pyMod;
    pyMod.takeOver(PyImport_ImportModule(pyModName.c_str()));
    if(!pyMod) {
        pythonCriticalError("PythonLayer: cannot import module '%s'", pyModName.c_str());
        return;
    }

    std::string pyConfig(paramPyModConfig(config));
    pyObject_.takeOver(Python::PyCallKw(pyMod, "SprintNnPythonLayer", "{s:s}", "config", pyConfig.c_str()));
    if(!pyObject_) {
        pythonCriticalError("PythonLayer: failed to call SprintNnPythonLayer");
        return;
    }

    // The output dimension is usually not changed anymore and this function would never be called otherwise.
    setOutputDimension(Precursor::outputDimension_);
    // Sprint will usually calculate dynamically all the input dimensions and then call setInputDimensions(),
    // followed by initializeNetworkParameters().
}

template<typename T>
PythonLayer<T>::~PythonLayer() {
    {
        require(Py_IsInitialized()); // should not happen. only via pythonInitializer_.
        Python::ScopedGIL gil;
        pyObject_.clear();
    }
    pythonInitializer_.uninit();
}



// Specialized over Core::Component::criticialError():
// Handles recent Python exceptions (prints them).
// Note that Py_Finalize() is not called here but registered via
// std::atexit(). See constructor code+comment.
template<typename T>
Core::Component::Message PythonLayer<T>::pythonCriticalError(const char* msg, ...) const {
    Python::handlePythonError();

    va_list ap;
    va_start(ap, msg);
    Core::Component::Message msgHelper = Core::Component::vCriticalError(msg, ap);
    va_end(ap);
    return msgHelper;
}

template<typename T>
Python::CriticalErrorFunc PythonLayer<T>::getPythonCriticalErrorFunc() const {
    return [this]() {
        return this->pythonCriticalError("PythonLayer: ");
    };
}



template<typename T>
void PythonLayer<T>::setInputDimension(u32 stream, u32 size) {
    Precursor::setInputDimension(stream, size);
    Python::ScopedGIL gil;
    Python::PyCallKw_IgnRet_HandleError(
                getPythonCriticalErrorFunc(),
                pyObject_, "setInputDimension", "{s:i,s:i}",
                "stream", stream,
                "size", size);
}

template<typename T>
void PythonLayer<T>::setOutputDimension(u32 size) {
    Precursor::setOutputDimension(size);
    Python::ScopedGIL gil;
    Python::PyCallKw_IgnRet_HandleError(
                getPythonCriticalErrorFunc(),
                pyObject_, "setOutputDimension", "{s:i}",
                "size", size);
}

template<typename T>
void PythonLayer<T>::initializeNetworkParameters() {
    Python::ScopedGIL gil;
    Python::PyCallKw_IgnRet_HandleError(
                getPythonCriticalErrorFunc(),
                pyObject_, "initializeNetworkParameters", "");
}

template<typename T>
void PythonLayer<T>::loadNetworkParameters(const std::string &filename) {
    Python::ScopedGIL gil;
    Python::PyCallKw_IgnRet_HandleError(
                getPythonCriticalErrorFunc(),
                pyObject_, "loadNetworkParameters", "{s:s}",
                "filename", filename.c_str());
}

template<typename T>
inline void PythonLayer<T>::saveNetworkParameters(const std::string &filename) const {
    Python::ScopedGIL gil;
    Python::PyCallKw_IgnRet_HandleError(
                getPythonCriticalErrorFunc(),
                pyObject_, "saveNetworkParameters", "{s:s}",
                "filename", filename.c_str());
}

template<typename T>
bool PythonLayer<T>::isTrainable() const {
    Python::ScopedGIL gil;
    Python::ObjRef res;
    res.takeOver(Python::PyCallKw(pyObject_, "isTrainable", ""));
    if(!res) {
        pythonCriticalError("PythonLayer: exception occured while calling 'isTrainable'");
        return false;
    }
    if(!PyBool_Check(res)) {
        pythonCriticalError("PythonLayer: 'isTrainable' did not return a bool.");
        return false;
    }
    if(res.obj == Py_True)
        return true;
    if(res.obj == Py_False)
        return false;
    pythonCriticalError("PythonLayer: 'isTrainable' did return an invalid bool.");
    return false;
}


template<typename T>
void PythonLayer<T>::forward(const std::vector<NnMatrix*>& input, NnMatrix& output) {
    timeval start, end;
    TIMER_START(start);
    Python::ScopedGIL gil;
    Python::ObjRef input_ls;
    input_ls.takeOver(PyList_New(input.size()));
    for(size_t i = 0; i < input.size(); ++i) {
        input[i]->finishComputation(true);
        PyObject* np_array = NULL;
        if(!Python::nnMatrix2numpy(getPythonCriticalErrorFunc(), np_array, *input[i])) return;
        PyList_SetItem(input_ls, i, np_array);  // overtake ref
        input[i]->initComputation(false);
    }
    Python::ObjRef res;
    res.takeOver(Python::PyCallKw(pyObject_, "forward", "{s:O}", "input", input_ls.obj));
    if(!res) {
        pythonCriticalError("PythonLayer: exception occured while calling 'forward'");
        return;
    }
    output.finishComputation(false);
    if(!Python::numpy2nnMatrix(getPythonCriticalErrorFunc(), res, output)) return;
    output.initComputation(true);
    TIMER_STOP(start, end, timeForward_);
}

template<typename T>
void PythonLayer<T>::backpropagateActivations(const NnMatrix& errorSignalIn, NnMatrix& errorSignalOut, const NnMatrix& activations) {
    // First, this is called, then backpropagateWeights.
    // Here we usually have &errorSignalIn == &errorSignalOut,
    // i.e. we are expected to handle that inplace,
    // and errorSignalOut still has our output-dimensions.
    // Note that in ase this is the lowest trainable layer,
    // backpropagateWeights might not be called anymore.
    // Thus, we have to catch the errorSignalIn always at this point.
    timeval start, end;
    TIMER_START(start);
    Python::ScopedGIL gil;
    Python::ObjRef in;
    errorSignalIn.finishComputation(true);
    if(!Python::nnMatrix2numpy(getPythonCriticalErrorFunc(), in.obj, errorSignalIn)) return;
    errorSignalIn.initComputation(false);
    backpropRes_.takeOver(Python::PyCallKw(pyObject_, "backpropagate", "{s:O}", "errorSignalIn", in.obj));
    if(!backpropRes_) {
        pythonCriticalError("PythonLayer: exception occured while calling 'backpropagate'");
        return;
    }
    if(!PyTuple_Check(backpropRes_)) {
        pythonCriticalError("PythonLayer: 'backpropagate' did not return a tuple");
        return;
    }
    TIMER_STOP(start, end, timeBackward_);
}

template<typename T>
void PythonLayer<T>::backpropagateWeights(const NnMatrix& errorSignalIn, std::vector<NnMatrix*>& errorSignalOut) {
    // backpropagateActivations was called before. See comment there.
    timeval start, end;
    TIMER_START(start);
    Python::ScopedGIL gil;
    if((size_t)PyTuple_Size(backpropRes_) != errorSignalOut.size()) {
        pythonCriticalError("PythonLayer: 'backpropagate' returned %zd items but we expected %zu items",
                            PyTuple_Size(backpropRes_), errorSignalOut.size());
        return;
    }
    for(size_t i = 0; i < errorSignalOut.size(); ++i) {
        require(errorSignalOut[i]);
        errorSignalOut[i]->finishComputation(false);
        PyObject* np_array = PyTuple_GetItem(backpropRes_, i); // borrowed
        if(!Python::numpy2nnMatrix(getPythonCriticalErrorFunc(), np_array, *errorSignalOut[i])) return;
        errorSignalOut[i]->initComputation(true);
    }
    backpropRes_.clear();
    TIMER_STOP(start, end, timeBackward_);
}


template<typename T>
void PythonLayer<T>::finalize() {
    {
        Python::ScopedGIL gil;
        Python::PyCallKw_IgnRet_HandleError(
                    getPythonCriticalErrorFunc(),
                    pyObject_, "finalize", "");
    }
    if(this->measureTime_) {
        this->log("Python layer: Time for forward pass: ") << timeForward_;
        this->log("Python layer: Time for backward pass: ") << timeBackward_;
    }
    Precursor::finalize();
}

template<typename T>
u32 PythonLayer<T>::getNumberOfFreeParameters() const {
    Python::ScopedGIL gil;
    Python::ObjRef res;
    res.takeOver(Python::PyCallKw(pyObject_, "getNumberOfFreeParameters", ""));
    if(!res) {
        pythonCriticalError("PythonLayer: exception occured while calling 'getNumberOfFreeParameters'");
        return false;
    }
    long n = PyLong_AsLong(res);
    if(PyErr_Occurred()) {
        pythonCriticalError("PythonLayer: 'getNumberOfFreeParameters' did not return an int/long.");
        return 0;
    }
    if(n < 0) {
        pythonCriticalError("PythonLayer: 'getNumberOfFreeParameters' did return a negative number");
        return 0;
    }
    return (u32) n;
}



// explicit template instantiation
template class PythonLayer<f32>;
template class PythonLayer<f64>;

}
