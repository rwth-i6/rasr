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
#include "Numpy.hh"

// Note that this should be the only file which includes the Numpy C headers directly.
// See the comment in initNumpy().
#include <Python.h>
#include <numpy/numpyconfig.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>  // must include Python.h before this

#include <Core/Debug.hh>

#include "Utilities.hh"

// For Numpy 1.6 or earlier.
#ifndef NPY_ARRAY_F_CONTIGUOUS
#define NPY_ARRAY_F_CONTIGUOUS NPY_FORTRAN
#endif
#ifndef NPY_ARRAY_FORCECAST
#define NPY_ARRAY_FORCECAST NPY_FORCECAST
#endif

namespace Python {

template<typename T>
struct NumpyType;
template<>
struct NumpyType<f32> {
    static constexpr int type() {
        return NPY_FLOAT;
    }
};
template<>
struct NumpyType<f64> {
    static constexpr int type() {
        return NPY_DOUBLE;
    }
};
template<>
struct NumpyType<u32> {
    static constexpr int type() {
        return NPY_UINT32;
    }
};

void initNumpy() {
    // Init NumPy C-API.
    // Note that it is important that we do the _import_array() call in the same C unit
    // where we use the Numpy C-API. This is because it will init a static variable
    // which lives in this unit.
    // Other units which would want to call the Numpy C-API directly would thus
    // also need to call _import_array() somewhere.
    // However, we should anyway wrap all Numpy calls in this unit.
    {
        // _import_array() will overwrite the original exception in case any happens.
        // We want to see it, thus do the import right here, and in case it fails, print the original exception.
        PyObject* numpy_mod = PyImport_ImportModule("numpy.core.multiarray");
        if (!numpy_mod) {
            Core::printLog("initNumpy: `import numpy.core.multiarray` failed.");
            goto error;
        }
        Py_DECREF(numpy_mod);
    }

    if (_import_array() < 0) {
        Core::printLog("initNumpy: _import_array() failed.");
        goto error;
    }

    return;

error:
    handlePythonError();
    dumpModulesEnv();
    criticalError("NumPy init failed");
}

bool isNumpyArrayType(PyObject* obj) {
    return PyArray_Check(obj);
}

bool isNumpyArrayTypeExact(PyObject* obj) {
    return PyArray_CheckExact(obj);
}

template<typename NumpyT, typename ContainerT>
void numpy2rawMat(PyArrayObject* nparr, ContainerT& nnmat) {
    require_eq(PyArray_TYPE(nparr), NumpyType<NumpyT>::type());
    require_eq(PyArray_NDIM(nparr), 2);
    if (std::is_same<decltype(nnmat.at(0, 0)), NumpyT&>::value &&
        PyArray_STRIDES(nparr)[0] == sizeof(NumpyT) &&
        PyArray_STRIDES(nparr)[1] == sizeof(NumpyT) * nnmat.nRows()) {
        // fast version. we can just use memcpy
        memcpy(&nnmat.at(0, 0), PyArray_BYTES(nparr), nnmat.nRows() * nnmat.nColumns() * sizeof(NumpyT));
    }
    else {
        for (u32 i = 0; i < nnmat.nRows(); ++i)
            for (u32 j = 0; j < nnmat.nColumns(); ++j) {
                NumpyT v       = *(NumpyT*)PyArray_GETPTR2(nparr, i, j);
                nnmat.at(i, j) = (typename ContainerT::value_type)v;
            }
    }
}

template<typename NumpyT, typename ContainerT>
void numpy2rawVec(PyArrayObject* nparr, ContainerT& vec) {
    require_eq(PyArray_TYPE(nparr), NumpyType<NumpyT>::type());
    require_eq(PyArray_NDIM(nparr), 1);
    if (std::is_same<decltype(vec.at(0)), NumpyT&>::value && PyArray_STRIDES(nparr)[0] == sizeof(NumpyT)) {
        // fast version. we can just use memcpy
        memcpy(&vec.at(0), PyArray_BYTES(nparr), vec.size() * sizeof(NumpyT));
    }
    else {
        for (u32 i = 0; i < vec.size(); ++i) {
            NumpyT v  = *(NumpyT*)PyArray_GETPTR1(nparr, i);
            vec.at(i) = (typename ContainerT::value_type)v;
        }
    }
}

template<typename NumpyT, typename ContainerT>
void rawMat2numpy(PyArrayObject* nparr, const ContainerT& nnmat) {
    require_eq(PyArray_TYPE(nparr), NumpyType<NumpyT>::type());
    require_eq(PyArray_NDIM(nparr), 2);
    if (std::is_same<decltype(nnmat.at(0, 0)), const NumpyT&>::value &&
        PyArray_STRIDES(nparr)[0] == sizeof(NumpyT) &&
        PyArray_STRIDES(nparr)[1] == sizeof(NumpyT) * nnmat.nRows()) {
        // fast version. we can just use memcpy
        memcpy(PyArray_BYTES(nparr), &nnmat.at(0, 0), nnmat.nRows() * nnmat.nColumns() * sizeof(NumpyT));
    }
    else {
        for (u32 i = 0; i < nnmat.nRows(); ++i)
            for (u32 j = 0; j < nnmat.nColumns(); ++j) {
                NumpyT* v = (NumpyT*)PyArray_GETPTR2(nparr, i, j);
                *v        = (NumpyT)nnmat.at(i, j);
            }
    }
}

template<typename NumpyT, typename ContainerT>
void rawVec2numpy(PyArrayObject* nparr, const ContainerT& vec) {
    require_eq(PyArray_TYPE(nparr), NumpyType<NumpyT>::type());
    require_eq(PyArray_NDIM(nparr), 1);
    if (std::is_same<decltype(vec.at(0)), const NumpyT&>::value && PyArray_STRIDES(nparr)[0] == sizeof(NumpyT)) {
        // fast version. we can just use memcpy
        memcpy(PyArray_BYTES(nparr), &vec.at(0), vec.size() * sizeof(NumpyT));
    }
    else {
        for (u32 i = 0; i < vec.size(); ++i) {
            NumpyT* v = (NumpyT*)PyArray_GETPTR1(nparr, i);
            *v        = (NumpyT)vec.at(i);
        }
    }
}

template<typename FastContainerType>
bool arraySanityChecks_base(CriticalErrorFunc        criticalErrorFunc,
                            PyObject*                nparr,
                            const FastContainerType& nnContainer,
                            int                      ndim,
                            const char*              func,
                            int                      numtype = -1) {
    if (!PyArray_Check(nparr)) {
        criticalErrorFunc() << Core::form("%s: expected PyArrayObject but got %s", func, nparr->ob_type->tp_name);
        return false;
    }

    if (PyArray_NDIM((PyArrayObject*)nparr) != ndim) {
        criticalErrorFunc() << Core::form("%s: expect %iD array, got %i dims", func, ndim, PyArray_NDIM((PyArrayObject*)nparr));
        return false;
    }

    // Need this to get well-behaved access to the internal data below.
    // See docs for PyArray_GETPTR2().
    if (!PyArray_ISBEHAVED_RO((PyArrayObject*)nparr)) {
        criticalErrorFunc() << Core::form("%s: expect behaved/standard Numpy array", func);
        return false;
    }

    int npTypeNum = PyArray_TYPE((PyArrayObject*)nparr);
    if (numtype == -1) {
        switch (npTypeNum) {
            case NPY_FLOAT:
            case NPY_DOUBLE:
                // ok
                break;
            default:
                criticalErrorFunc() << Core::form("%s: expect float/double Numpy array. got type %i", func, npTypeNum);
                return false;
        }
    }
    else if (numtype != npTypeNum) {
        criticalErrorFunc() << Core::form("%s: expect type %i Numpy array. got type %i", func, numtype, npTypeNum);
        return false;
    }

    return true;
}

template<typename NnContainerType>
bool arraySanityChecks(CriticalErrorFunc      criticalErrorFunc,
                       PyObject*              nparr,
                       const NnContainerType& nnContainer,
                       int                    ndim,
                       const char*            func,
                       int                    numtype = -1) {
    require(!nnContainer.isComputing());

    return arraySanityChecks_base(criticalErrorFunc, nparr, nnContainer, ndim, func, numtype);
}

template<typename T>
bool numpy2nnMatrix(CriticalErrorFunc criticalErrorFunc, PyObject* nparr, Math::CudaMatrix<T>& nnmat) {
    if (!arraySanityChecks(criticalErrorFunc, nparr, nnmat, 2, "numpy2nnMatrix"))
        return false;

    size_t s1 = PyArray_DIM((PyArrayObject*)nparr, 0);
    size_t s2 = PyArray_DIM((PyArrayObject*)nparr, 1);
    nnmat.resize(s1, s2);

    Python::ObjRef _nparr;
    if (PyArray_TYPE((PyArrayObject*)nparr) != NumpyType<T>::type() ||
        PyArray_STRIDES((PyArrayObject*)nparr)[0] != sizeof(T) ||
        PyArray_STRIDES((PyArrayObject*)nparr)[1] != sizeof(T) * nnmat.nRows()) {
        // numpy2rawMat has a slow fallback which works in all cases.
        // However, we can speed it up by converting it to our prefered format which we can handle much faster.
        // So we ultimatively hope that Numpy can do that faster than our slow fallback would be.
        // Sprint has the matrix data in the format of a Fortran array.
        _nparr.takeOver(PyArray_FROM_OTF(nparr, NumpyType<T>::type(), NPY_ARRAY_FORCECAST | NPY_ARRAY_F_CONTIGUOUS));
    }
    if (!_nparr)
        _nparr.copyRef(nparr);
    // Just handle both cases without warning about casts.
    int npTypeNum = PyArray_TYPE((PyArrayObject*) _nparr.obj);
    if(npTypeNum == NPY_FLOAT)
        numpy2rawMat<float, Math::CudaMatrix<T> >((PyArrayObject*) _nparr.obj, nnmat);
    else if(npTypeNum == NPY_DOUBLE)
        numpy2rawMat<double, Math::CudaMatrix<T> >((PyArrayObject*) _nparr.obj, nnmat);
    else
        assert(false);

    return true;
}

template<typename T>
bool numpy2nnVector(CriticalErrorFunc criticalErrorFunc, PyObject* nparr, Math::CudaVector<T>& nnvec) {
    if (!arraySanityChecks(criticalErrorFunc, nparr, nnvec, 1, "numpy2nnVector"))
        return false;

    size_t s1 = PyArray_DIM((PyArrayObject*)nparr, 0);
    nnvec.resize(s1);

    // Just handle both cases without warning about casts.
    int npTypeNum = PyArray_TYPE((PyArrayObject*) nparr);
    if(npTypeNum == NPY_FLOAT)
        numpy2rawVec<float, Math::CudaVector<T> >((PyArrayObject*) nparr, nnvec);
    else if(npTypeNum == NPY_DOUBLE)
        numpy2rawVec<double, Math::CudaVector<T> >((PyArrayObject*) nparr, nnvec);
    else
        assert(false);

    return true;
}


template<typename T>
bool numpy2stdVec(CriticalErrorFunc criticalErrorFunc, PyObject* nparr, std::vector<T>& vec) {
    if(!arraySanityChecks_base(criticalErrorFunc, nparr, vec, 1, "numpy2nnVector"))
        return false;

    size_t s1 = PyArray_DIM((PyArrayObject*) nparr, 0);
    vec.resize(s1);

    // Just handle both cases without warning about casts.
    int npTypeNum = PyArray_TYPE((PyArrayObject*) nparr);
    if(npTypeNum == NPY_FLOAT)
        numpy2rawVec<float, std::vector<T> >((PyArrayObject*) nparr, vec);
    else if(npTypeNum == NPY_DOUBLE)
        numpy2rawVec<double, std::vector<T> >((PyArrayObject*) nparr, vec);
    else
        assert(false);

    return true;
}

template<typename T>
static PyObject* newNumpy2D(npy_intp* dims) {
    npy_intp strides[] = {sizeof(T), (npy_intp)sizeof(T) * dims[0]};
    int      npTypeNum = NumpyType<T>::type();
    return PyArray_New(&PyArray_Type, 2, dims, npTypeNum, strides, nullptr, 0, 0, nullptr);
}

template<typename T>
static PyObject* newNumpy1D(npy_intp* dims) {
    int npTypeNum = NumpyType<T>::type();
    return PyArray_SimpleNew(1, dims, npTypeNum);
}

template<typename T>
bool nnMatrix2numpy(CriticalErrorFunc criticalErrorFunc, PyObject*& nparr, const Math::CudaMatrix<T>& nnmat) {
    Py_CLEAR(nparr);

    npy_intp dims[] = {nnmat.nRows(), nnmat.nColumns()};
    nparr           = newNumpy2D<T>(dims);
    if (!nparr) {
        criticalErrorFunc() << "failed to create Numpy array";
        return false;
    }

    if (!arraySanityChecks(criticalErrorFunc, nparr, nnmat, 2, "nnMatrix2numpy", NumpyType<T>::type()))
        return false;

    rawMat2numpy<T, Math::CudaMatrix<T>>((PyArrayObject*)nparr, nnmat);

    return true;
}

template<typename T>
bool fastMatrix2numpy(CriticalErrorFunc criticalErrorFunc, PyObject*& nparr, const Math::FastMatrix<T>& fastmat) {
    Py_CLEAR(nparr);

    npy_intp dims[] = {fastmat.nRows(), fastmat.nColumns()};
    nparr           = newNumpy2D<T>(dims);
    if (!nparr) {
        criticalErrorFunc() << "failed to create Numpy array";
        return false;
    }

    if (!arraySanityChecks_base(criticalErrorFunc, nparr, fastmat, 2, "fastMatrix2numpy", NumpyType<T>::type()))
        return false;

    rawMat2numpy<T, Math::FastMatrix<T>>((PyArrayObject*)nparr, fastmat);

    return true;
}

template<typename T>
bool nnVec2numpy(CriticalErrorFunc criticalErrorFunc, PyObject*& nparr, const Math::CudaVector<T>& nnvec) {
    Py_CLEAR(nparr);

    npy_intp dims[] = {nnvec.nRows()};
    nparr           = newNumpy1D<T>(dims);
    if (!nparr) {
        criticalErrorFunc() << "failed to create Numpy array";
        return false;
    }

    if (!arraySanityChecks(criticalErrorFunc, nparr, nnvec, 1, "nnVec2numpy", NumpyType<T>::type()))
        return false;

    rawVec2numpy<T, Math::CudaVector<T>>((PyArrayObject*)nparr, nnvec);

    return true;
}

template<typename T>
bool stdVec2numpy(CriticalErrorFunc criticalErrorFunc, PyObject*& nparr, const std::vector<T>& stdvec) {
    Py_CLEAR(nparr);

    npy_intp dims[] = {(npy_intp)stdvec.size()};
    nparr           = newNumpy1D<T>(dims);
    if (!nparr) {
        criticalErrorFunc() << "failed to create Numpy array";
        return false;
    }

    if (!arraySanityChecks_base(criticalErrorFunc, nparr, stdvec, 1, "stdVec2numpy", NumpyType<T>::type()))
        return false;

    rawVec2numpy<T, std::vector<T>>((PyArrayObject*)nparr, stdvec);

    return true;
}

// explicit template instantiation
template bool numpy2nnMatrix<f32>(CriticalErrorFunc, PyObject*, Math::CudaMatrix<f32>&);
template bool numpy2nnMatrix<f64>(CriticalErrorFunc, PyObject*, Math::CudaMatrix<f64>&);
template bool numpy2nnVector<f32>(CriticalErrorFunc, PyObject*, Math::CudaVector<f32>&);
template bool numpy2nnVector<f64>(CriticalErrorFunc, PyObject*, Math::CudaVector<f64>&);
template bool numpy2nnVector<u32>(CriticalErrorFunc, PyObject*, Math::CudaVector<u32>&);
template bool numpy2stdVec<f32>(CriticalErrorFunc, PyObject*, std::vector<f32>&);
template bool numpy2stdVec<f64>(CriticalErrorFunc, PyObject*, std::vector<f64>&);
template bool numpy2stdVec<u32>(CriticalErrorFunc, PyObject*, std::vector<u32>&);
template bool nnMatrix2numpy<f32>(CriticalErrorFunc, PyObject*&, const Math::CudaMatrix<f32>&);
template bool nnMatrix2numpy<f64>(CriticalErrorFunc, PyObject*&, const Math::CudaMatrix<f64>&);
template bool fastMatrix2numpy<f32>(CriticalErrorFunc, PyObject*&, const Math::FastMatrix<f32>&);
template bool fastMatrix2numpy<f64>(CriticalErrorFunc, PyObject*&, const Math::FastMatrix<f64>&);
template bool nnVec2numpy<f32>(CriticalErrorFunc, PyObject*&, const Math::CudaVector<f32>&);
template bool nnVec2numpy<f64>(CriticalErrorFunc, PyObject*&, const Math::CudaVector<f64>&);
template bool nnVec2numpy<u32>(CriticalErrorFunc, PyObject*&, const Math::CudaVector<u32>&);
template bool stdVec2numpy<f32>(CriticalErrorFunc, PyObject*&, const std::vector<f32>&);
template bool stdVec2numpy<f64>(CriticalErrorFunc, PyObject*&, const std::vector<f64>&);
template bool stdVec2numpy<u32>(CriticalErrorFunc, PyObject*&, const std::vector<u32>&);

}  // namespace Python
