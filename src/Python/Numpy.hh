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
#ifndef _PYTHON_NUMPY_HH
#define _PYTHON_NUMPY_HH

#include <vector>

#include <Core/Component.hh>
#include <Core/StringUtilities.hh>
#include <Core/Types.hh>
#include <Math/CudaMatrix.hh>
#include <Math/CudaVector.hh>
#ifndef CMAKE_DISABLE_MODULES_HH
#include <Modules.hh>
#endif
#include <Python/Utilities.hh>
#include <Python.h>

namespace Python {

// All these expect to have the GIL.

void initNumpy();

bool isNumpyArrayType(PyObject* obj);
bool isNumpyArrayTypeExact(PyObject* obj);

template<typename T>
bool numpy2nnMatrix(CriticalErrorFunc criticalErrorFunc, PyObject* nparr, Math::CudaMatrix<T>& nnmat);

template<typename T>
bool numpy2nnVector(CriticalErrorFunc criticalErrorFunc, PyObject* nparr, Math::CudaVector<T>& nnvec);

template<typename T>
bool numpy2stdVec(CriticalErrorFunc criticalErrorFunc, PyObject* nparr, std::vector<T>& vec);

template<typename T>
bool nnMatrix2numpy(CriticalErrorFunc criticalErrorFunc, PyObject*& nparr, const Math::CudaMatrix<T>& nnmat);

template<typename T>
bool fastMatrix2numpy(CriticalErrorFunc criticalErrorFunc, PyObject*& nparr, const Math::FastMatrix<T>& fastmat);

template<typename T>
bool nnVec2numpy(CriticalErrorFunc criticalErrorFunc, PyObject*& nparr, const Math::CudaVector<T>& nnvec);

template<typename T>
bool stdVec2numpy(CriticalErrorFunc criticalErrorFunc, PyObject*& nparr, const std::vector<T>& stdvec);

}  // namespace Python

#endif  // NUMPY_HH
