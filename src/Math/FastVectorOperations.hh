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
#ifndef FAST_VECTOR_OPERATIONS_HH_
#define FAST_VECTOR_OPERATIONS_HH_

#ifndef CMAKE_DISABLE_MODULE_HH
#include <Modules.hh>
#endif
#include <Core/OpenMPWrapper.hh>
#include <Math/MultithreadingHelper.hh>
#include <cmath>
#include <functional>

#ifdef MODULE_ACML
#include <acml_mv.h>
#endif

namespace Math {

// TODO add multithreading for all methods

/*
 *  y = exp(x) (componentwise)
 *
 */

template<typename T>
inline void vr_exp(int n, T* x, T* y) {
    for (int i = 0; i < n; i++) {
        y[i] = exp(x[i]);
    }
}

#ifdef MODULE_ACML

template<>
inline void vr_exp(int n, float* x, float* y) {
    vrsa_expf(n, x, y);
}

template<>
inline void vr_exp(int n, double* x, double* y) {
    vrda_exp(n, x, y);
}

#endif

template<typename T>
inline void mt_vr_exp(int n, T* x, T* y, int nThreads) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] = exp(x[i]);
    }
}

#ifdef MODULE_ACML

template<>
inline void mt_vr_exp(int n, float* x, float* y, int nThreads) {
    mt_v2v(n, x, y, vrsa_expf, nThreads);
}

template<>
inline void mt_vr_exp(int n, double* x, double* y, int nThreads) {
    mt_v2v(n, x, y, vrda_exp, nThreads);
}

#endif

// TODO add Intel MKL

/*
 *  y = log(x) (componentwise)
 */

template<typename T>
inline void vr_log(int n, T* x, T* y) {
    for (int i = 0; i < n; i++) {
        y[i] = log(x[i]);
    }
}

#ifdef MODULE_ACML

template<>
inline void vr_log(int n, float* x, float* y) {
    vrsa_logf(n, x, y);
}

template<>
inline void vr_log(int n, double* x, double* y) {
    vrda_log(n, x, y);
}

#endif

/*
 *  z = x**y (componentwise)
 */

template<typename T>
inline void vr_powx(int n, T* x, T y, T* z) {
    for (int i = 0; i < n; i++) {
        z[i] = pow(x[i], y);
    }
}

#ifdef MODULE_ACML

template<>
inline void vr_powx(int n, float* x, float y, float* z) {
    vrsa_powxf(n, x, y, z);
}

template<>
inline void vr_powx(int n, double* x, double y, double* z) {
    for (int i = 0; i < n; i++) {
        z[i] = fastpow(x[i], y);
    }
}

#endif

}  // namespace Math

#endif
