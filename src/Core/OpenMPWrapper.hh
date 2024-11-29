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
#ifndef OPENMP_WRAPPER_HH_
#define OPENMP_WRAPPER_HH_

#include <Core/Types.hh>
#ifndef CMAKE_DISABLE_MODULES_HH
#include <Modules.hh>
#endif

#ifdef MODULE_OPENMP
#include <omp.h>
#endif

namespace Core {

namespace omp {

inline s32 get_max_threads() {
#ifdef MODULE_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

inline s32 get_num_threads() {
#ifdef MODULE_OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}

inline void set_num_threads(int nThreads) {
#ifdef MODULE_OPENMP
    omp_set_num_threads(nThreads);
#endif
}

}  // namespace omp

}  // namespace Core

#endif
