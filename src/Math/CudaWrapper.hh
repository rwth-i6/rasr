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
#ifndef CUDAWRAPPER_HH_
#define CUDAWRAPPER_HH_

#include <Core/Application.hh>
#include <Core/Component.hh>
#ifndef CMAKE_DISABLE_MODULES_HH
#include <Modules.hh>
#endif

#ifdef MODULE_CUDA
#include <cuda_runtime.h>
#include <curand.h>
#endif

/*
 * wrapper for CUDA routines
 */

namespace Math {

#ifndef MODULE_CUDA
struct curandGenerator_t {
    int dummyGenerator;
};
typedef int   curandRngType_t;
typedef void* cudaStream_t;
#define CURAND_RNG_PSEUDO_DEFAULT 0

enum cudaError_t {
    cudaSuccess = 0
};
enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS = 0
};
enum curandStatus_t {
    CURAND_STATUS_SUCCESS = 0,
};
#endif

namespace Cuda {

inline cudaError_t getNumberOfGpus(int& count, bool& hasCuda) {
    cudaError_t success = cudaError_t();
    hasCuda             = false;
    count               = 0;
#ifdef MODULE_CUDA
    success = cudaGetDeviceCount(&count);
    hasCuda = true;
#endif
    return success;
}

inline curandStatus_t createRandomNumberGenerator(curandGenerator_t& generator, curandRngType_t rng_type) {
#ifdef MODULE_CUDA
    return curandCreateGenerator(&generator, rng_type);
#else
    Core::Application::us()->criticalError("Calling gpu method 'createRandomNumberGenerator' in binary without gpu support!");
    return curandStatus_t(1);
#endif
}

inline curandStatus_t setSeed(curandGenerator_t& generator, unsigned long long seed) {
#ifdef MODULE_CUDA
    return curandSetPseudoRandomGeneratorSeed(generator, seed);
#else
    Core::Application::us()->criticalError("Calling gpu method 'setSeed' in binary without gpu support!");
    return curandStatus_t(1);
#endif
}

inline unsigned int deviceSync(bool hasGpu = true) {
    int result = 0;
#ifdef MODULE_CUDA
    result = hasGpu ? cudaDeviceSynchronize() : true;
#endif
    return result;
}

inline unsigned int deviceReset(bool hasGpu = true) {
    int result = 0;
#ifdef MODULE_CUDA
    result = hasGpu ? cudaDeviceReset() : true;
#endif
    return result;
}

inline void printError(cudaError_t err) {
#ifdef MODULE_CUDA
    std::cout << "Error:\t" << cudaGetErrorString(err) << std::endl;
#else
    std::cout << "Error:\t" << err << std::endl;
#endif
}

// for cublasStatus_t use Cuda::cublasGetErrorString()
inline const char* getErrorString(cudaError_t err) {
#ifdef MODULE_CUDA
    return cudaGetErrorString(err);
#else
    char* buf = new char[512];
    sprintf(buf, "Could not convert error code '%d' to string w/o MODULE_CUDA enabled.", err);
    return buf;
#endif
}

inline const char* curandGetErrorString(curandStatus_t status) {
#ifdef MODULE_CUDA
    switch (status) {
        case CURAND_STATUS_SUCCESS:
            return "No errors.";
        case CURAND_STATUS_VERSION_MISMATCH:
            return "Header file and linked library version do not match.";
        case CURAND_STATUS_NOT_INITIALIZED:
            return "Generator not initialized.";
        case CURAND_STATUS_ALLOCATION_FAILED:
            return "Memory allocation failed.";
        case CURAND_STATUS_TYPE_ERROR:
            return "Generator is wrong type.";
        case CURAND_STATUS_OUT_OF_RANGE:
            return "Argument out of range.";
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "Length requested is not a multple of dimension.";
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "GPU does not have double precision required by MRG32k3a.";
        case CURAND_STATUS_LAUNCH_FAILURE:
            return "Kernel launch failure.";
        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "Preexisting failure on library entry.";
        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "Initialization of CUDA failed.";
        case CURAND_STATUS_ARCH_MISMATCH:
            return "Architecture mismatch, GPU does not support requested feature.";
        case CURAND_STATUS_INTERNAL_ERROR:
            return "Internal library error.";
        default:
            return "Unknown Curand error";
    }
#else
    return "Curand error unknown with MODULE_CUDA disabled.";
#endif
}

inline cudaError_t getMemoryInfo(size_t* free, size_t* total) {
#ifdef MODULE_CUDA
    return cudaMemGetInfo(free, total);
#else
    Core::Application::us()->criticalError("Calling gpu method 'gpuGetMemoryInfo' in binary without gpu support!");
    return cudaError_t(1);
#endif
}

template<typename T>
cudaError_t alloc(T*& devPtr, size_t nElements) {
#ifdef MODULE_CUDA
    return cudaMalloc((void**)&devPtr, nElements * sizeof(T));
#else
    Core::Application::us()->criticalError("Calling gpu method 'gpuAlloc' in binary without gpu support!");
    return cudaError_t(1);
#endif
}

template<typename T>
cudaError_t free(T* devPtr) {
#ifdef MODULE_CUDA
    return cudaFree((void*)devPtr);
#else
    Core::Application::us()->criticalError("Calling gpu method 'gpuFree' in binary without gpu support!");
    return cudaError_t(1);
#endif
}

template<typename T>
cudaError_t copyFromGpu(T* dst, const T* src, size_t nElements) {
#ifdef MODULE_CUDA
    return cudaMemcpy(dst, src, nElements * sizeof(T), cudaMemcpyDeviceToHost);
#else
    Core::Application::us()->criticalError("Calling gpu method 'cppyFromGpu' in binary without gpu support!");
    return cudaError_t(1);
#endif
}

template<typename T>
cudaError_t copyToGpu(T* dst, const T* src, size_t nElements) {
#ifdef MODULE_CUDA
    return cudaMemcpy(dst, src, nElements * sizeof(T), cudaMemcpyHostToDevice);
#else
    Core::Application::us()->criticalError("Calling gpu method 'copyToGpu' in binary without gpu support!");
    return cudaError_t(1);
#endif
}

template<typename T>
cudaError_t memcpy(T* dst, const T* src, size_t nElements) {
#ifdef MODULE_CUDA
    return cudaMemcpy(dst, src, nElements * sizeof(T), cudaMemcpyDeviceToDevice);
#else
    Core::Application::us()->criticalError("Calling gpu method 'memcpy' in binary without gpu support!");
    return cudaError_t(1);
#endif
}

template<typename T>
int memSet(T* devPtr, int value, size_t count, cudaStream_t stream = 0) {
    int result = 0;
#ifdef MODULE_CUDA
    if (stream) {
        result = cudaMemsetAsync(devPtr, value, count * sizeof(T), stream);
    }
    else {
        result = cudaMemset(devPtr, value, count * sizeof(T));
    }
#else
    Core::Application::us()->criticalError("Calling gpu method 'memSet' in binary without gpu support!");
#endif
    return result;
}

template<typename T>
inline int generateUniform(curandGenerator_t& generator, T* outputPtr, size_t num);

template<>
inline int generateUniform(curandGenerator_t& generator, float* outputPtr, size_t num) {
    int result = 0;
#ifdef MODULE_CUDA
    result = curandGenerateUniform(generator, outputPtr, num);
#else
    Core::Application::us()->criticalError("Calling gpu method 'generateUniform' in binary without gpu support!");
#endif
    return result;
}

template<>
inline int generateUniform(curandGenerator_t& generator, double* outputPtr, size_t num) {
    int result = 0;
#ifdef MODULE_CUDA
    result = curandGenerateUniformDouble(generator, outputPtr, num);
#else
    Core::Application::us()->criticalError("Calling gpu method 'generateUniform' in binary without gpu support!");
#endif
    return result;
}

template<typename T>
inline int generateNormal(curandGenerator_t& generator, T* outputPtr, size_t num, T mean, T stddev);

template<>
inline int generateNormal(curandGenerator_t& generator, float* outputPtr, size_t num, float mean, float stddev) {
    int result = 0;
#ifdef MODULE_CUDA
    result = curandGenerateNormal(generator, outputPtr, num, mean, stddev);
#else
    Core::Application::us()->criticalError("Calling gpu method 'generateNormal' in binary without gpu support!");
#endif
    return result;
}

template<>
inline int generateNormal(curandGenerator_t& generator, double* outputPtr, size_t num, double mean, double stddev) {
    int result = 0;
#ifdef MODULE_CUDA
    result = curandGenerateNormalDouble(generator, outputPtr, num, mean, stddev);
#else
    Core::Application::us()->criticalError("Calling gpu method 'generateNormal' in binary without gpu support!");
#endif
    return result;
}

inline cudaError_t getLastError() {
#ifdef MODULE_CUDA
    return cudaGetLastError();
#else
    return cudaError_t(0);
#endif
}

inline void checkForLastError() {
#ifdef MODULE_CUDA
    cudaError_t error = Cuda::getLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
#endif
}

template<typename T>
inline cudaError_t hostRegister(T** ptr, size_t bytes) {
#ifdef MODULE_CUDA
    return cudaHostRegister(ptr, bytes, cudaHostRegisterPortable);
#else
    return cudaError_t(0);
#endif
}

inline cudaStream_t* streamCreate() {
#ifdef MODULE_CUDA
    cudaStream_t* str = new cudaStream_t;
    cudaStreamCreate(str);
    return str;
#else
    return NULL;
#endif
}

}  // namespace Cuda

}  // namespace Math

#ifdef MODULE_CUDA
#define TIMER_GPU_STOP(startTime, endTime, condition, sum)                 \
    Math::Cuda::deviceSync(condition&& Math::CudaDataStructure::hasGpu()); \
    gettimeofday(&endTime, NULL);                                          \
    sum += Core::timeDiff(startTime, endTime);
#else
#define TIMER_GPU_STOP(startTime, endTime, condition, sum) \
    gettimeofday(&endTime, NULL);                          \
    sum += Core::timeDiff(startTime, endTime);
#endif

#ifdef MODULE_CUDA
#define TIMER_GPU_STOP_SUM2(startTime, endTime, condition, sum1, sum2)     \
    Math::Cuda::deviceSync(condition&& Math::CudaDataStructure::hasGpu()); \
    gettimeofday(&endTime, NULL);                                          \
    sum1 += Core::timeDiff(startTime, endTime);                            \
    sum2 += Core::timeDiff(startTime, endTime);
#else
#define TIMER_GPU_STOP_SUM2(startTime, endTime, condition, sum1, sum2) \
    gettimeofday(&endTime, NULL);                                      \
    sum1 += Core::timeDiff(startTime, endTime);                        \
    sum2 += Core::timeDiff(startTime, endTime);
#endif

#endif /* CUDAWRAPPER_HH_ */
