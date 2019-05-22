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
#include "CudaDataStructure.hh"
#include <Core/Application.hh>
#include <Math/Random.hh>
#include <string>

using namespace Math;

bool              CudaDataStructure::initialized = false;
bool              CudaDataStructure::hasGpu_     = false;
int               CudaDataStructure::activeGpu_  = -1;
cublasHandle_t    CudaDataStructure::cublasHandle;
curandGenerator_t CudaDataStructure::randomNumberGenerator;
u32               CudaDataStructure::multiPrecisionBunchSize = 8;

enum UseCudaMode {
    useCuda,
    dontUseCuda,
    autoUseCuda
};

static const Core::Choice choiceUseCuda(
        "true", useCuda,
        "false", dontUseCuda,
        "auto", autoUseCuda,
        Core::Choice::endMark());

static const Core::ParameterChoice paramUseCuda(
        "use-cuda", &choiceUseCuda,
        "Specify whether you want to force usage of CUDA, "
        "or just automatically use it if available.",
        autoUseCuda);

void CudaDataStructure::initialize() {
    if (initialized)
        return;
    initialized = true;

    UseCudaMode useCudaMode = autoUseCuda;
    if (!Core::Application::us())
        warning("CudaDataStructure: no application, use CUDA if available");
    else
        useCudaMode = (UseCudaMode)paramUseCuda(Core::Application::us()->getConfiguration());

    // check whether MODULE_CUDA is active and a GPU is available
    int  nGpus   = 0;
    bool hasCuda = false;
    if (useCudaMode == dontUseCuda) {
        hasGpu_ = false;
        log("CUDA is disabled via config");
    }
    else {
        cudaError_t success = Cuda::getNumberOfGpus(nGpus, hasCuda);
        if (!hasCuda)
            hasGpu_ = false;
        else if (success != cudaSuccess || nGpus <= 0) {
            // no GPU available, or some error occured
            hasGpu_ = false;
            std::ostringstream ss;
            ss << "Using binary with GPU support, but no GPU available.";
            if (success != cudaSuccess) {
                ss << " Error code is: " << success
                   << " (" << Cuda::getErrorString(success) << ")";
            }
            if (nGpus != 0) {
                // this should never occur (when no GPU is available, a non-zero error code is returned)
                ss << " Strange, number of GPUs is: " << nGpus;
            }
            if (useCudaMode == autoUseCuda) {
                log(ss.str());
            }
            else {
                ss << " This is critical with use-cuda = true.";
                criticalError(ss.str());
            }
        }
        else {
            // We have some GPUs on this system.
            hasGpu_ = true;
        }
    }

    if (hasGpu_) {
#ifdef MODULE_CUDA
        // find free device first
        {
            cudaError_t        status = cudaErrorInvalidDevice;
            std::ostringstream ss;

            for (int d = 0; d < nGpus; ++d) {
                ss.str("");
                ss.clear();
                ss << "Trying to cudaSetDevice on GPU " << d;
                log(ss.str());
                cudaSetDevice(d);      // this sometimes returns 0 on occupied GPUs
                status = cudaFree(0);  // this always fails on occupied GPUs
                if (status == cudaSuccess)
                    break;
            }
            if (status != cudaSuccess)
                criticalError("Failed to get a GPU handle.");
        }
#endif
        // initialize cuBLAS and cuRAND
        {
            cublasStatus_t success = Cuda::createCublasHandle(cublasHandle);
            if (success != CUBLAS_STATUS_SUCCESS) {
                std::ostringstream serr;
                serr << "Failed to initialize cuBLAS library: Error code is: " << success;
                serr << " (" << Cuda::cublasGetErrorString(success) << ")";
                criticalError(serr.str());
            }
        }
        {
            curandStatus_t success = Cuda::createRandomNumberGenerator(randomNumberGenerator, CURAND_RNG_PSEUDO_DEFAULT);
            if (success != CURAND_STATUS_SUCCESS) {
                std::ostringstream serr;
                serr << "Failed to initialize cuRAND random number generator library: Error code is: " << success;
                serr << " (" << Cuda::curandGetErrorString(success) << ")";
                criticalError(serr.str());
            }
        }
        {
            curandStatus_t success = Cuda::setSeed(randomNumberGenerator, (unsigned long long)Math::rand());
            if (success != CURAND_STATUS_SUCCESS) {
                std::ostringstream serr;
                serr << "Failed to set seed for cuRAND random number generator: Error code is: " << success;
                serr << " (" << Cuda::curandGetErrorString(success) << ")";
                criticalError(serr.str());
            }
        }

#ifdef MODULE_CUDA
        // Get the current active GPU which we are going to use.
        // Do this after CuBLAS init because otherwise it might return the wrong one.
        // See related comment in Theano CUDA init code:
        //   https://github.com/Theano/Theano/blob/master/theano/sandbox/cuda/__init__.py
        cudaError_t ret = cudaGetDevice(&activeGpu_);
        if (ret != cudaSuccess) {
            criticalError(std::string() + "Cannot query current active GPU, error " + std::to_string(ret) + " (" + Cuda::getErrorString(ret) + ")");
        }
        else {
            log(std::string() + "Using GPU " + std::to_string(activeGpu_ + 1) + " (= idx + 1) of " + std::to_string(nGpus) + " GPUs");
        }
#endif
    }
}

bool CudaDataStructure::hasGpu() {
    if (!initialized)
        initialize();
    return hasGpu_;
}

int CudaDataStructure::getActiveGpu() {
    require(hasGpu_);
    return activeGpu_;
}

void CudaDataStructure::setMultiprecisionBunchSize(u32 val) {
    verify_ge(val, 1);
    multiPrecisionBunchSize = val;
}

CudaDataStructure::CudaDataStructure()
        : gpuMode_(hasGpu()) {}

CudaDataStructure::CudaDataStructure(const CudaDataStructure& x)
        : gpuMode_(x.gpuMode_) {}

void CudaDataStructure::log(const std::string& msg) {
    if (Core::Application::us())
        Core::Application::us()->log() << msg;
    else
        std::cerr << msg << std::endl;
}

void CudaDataStructure::warning(const std::string& msg) {
    if (Core::Application::us())
        Core::Application::us()->warning() << msg;
    else
        std::cerr << msg << std::endl;
}

void CudaDataStructure::error(const std::string& msg) {
    if (Core::Application::us())
        Core::Application::us()->error() << msg;
    else
        std::cerr << msg << std::endl;
}

void CudaDataStructure::criticalError(const std::string& msg) {
    if (Core::Application::us())
        Core::Application::us()->criticalError() << msg;
    else
        std::cerr << msg << std::endl;
}
