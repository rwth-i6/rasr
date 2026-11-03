/** Copyright 2026 RWTH Aachen University. All rights reserved.
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
#include <cuda_runtime.h>

#include "StateCopyKernels.hh"

namespace Onnx {
namespace CudaKernels {

namespace {

// One CUDA block per task, its threads copy the elements of the time step
__global__ void stateCopyKernel(StateCopyTask const* tasks,
                                size_t const*        runOffsets,
                                size_t               runLength,
                                size_t               stepSize,
                                bool                 toTensor) {
    StateCopyTask const task = tasks[blockIdx.x];
    for (size_t i = threadIdx.x; i < stepSize; i += blockDim.x) {
        size_t const tensorIdx = runOffsets[i / runLength] + i % runLength;
        if (toTensor) {
            task.dst[tensorIdx] = task.src[i];
        }
        else {
            task.dst[i] = task.src[tensorIdx];
        }
    }
}

constexpr unsigned threadsPerBlock = 256u;

}  // namespace

cudaError_t launchStateCopy(StateCopyTask const* tasks,
                            size_t               numTasks,
                            size_t const*        runOffsets,
                            size_t               runLength,
                            size_t               stepSize,
                            StateCopyDirection   direction,
                            cudaStream_t         stream) {
    if (numTasks == 0ul or stepSize == 0ul) {
        return cudaSuccess;
    }
    bool const toTensor = direction == StateCopyDirection::StateToTensor;
    stateCopyKernel<<<static_cast<unsigned>(numTasks), threadsPerBlock, 0, stream>>>(tasks, runOffsets, runLength, stepSize, toTensor);
    return cudaGetLastError();
}

}  // namespace CudaKernels
}  // namespace Onnx
