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
#include <algorithm>

#include <cuda_runtime.h>

#include "TransformerGpuKernels.hh"

namespace Onnx {
namespace CudaKernels {

namespace {

// One CUDA block per task, its threads copy the elements of the time step
__global__ void blockCopyKernel(BlockCopyTask const* tasks,
                                size_t const*        blockOffsets,
                                size_t               blockSize,
                                size_t               stepSize,
                                bool                 toTensor) {
    BlockCopyTask const task = tasks[blockIdx.x];
    for (size_t i = threadIdx.x; i < stepSize; i += blockDim.x) {
        size_t const tensorIdx = blockOffsets[i / blockSize] + i % blockSize;
        if (toTensor) {
            task.dst[tensorIdx] = task.src[i];
        }
        else {
            task.dst[i] = task.src[tensorIdx];
        }
    }
}

constexpr unsigned kThreadsPerBlock = 256u;
constexpr size_t   kMaxGridSize     = 1ul << 30;  // well below the limit of gridDim.x (2^31 - 1)

}  // namespace

int launchBlockCopy(BlockCopyTask const* tasks,
                    size_t               numTasks,
                    size_t const*        blockOffsets,
                    size_t               blockSize,
                    size_t               stepSize,
                    BlockCopyDirection   direction,
                    void*                stream) {
    if (numTasks == 0ul or stepSize == 0ul) {
        return cudaSuccess;
    }
    bool const toTensor = direction == BlockCopyDirection::BlockToTensor;
    for (size_t start = 0ul; start < numTasks; start += kMaxGridSize) {
        unsigned const gridSize = static_cast<unsigned>(std::min(kMaxGridSize, numTasks - start));
        blockCopyKernel<<<gridSize, kThreadsPerBlock, 0, static_cast<cudaStream_t>(stream)>>>(
                tasks + start, blockOffsets, blockSize, stepSize, toTensor);
    }
    return cudaGetLastError();
}

}  // namespace CudaKernels
}  // namespace Onnx
