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
#ifndef _ONNX_TRANSFORMER_GPU_KERNELS_HH
#define _ONNX_TRANSFORMER_GPU_KERNELS_HH

#include <cstddef>

/*
 * CUDA kernels of the `transformer-gpu` state manager. Plain interface without RASR headers, compiled by nvcc.
 */
namespace Onnx {
namespace CudaKernels {

// One time step of one state variable: device pointers to the block and to the time step start in the tensor
struct BlockCopyTask {
    float const* src;
    float*       dst;
};

enum class BlockCopyDirection {
    BlockToTensor,  // merge: contiguous block -> strided tensor
    TensorToBlock,  // split: strided tensor -> contiguous block
};

/*
 * For each task copy `stepSize` floats between block and tensor. Block element i is tensor element
 * `blockOffsets[i / blockSize] + i % blockSize`. `tasks`/`blockOffsets` are device pointers, returns a cudaError_t.
 */
int launchBlockCopy(BlockCopyTask const* tasks,
                    size_t               numTasks,
                    size_t const*        blockOffsets,
                    size_t               blockSize,
                    size_t               stepSize,
                    BlockCopyDirection   direction,
                    void*                stream);

}  // namespace CudaKernels
}  // namespace Onnx

#endif  // _ONNX_TRANSFORMER_GPU_KERNELS_HH
