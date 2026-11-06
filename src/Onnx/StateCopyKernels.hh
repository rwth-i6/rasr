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
#ifndef _ONNX_STATE_COPY_KERNELS_HH
#define _ONNX_STATE_COPY_KERNELS_HH

#include <cstddef>

#include <cuda_runtime.h>

/*
 * CUDA kernels of the GPU state managers (see `StateCopier`). Plain interface without RASR headers, compiled by nvcc.
 */
namespace Onnx {
namespace CudaKernels {

// Device pointers to the state and to the time step start in the tensor
struct StateCopyTask {
    float const* src;
    float*       dst;
};

enum class StateCopyDirection {
    StateToTensor,  // merge: contiguous state -> strided tensor
    TensorToState,  // split: strided tensor -> contiguous state
};

/*
 * For each task copy `stepSize` floats between state and tensor. State element i is tensor element
 * `runOffsets[i / runLength] + i % runLength`. `tasks`/`runOffsets` are device pointers.
 */
cudaError_t launchStateCopy(StateCopyTask const* tasks,
                            size_t               numTasks,
                            size_t const*        runOffsets,
                            size_t               runLength,
                            size_t               stepSize,
                            StateCopyDirection   direction,
                            cudaStream_t         stream);

/*
 * Layout of the merged tensor of one state variable in `launchStateMerge`: the time step `t` of batch entry `b`
 * starts at `tensor + b * batchStride + t * timeStride` and consists of `stepSize / runLength` contiguous runs of
 * `runLength` floats at `runOffsets[runOffsetsStart + i]` (relative to the start of the time step).
 */
struct MergeLayout {
    float* tensor;
    size_t batchStride;
    size_t timeStride;
    size_t stepSize;
    size_t runLength;
    size_t runOffsetsStart;
};

/*
 * Merge the states of all state variables into their left-padded tensors in one launch, without a separate memset.
 * Batch entry `b` has `prefixLengths[b]` states, right-aligned to `maxPrefix`; the time steps before them are zeroed.
 * The state of variable `v`, batch entry `b`, prefix position `p` is `states[v * numStates + batchStarts[b] + p]`
 * (`numStates` = sum of the prefix lengths). All arrays are device pointers.
 * Requires `numBatch` and `numVars` <= 65535 (grid dimensions).
 */
cudaError_t launchStateMerge(float const* const* states,
                             size_t const*       prefixLengths,
                             size_t const*       batchStarts,
                             MergeLayout const*  layouts,
                             size_t const*       runOffsets,
                             size_t              numStates,
                             size_t              maxPrefix,
                             size_t              numBatch,
                             size_t              numVars,
                             cudaStream_t        stream);

}  // namespace CudaKernels
}  // namespace Onnx

#endif  // _ONNX_STATE_COPY_KERNELS_HH
