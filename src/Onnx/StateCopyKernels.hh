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

}  // namespace CudaKernels
}  // namespace Onnx

#endif  // _ONNX_STATE_COPY_KERNELS_HH
