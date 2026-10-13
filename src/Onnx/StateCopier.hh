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
#ifndef _ONNX_STATE_COPIER_HH
#define _ONNX_STATE_COPIER_HH

#include <vector>

#include "DeviceVector.hh"
#include "StateCopyKernels.hh"
#include "Value.hh"

namespace Onnx {

/*
 * Copies states between `DeviceVector`s and the (merged) state tensors of a model on the GPU, for the GPU state managers.
 * Everything runs asynchronously on `Session::sharedCudaStream()`.
 */
class StateCopier {
public:
    // Copies of one state variable
    // All states have the same layout within the tensor
    struct Batch {
        std::vector<CudaKernels::BlockCopyTask> tasks;
        std::vector<size_t>                     blockOffsets;     // contiguous runs of one state in the tensor, relative to its start
        size_t                                  blockSize = 0ul;  // elements per run
        size_t                                  stepSize  = 0ul;  // elements per state
    };

    // Upload the tasks of all batches at once and launch the copies, one launch per batch
    void copy(std::vector<Batch> const& batches, CudaKernels::BlockCopyDirection direction);

    // Device pointers to the data of the float `tensors`, tensors in host memory are uploaded into a staging buffer,
    // valid until the next call
    std::vector<float const*> deviceData(std::vector<Value> const& tensors);

private:
    DeviceBuffer taskBuffer_;
    DeviceBuffer stagingBuffer_;
};

}  // namespace Onnx

#endif  // _ONNX_STATE_COPIER_HH
