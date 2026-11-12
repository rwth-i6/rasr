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
#include "StateCopier.hh"

#include <cstring>

#include <cuda_runtime.h>

namespace Onnx {

void StateCopier::copy(std::vector<Batch> const& batches, CudaKernels::StateCopyDirection direction) {
    // Tasks of all batches first, then their run offsets
    size_t numTasks   = 0ul;
    size_t numOffsets = 0ul;
    for (auto const& batch : batches) {
        numTasks += batch.tasks.size();
        numOffsets += batch.runOffsets.size();
    }
    if (numTasks == 0ul) {
        return;
    }
    size_t const      taskBytes = numTasks * sizeof(CudaKernels::StateCopyTask);
    std::vector<char> hostTasks(taskBytes + numOffsets * sizeof(size_t));
    char*             taskPos   = hostTasks.data();
    char*             offsetPos = hostTasks.data() + taskBytes;
    for (auto const& batch : batches) {
        std::memcpy(taskPos, batch.tasks.data(), batch.tasks.size() * sizeof(CudaKernels::StateCopyTask));
        taskPos += batch.tasks.size() * sizeof(CudaKernels::StateCopyTask);
        std::memcpy(offsetPos, batch.runOffsets.data(), batch.runOffsets.size() * sizeof(size_t));
        offsetPos += batch.runOffsets.size() * sizeof(size_t);
    }

    char* deviceTasks = static_cast<char*>(taskBuffer_.allocate(hostTasks.size()));
    // hostTasks is pageable memory, so cudaMemcpyAsync returns only after it has been copied to a staging buffer
    // it may be freed at the end of this function although the copy to the device can still be pending
    checkCudaStatus(cudaMemcpyAsync(deviceTasks, hostTasks.data(), hostTasks.size(), cudaMemcpyHostToDevice, sharedStream()), "Uploading copy tasks");

    auto const* deviceTaskArray   = reinterpret_cast<CudaKernels::StateCopyTask const*>(deviceTasks);
    auto const* deviceOffsetArray = reinterpret_cast<size_t const*>(deviceTasks + taskBytes);
    for (auto const& batch : batches) {
        checkCudaStatus(CudaKernels::launchStateCopy(deviceTaskArray, batch.tasks.size(), deviceOffsetArray, batch.runLength, batch.stepSize, direction, sharedStream()),
                        "Copying states on the device");
        deviceTaskArray += batch.tasks.size();
        deviceOffsetArray += batch.runOffsets.size();
    }
}

}  // namespace Onnx
