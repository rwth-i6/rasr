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
#include <limits>

#include <cuda_runtime.h>

#include "Session.hh"

namespace Onnx {

namespace {

cudaStream_t sharedStream() {
    return static_cast<cudaStream_t>(Session::sharedCudaStream());
}

}  // namespace

void StateCopier::copy(std::vector<Batch> const& batches, CudaKernels::BlockCopyDirection direction) {
    // Tasks of all batches first, then their block offsets
    size_t numTasks   = 0ul;
    size_t numOffsets = 0ul;
    for (auto const& batch : batches) {
        numTasks += batch.tasks.size();
        numOffsets += batch.blockOffsets.size();
    }
    if (numTasks == 0ul) {
        return;
    }
    size_t const      taskBytes = numTasks * sizeof(CudaKernels::BlockCopyTask);
    std::vector<char> hostTasks(taskBytes + numOffsets * sizeof(size_t));
    char*             taskPos   = hostTasks.data();
    char*             offsetPos = hostTasks.data() + taskBytes;
    for (auto const& batch : batches) {
        std::memcpy(taskPos, batch.tasks.data(), batch.tasks.size() * sizeof(CudaKernels::BlockCopyTask));
        taskPos += batch.tasks.size() * sizeof(CudaKernels::BlockCopyTask);
        std::memcpy(offsetPos, batch.blockOffsets.data(), batch.blockOffsets.size() * sizeof(size_t));
        offsetPos += batch.blockOffsets.size() * sizeof(size_t);
    }

    char* deviceTasks = static_cast<char*>(taskBuffer_.allocate(hostTasks.size()));
    // hostTasks is pageable memory, so cudaMemcpyAsync returns only after it has been copied to a staging buffer
    // it may be freed at the end of this function although the copy to the device can still be pending
    checkCudaStatus(cudaMemcpyAsync(deviceTasks, hostTasks.data(), hostTasks.size(), cudaMemcpyHostToDevice, sharedStream()), "Uploading copy tasks");

    auto const* deviceTaskArray   = reinterpret_cast<CudaKernels::BlockCopyTask const*>(deviceTasks);
    auto const* deviceOffsetArray = reinterpret_cast<size_t const*>(deviceTasks + taskBytes);
    for (auto const& batch : batches) {
        checkCudaStatus(CudaKernels::launchBlockCopy(deviceTaskArray, batch.tasks.size(), deviceOffsetArray, batch.blockSize, batch.stepSize, direction, sharedStream()),
                        "Copying states on the device");
        deviceTaskArray += batch.tasks.size();
        deviceOffsetArray += batch.blockOffsets.size();
    }
}

std::vector<float const*> StateCopier::deviceData(std::vector<Value> const& tensors) {
    std::vector<size_t> numElements(tensors.size());
    std::vector<size_t> stagingOffsets(tensors.size(), std::numeric_limits<size_t>::max());
    size_t              stagingFloats = 0ul;
    for (size_t i = 0ul; i < tensors.size(); ++i) {
        numElements[i] = 1ul;
        for (int d = 0; d < tensors[i].numDims(); ++d) {
            numElements[i] *= tensors[i].dimSize(d);
        }
        if (not tensors[i].isOnDevice()) {
            stagingOffsets[i] = stagingFloats;
            stagingFloats += alignDeviceFloats(numElements[i]);
        }
    }

    std::vector<float const*> result(tensors.size());
    float*                    staging = stagingFloats > 0ul ? static_cast<float*>(stagingBuffer_.allocate(stagingFloats * sizeof(float))) : nullptr;
    for (size_t i = 0ul; i < tensors.size(); ++i) {
        result[i] = tensors[i].rawData<float>();
        if (stagingOffsets[i] != std::numeric_limits<size_t>::max()) {
            checkCudaStatus(cudaMemcpyAsync(staging + stagingOffsets[i], result[i], numElements[i] * sizeof(float), cudaMemcpyHostToDevice, sharedStream()),
                            "Uploading states");
            result[i] = staging + stagingOffsets[i];
        }
    }
    return result;
}

}  // namespace Onnx
