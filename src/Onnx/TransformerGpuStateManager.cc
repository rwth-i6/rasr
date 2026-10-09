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
#include "TransformerGpuStateManager.hh"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <valarray>

#include <cuda_runtime.h>

#include <Core/Application.hh>

#include "Session.hh"
#include "TransformerGpuKernels.hh"

namespace Onnx {

namespace {

// Alignment of the merged tensors within the merged buffer (in floats, i.e. 256 bytes)
constexpr size_t kAlignFloats = 64ul;

size_t alignUp(size_t numFloats) {
    return (numFloats + kAlignFloats - 1ul) / kAlignFloats * kAlignFloats;
}

// Layout of one time step in a tensor with shape `dims` (batch first), like in `Nn::TransformerStateManager`
struct StepLayout {
    std::vector<int64_t> dims;
    size_t               batchStride = 1ul;  // elements per batch entry
    size_t               timeStride  = 0ul;  // elements per time step
    std::vector<size_t>  blockOffsets;       // contiguous runs of one time step, relative to its start
    size_t               blockSize = 0ul;    // elements per run
    size_t               stepSize  = 0ul;    // elements per time step

    size_t numElements() const {
        return static_cast<size_t>(dims[0]) * batchStride;
    }
};

StepLayout computeStepLayout(std::vector<int64_t> const& dims, size_t timeAxis) {
    require_ge(dims.size(), 2ul);
    require(timeAxis >= 1ul and timeAxis < dims.size());

    StepLayout layout;
    layout.dims = dims;

    size_t const          n = dims.size() - 1ul;  // without the batch axis
    std::valarray<size_t> sizes(n);
    std::valarray<size_t> strides(n);
    for (size_t d = 1ul; d < dims.size(); ++d) {
        sizes[d - 1ul] = d == timeAxis ? 1ul : static_cast<size_t>(dims[d]);
        layout.batchStride *= static_cast<size_t>(dims[d]);
    }
    strides[n - 1ul] = 1ul;
    for (size_t d = n - 1ul; d > 0ul; --d) {
        strides[d - 1ul] = static_cast<size_t>(dims[d + 1ul]) * strides[d];
    }
    layout.timeStride = strides[timeAxis - 1ul];

    Nn::ContiguousBlockInfo blockInfo(std::gslice(0ul, sizes, strides));
    layout.blockOffsets.resize(blockInfo.numBlocks());
    for (size_t i = 0ul; i < layout.blockOffsets.size(); ++i) {
        layout.blockOffsets[i] = blockInfo.blockOffset(i);
    }
    layout.blockSize = blockInfo.blockSize();
    layout.stepSize  = blockInfo.totalSize();
    return layout;
}

// Index of the time axis (the dynamic axis besides the batch axis) in the shape of a state variable
size_t timeAxisOf(OnnxStateVariable const& var) {
    for (size_t d = 1ul; d < var.shape.size(); ++d) {
        if (var.shape[d] < 0l) {
            return d;
        }
    }
    Core::Application::us()->criticalError("State variable %s has no time axis", var.input_state_key.c_str());
    return 0ul;
}

// Copy tasks of one launch (one state variable)
struct LaunchInfo {
    size_t firstTask   = 0ul;
    size_t numTasks    = 0ul;
    size_t firstOffset = 0ul;  // in the block offsets of all launches
    size_t blockSize   = 0ul;
    size_t stepSize    = 0ul;
};

}  // namespace

const Core::ParameterInt TransformerGpuStateManager::paramMaxHistoryLength(
        "max-history",
        "maximum length of the history to feed to the transformer",
        std::numeric_limits<int>::max(),
        0);

const Core::ParameterBool TransformerGpuStateManager::paramAlwaysIncludeFirstTokenState(
        "always-include-first-token-state",
        "whether to always include the state of the first token, even if history is restricted by max-history",
        false);

const Core::ParameterInt TransformerGpuStateManager::paramBlocksPerChunk(
        "blocks-per-chunk",
        "number of states (one time step of one state variable each) per device memory allocation of the state pool",
        4096,
        1);

TransformerGpuStateManager::TransformerGpuStateManager(Core::Configuration const& config)
        : Precursor(config),
          maxHistory_(paramMaxHistoryLength(config)),
          alwaysIncludeFirstTokenState_(paramAlwaysIncludeFirstTokenState(config)),
          blocksPerChunk_(paramBlocksPerChunk(config)) {
}

TransformerGpuStateManager::~TransformerGpuStateManager() {
    // Errors are ignored here: at program exit the CUDA context may already be destroyed
    for (auto* buffer : {&mergedBuffer_, &taskBuffer_, &stagingBuffer_}) {
        if (buffer->data != nullptr) {
            cudaFree(buffer->data);
        }
    }
}

bool TransformerGpuStateManager::requiresAllParentStates() const {
    return true;
}

void TransformerGpuStateManager::checkCuda(int status, char const* what) const {
    if (static_cast<cudaError_t>(status) != cudaSuccess) {
        criticalError("%s failed: %s", what, cudaGetErrorString(static_cast<cudaError_t>(status)));
    }
}

void* TransformerGpuStateManager::ensureCapacity(DeviceBuffer& buffer, size_t bytes, char const* what) {
    bytes = std::max(bytes, sizeof(float));
    if (buffer.capacity < bytes) {
        if (buffer.data != nullptr) {
            // cudaFree waits for pending work on the buffer
            checkCuda(cudaFree(buffer.data), what);
            buffer.data     = nullptr;
            buffer.capacity = 0ul;
        }
        size_t capacity = std::max(bytes, buffer.capacity + buffer.capacity / 2ul);
        checkCuda(cudaMalloc(&buffer.data, capacity), what);
        buffer.capacity = capacity;
    }
    return buffer.data;
}

std::shared_ptr<DeviceBlockPool> const& TransformerGpuStateManager::pool(size_t blockSize) {
    auto iter = pools_.find(blockSize);
    if (iter == pools_.end()) {
        iter = pools_.emplace(blockSize, std::make_shared<DeviceBlockPool>(blockSize, blocksPerChunk_)).first;
    }
    return iter->second;
}

TransformerGpuStateManager::HistoryState TransformerGpuStateManager::initialState(StateVariables const& vars, Nn::CompressedVectorFactory<float> const&) {
    HistoryState result;
    result.reserve(vars.size());
    for (size_t v = 0ul; v < vars.size(); ++v) {
        result.emplace_back(new DeviceVector(nullptr, 0ul));
    }
    return result;
}

void TransformerGpuStateManager::mergeStates(StateVariables const&                   vars,
                                             std::vector<size_t>&                    prefix_lengths,
                                             std::vector<HistoryState const*> const& prefix_states,
                                             FeedDict&                               feed_dict,
                                             TargetList&                             targets) {
    cudaStream_t stream = static_cast<cudaStream_t>(Session::sharedCudaStream());

    std::vector<size_t> original_prefix_lengths(prefix_lengths);

    size_t max_prefix = 0ul;
    for (size_t& len : prefix_lengths) {
        len        = std::min(len, maxHistory_);
        max_prefix = std::max(max_prefix, len);
    }
    // Padding (zeros before shorter prefixes) is only needed if the prefixes differ in length
    bool needs_padding = std::any_of(prefix_lengths.begin(), prefix_lengths.end(), [max_prefix](size_t len) { return len != max_prefix; });

    // Layouts and positions of the merged tensors in the merged buffer
    std::vector<StepLayout> layouts;
    std::vector<size_t>     tensorOffsets;
    layouts.reserve(vars.size());
    tensorOffsets.reserve(vars.size());
    size_t totalFloats = 0ul;
    for (auto const& var : vars) {
        size_t const         timeAxis = timeAxisOf(var);
        std::vector<int64_t> dims(var.shape.begin(), var.shape.end());
        dims[0]        = static_cast<int64_t>(prefix_lengths.size());
        dims[timeAxis] = static_cast<int64_t>(max_prefix);
        layouts.push_back(computeStepLayout(dims, timeAxis));
        tensorOffsets.push_back(totalFloats);
        totalFloats += alignUp(layouts.back().numElements());
    }
    float* merged = static_cast<float*>(ensureCapacity(mergedBuffer_, totalFloats * sizeof(float), "Allocating device memory for merged states"));

    // Copy tasks: one per time step of each state variable
    std::vector<CudaKernels::BlockCopyTask> tasks;
    std::vector<size_t>                     blockOffsets;
    std::vector<LaunchInfo>                 launches(vars.size());
    for (size_t v = 0ul; v < vars.size(); ++v) {
        auto const& layout = layouts[v];
        auto&       launch = launches[v];
        launch.firstTask   = tasks.size();
        launch.firstOffset = blockOffsets.size();
        launch.blockSize   = layout.blockSize;
        launch.stepSize    = layout.stepSize;
        blockOffsets.insert(blockOffsets.end(), layout.blockOffsets.begin(), layout.blockOffsets.end());

        size_t state_offset = 0ul;
        for (size_t b = 0ul; b < prefix_lengths.size(); ++b) {
            size_t prefix_length = prefix_lengths[b];
            size_t prefix_offset = original_prefix_lengths[b] - prefix_length;
            for (size_t p = 0ul; p < prefix_length; ++p) {
                size_t start = b * layout.batchStride + (max_prefix - prefix_length + p) * layout.timeStride;
                size_t idx   = state_offset;
                if (not alwaysIncludeFirstTokenState_ or p != 0ul) {
                    idx += prefix_offset + p;
                }
                auto const* state = dynamic_cast<DeviceVector const*>(prefix_states[idx]->at(v).get());
                if (state == nullptr) {
                    criticalError("State of %s was not created by the transformer-gpu state manager", vars[v].input_state_key.c_str());
                }
                require_eq(state->size(), layout.stepSize);
                tasks.push_back({state->deviceData(), merged + tensorOffsets[v] + start});
            }
            state_offset += original_prefix_lengths[b];
        }
        launch.numTasks = tasks.size() - launch.firstTask;
    }

    // Upload the tasks and block offsets at once: tasks first, then offsets
    size_t const taskBytes   = tasks.size() * sizeof(CudaKernels::BlockCopyTask);
    size_t const offsetBytes = blockOffsets.size() * sizeof(size_t);
    char*        deviceTasks = static_cast<char*>(ensureCapacity(taskBuffer_, taskBytes + offsetBytes, "Allocating device memory for copy tasks"));
    std::vector<char> hostTasks(taskBytes + offsetBytes);
    std::memcpy(hostTasks.data(), tasks.data(), taskBytes);
    std::memcpy(hostTasks.data() + taskBytes, blockOffsets.data(), offsetBytes);
    if (not hostTasks.empty()) {
        // hostTasks is pageable memory, so cudaMemcpyAsync returns only after it has been copied to a staging buffer
        // it may be freed at the end of this function although the copy to the device can still be pending
        checkCuda(cudaMemcpyAsync(deviceTasks, hostTasks.data(), hostTasks.size(), cudaMemcpyHostToDevice, stream), "Uploading copy tasks");
    }

    if (needs_padding) {
        checkCuda(cudaMemsetAsync(merged, 0, totalFloats * sizeof(float), stream), "Zeroing merged states");
    }

    auto const* deviceTaskArray   = reinterpret_cast<CudaKernels::BlockCopyTask const*>(deviceTasks);
    auto const* deviceOffsetArray = reinterpret_cast<size_t const*>(deviceTasks + taskBytes);
    for (auto const& launch : launches) {
        checkCuda(CudaKernels::launchBlockCopy(deviceTaskArray + launch.firstTask, launch.numTasks,
                                               deviceOffsetArray + launch.firstOffset, launch.blockSize, launch.stepSize,
                                               CudaKernels::BlockCopyDirection::BlockToTensor, stream),
                  "Merging states on the device");
    }

    feed_dict.reserve(feed_dict.size() + vars.size());
    targets.reserve(targets.size() + vars.size());
    for (size_t v = 0ul; v < vars.size(); ++v) {
        feed_dict.emplace_back(vars[v].input_state_key, Value::wrapDeviceMemory<float>(merged + tensorOffsets[v], layouts[v].dims));
        targets.emplace_back(vars[v].output_state_key);
    }
}

std::vector<TransformerGpuStateManager::HistoryState> TransformerGpuStateManager::splitStates(
        StateVariables const&                     vars,
        std::vector<size_t>&                      suffix_lengths,
        std::vector<Value> const&                 state_tensors,
        Nn::CompressedVectorFactory<float> const& vector_factory) {
    require_eq(vars.size(), state_tensors.size());
    cudaStream_t stream = static_cast<cudaStream_t>(Session::sharedCudaStream());

    size_t max_suffix = *std::max_element(suffix_lengths.begin(), suffix_lengths.end());
    size_t sum_suffix = std::accumulate(suffix_lengths.begin(), suffix_lengths.end(), 0ul);

    // Layouts of the state outputs; outputs in host memory are uploaded into the staging buffer
    std::vector<StepLayout> layouts;
    std::vector<size_t>     stagingOffsets(vars.size(), std::numeric_limits<size_t>::max());
    layouts.reserve(vars.size());
    size_t stagingFloats = 0ul;
    for (size_t v = 0ul; v < vars.size(); ++v) {
        auto const& tensor = state_tensors[v];
        if (tensor.dataType() != ValueDataType::FLOAT) {
            criticalError("transformer-gpu state manager: state %s is not float", vars[v].output_state_key.c_str());
        }
        size_t const         timeAxis = timeAxisOf(vars[v]);
        std::vector<int64_t> dims(tensor.numDims());
        for (int d = 0; d < tensor.numDims(); ++d) {
            dims[d] = tensor.dimSize(d);
        }
        require_eq(dims.size(), vars[v].shape.size());
        for (size_t d = 1ul; d < dims.size(); ++d) {
            if (d != timeAxis) {
                require_eq(vars[v].shape[d], dims[d]);
            }
        }
        layouts.push_back(computeStepLayout(dims, timeAxis));
        if (not tensor.isOnDevice()) {
            stagingOffsets[v] = stagingFloats;
            stagingFloats += alignUp(layouts.back().numElements());
        }
    }
    float* staging = nullptr;
    if (stagingFloats > 0ul) {
        staging = static_cast<float*>(ensureCapacity(stagingBuffer_, stagingFloats * sizeof(float), "Allocating device memory for staging states"));
        for (size_t v = 0ul; v < vars.size(); ++v) {
            if (stagingOffsets[v] != std::numeric_limits<size_t>::max()) {
                checkCuda(cudaMemcpyAsync(staging + stagingOffsets[v], state_tensors[v].rawData<float>(), layouts[v].numElements() * sizeof(float), cudaMemcpyHostToDevice, stream),
                          "Uploading states");
            }
        }
    }

    std::vector<HistoryState> result(sum_suffix);
    for (auto& state : result) {
        state.reserve(vars.size());
    }

    std::vector<CudaKernels::BlockCopyTask> tasks;
    std::vector<size_t>                     blockOffsets;
    std::vector<LaunchInfo>                 launches(vars.size());
    for (size_t v = 0ul; v < vars.size(); ++v) {
        auto const& layout = layouts[v];
        auto&       launch = launches[v];
        launch.firstTask   = tasks.size();
        launch.firstOffset = blockOffsets.size();
        launch.blockSize   = layout.blockSize;
        launch.stepSize    = layout.stepSize;
        blockOffsets.insert(blockOffsets.end(), layout.blockOffsets.begin(), layout.blockOffsets.end());

        float const* base = stagingOffsets[v] != std::numeric_limits<size_t>::max() ? staging + stagingOffsets[v] : state_tensors[v].rawData<float>();

        size_t const timeAxis   = timeAxisOf(vars[v]);
        size_t const max_prefix = static_cast<size_t>(layout.dims[timeAxis]) - max_suffix;
        size_t       output_idx = 0ul;
        for (size_t b = 0ul; b < suffix_lengths.size(); ++b) {
            for (size_t p = 0ul; p < suffix_lengths[b]; ++p) {
                size_t start = b * layout.batchStride + (max_prefix + p) * layout.timeStride;
                auto*  state = new DeviceVector(pool(layout.stepSize), layout.stepSize);
                result[output_idx].emplace_back(state);
                tasks.push_back({base + start, state->deviceData()});
                output_idx += 1ul;
            }
        }
        launch.numTasks = tasks.size() - launch.firstTask;
    }

    size_t const taskBytes   = tasks.size() * sizeof(CudaKernels::BlockCopyTask);
    size_t const offsetBytes = blockOffsets.size() * sizeof(size_t);
    char*        deviceTasks = static_cast<char*>(ensureCapacity(taskBuffer_, taskBytes + offsetBytes, "Allocating device memory for copy tasks"));
    std::vector<char> hostTasks(taskBytes + offsetBytes);
    std::memcpy(hostTasks.data(), tasks.data(), taskBytes);
    std::memcpy(hostTasks.data() + taskBytes, blockOffsets.data(), offsetBytes);
    if (not hostTasks.empty()) {
        // hostTasks is pageable memory, so cudaMemcpyAsync returns only after it has been copied to a staging buffer
        // it may be freed at the end of this function although the copy to the device can still be pending
        checkCuda(cudaMemcpyAsync(deviceTasks, hostTasks.data(), hostTasks.size(), cudaMemcpyHostToDevice, stream), "Uploading copy tasks");
    }

    auto const* deviceTaskArray   = reinterpret_cast<CudaKernels::BlockCopyTask const*>(deviceTasks);
    auto const* deviceOffsetArray = reinterpret_cast<size_t const*>(deviceTasks + taskBytes);
    for (auto const& launch : launches) {
        checkCuda(CudaKernels::launchBlockCopy(deviceTaskArray + launch.firstTask, launch.numTasks,
                                               deviceOffsetArray + launch.firstOffset, launch.blockSize, launch.stepSize,
                                               CudaKernels::BlockCopyDirection::TensorToBlock, stream),
                  "Splitting states on the device");
    }

    // The caller may release the output tensors right after this call, so the copies must be done
    checkCuda(cudaStreamSynchronize(stream), "Synchronizing the shared CUDA stream after splitting states");
    return result;
}

}  // namespace Onnx
