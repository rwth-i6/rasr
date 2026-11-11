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
#include <Nn/DummyCompressedVectorFactory.hh>

namespace Onnx {

namespace {

// Layout of one time step in a tensor with shape `dims` (batch first), like in `Nn::TransformerStateManager`
struct StepLayout {
    std::vector<int64_t> dims;
    size_t               timeAxis    = 0ul;
    size_t               batchStride = 1ul;  // elements per batch entry
    size_t               timeStride  = 0ul;  // elements per time step
    std::vector<size_t>  runOffsets;         // contiguous runs of one time step, relative to its start
    size_t               runLength = 0ul;    // elements per run
    size_t               stepSize  = 0ul;    // elements per time step

    size_t numElements() const {
        return static_cast<size_t>(dims[0]) * batchStride;
    }
};

StepLayout computeStepLayout(std::vector<int64_t> const& dims, size_t timeAxis) {
    require_ge(dims.size(), 2ul);
    require(timeAxis >= 1ul and timeAxis < dims.size());

    StepLayout layout;
    layout.dims     = dims;
    layout.timeAxis = timeAxis;

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
    layout.runOffsets.resize(blockInfo.numBlocks());
    for (size_t i = 0ul; i < layout.runOffsets.size(); ++i) {
        layout.runOffsets[i] = blockInfo.blockOffset(i);
    }
    layout.runLength = blockInfo.blockSize();
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

// Copy batch of one state variable with the given layout, without tasks yet
StateCopier::Batch makeBatch(StepLayout const& layout) {
    StateCopier::Batch batch;
    batch.runOffsets = layout.runOffsets;
    batch.runLength  = layout.runLength;
    batch.stepSize   = layout.stepSize;
    return batch;
}

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

const Core::ParameterInt TransformerGpuStateManager::paramSlotsPerChunk(
        "slots-per-chunk",
        "number of states (one time step of one state variable each) per device memory allocation of the state pool",
        4096,
        1);

TransformerGpuStateManager::TransformerGpuStateManager(Core::Configuration const& config)
        : Precursor(config),
          maxHistory_(paramMaxHistoryLength(config)),
          alwaysIncludeFirstTokenState_(paramAlwaysIncludeFirstTokenState(config)),
          compressionWarningLogged_(false),
          pools_(paramSlotsPerChunk(config)) {
}

TransformerGpuStateManager::~TransformerGpuStateManager() = default;

bool TransformerGpuStateManager::requiresAllParentStates() const {
    return true;
}

void TransformerGpuStateManager::warnIfCompressed(Nn::CompressedVectorFactory<float> const& vectorFactory) {
    if (not compressionWarningLogged_ and dynamic_cast<Nn::DummyCompressedVectorFactory<float> const*>(&vectorFactory) == nullptr) {
        warning("The transformer-cuda state manager ignores the configured state-compression, its states are uncompressed floats on the device");
        compressionWarningLogged_ = true;
    }
}

TransformerGpuStateManager::HistoryState TransformerGpuStateManager::initialState(StateVariables const& vars, Nn::CompressedVectorFactory<float> const& vectorFactory) {
    warnIfCompressed(vectorFactory);
    HistoryState result;
    result.reserve(vars.size());
    for (size_t v = 0ul; v < vars.size(); ++v) {
        result.push_back(std::make_unique<DeviceVector>(nullptr, 0ul));
    }
    return result;
}

void TransformerGpuStateManager::mergeStates(StateVariables const&                   vars,
                                             std::vector<size_t>&                    prefixLengths,
                                             std::vector<HistoryState const*> const& prefixStates,
                                             FeedDict&                               feedDict,
                                             TargetList&                             targets) {
    std::vector<size_t> originalPrefixLengths(prefixLengths);

    size_t maxPrefix = 0ul;
    for (size_t& len : prefixLengths) {
        len       = std::min(len, maxHistory_);
        maxPrefix = std::max(maxPrefix, len);
    }
    size_t const numBatch = prefixLengths.size();
    size_t const numVars  = vars.size();
    // Grid dimensions y and z of the merge kernel
    constexpr size_t maxGridDim = 65535ul;
    if (numBatch > maxGridDim or numVars > maxGridDim) {
        criticalError("transformer-gpu state manager: at most %zu batch entries and state variables are supported", maxGridDim);
    }

    // Layouts and positions of the merged tensors in the merged buffer
    std::vector<StepLayout> layouts;
    std::vector<size_t>     tensorOffsets;
    layouts.reserve(numVars);
    tensorOffsets.reserve(numVars);
    size_t totalFloats = 0ul;
    for (auto const& var : vars) {
        size_t const         timeAxis = timeAxisOf(var);
        std::vector<int64_t> dims(var.shape.begin(), var.shape.end());
        dims[0]        = static_cast<int64_t>(numBatch);
        dims[timeAxis] = static_cast<int64_t>(maxPrefix);
        layouts.push_back(computeStepLayout(dims, timeAxis));
        tensorOffsets.push_back(totalFloats);
        totalFloats += alignDeviceFloats(layouts.back().numElements());
    }
    float* merged = static_cast<float*>(mergedBuffer_.allocate(totalFloats * sizeof(float)));

    // Which stored state goes to which prefix position, the same for all state variables:
    // the states of batch entry b are at batchStarts[b] ... batchStarts[b] + prefixLengths[b] - 1
    std::vector<size_t> batchStarts(numBatch);
    std::vector<size_t> stateIndices;
    stateIndices.reserve(std::accumulate(prefixLengths.begin(), prefixLengths.end(), 0ul));
    size_t stateOffset = 0ul;
    for (size_t b = 0ul; b < numBatch; ++b) {
        size_t prefixLength = prefixLengths[b];
        size_t prefixOffset = originalPrefixLengths[b] - prefixLength;
        batchStarts[b]      = stateIndices.size();
        for (size_t p = 0ul; p < prefixLength; ++p) {
            size_t idx = stateOffset;
            if (not alwaysIncludeFirstTokenState_ or p != 0ul) {
                idx += prefixOffset + p;
            }
            stateIndices.push_back(idx);
        }
        stateOffset += originalPrefixLengths[b];
    }
    size_t const numStates = stateIndices.size();

    // Merge inputs, uploaded at once (all entries are 8 bytes, so the arrays stay aligned):
    // state pointers [numVars * numStates], prefix lengths [numBatch], batch starts [numBatch], layouts [numVars], run offsets
    std::vector<CudaKernels::MergeLayout> mergeLayouts(numVars);
    std::vector<size_t>                   runOffsets;
    for (size_t v = 0ul; v < numVars; ++v) {
        auto const& layout = layouts[v];
        mergeLayouts[v]    = {merged + tensorOffsets[v], layout.batchStride, layout.timeStride, layout.stepSize, layout.runLength, runOffsets.size()};
        runOffsets.insert(runOffsets.end(), layout.runOffsets.begin(), layout.runOffsets.end());
    }
    size_t const      pointerBytes = numVars * numStates * sizeof(float const*);
    size_t const      batchBytes   = numBatch * sizeof(size_t);
    size_t const      layoutBytes  = numVars * sizeof(CudaKernels::MergeLayout);
    size_t const      offsetBytes  = runOffsets.size() * sizeof(size_t);
    std::vector<char> hostInput(pointerBytes + 2ul * batchBytes + layoutBytes + offsetBytes);

    auto* statePointers = reinterpret_cast<float const**>(hostInput.data());
    for (size_t v = 0ul; v < numVars; ++v) {
        // All states come from the same state manager, so checking the type of the first one is enough
        if (numStates > 0ul and dynamic_cast<DeviceVector const*>(prefixStates[stateIndices.front()]->at(v).get()) == nullptr) {
            criticalError("State of %s was not created by the transformer-gpu state manager", vars[v].input_state_key.c_str());
        }
        size_t const  stepSize = layouts[v].stepSize;
        float const** pointers = statePointers + v * numStates;
        for (size_t i = 0ul; i < numStates; ++i) {
            auto const* state = static_cast<DeviceVector const*>(prefixStates[stateIndices[i]]->at(v).get());
            require_eq(state->size(), stepSize);
            pointers[i] = state->deviceData();
        }
    }
    char* pos = hostInput.data() + pointerBytes;
    std::memcpy(pos, prefixLengths.data(), batchBytes);
    std::memcpy(pos + batchBytes, batchStarts.data(), batchBytes);
    std::memcpy(pos + 2ul * batchBytes, mergeLayouts.data(), layoutBytes);
    std::memcpy(pos + 2ul * batchBytes + layoutBytes, runOffsets.data(), offsetBytes);

    char* deviceInput = static_cast<char*>(mergeInputBuffer_.allocate(hostInput.size()));
    // hostInput is pageable memory, so cudaMemcpyAsync returns only after it has been copied to a staging buffer
    checkCudaStatus(cudaMemcpyAsync(deviceInput, hostInput.data(), hostInput.size(), cudaMemcpyHostToDevice, sharedStream()), "Uploading merge inputs");
    char const* devicePos = deviceInput + pointerBytes;
    checkCudaStatus(CudaKernels::launchStateMerge(reinterpret_cast<float const* const*>(deviceInput),
                                                  reinterpret_cast<size_t const*>(devicePos),
                                                  reinterpret_cast<size_t const*>(devicePos + batchBytes),
                                                  reinterpret_cast<CudaKernels::MergeLayout const*>(devicePos + 2ul * batchBytes),
                                                  reinterpret_cast<size_t const*>(devicePos + 2ul * batchBytes + layoutBytes),
                                                  numStates,
                                                  maxPrefix,
                                                  numBatch,
                                                  numVars,
                                                  sharedStream()),
                    "Merging states on the device");

    feedDict.reserve(feedDict.size() + numVars);
    targets.reserve(targets.size() + numVars);
    for (size_t v = 0ul; v < numVars; ++v) {
        feedDict.emplace_back(vars[v].input_state_key, Value::wrapDeviceMemory<float>(merged + tensorOffsets[v], layouts[v].dims));
        targets.emplace_back(vars[v].output_state_key);
    }
}

std::vector<TransformerGpuStateManager::HistoryState> TransformerGpuStateManager::splitStates(
        StateVariables const&                     vars,
        std::vector<size_t>&                      suffixLengths,
        std::vector<Value> const&                 stateTensors,
        Nn::CompressedVectorFactory<float> const& vectorFactory) {
    warnIfCompressed(vectorFactory);
    require_eq(vars.size(), stateTensors.size());

    size_t maxSuffix = *std::max_element(suffixLengths.begin(), suffixLengths.end());
    size_t sumSuffix = std::accumulate(suffixLengths.begin(), suffixLengths.end(), 0ul);

    // Layouts of the state outputs
    std::vector<StepLayout> layouts;
    layouts.reserve(vars.size());
    for (size_t v = 0ul; v < vars.size(); ++v) {
        auto const& tensor = stateTensors[v];
        if (tensor.dataType() != ValueDataType::FLOAT) {
            criticalError("transformer-gpu state manager: state %s is not float", vars[v].output_state_key.c_str());
        }
        // The merged state inputs are on the device, so the session puts the state outputs there as well
        require(tensor.isOnDevice());
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
    }

    std::vector<HistoryState> result(sumSuffix);
    for (auto& state : result) {
        state.reserve(vars.size());
    }

    std::vector<StateCopier::Batch> batches;
    batches.reserve(vars.size());
    for (size_t v = 0ul; v < vars.size(); ++v) {
        auto const& layout = layouts[v];
        batches.push_back(makeBatch(layout));
        auto&        tasks = batches.back().tasks;
        float const* base  = stateTensors[v].rawData<float>();

        size_t const maxPrefix = static_cast<size_t>(layout.dims[layout.timeAxis]) - maxSuffix;
        size_t       outputIdx = 0ul;
        for (size_t b = 0ul; b < suffixLengths.size(); ++b) {
            for (size_t p = 0ul; p < suffixLengths[b]; ++p) {
                size_t start = b * layout.batchStride + (maxPrefix + p) * layout.timeStride;
                auto   state = std::make_unique<DeviceVector>(pools_.get(layout.stepSize), layout.stepSize);
                tasks.push_back({base + start, state->deviceData()});
                result[outputIdx].push_back(std::move(state));
                outputIdx += 1ul;
            }
        }
    }

    copier_.copy(batches, CudaKernels::StateCopyDirection::TensorToState);

    // The caller may release the output tensors right after this call, so the copies must be done
    checkCudaStatus(cudaStreamSynchronize(sharedStream()), "Synchronizing the shared CUDA stream after splitting states");
    return result;
}

}  // namespace Onnx
