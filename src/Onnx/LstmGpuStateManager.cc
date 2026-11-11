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
#include "LstmGpuStateManager.hh"

#include <Nn/DummyCompressedVectorFactory.hh>

namespace Onnx {

namespace {

// Copy batch of one state variable (a state is one contiguous row of the tensor)
StateCopier::Batch makeBatch(size_t stateSize, size_t batchSize) {
    StateCopier::Batch batch;
    batch.runOffsets = {0ul};
    batch.runLength  = stateSize;
    batch.stepSize   = stateSize;
    batch.tasks.reserve(batchSize);
    return batch;
}

}  // namespace

const Core::ParameterInt LstmGpuStateManager::paramSlotsPerChunk(
        "slots-per-chunk",
        "number of states (of one state variable each) per device memory allocation of the state pool",
        4096,
        1);

LstmGpuStateManager::LstmGpuStateManager(Core::Configuration const& config)
        : Precursor(config),
          compressionWarningLogged_(false),
          pools_(paramSlotsPerChunk(config)) {
}

void LstmGpuStateManager::warnIfCompressed(Nn::CompressedVectorFactory<float> const& vectorFactory) {
    if (not compressionWarningLogged_ and dynamic_cast<Nn::DummyCompressedVectorFactory<float> const*>(&vectorFactory) == nullptr) {
        warning("The lstm-cuda state manager ignores the configured state-compression, its states are uncompressed floats on the device");
        compressionWarningLogged_ = true;
    }
}

LstmGpuStateManager::HistoryState LstmGpuStateManager::initialState(StateVariables const& vars, Nn::CompressedVectorFactory<float> const& vectorFactory) {
    warnIfCompressed(vectorFactory);
    HistoryState result;
    result.reserve(vars.size());
    for (auto const& var : vars) {
        require_gt(var.shape.size(), 0ul);
        int64_t stateSize = var.shape.back();
        if (stateSize <= 0l) {
            criticalError("lstm-cuda state manager: state %s must have a fixed size", var.input_state_key.c_str());
        }
        auto state = std::make_unique<DeviceVector>(pools_.get(stateSize), stateSize);
        checkCudaStatus(cudaMemsetAsync(state->deviceData(), 0, stateSize * sizeof(float), sharedStream()), "Zeroing initial state");
        result.push_back(std::move(state));
    }
    return result;
}

void LstmGpuStateManager::mergeStates(StateVariables const&                   vars,
                                      std::vector<size_t>&                    prefixLengths,
                                      std::vector<HistoryState const*> const& prefixStates,
                                      FeedDict&                               feedDict,
                                      TargetList&                             targets) {
    require_eq(prefixStates.size(), prefixLengths.size());
    size_t const batchSize = prefixStates.size();

    // Positions of the merged tensors [batchSize, stateSize] in the merged buffer
    std::vector<size_t> stateSizes(vars.size());
    std::vector<size_t> tensorOffsets(vars.size());
    size_t              totalFloats = 0ul;
    for (size_t v = 0ul; v < vars.size(); ++v) {
        stateSizes[v]    = batchSize > 0ul ? prefixStates.front()->at(v)->size() : 0ul;
        tensorOffsets[v] = totalFloats;
        totalFloats += alignDeviceFloats(batchSize * stateSizes[v]);
    }
    float* merged = static_cast<float*>(mergedBuffer_.allocate(totalFloats * sizeof(float)));

    std::vector<StateCopier::Batch> batches;
    batches.reserve(vars.size());
    for (size_t v = 0ul; v < vars.size(); ++v) {
        batches.push_back(makeBatch(stateSizes[v], batchSize));

        // All states come from the same state manager, so checking the type of the first one is enough
        if (batchSize > 0ul and dynamic_cast<DeviceVector const*>(prefixStates.front()->at(v).get()) == nullptr) {
            criticalError("State of %s was not created by the lstm-cuda state manager", vars[v].input_state_key.c_str());
        }

        for (size_t b = 0ul; b < batchSize; ++b) {
            auto const* state = static_cast<DeviceVector const*>(prefixStates[b]->at(v).get());
            require_eq(state->size(), stateSizes[v]);
            batches.back().tasks.push_back({state->deviceData(), merged + tensorOffsets[v] + b * stateSizes[v]});
        }
    }
    copier_.copy(batches, CudaKernels::StateCopyDirection::StateToTensor);

    feedDict.reserve(feedDict.size() + vars.size());
    targets.reserve(targets.size() + vars.size());
    for (size_t v = 0ul; v < vars.size(); ++v) {
        std::vector<int64_t> dims = {static_cast<int64_t>(batchSize), static_cast<int64_t>(stateSizes[v])};
        feedDict.emplace_back(vars[v].input_state_key, Value::wrapDeviceMemory<float>(merged + tensorOffsets[v], dims));
        targets.emplace_back(vars[v].output_state_key);
    }
}

std::vector<LstmGpuStateManager::HistoryState> LstmGpuStateManager::splitStates(
        StateVariables const&                     vars,
        std::vector<size_t>&                      suffixLengths,
        std::vector<Value> const&                 stateTensors,
        Nn::CompressedVectorFactory<float> const& vectorFactory) {
    warnIfCompressed(vectorFactory);
    require_eq(vars.size(), stateTensors.size());
    size_t const batchSize = suffixLengths.size();

    std::vector<size_t> stateSizes(vars.size());
    for (size_t v = 0ul; v < vars.size(); ++v) {
        auto const& tensor = stateTensors[v];
        if (tensor.dataType() != ValueDataType::FLOAT or tensor.numDims() != 2) {
            criticalError("lstm-cuda state manager: state %s must be a float tensor of shape [batch, size]", vars[v].output_state_key.c_str());
        }
        // The merged state inputs are on the device, so the session puts the state outputs there as well
        require(tensor.isOnDevice());
        require_eq(static_cast<size_t>(tensor.dimSize(0)), batchSize);
        stateSizes[v] = tensor.dimSize(1);
    }

    std::vector<HistoryState> result(batchSize);
    for (auto& state : result) {
        state.reserve(vars.size());
    }
    std::vector<StateCopier::Batch> batches;
    batches.reserve(vars.size());
    for (size_t v = 0ul; v < vars.size(); ++v) {
        batches.push_back(makeBatch(stateSizes[v], batchSize));
        float const* base = stateTensors[v].rawData<float>();
        for (size_t b = 0ul; b < batchSize; ++b) {
            auto state = std::make_unique<DeviceVector>(pools_.get(stateSizes[v]), stateSizes[v]);
            batches.back().tasks.push_back({base + b * stateSizes[v], state->deviceData()});
            result[b].push_back(std::move(state));
        }
    }
    copier_.copy(batches, CudaKernels::StateCopyDirection::TensorToState);

    // One state per batch entry, like the `lstm-cpu` state manager
    for (auto& suffixLength : suffixLengths) {
        suffixLength = 1ul;
    }

    // The caller may release the output tensors right after this call, so the copies must be done
    checkCudaStatus(cudaStreamSynchronize(sharedStream()), "Synchronizing the shared CUDA stream after splitting states");
    return result;
}

}  // namespace Onnx
