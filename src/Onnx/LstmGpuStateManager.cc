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

#include <cuda_runtime.h>

#include "Session.hh"

namespace Onnx {

namespace {

cudaStream_t sharedStream() {
    return static_cast<cudaStream_t>(Session::sharedCudaStream());
}

// Copy batch of one state variable (a state is one contiguous row of the tensor)
StateCopier::Batch makeBatch(size_t stateSize, size_t batchSize) {
    StateCopier::Batch batch;
    batch.blockOffsets = {0ul};
    batch.blockSize    = stateSize;
    batch.stepSize     = stateSize;
    batch.tasks.reserve(batchSize);
    return batch;
}

}  // namespace

const Core::ParameterInt LstmGpuStateManager::paramBlocksPerChunk(
        "blocks-per-chunk",
        "number of states (of one state variable each) per device memory allocation of the state pool",
        4096,
        1);

LstmGpuStateManager::LstmGpuStateManager(Core::Configuration const& config)
        : Precursor(config),
          pools_(paramBlocksPerChunk(config)) {
}

LstmGpuStateManager::HistoryState LstmGpuStateManager::initialState(StateVariables const& vars, Nn::CompressedVectorFactory<float> const&) {
    HistoryState result;
    result.reserve(vars.size());
    for (auto const& var : vars) {
        require_gt(var.shape.size(), 0ul);
        int64_t stateSize = var.shape.back();
        if (stateSize <= 0l) {
            criticalError("lstm-gpu state manager: state %s must have a fixed size", var.input_state_key.c_str());
        }
        auto* state = new DeviceVector(pools_.get(stateSize), stateSize);
        checkCudaStatus(cudaMemsetAsync(state->deviceData(), 0, stateSize * sizeof(float), sharedStream()), "Zeroing initial state");
        result.emplace_back(state);
    }
    return result;
}

void LstmGpuStateManager::mergeStates(StateVariables const&                   vars,
                                      std::vector<size_t>&                    prefix_lengths,
                                      std::vector<HistoryState const*> const& prefix_states,
                                      FeedDict&                               feed_dict,
                                      TargetList&                             targets) {
    require_eq(prefix_states.size(), prefix_lengths.size());
    size_t const batchSize = prefix_states.size();

    // Positions of the merged tensors [batchSize, stateSize] in the merged buffer
    std::vector<size_t> stateSizes(vars.size());
    std::vector<size_t> tensorOffsets(vars.size());
    size_t              totalFloats = 0ul;
    for (size_t v = 0ul; v < vars.size(); ++v) {
        stateSizes[v]    = batchSize > 0ul ? prefix_states.front()->at(v)->size() : 0ul;
        tensorOffsets[v] = totalFloats;
        totalFloats += alignDeviceFloats(batchSize * stateSizes[v]);
    }
    float* merged = static_cast<float*>(mergedBuffer_.allocate(totalFloats * sizeof(float)));

    std::vector<StateCopier::Batch> batches;
    batches.reserve(vars.size());
    for (size_t v = 0ul; v < vars.size(); ++v) {
        batches.push_back(makeBatch(stateSizes[v], batchSize));
        for (size_t b = 0ul; b < batchSize; ++b) {
            auto const* state = dynamic_cast<DeviceVector const*>(prefix_states[b]->at(v).get());
            if (state == nullptr) {
                criticalError("State of %s was not created by the lstm-gpu state manager", vars[v].input_state_key.c_str());
            }
            require_eq(state->size(), stateSizes[v]);
            batches.back().tasks.push_back({state->deviceData(), merged + tensorOffsets[v] + b * stateSizes[v]});
        }
    }
    copier_.copy(batches, CudaKernels::BlockCopyDirection::BlockToTensor);

    feed_dict.reserve(feed_dict.size() + vars.size());
    targets.reserve(targets.size() + vars.size());
    for (size_t v = 0ul; v < vars.size(); ++v) {
        std::vector<int64_t> dims = {static_cast<int64_t>(batchSize), static_cast<int64_t>(stateSizes[v])};
        feed_dict.emplace_back(vars[v].input_state_key, Value::wrapDeviceMemory<float>(merged + tensorOffsets[v], dims));
        targets.emplace_back(vars[v].output_state_key);
    }
}

std::vector<LstmGpuStateManager::HistoryState> LstmGpuStateManager::splitStates(
        StateVariables const&                     vars,
        std::vector<size_t>&                      suffix_lengths,
        std::vector<Value> const&                 state_tensors,
        Nn::CompressedVectorFactory<float> const& vector_factory) {
    require_eq(vars.size(), state_tensors.size());
    size_t const batchSize = suffix_lengths.size();

    std::vector<size_t> stateSizes(vars.size());
    for (size_t v = 0ul; v < vars.size(); ++v) {
        auto const& tensor = state_tensors[v];
        if (tensor.dataType() != ValueDataType::FLOAT or tensor.numDims() != 2) {
            criticalError("lstm-gpu state manager: state %s must be a float tensor of shape [batch, size]", vars[v].output_state_key.c_str());
        }
        require_eq(static_cast<size_t>(tensor.dimSize(0)), batchSize);
        stateSizes[v] = tensor.dimSize(1);
    }
    // Outputs in host memory are uploaded into a staging buffer
    std::vector<float const*> bases = copier_.deviceData(state_tensors);

    std::vector<HistoryState> result(batchSize);
    for (auto& state : result) {
        state.reserve(vars.size());
    }
    std::vector<StateCopier::Batch> batches;
    batches.reserve(vars.size());
    for (size_t v = 0ul; v < vars.size(); ++v) {
        batches.push_back(makeBatch(stateSizes[v], batchSize));
        for (size_t b = 0ul; b < batchSize; ++b) {
            auto* state = new DeviceVector(pools_.get(stateSizes[v]), stateSizes[v]);
            result[b].emplace_back(state);
            batches.back().tasks.push_back({bases[v] + b * stateSizes[v], state->deviceData()});
        }
    }
    copier_.copy(batches, CudaKernels::BlockCopyDirection::TensorToBlock);

    // One state per batch entry, like the `lstm` state manager
    for (auto& suffixLength : suffix_lengths) {
        suffixLength = 1ul;
    }

    // The caller may release the output tensors right after this call, so the copies must be done
    checkCudaStatus(cudaStreamSynchronize(sharedStream()), "Synchronizing the shared CUDA stream after splitting states");
    return result;
}

}  // namespace Onnx
