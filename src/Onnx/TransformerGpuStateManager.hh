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
#ifndef _ONNX_TRANSFORMER_GPU_STATE_MANAGER_HH
#define _ONNX_TRANSFORMER_GPU_STATE_MANAGER_HH

#include <memory>
#include <unordered_map>

#include <Nn/AbstractStateManager.hh>
#include <Onnx/OnnxStateVariable.hh>
#include <Onnx/Value.hh>

#include "DeviceVector.hh"

namespace Onnx {

/*
 * Like the `transformer` state manager, but keeps the states (`DeviceVector`s) in CUDA device memory and merges and
 * splits them on the GPU. Only float states. Requires MODULE_CUDA and the CUDA execution provider.
 */
class TransformerGpuStateManager : public Nn::AbstractStateManager<Value, OnnxStateVariable> {
public:
    using Precursor = Nn::AbstractStateManager<Value, OnnxStateVariable>;

    static const Core::ParameterInt  paramMaxHistoryLength;
    static const Core::ParameterBool paramAlwaysIncludeFirstTokenState;
    static const Core::ParameterInt  paramBlocksPerChunk;

    TransformerGpuStateManager(Core::Configuration const& config);
    ~TransformerGpuStateManager() override;

    bool requiresAllParentStates() const override;

    HistoryState initialState(StateVariables const& vars, Nn::CompressedVectorFactory<float> const& vector_factory) override;

    void mergeStates(StateVariables const&                   vars,
                     std::vector<size_t>&                    prefix_lengths,
                     std::vector<HistoryState const*> const& prefix_states,
                     FeedDict&                               feed_dict,
                     TargetList&                             targets) override;

    std::vector<HistoryState> splitStates(StateVariables const&                     vars,
                                          std::vector<size_t>&                      suffix_lengths,
                                          std::vector<Value> const&                 state_tensors,
                                          Nn::CompressedVectorFactory<float> const& vector_factory) override;

private:
    // Device memory that grows on demand and is reused between calls
    struct DeviceBuffer {
        void*  data     = nullptr;
        size_t capacity = 0ul;  // in bytes
    };

    const size_t maxHistory_;
    const bool   alwaysIncludeFirstTokenState_;
    const size_t blocksPerChunk_;

    std::unordered_map<size_t, std::shared_ptr<DeviceBlockPool>> pools_;  // by block size (floats per state)

    DeviceBuffer mergedBuffer_;   // merged state inputs of the model
    DeviceBuffer taskBuffer_;     // copy tasks and block offsets for the kernel
    DeviceBuffer stagingBuffer_;  // state outputs in host memory, uploaded for splitting

    std::shared_ptr<DeviceBlockPool> const& pool(size_t blockSize);

    void* ensureCapacity(DeviceBuffer& buffer, size_t bytes, char const* what);
    void  checkCuda(int status, char const* what) const;
};

}  // namespace Onnx

#endif  // _ONNX_TRANSFORMER_GPU_STATE_MANAGER_HH
