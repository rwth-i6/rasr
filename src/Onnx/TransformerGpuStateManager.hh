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
#include "StateCopier.hh"

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
    static const Core::ParameterInt  paramSlotsPerChunk;

    TransformerGpuStateManager(Core::Configuration const& config);
    ~TransformerGpuStateManager() override;

    bool requiresAllParentStates() const override;

    HistoryState initialState(StateVariables const& vars, Nn::CompressedVectorFactory<float> const& vectorFactory) override;

    void mergeStates(StateVariables const&                   vars,
                     std::vector<size_t>&                    prefixLengths,
                     std::vector<HistoryState const*> const& prefixStates,
                     FeedDict&                               feedDict,
                     TargetList&                             targets) override;

    std::vector<HistoryState> splitStates(StateVariables const&                     vars,
                                          std::vector<size_t>&                      suffixLengths,
                                          std::vector<Value> const&                 stateTensors,
                                          Nn::CompressedVectorFactory<float> const& vectorFactory) override;

private:
    const size_t maxHistory_;
    const bool   alwaysIncludeFirstTokenState_;
    bool         compressionWarningLogged_;

    // Warn (once) if a state compression is configured: the states stay uncompressed floats on the device
    void warnIfCompressed(Nn::CompressedVectorFactory<float> const& vectorFactory);

    DeviceSlotPools pools_;             // by slot size (floats per state)
    DeviceBuffer    mergedBuffer_;      // merged state inputs of the model
    DeviceBuffer    mergeInputBuffer_;  // state pointers, batch and layout data of `mergeStates`
    StateCopier     copier_;            // copies of `splitStates`
};

}  // namespace Onnx

#endif  // _ONNX_TRANSFORMER_GPU_STATE_MANAGER_HH
