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
#ifndef _ONNX_LSTM_GPU_STATE_MANAGER_HH
#define _ONNX_LSTM_GPU_STATE_MANAGER_HH

#include <Nn/AbstractStateManager.hh>
#include <Onnx/OnnxStateVariable.hh>
#include <Onnx/Value.hh>

#include "DeviceVector.hh"
#include "StateCopier.hh"

namespace Onnx {

/*
 * Like the `lstm` state manager, but keeps the states (`DeviceVector`s) in CUDA device memory and merges and splits them
 * on the GPU. States must be float tensors of shape [batch, size]. Requires MODULE_CUDA and the CUDA execution provider.
 */
class LstmGpuStateManager : public Nn::AbstractStateManager<Value, OnnxStateVariable> {
public:
    using Precursor = Nn::AbstractStateManager<Value, OnnxStateVariable>;

    static const Core::ParameterInt paramBlocksPerChunk;

    LstmGpuStateManager(Core::Configuration const& config);
    ~LstmGpuStateManager() override = default;

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
    DeviceBlockPools pools_;         // by state size (floats per state)
    DeviceBuffer     mergedBuffer_;  // merged state inputs of the model
    StateCopier      copier_;
};

}  // namespace Onnx

#endif  // _ONNX_LSTM_GPU_STATE_MANAGER_HH
