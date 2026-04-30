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

#ifndef _TORCH_CONFORMERSTATEMANAGER_HH
#define _TORCH_CONFORMERSTATEMANAGER_HH

#include "StateManager.hh"

namespace Torch {

// State manager for conformer-style streaming caches.
// Keeps the current state tensors, trims them according to the configured
// left-context policy, appends them to the model inputs, and updates them
// from the corresponding model outputs.
class ConformerStateManager : public StateManager {
public:
    using Precursor = StateManager;
    using InputList = Precursor::InputList;

    static const Core::ParameterInt paramAttentionContextSize;
    static const Core::ParameterInt paramConvContextSize;
    static const Core::ParameterInt paramDiscardSuffixLength;

    explicit ConformerStateManager(Core::Configuration const& config);
    virtual ~ConformerStateManager() = default;

    virtual void setInitialStates(std::vector<TorchStateVariable> const& stateVars) override;
    virtual void extendInputs(InputList& inputs, std::vector<TorchStateVariable> const& stateVars) override;
    virtual void updateStates(std::vector<torch::Tensor> const& outputs, std::vector<TorchStateVariable> const& stateVars) override;

private:
    const int64_t attentionContextSize_;
    const int64_t convContextSize_;
    const int64_t discardSuffixLength_;

    std::vector<torch::Tensor> states_;

    torch::Tensor trimStates(torch::Tensor const& state, TorchStateVariable const& stateVar) const;
};

}  // namespace Torch

#endif  // _TORCH_CONFORMERSTATEMANAGER_HH