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

#ifndef _TORCH_STATEMANAGER_HH
#define _TORCH_STATEMANAGER_HH

#include <Core/Component.hh>

#pragma push_macro("ensure")
#undef ensure
#include <torch/torch.h>
#pragma pop_macro("ensure")

namespace Torch {

// Description of one explicit runtime state tensor
struct TorchStateVariable {
    std::string          name;                     // semantic state name
    std::string          kind;                     // state type, e.g. conv/mhsa
    int                  inputIndex  = -1;         // absolute runtime input index
    int                  outputIndex = -1;         // absolute runtime output index
    int                  layer       = -1;         // owning layer index
    std::vector<int64_t> shape;                    // expected tensor shape
    torch::ScalarType    dtype = torch::kFloat32;  // expected tensor dtype
};

/*
 * Base interface for Torch state handling
 */
class StateManager : public Core::Component {
public:
    using Precursor = Core::Component;
    using InputList = std::vector<torch::Tensor>;

    // Creates a state manager from config
    static std::unique_ptr<StateManager> create(Core::Configuration const& config);

    StateManager(Core::Configuration const& config);
    virtual ~StateManager() = default;

    // Initializes all managed states
    virtual void setInitialStates(std::vector<TorchStateVariable> const& stateVars) = 0;

    // Appends the current states to the positional model input list
    virtual void extendInputs(InputList& inputs, std::vector<TorchStateVariable> const& stateVars) = 0;

    // Updates the stored states from the model outputs
    virtual void updateStates(std::vector<torch::Tensor> const& outputs, std::vector<TorchStateVariable> const& stateVars) = 0;
};

}  // namespace Torch

#endif  // _TORCH_STATEMANAGER_HH