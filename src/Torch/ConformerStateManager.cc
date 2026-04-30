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

#include "ConformerStateManager.hh"

#include <algorithm>

namespace Torch {

const Core::ParameterInt ConformerStateManager::paramAttentionContextSize(
        "attention-context-size",
        "left-context size for attention states (in frames)",
        100,
        0);

const Core::ParameterInt ConformerStateManager::paramConvContextSize(
        "conv-context-size",
        "left-context size for convolution states (in frames)",
        100,
        0);

const Core::ParameterInt ConformerStateManager::paramDiscardSuffixLength(
        "discard-suffix-length",
        "how many frames to drop from the end of the new state (useful for overlapping chunks)",
        0,
        0);

ConformerStateManager::ConformerStateManager(Core::Configuration const& config)
        : Precursor(config),
          attentionContextSize_(paramAttentionContextSize(config)),
          convContextSize_(paramConvContextSize(config)),
          discardSuffixLength_(paramDiscardSuffixLength(config)),
          states_() {}

void ConformerStateManager::setInitialStates(std::vector<TorchStateVariable> const& stateVars) {
    if (stateVars.empty()) {
        criticalError(
                "ConformerStateManager requires non-empty state variables. "
                "This usually means that no valid state specification was provided.");
    }

    states_.clear();
    states_.reserve(stateVars.size());

    for (auto const& stateVar : stateVars) {
        if (stateVar.inputIndex < 0) {
            criticalError("State variable '%s' has invalid input index %d", stateVar.name.c_str(), stateVar.inputIndex);
        }
        if (stateVar.outputIndex < 0) {
            criticalError("State variable '%s' has invalid output index %d", stateVar.name.c_str(), stateVar.outputIndex);
        }
        if (stateVar.shape.empty()) {
            criticalError("State variable '%s' has empty shape", stateVar.name.c_str());
        }

        std::vector<int64_t> shape;
        shape.reserve(stateVar.shape.size());

        for (size_t i = 0; i < stateVar.shape.size(); ++i) {
            const int64_t dim = stateVar.shape[i];

            // Batch axis stays fixed at 1. Dynamic non-batch axes start empty.
            if (i != 0 && dim < 0) {
                shape.push_back(0);
            }
            else {
                shape.push_back(dim);
            }
        }

        shape[0] = 1;

        auto options = torch::TensorOptions().dtype(stateVar.dtype);
        states_.push_back(torch::zeros(shape, options));
    }
}

void ConformerStateManager::extendInputs(InputList& inputs, std::vector<TorchStateVariable> const& stateVars) {
    if (states_.empty() && !stateVars.empty()) {
        setInitialStates(stateVars);
    }

    if (states_.size() != stateVars.size()) {
        criticalError("Number of stored states (%zu) does not match number of state variables (%zu)", states_.size(), stateVars.size());
    }

    // Append states in the order given by stateVars
    // This should correspond to the expected input state tuple order
    for (size_t i = 0; i < states_.size(); ++i) {
        inputs.push_back(trimStates(states_[i], stateVars[i]));
    }
}

void ConformerStateManager::updateStates(std::vector<torch::Tensor> const& outputs, std::vector<TorchStateVariable> const& stateVars) {
    if (stateVars.empty()) {
        return;
    }

    states_.clear();
    states_.reserve(stateVars.size());

    for (size_t i = 0; i < stateVars.size(); ++i) {
        const size_t outputIdx = static_cast<size_t>(stateVars[i].outputIndex);

        if (outputIdx >= outputs.size()) {
            criticalError("Expected output index %zu for state '%s', but only %zu state outputs are available", outputIdx, stateVars[i].name.c_str(), outputs.size());
        }

        torch::Tensor const& state = outputs[outputIdx];

        if (!state.defined()) {
            criticalError("Output state '%s' (index %zu) is undefined", stateVars[i].name.c_str(), outputIdx);
        }

        states_.push_back(state.contiguous());
    }
}

torch::Tensor ConformerStateManager::trimStates(torch::Tensor const& state, TorchStateVariable const& stateVar) const {
    if (state.dim() < 2) {
        return state;
    }

    // Assume time axis is at index 1
    const int64_t dimSize = state.size(1);

    int64_t sliceStart = 0;
    int64_t sliceEnd   = dimSize;

    if (stateVar.kind == "mhsa") {
        sliceStart = std::max<int64_t>(dimSize - discardSuffixLength_ - attentionContextSize_, 0);
        sliceEnd   = std::max<int64_t>(dimSize - discardSuffixLength_, 0);
    }
    else if (stateVar.kind == "conv") {
        sliceStart = std::max<int64_t>(dimSize - discardSuffixLength_ - convContextSize_, 0);
        sliceEnd   = std::max<int64_t>(dimSize - discardSuffixLength_, 0);
    }
    else {
        return state;
    }

    sliceStart = std::min<int64_t>(sliceStart, dimSize);
    sliceEnd   = std::min<int64_t>(sliceEnd, dimSize);

    return state.slice(1, sliceStart, sliceEnd);
}

}  // namespace Torch