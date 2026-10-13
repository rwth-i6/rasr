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

#include "DummyStateManager.hh"

namespace Torch {

DummyStateManager::DummyStateManager(Core::Configuration const& config)
        : Precursor(config) {
}

void DummyStateManager::setInitialStates(std::vector<TorchStateVariable> const& stateVars) {
}

void DummyStateManager::extendInputs(InputList& inputs, std::vector<TorchStateVariable> const& stateVars) {
}

void DummyStateManager::updateStates(std::vector<torch::Tensor> const& outputs, std::vector<TorchStateVariable> const& stateVars) {
}

}  // namespace Torch