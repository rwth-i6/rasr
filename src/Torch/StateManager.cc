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

#include "StateManager.hh"
#include "ConformerStateManager.hh"
#include "DummyStateManager.hh"

namespace {
enum StateManagerType : int {
    Dummy,
    Conformer,
};

const Core::Choice stateManagerTypeChoice(
        "dummy", StateManagerType::Dummy,
        "conformer", StateManagerType::Conformer,
        Core::Choice::endMark());

const Core::ParameterChoice stateManagerTypeParam(
        "type", &stateManagerTypeChoice, "type of stateManager", StateManagerType::Dummy);

}  // namespace

namespace Torch {

std::unique_ptr<StateManager> StateManager::create(Core::Configuration const& config) {
    switch (stateManagerTypeParam(config)) {
        case Conformer:
            return std::unique_ptr<StateManager>(new ConformerStateManager(config));
        case Dummy:
        default:
            return std::unique_ptr<StateManager>(new DummyStateManager(config));
    }
    return std::unique_ptr<StateManager>();
}

StateManager::StateManager(Core::Configuration const& config)
        : Precursor(config) {}

}  // namespace Torch