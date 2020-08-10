/** Copyright 2020 RWTH Aachen University. All rights reserved.
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
#ifndef _LM_STATE_MANAGER_HH
#define _LM_STATE_MANAGER_HH

#include <Core/Component.hh>
#include <Tensorflow/Graph.hh>
#include <Tensorflow/Tensor.hh>

#include "CompressedVector.hh"

namespace Lm {

class StateManager : public Core::Component {
public:
    using Precursor      = Core::Component;
    using FeedDict       = std::vector<std::pair<std::string, Tensorflow::Tensor>>;
    using TargetList     = std::vector<std::string>;
    using StateVariables = std::vector<Tensorflow::Variable>;
    using HistoryState   = std::vector<CompressedVectorPtr<float>>;

    StateManager(Core::Configuration const& config);
    virtual ~StateManager() = default;

    virtual bool requiresAllParentStates() const;

    virtual HistoryState              initialState(StateVariables const& vars, CompressedVectorFactory<float> const& vector_factory) = 0;
    virtual void                      mergeStates(StateVariables const&                   vars,
                                                  std::vector<size_t>&                    prefix_lengths,
                                                  std::vector<HistoryState const*> const& prefix_states,
                                                  FeedDict&                               feed_dict,
                                                  TargetList&                             targets)                                                               = 0;
    virtual std::vector<HistoryState> splitStates(StateVariables const&                  vars,
                                                  std::vector<size_t>&                   suffix_lengths,
                                                  std::vector<Tensorflow::Tensor> const& state_tensors,
                                                  CompressedVectorFactory<float> const&  vector_factory)                              = 0;
};

// inline implementations

inline bool StateManager::requiresAllParentStates() const {
    return false;
}

inline StateManager::StateManager(Core::Configuration const& config)
        : Precursor(config) {
}

}  // namespace Lm

#endif  // _LM_STATE_MANAGER_HH
