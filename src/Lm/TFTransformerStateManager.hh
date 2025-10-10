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
#ifndef _LM_TRANSFORMER_STATE_MANAGER_HH
#define _LM_TRANSFORMER_STATE_MANAGER_HH

#include "AbstractStateManager.hh"

#include <Tensorflow/Graph.hh>
#include <Tensorflow/Tensor.hh>

#include <Bliss/Symbol.hh>

namespace Lm {

template<typename T>
class TFTransformerStateManager : public AbstractStateManager<Tensorflow::Tensor, Tensorflow::Variable> {
public:
    using Precursor = AbstractStateManager<Tensorflow::Tensor, Tensorflow::Variable>;

    static const Core::ParameterInt  paramMaxHistoryLength;
    static const Core::ParameterBool paramAlwaysIncludeFirstTokenState;

    TFTransformerStateManager(Core::Configuration const& config);
    virtual ~TFTransformerStateManager() = default;

    virtual bool requiresAllParentStates() const;

    virtual HistoryState              initialState(StateVariables const& vars, CompressedVectorFactory<float> const& vector_factory);
    virtual void                      mergeStates(StateVariables const&                   vars,
                                                  std::vector<size_t>&                    prefix_lengths,
                                                  std::vector<HistoryState const*> const& prefix_states,
                                                  FeedDict&                               feed_dict,
                                                  TargetList&                             targets);
    virtual std::vector<HistoryState> splitStates(StateVariables const&                  vars,
                                                  std::vector<size_t>&                   suffix_lengths,
                                                  std::vector<Tensorflow::Tensor> const& state_tensors,
                                                  CompressedVectorFactory<float> const&  vector_factory);

protected:
    const size_t maxHistory_;
    const bool   alwaysIncludeFirstTokenState_;
};

template<typename T>
class TFTransformerStateManagerWithCommonPrefix : public TFTransformerStateManager<T> {
public:
    using Precursor = TFTransformerStateManager<T>;

    static const Core::ParameterString paramVarName;
    static const Core::ParameterString paramCommonPrefixInitialValue;
    static const Core::ParameterString paramCommonPrefixInitializer;
    static const Core::ParameterBool   paramCachePrefix;
    static const Core::ParameterInt    paramMinBatchSize;
    static const Core::ParameterInt    paramMinCommonPrefixLength;
    static const Core::ParameterInt    paramMaxCommonPrefixLength;

    TFTransformerStateManagerWithCommonPrefix(Core::Configuration const& config);
    virtual ~TFTransformerStateManagerWithCommonPrefix() = default;

    virtual void mergeStates(typename Precursor::StateVariables const&                   vars,
                             std::vector<size_t>&                                        prefix_lengths,
                             std::vector<typename Precursor::HistoryState const*> const& prefix_states,
                             typename Precursor::FeedDict&                               feed_dict,
                             typename Precursor::TargetList&                             targets);

protected:
    std::unordered_map<std::string, std::pair<std::string, std::string>> varMap_;

    const bool   cachePrefix_;
    const size_t minBatchSize_;
    const size_t minCommonPrefixLength_;
    const size_t maxCommonPrefixLength_;

    std::vector<typename Precursor::HistoryState const*> previousPrefix_;
};

// inline implementations

template<typename T>
inline TFTransformerStateManager<T>::TFTransformerStateManager(Core::Configuration const& config)
        : Precursor(config),
          maxHistory_(paramMaxHistoryLength(config)),
          alwaysIncludeFirstTokenState_(paramAlwaysIncludeFirstTokenState(config)) {
}

template<typename T>
inline bool TFTransformerStateManager<T>::requiresAllParentStates() const {
    return true;
}

template<typename T>
inline TFTransformerStateManagerWithCommonPrefix<T>::TFTransformerStateManagerWithCommonPrefix(Core::Configuration const& config)
        : Precursor(config),
          cachePrefix_(paramCachePrefix(config)),
          minBatchSize_(paramMinBatchSize(config)),
          minCommonPrefixLength_(paramMinCommonPrefixLength(config)),
          maxCommonPrefixLength_(paramMaxCommonPrefixLength(config)) {
    Core::Configuration varmap_config = this->select("var-map");
    for (size_t i = 0ul; true; i++) {
        Core::Configuration idx_config(varmap_config, std::string("item-") + std::to_string(i));
        std::string         var_name      = paramVarName(idx_config);
        std::string         initial_value = paramCommonPrefixInitialValue(idx_config);
        std::string         initializer   = paramCommonPrefixInitializer(idx_config);
        if (not var_name.empty()) {
            varMap_[var_name] = std::make_pair<>(initial_value, initializer);
        }
        else {
            break;
        }
    }
}

}  // namespace Lm

#endif  // _LM_TRANSFORMER_STATE_MANAGER_HH
