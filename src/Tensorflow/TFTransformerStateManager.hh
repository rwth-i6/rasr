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
#ifndef _TF_TRANSFORMER_STATE_MANAGER_HH
#define _TF_TRANSFORMER_STATE_MANAGER_HH

#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <valarray>

#include <Tensorflow/Graph.hh>
#include <Tensorflow/Tensor.hh>

#include <Bliss/Symbol.hh>

#include <Nn/TransformerStateManager.hh>

namespace Tensorflow {

template<typename T>
class TFTransformerStateManager : public Nn::TransformerStateManager<T, Tensorflow::Tensor, Tensorflow::Variable> {
public:
    using Precursor = Nn::TransformerStateManager<T, Tensorflow::Tensor, Tensorflow::Variable>;

    TFTransformerStateManager(Core::Configuration const& config);
    virtual ~TFTransformerStateManager() = default;

protected:
    virtual void extendFeedDict(typename Precursor::FeedDict& feed_dict, Tensorflow::Variable const& state_var, Tensorflow::Tensor& var);
    virtual void extendTargets(typename Precursor::TargetList& targets, Tensorflow::Variable const& state_var);
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
        : Precursor(config) {
}

template<typename T>
void TFTransformerStateManager<T>::extendFeedDict(typename Tensorflow::TFTransformerStateManager<T>::Precursor::FeedDict& feed_dict, Tensorflow::Variable const& state_var, Tensorflow::Tensor& var) {
    feed_dict.emplace_back(state_var.initial_value_name, var);
}

template<typename T>
void TFTransformerStateManager<T>::extendTargets(typename Tensorflow::TFTransformerStateManager<T>::Precursor::TargetList& targets, Tensorflow::Variable const& state_var) {
    targets.emplace_back(state_var.initializer_name);
}

template<typename T>
const Core::ParameterString TFTransformerStateManagerWithCommonPrefix<T>::paramVarName("var-name", "the name of the original state variable", "");

template<typename T>
const Core::ParameterString TFTransformerStateManagerWithCommonPrefix<T>::paramCommonPrefixInitialValue("common-prefix-initial-value",
                                                                                                        "the name the initial-value of the corresponding common-prefix variable",
                                                                                                        "");

template<typename T>
const Core::ParameterString TFTransformerStateManagerWithCommonPrefix<T>::paramCommonPrefixInitializer("common-prefix-initializer",
                                                                                                       "the name of the initializer of the corresponding common-prefix variable",
                                                                                                       "");

template<typename T>
const Core::ParameterBool TFTransformerStateManagerWithCommonPrefix<T>::paramCachePrefix("cache-prefix",
                                                                                         "whether to reuse the prefix if it's the same",
                                                                                         false);

template<typename T>
const Core::ParameterInt TFTransformerStateManagerWithCommonPrefix<T>::paramMinBatchSize("min-batch-size",
                                                                                         "for batches smaller than the given size we set the common-prefix length to 0",
                                                                                         2, 0);

template<typename T>
const Core::ParameterInt TFTransformerStateManagerWithCommonPrefix<T>::paramMinCommonPrefixLength("min-common-prefix-length",
                                                                                                  "if the common-prefix length is smaller than this value, set it to 0",
                                                                                                  1, 0);

template<typename T>
const Core::ParameterInt TFTransformerStateManagerWithCommonPrefix<T>::paramMaxCommonPrefixLength("max-common-prefix-length",
                                                                                                  "Truncate the common prefix to this length. Observes always-include-first-token-state.",
                                                                                                  std::numeric_limits<int>::max(), 0);

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

template<typename T>
void TFTransformerStateManagerWithCommonPrefix<T>::mergeStates(typename Precursor::StateVariables const&                   vars,
                                                               std::vector<size_t>&                                        prefix_lengths,
                                                               std::vector<typename Precursor::HistoryState const*> const& prefix_states,
                                                               typename Precursor::FeedDict&                               feed_dict,
                                                               typename Precursor::TargetList&                             targets) {
    std::vector<size_t> original_prefix_lengths(prefix_lengths);
    std::vector<size_t> batch_offsets;
    batch_offsets.reserve(prefix_lengths.size() + 1ul);
    batch_offsets.push_back(0ul);

    size_t max_prefix = 0ul;
    size_t min_prefix = std::numeric_limits<size_t>::max();
    for (size_t len : prefix_lengths) {
        max_prefix = std::max(max_prefix, len);
        min_prefix = std::min(min_prefix, len);
        batch_offsets.push_back(batch_offsets.back() + len);
    }

    bool                                                 reset_common_prefix = true;
    std::vector<typename Precursor::HistoryState const*> current_prefix;
    size_t                                               common_prefix_length = 0ul;
    if (prefix_lengths.size() >= minBatchSize_) {
        for (size_t p = 0ul; p < min_prefix; p++) {
            typename Precursor::HistoryState const* hs = prefix_states[batch_offsets[0] + p];
            for (size_t b = 1ul; b < prefix_lengths.size(); b++) {
                if (hs != prefix_states[batch_offsets[b] + p]) {
                    goto common_prefix_length_computation_finished;
                }
            }
            current_prefix.push_back(hs);
            common_prefix_length = p + 1;
        }
    }
common_prefix_length_computation_finished:
    if (common_prefix_length < minCommonPrefixLength_) {
        common_prefix_length = 0ul;
        current_prefix.clear();
    }

    if (cachePrefix_ and current_prefix.size() >= previousPrefix_.size()) {
        bool previous_prefix_valid = true;
        for (size_t p = 0ul; p < previousPrefix_.size(); p++) {
            if (current_prefix[p] != previousPrefix_[p]) {
                previous_prefix_valid = false;
                break;
            }
        }
        reset_common_prefix = not(previous_prefix_valid and current_prefix.size() == previousPrefix_.size());
    }
    if (reset_common_prefix) {
        std::swap(previousPrefix_, current_prefix);
    }

    max_prefix -= common_prefix_length;
    for (size_t& len : prefix_lengths) {
        len -= common_prefix_length;
    }

    feed_dict.reserve(vars.size() * 2);
    targets.reserve(vars.size() * 2);

    for (size_t v = 0ul; v < vars.size(); v++) {
        auto const& var = vars[v];
        require_ge(var.shape.size(), 2);

        std::vector<Tensorflow::int64> tensor_dim(var.shape.size());
        tensor_dim[0]                      = prefix_lengths.size();
        size_t                batch_stride = 1ul;
        size_t                time_dim     = static_cast<size_t>(-1);
        std::valarray<size_t> sizes(var.shape.size() - 1);
        std::valarray<size_t> strides(var.shape.size() - 1);

        for (size_t d = 1ul; d < var.shape.size(); d++) {
            bool is_time_dim = var.shape[d] < 0l;
            tensor_dim[d]    = is_time_dim ? max_prefix : var.shape[d];
            sizes[d - 1ul]   = is_time_dim ? std::min(max_prefix, 1ul) : var.shape[d];
            if (is_time_dim) {
                time_dim = d - 1ul;
            }
            else {
                require_eq(var.shape[d], var.shape[d]);
            }
            batch_stride *= tensor_dim[d];
        }
        require_lt(time_dim, static_cast<size_t>(-1));

        strides[strides.size() - 1ul] = 1ul;
        for (size_t d = strides.size() - 1ul; d > 0ul; d--) {
            strides[d - 1ul] = tensor_dim[d + 1] * strides[d];
        }

        Tensorflow::Tensor var_tensor = Tensorflow::Tensor::zeros<T>(tensor_dim);

        size_t state_offset = 0ul;
        for (size_t b = 0ul; b < prefix_lengths.size(); b++) {
            size_t prefix_length = prefix_lengths[b];
            size_t prefix_offset = original_prefix_lengths[b] - prefix_length;
            for (size_t p = 0ul; p < prefix_length; p++) {
                std::gslice         slice(b * batch_stride + (max_prefix - prefix_length + p) * strides[time_dim], sizes, strides);
                ContiguousBlockInfo block_info(slice);
                size_t              idx = state_offset + prefix_offset + p;
                detail::uncompress(prefix_states[idx]->at(v).get(), var_tensor.data<T>(), block_info);
            }
            state_offset += original_prefix_lengths[b];
        }
        feed_dict.emplace_back(vars[v].initial_value_name, var_tensor);
        targets.emplace_back(vars[v].initializer_name);

        if (reset_common_prefix) {
            const size_t truncated_prefix_length = std::min(common_prefix_length, maxCommonPrefixLength_);
            const size_t common_prefix_offset    = common_prefix_length - truncated_prefix_length;

            tensor_dim[0]                 = 1ul;
            tensor_dim[time_dim + 1ul]    = truncated_prefix_length;
            sizes[time_dim]               = std::min(truncated_prefix_length, 1ul);
            strides[strides.size() - 1ul] = 1ul;
            for (size_t d = strides.size() - 1ul; d > 0ul; d--) {
                strides[d - 1ul] = tensor_dim[d + 1] * strides[d];
            }

            Tensorflow::Tensor common_prefix_tensor = Tensorflow::Tensor::zeros<T>(tensor_dim);

            for (size_t p = 0ul; p < truncated_prefix_length; p++) {
                size_t pos = p;
                if (not Precursor::alwaysIncludeFirstTokenState_ or p != 0) {
                    pos += common_prefix_offset;
                }
                std::gslice         slice(p * strides[time_dim], sizes, strides);
                ContiguousBlockInfo block_info(slice);
                detail::uncompress(prefix_states[pos]->at(v).get(), common_prefix_tensor.data<T>(), block_info);
            }
            auto iter = varMap_.find(vars[v].name);
            require(iter != varMap_.end());
            feed_dict.emplace_back(iter->second.first, common_prefix_tensor);
            targets.emplace_back(iter->second.second);
        }
    }
}

}  // namespace Tensorflow

#endif  // _TF_TRANSFORMER_STATE_MANAGER_HH
