#include "TransformerStateManager.hh"

#include "FixedQuantizationCompressedVectorFactory.hh"

namespace {

void print_slice(std::gslice const& slice) {
    std::cerr << "start: " << slice.start() << " sizes: [";
    for (auto s : slice.size()) {
        std::cerr << s << ",";
    }
    std::cerr << "] strides: [";
    for (auto s : slice.stride()) {
        std::cerr << s << ",";
    }
    std::cerr << "]" << std::endl;
}

template<typename B>
Lm::CompressedVectorPtr<float> compress(float const* data, B const& b, Lm::CompressedVectorFactory<float> const& vector_factory, Lm::CompressionParameters const* parameters) {
    return vector_factory.compress(data, b, parameters);
}

template<typename B>
Lm::CompressedVectorPtr<float> compress(int16_t const* data, B const& b, Lm::CompressedVectorFactory<float> const& vector_factory, Lm::CompressionParameters const* parameters) {
    Lm::QuantizedFloatVector16Bits* res = new Lm::QuantizedFloatVector16Bits(0.001);
    res->store(data, b);
    return Lm::CompressedVectorPtr<float>(res);
}

template<typename B>
Lm::CompressedVectorPtr<float> compress(int8_t const* data, B const& b, Lm::CompressedVectorFactory<float> const& vector_factory, Lm::CompressionParameters const* parameters) {
    Lm::QuantizedFloatVector8Bits* res = new Lm::QuantizedFloatVector8Bits(0.05);
    res->store(data, b);
    return Lm::CompressedVectorPtr<float>(res);
}

template<typename B>
void uncompress(Lm::CompressedVector<float> const* vec, float* dst, B const& b) {
    vec->uncompress(dst, b);
}

template<typename B>
void uncompress(Lm::CompressedVector<float> const* vec, int16_t* dst, B const& b) {
    Lm::QuantizedFloatVector16Bits const* qvec = dynamic_cast<Lm::QuantizedFloatVector16Bits const*>(vec);
    require(qvec != nullptr);
    qvec->load(dst, b);
}

template<typename B>
void uncompress(Lm::CompressedVector<float> const* vec, int8_t* dst, B const& b) {
    Lm::QuantizedFloatVector8Bits const* qvec = dynamic_cast<Lm::QuantizedFloatVector8Bits const*>(vec);
    require(qvec != nullptr);
    qvec->load(dst, b);
}

}  // namespace

namespace Lm {

/* ----------------------------------- TransformerStateManager ---------------------------------- */

template<typename T>
const Core::ParameterInt          TransformerStateManager<T>::paramMaxHistoryLength("max-history",
                                                                                    "maximum length of the history to feed to the transformer",
                                                                                    std::numeric_limits<int>::max(),
                                                                                    0);
template const Core::ParameterInt TransformerStateManager<float>::paramMaxHistoryLength;
template const Core::ParameterInt TransformerStateManager<int16_t>::paramMaxHistoryLength;
template const Core::ParameterInt TransformerStateManager<int8_t>::paramMaxHistoryLength;

template<typename T>
const Core::ParameterBool          TransformerStateManager<T>::paramAlwaysIncludeFirstTokenState("always-include-first-token-state",
                                                                                                 "wether to always include the state of the first token, even if history is restricted by max-history",
                                                                                                 false);
template const Core::ParameterBool TransformerStateManager<float>::paramAlwaysIncludeFirstTokenState;
template const Core::ParameterBool TransformerStateManager<int16_t>::paramAlwaysIncludeFirstTokenState;
template const Core::ParameterBool TransformerStateManager<int8_t>::paramAlwaysIncludeFirstTokenState;

template<typename T>
typename TransformerStateManager<T>::HistoryState TransformerStateManager<T>::initialState(StateVariables const& vars, CompressedVectorFactory<float> const& vector_factory) {
    TransformerStateManager::HistoryState result;
    result.reserve(vars.size());

    std::vector<float> vec(0, 0.0f);
    auto               compression_param_estimator = vector_factory.getEstimator();
    compression_param_estimator->accumulate(vec.data(), vec.size());
    auto compression_params = compression_param_estimator->estimate();

    for (size_t i = 0ul; i < vars.size(); i++) {
        result.emplace_back(compress<T>(vec.data(), vec.size(), vector_factory, compression_params.get()));
    }

    return result;
}
template typename TransformerStateManager<float>::HistoryState   TransformerStateManager<float>::initialState(StateVariables const& vars, CompressedVectorFactory<float> const& vector_factory);
template typename TransformerStateManager<int16_t>::HistoryState TransformerStateManager<int16_t>::initialState(StateVariables const& vars, CompressedVectorFactory<float> const& vector_factory);
template typename TransformerStateManager<int8_t>::HistoryState  TransformerStateManager<int8_t>::initialState(StateVariables const& vars, CompressedVectorFactory<float> const& vector_factory);

template<typename T>
void TransformerStateManager<T>::mergeStates(StateVariables const&                   vars,
                                             std::vector<size_t>&                    prefix_lengths,
                                             std::vector<HistoryState const*> const& prefix_states,
                                             FeedDict&                               feed_dict,
                                             TargetList&                             targets) {
    std::vector<size_t> original_prefix_lengths(prefix_lengths);

    size_t max_prefix = 0ul;
    for (size_t& len : prefix_lengths) {
        len        = std::min(len, maxHistory_);
        max_prefix = std::max(max_prefix, len);
    }

    feed_dict.reserve(vars.size());
    targets.reserve(vars.size());

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
            if (is_time_dim) {
                time_dim = d - 1ul;
            }
            else {
                require_eq(var.shape[d], var.shape[d]);
            }
            sizes[d - 1ul] = is_time_dim ? std::min(max_prefix, 1ul) : var.shape[d];
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
                size_t              idx = state_offset;
                if (not alwaysIncludeFirstTokenState_ or p != 0ul) {
                    idx += prefix_offset + p;
                }
                uncompress(prefix_states[idx]->at(v).get(), var_tensor.data<T>(), block_info);
            }
            state_offset += original_prefix_lengths[b];
        }

        feed_dict.emplace_back(vars[v].initial_value_name, var_tensor);
        targets.emplace_back(vars[v].initializer_name);
    }
}

template void TransformerStateManager<float>::mergeStates(StateVariables const&                   vars,
                                                          std::vector<size_t>&                    prefix_lengths,
                                                          std::vector<HistoryState const*> const& prefix_states,
                                                          FeedDict&                               feed_dict,
                                                          TargetList&                             targets);
template void TransformerStateManager<int16_t>::mergeStates(StateVariables const&                   vars,
                                                            std::vector<size_t>&                    prefix_lengths,
                                                            std::vector<HistoryState const*> const& prefix_states,
                                                            FeedDict&                               feed_dict,
                                                            TargetList&                             targets);
template void TransformerStateManager<int8_t>::mergeStates(StateVariables const&                   vars,
                                                           std::vector<size_t>&                    prefix_lengths,
                                                           std::vector<HistoryState const*> const& prefix_states,
                                                           FeedDict&                               feed_dict,
                                                           TargetList&                             targets);

template<typename T>
std::vector<typename TransformerStateManager<T>::HistoryState>
        TransformerStateManager<T>::splitStates(StateVariables const&                  vars,
                                                std::vector<size_t>&                   suffix_lengths,
                                                std::vector<Tensorflow::Tensor> const& state_tensors,
                                                CompressedVectorFactory<float> const&  vector_factory) {
    require_eq(vars.size(), state_tensors.size());

    size_t max_suffix = *std::max_element(suffix_lengths.begin(), suffix_lengths.end());
    size_t sum_suffix = std::accumulate(suffix_lengths.begin(), suffix_lengths.end(), 0ul);

    std::vector<HistoryState> result(sum_suffix);

    for (size_t v = 0ul; v < vars.size(); v++) {
        auto const& var    = vars[v];
        auto const& tensor = state_tensors[v];

        require_ge(var.shape.size(), 2);

        size_t batch_stride = 1ul;
        size_t time_dim     = static_cast<size_t>(-1);

        std::valarray<size_t> sizes(var.shape.size() - 1);  // ignore first dimension as that is the batch dimension
        std::valarray<size_t> strides(var.shape.size() - 1);
        for (size_t d = 1ul; d < var.shape.size(); d++) {
            bool is_time_dim = var.shape[d] < 0l;
            if (is_time_dim) {
                time_dim = d - 1;
            }
            else {
                require_eq(var.shape[d], tensor.dimSize(d));
            }
            sizes[d - 1] = is_time_dim ? 1ul : var.shape[d];
            batch_stride *= tensor.dimSize(d);
        }
        require_lt(time_dim, static_cast<size_t>(-1));
        size_t max_prefix = tensor.dimSize(time_dim + 1) - max_suffix;

        strides[strides.size() - 1ul] = 1ul;
        for (size_t d = strides.size() - 1ul; d > 0ul; d--) {
            strides[d - 1ul] = tensor.dimSize(d + 1) * strides[d];
        }

        size_t output_idx = 0ul;
        for (size_t b = 0ul; b < suffix_lengths.size(); b++) {
            for (size_t p = 0ul; p < suffix_lengths[b]; p++) {
                std::gslice         slice(b * batch_stride + (max_prefix + p) * strides[time_dim], sizes, strides);
                ContiguousBlockInfo block_info(slice);
                auto                compression_param_estimator = vector_factory.getEstimator();
                if (std::is_same<T, float>::value) {
                    compression_param_estimator->accumulate(tensor.data<float>(), block_info);
                }
                auto compression_params = compression_param_estimator->estimate();
                result[output_idx].emplace_back(compress<ContiguousBlockInfo>(tensor.data<T>(), block_info, vector_factory, compression_params.get()));
                output_idx += 1ul;
            }
        }
    }

    return result;
}

template std::vector<typename TransformerStateManager<float>::HistoryState>
        TransformerStateManager<float>::splitStates(StateVariables const&                  vars,
                                                    std::vector<size_t>&                   suffix_lengths,
                                                    std::vector<Tensorflow::Tensor> const& state_tensors,
                                                    CompressedVectorFactory<float> const&  vector_factory);
template std::vector<typename TransformerStateManager<int16_t>::HistoryState>
        TransformerStateManager<int16_t>::splitStates(StateVariables const&                  vars,
                                                      std::vector<size_t>&                   suffix_lengths,
                                                      std::vector<Tensorflow::Tensor> const& state_tensors,
                                                      CompressedVectorFactory<float> const&  vector_factory);
template std::vector<typename TransformerStateManager<int8_t>::HistoryState>
        TransformerStateManager<int8_t>::splitStates(StateVariables const&                  vars,
                                                     std::vector<size_t>&                   suffix_lengths,
                                                     std::vector<Tensorflow::Tensor> const& state_tensors,
                                                     CompressedVectorFactory<float> const&  vector_factory);

/* ----------------------------------- TransformerStateManagerWithCommonPrefix ---------------------------------- */

template<typename T>
const Core::ParameterString          TransformerStateManagerWithCommonPrefix<T>::paramVarName("var-name", "the name of the original state variable", "");
template const Core::ParameterString TransformerStateManagerWithCommonPrefix<float>::paramVarName;
template const Core::ParameterString TransformerStateManagerWithCommonPrefix<int16_t>::paramVarName;
template const Core::ParameterString TransformerStateManagerWithCommonPrefix<int8_t>::paramVarName;

template<typename T>
const Core::ParameterString          TransformerStateManagerWithCommonPrefix<T>::paramCommonPrefixInitialValue("common-prefix-initial-value", "the name the initial-value of the corresponding common-prefix variable", "");
template const Core::ParameterString TransformerStateManagerWithCommonPrefix<float>::paramCommonPrefixInitialValue;
template const Core::ParameterString TransformerStateManagerWithCommonPrefix<int16_t>::paramCommonPrefixInitialValue;
template const Core::ParameterString TransformerStateManagerWithCommonPrefix<int8_t>::paramCommonPrefixInitialValue;

template<typename T>
const Core::ParameterString          TransformerStateManagerWithCommonPrefix<T>::paramCommonPrefixInitializer("common-prefix-initializer", "the name of the initializer of the corresponding common-prefix variable", "");
template const Core::ParameterString TransformerStateManagerWithCommonPrefix<float>::paramCommonPrefixInitializer;
template const Core::ParameterString TransformerStateManagerWithCommonPrefix<int16_t>::paramCommonPrefixInitializer;
template const Core::ParameterString TransformerStateManagerWithCommonPrefix<int8_t>::paramCommonPrefixInitializer;

template<typename T>
const Core::ParameterBool          TransformerStateManagerWithCommonPrefix<T>::paramCachePrefix("cache-prefix", "wether to reuse the prefix if it's the same", false);
template const Core::ParameterBool TransformerStateManagerWithCommonPrefix<float>::paramCachePrefix;
template const Core::ParameterBool TransformerStateManagerWithCommonPrefix<int16_t>::paramCachePrefix;
template const Core::ParameterBool TransformerStateManagerWithCommonPrefix<int8_t>::paramCachePrefix;

template<typename T>
const Core::ParameterInt          TransformerStateManagerWithCommonPrefix<T>::paramMinBatchSize("min-batch-size",
                                                                                                "for batches smaller than the given size we set the common-prefix length to 0",
                                                                                                2, 0);
template const Core::ParameterInt TransformerStateManagerWithCommonPrefix<float>::paramMinBatchSize;
template const Core::ParameterInt TransformerStateManagerWithCommonPrefix<int16_t>::paramMinBatchSize;
template const Core::ParameterInt TransformerStateManagerWithCommonPrefix<int8_t>::paramMinBatchSize;

template<typename T>
const Core::ParameterInt          TransformerStateManagerWithCommonPrefix<T>::paramMinCommonPrefixLength("min-common-prefix-length",
                                                                                                         "if the common-prefix length is smaller than this value, set it to 0",
                                                                                                         1, 0);
template const Core::ParameterInt TransformerStateManagerWithCommonPrefix<float>::paramMinCommonPrefixLength;
template const Core::ParameterInt TransformerStateManagerWithCommonPrefix<int16_t>::paramMinCommonPrefixLength;
template const Core::ParameterInt TransformerStateManagerWithCommonPrefix<int8_t>::paramMinCommonPrefixLength;

template<typename T>
const Core::ParameterInt          TransformerStateManagerWithCommonPrefix<T>::paramMaxCommonPrefixLength("max-common-prefix-length",
                                                                                                         "Truncate the common prefix to this length. Observes always-include-first-token-state.",
                                                                                                         std::numeric_limits<int>::max(), 0);
template const Core::ParameterInt TransformerStateManagerWithCommonPrefix<float>::paramMaxCommonPrefixLength;
template const Core::ParameterInt TransformerStateManagerWithCommonPrefix<int16_t>::paramMaxCommonPrefixLength;
template const Core::ParameterInt TransformerStateManagerWithCommonPrefix<int8_t>::paramMaxCommonPrefixLength;


template<typename T>
void TransformerStateManagerWithCommonPrefix<T>::mergeStates(typename Precursor::StateVariables const&                   vars,
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
                uncompress(prefix_states[idx]->at(v).get(), var_tensor.data<T>(), block_info);
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
                uncompress(prefix_states[pos]->at(v).get(), common_prefix_tensor.data<T>(), block_info);
            }
            auto iter = varMap_.find(vars[v].name);
            require(iter != varMap_.end());
            feed_dict.emplace_back(iter->second.first, common_prefix_tensor);
            targets.emplace_back(iter->second.second);
        }
    }
}

template void TransformerStateManagerWithCommonPrefix<float>::mergeStates(StateVariables const&                   vars,
                                                                          std::vector<size_t>&                    prefix_lengths,
                                                                          std::vector<HistoryState const*> const& prefix_states,
                                                                          FeedDict&                               feed_dict,
                                                                          TargetList&                             targets);
template void TransformerStateManagerWithCommonPrefix<int16_t>::mergeStates(StateVariables const&                   vars,
                                                                            std::vector<size_t>&                    prefix_lengths,
                                                                            std::vector<HistoryState const*> const& prefix_states,
                                                                            FeedDict&                               feed_dict,
                                                                            TargetList&                             targets);
template void TransformerStateManagerWithCommonPrefix<int8_t>::mergeStates(StateVariables const&                   vars,
                                                                           std::vector<size_t>&                    prefix_lengths,
                                                                           std::vector<HistoryState const*> const& prefix_states,
                                                                           FeedDict&                               feed_dict,
                                                                           TargetList&                             targets);

}  // namespace Lm
