#ifndef _NN_TRANSFORMER_STATE_MANAGER_HH
#define _NN_TRANSFORMER_STATE_MANAGER_HH

#include <algorithm>
#include <limits>
#include <numeric>
#include <thread>
#include <type_traits>
#include <valarray>

#include "AbstractStateManager.hh"
#include "CompressedVector.hh"
#include "DummyCompressedVectorFactory.hh"
#include "FixedQuantizationCompressedVectorFactory.hh"

namespace Nn {

namespace detail {

template<typename B>
Nn::CompressedVectorPtr<float> compress(float const* data, B const& b, Nn::CompressedVectorFactory<float> const& vector_factory, Nn::CompressionParameters const* parameters) {
    return vector_factory.compress(data, b, parameters);
}

template<typename B>
Nn::CompressedVectorPtr<float> compress(int16_t const* data, B const& b, Nn::CompressedVectorFactory<float> const& vector_factory, Nn::CompressionParameters const* parameters) {
    Nn::QuantizedFloatVector16Bits* res = new Nn::QuantizedFloatVector16Bits(0.001);
    res->store(data, b);
    return Nn::CompressedVectorPtr<float>(res);
}

template<typename B>
Nn::CompressedVectorPtr<float> compress(int8_t const* data, B const& b, Nn::CompressedVectorFactory<float> const& vector_factory, Nn::CompressionParameters const* parameters) {
    Nn::QuantizedFloatVector8Bits* res = new Nn::QuantizedFloatVector8Bits(0.05);
    res->store(data, b);
    return Nn::CompressedVectorPtr<float>(res);
}

template<typename B>
void uncompress(Nn::CompressedVector<float> const* vec, float* dst, B const& b) {
    vec->uncompress(dst, b);
}

template<typename B>
void uncompress(Nn::CompressedVector<float> const* vec, int16_t* dst, B const& b) {
    Nn::QuantizedFloatVector16Bits const* qvec = dynamic_cast<Nn::QuantizedFloatVector16Bits const*>(vec);
    require(qvec != nullptr);
    qvec->load(dst, b);
}

template<typename B>
void uncompress(Nn::CompressedVector<float> const* vec, int8_t* dst, B const& b) {
    Nn::QuantizedFloatVector8Bits const* qvec = dynamic_cast<Nn::QuantizedFloatVector8Bits const*>(vec);
    require(qvec != nullptr);
    qvec->load(dst, b);
}

}  // namespace detail

template<typename T, typename value_t, typename state_variable_t>
class TransformerStateManager : public AbstractStateManager<value_t, state_variable_t> {
public:
    using Precursor = AbstractStateManager<value_t, state_variable_t>;

    static const Core::ParameterInt  paramMaxHistoryLength;
    static const Core::ParameterBool paramAlwaysIncludeFirstTokenState;
    static const Core::ParameterInt  paramNumMergeThreads;

    TransformerStateManager(Core::Configuration const& config);
    virtual ~TransformerStateManager() = default;

    virtual bool requiresAllParentStates() const;

    virtual typename Precursor::HistoryState initialState(typename Precursor::StateVariables const& vars, CompressedVectorFactory<float> const& vector_factory);

    virtual void mergeStates(typename Precursor::StateVariables const&                   vars,
                             std::vector<size_t>&                                        prefix_lengths,
                             std::vector<typename Precursor::HistoryState const*> const& prefix_states,
                             typename Precursor::FeedDict&                               feed_dict,
                             typename Precursor::TargetList&                             targets);

    virtual std::vector<typename Precursor::HistoryState> splitStates(
            typename Precursor::StateVariables const& vars,
            std::vector<size_t>&                      suffix_lengths,
            std::vector<value_t> const&               state_tensors,
            CompressedVectorFactory<float> const&     vector_factory);

protected:
    virtual void extendFeedDict(typename Precursor::FeedDict& feed_dict, state_variable_t const& state_var, value_t& var) = 0;
    virtual void extendTargets(typename Precursor::TargetList& targets, state_variable_t const& state_var)                = 0;

    const size_t maxHistory_;
    const bool   alwaysIncludeFirstTokenState_;
    const size_t numMergeThreads_;
};

template<typename T, typename value_t, typename state_variable_t>
const Core::ParameterInt TransformerStateManager<T, value_t, state_variable_t>::paramMaxHistoryLength("max-history", "maximum length of the history to feed to the transformer", std::numeric_limits<int>::max(), 0);

template<typename T, typename value_t, typename state_variable_t>
const Core::ParameterBool TransformerStateManager<T, value_t, state_variable_t>::paramAlwaysIncludeFirstTokenState("always-include-first-token-state", "wether to always include the state of the first token, even if history is restricted by max-history", false);

template<typename T, typename value_t, typename state_variable_t>
const Core::ParameterInt TransformerStateManager<T, value_t, state_variable_t>::paramNumMergeThreads(
        "num-merge-threads",
        "number of threads that copy the prefix states into the batched input tensors (one tensor per state variable each); 1 = no extra threads",
        1,
        1);

template<typename T, typename value_t, typename state_variable_t>
TransformerStateManager<T, value_t, state_variable_t>::TransformerStateManager(Core::Configuration const& config)
        : Precursor(config),
          maxHistory_(paramMaxHistoryLength(config)),
          alwaysIncludeFirstTokenState_(paramAlwaysIncludeFirstTokenState(config)),
          numMergeThreads_(paramNumMergeThreads(config)) {
}

template<typename T, typename value_t, typename state_variable_t>
bool TransformerStateManager<T, value_t, state_variable_t>::requiresAllParentStates() const {
    return true;
}

template<typename T, typename value_t, typename state_variable_t>
typename TransformerStateManager<T, value_t, state_variable_t>::Precursor::HistoryState TransformerStateManager<T, value_t, state_variable_t>::initialState(
        typename Precursor::StateVariables const& vars,
        CompressedVectorFactory<float> const&     vector_factory) {
    typename Precursor::HistoryState result;
    result.reserve(vars.size());

    std::vector<float> vec(0, 0.0f);
    auto               compression_param_estimator = vector_factory.getEstimator();
    compression_param_estimator->accumulate(vec.data(), vec.size());
    auto compression_params = compression_param_estimator->estimate();

    for (size_t i = 0ul; i < vars.size(); i++) {
        result.emplace_back(detail::compress<T>(vec.data(), vec.size(), vector_factory, compression_params.get()));
    }

    return result;
}

template<typename T, typename value_t, typename state_variable_t>
void TransformerStateManager<T, value_t, state_variable_t>::mergeStates(
        typename TransformerStateManager<T, value_t, state_variable_t>::Precursor::StateVariables const&                   vars,
        std::vector<size_t>&                                                                                               prefix_lengths,
        std::vector<typename TransformerStateManager<T, value_t, state_variable_t>::Precursor::HistoryState const*> const& prefix_states,
        typename TransformerStateManager<T, value_t, state_variable_t>::Precursor::FeedDict&                               feed_dict,
        typename TransformerStateManager<T, value_t, state_variable_t>::Precursor::TargetList&                             targets) {
    std::vector<size_t> original_prefix_lengths(prefix_lengths);

    size_t max_prefix = 0ul;
    for (size_t& len : prefix_lengths) {
        len        = std::min(len, maxHistory_);
        max_prefix = std::max(max_prefix, len);
    }

    feed_dict.reserve(vars.size());
    targets.reserve(vars.size());

    // Padding (zeros before shorter prefixes) is only needed if the prefixes differ in length,
    // otherwise every element of the merged tensors is overwritten below
    bool needs_padding = std::any_of(prefix_lengths.begin(), prefix_lengths.end(), [max_prefix](size_t len) { return len != max_prefix; });

    // Layout of the merged tensor of one state variable
    struct MergeLayout {
        std::valarray<size_t> sizes;
        std::valarray<size_t> strides;
        size_t                batch_stride;
        size_t                time_stride;
        std::vector<size_t>   block_offsets;  // offsets of the contiguous blocks of one time step, relative to its start
        size_t                block_size;
        size_t                step_size;  // number of elements of one time step
    };

    // Compute the layouts and allocate the merged tensors sequentially, the copying below can then run in parallel
    std::vector<MergeLayout> layouts(vars.size());
    std::vector<value_t>     var_tensors(vars.size());
    for (size_t v = 0ul; v < vars.size(); v++) {
        auto const& var = vars[v];
        require_ge(var.shape.size(), 2);

        std::vector<s64> tensor_dim(var.shape.size());
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

        var_tensors[v] = needs_padding ? value_t::template zeros<T>(tensor_dim) : value_t::template createEmpty<T>(tensor_dim);

        // All time steps have the same layout of contiguous blocks, only the start offset differs
        auto&               layout = layouts[v];
        ContiguousBlockInfo step_layout(std::gslice(0ul, sizes, strides));
        layout.block_offsets.resize(step_layout.numBlocks());
        for (size_t i = 0ul; i < layout.block_offsets.size(); i++) {
            layout.block_offsets[i] = step_layout.blockOffset(i);
        }
        layout.block_size   = step_layout.blockSize();
        layout.step_size    = step_layout.totalSize();
        layout.batch_stride = batch_stride;
        layout.time_stride  = strides[time_dim];
        layout.sizes        = sizes;
        layout.strides      = strides;
    }

    // Copy the prefix states of state variable `v` into its merged tensor
    auto merge_var = [&](size_t v) {
        auto const& layout      = layouts[v];
        T*          tensor_data = var_tensors[v].template data<T>();

        size_t state_offset = 0ul;
        for (size_t b = 0ul; b < prefix_lengths.size(); b++) {
            size_t prefix_length = prefix_lengths[b];
            size_t prefix_offset = original_prefix_lengths[b] - prefix_length;
            for (size_t p = 0ul; p < prefix_length; p++) {
                size_t start = b * layout.batch_stride + (max_prefix - prefix_length + p) * layout.time_stride;
                size_t idx   = state_offset;
                if (not alwaysIncludeFirstTokenState_ or p != 0ul) {
                    idx += prefix_offset + p;
                }
                auto const* state_vector = prefix_states[idx]->at(v).get();

                // Fast path if the stored values already have type T: copy the blocks directly
                T const* src = nullptr;
                if constexpr (std::is_same<T, float>::value) {
                    if (auto const* uncompressed = dynamic_cast<UncompressedVector<float> const*>(state_vector)) {
                        src = uncompressed->data();
                    }
                }
                else {
                    if (auto const* quantized = dynamic_cast<QuantizedFloatVectorFixedBits<T> const*>(state_vector)) {
                        src = quantized->data().data();
                    }
                }
                if (src != nullptr) {
                    require_eq(state_vector->size(), layout.step_size);
                    for (size_t i = 0ul; i < layout.block_offsets.size(); i++) {
                        std::copy(src + i * layout.block_size, src + (i + 1ul) * layout.block_size, tensor_data + start + layout.block_offsets[i]);
                    }
                    continue;
                }

                std::gslice         slice(start, layout.sizes, layout.strides);
                ContiguousBlockInfo block_info(slice);
                detail::uncompress(state_vector, tensor_data, block_info);
            }
            state_offset += original_prefix_lengths[b];
        }
    };

    // Each thread merges every num_threads-th state variable, i.e. writes only into its own tensors
    size_t num_threads = std::min(numMergeThreads_, vars.size());
    if (num_threads > 1ul) {
        std::vector<std::thread> threads;
        threads.reserve(num_threads - 1ul);
        for (size_t t = 1ul; t < num_threads; t++) {
            threads.emplace_back([&merge_var, &vars, t, num_threads]() {
                for (size_t v = t; v < vars.size(); v += num_threads) {
                    merge_var(v);
                }
            });
        }
        for (size_t v = 0ul; v < vars.size(); v += num_threads) {
            merge_var(v);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }
    else {
        for (size_t v = 0ul; v < vars.size(); v++) {
            merge_var(v);
        }
    }

    for (size_t v = 0ul; v < vars.size(); v++) {
        extendFeedDict(feed_dict, vars[v], var_tensors[v]);
        extendTargets(targets, vars[v]);
    }
}

template<typename T, typename value_t, typename state_variable_t>
std::vector<typename TransformerStateManager<T, value_t, state_variable_t>::Precursor::HistoryState> TransformerStateManager<T, value_t, state_variable_t>::splitStates(
        typename TransformerStateManager<T, value_t, state_variable_t>::Precursor::StateVariables const& vars,
        std::vector<size_t>&                                                                             suffix_lengths,
        std::vector<value_t> const&                                                                      state_tensors,
        CompressedVectorFactory<float> const&                                                            vector_factory) {
    require_eq(vars.size(), state_tensors.size());

    size_t max_suffix = *std::max_element(suffix_lengths.begin(), suffix_lengths.end());
    size_t sum_suffix = std::accumulate(suffix_lengths.begin(), suffix_lengths.end(), 0ul);

    std::vector<typename Precursor::HistoryState> result(sum_suffix);

    for (size_t v = 0ul; v < vars.size(); v++) {
        auto const&    var    = vars[v];
        value_t const& tensor = state_tensors[v];

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
                    compression_param_estimator->accumulate(tensor.template data<float>(), block_info);
                }
                auto compression_params = compression_param_estimator->estimate();
                result[output_idx].emplace_back(detail::compress<ContiguousBlockInfo>(tensor.template data<T>(), block_info, vector_factory, compression_params.get()));
                output_idx += 1ul;
            }
        }
    }

    return result;
}

}  // namespace Nn

#endif  // _NN_TRANSFORMER_STATE_MANAGER_HH
