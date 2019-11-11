#include "TransformerStateManager.hh"

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

}

namespace Lm {

/* ----------------------------------- TransformerStateManager ---------------------------------- */

const Core::ParameterInt TransformerStateManager::paramMaxHistoryLength("max-history",
                                                                        "maximum length of the history to feed to the transformer",
                                                                        std::numeric_limits<int>::max(),
                                                                        0);

const Core::ParameterBool TransformerStateManager::paramAlwaysIncludeFirstTokenState("always-include-first-token-state",
                                                                                     "wether to always include the state of the first token, even if history is restricted by max-history",
                                                                                     false);


TransformerStateManager::HistoryState TransformerStateManager::initialState(StateVariables const& vars, CompressedVectorFactory<float> const& vector_factory) {
    TransformerStateManager::HistoryState result;
    result.reserve(vars.size());

    std::vector<float> vec(0, 0.0f);
    auto               compression_param_estimator = vector_factory.getEstimator();
    compression_param_estimator->accumulate(vec.data(), vec.size());
    auto compression_params = compression_param_estimator->estimate();

    for (size_t i = 0ul; i < vars.size(); i++) {
        result.emplace_back(vector_factory.compress(vec.data(), vec.size(), compression_params.get()));
    }

    return result;
}

TransformerStateManager::FeedDict TransformerStateManager::mergeStates(StateVariables const& vars,
                                                                       std::vector<size_t>& prefix_lengths,
                                                                       std::vector<HistoryState const*> const& prefix_states) {
    std::vector<size_t> original_prefix_lengths(prefix_lengths);

    size_t max_prefix = 0ul;
    for (size_t& len : prefix_lengths) {
        len = std::min(len, maxHistory_);
        max_prefix = std::max(max_prefix, len);
    }

    FeedDict result;
    result.reserve(vars.size());

    for (size_t v = 0ul; v < vars.size(); v++) {
        auto const& var = vars[v];
        require_ge(var.shape.size(), 2);

        std::vector<Tensorflow::int64> tensor_dim(var.shape.size());
        tensor_dim[0] = prefix_lengths.size();
        size_t batch_stride = 1ul;
        size_t time_dim = static_cast<size_t>(-1);
        std::valarray<size_t> sizes(var.shape.size()-1);
        std::valarray<size_t> strides(var.shape.size()-1);
        for (size_t d = 1ul; d < var.shape.size(); d++) {
            bool is_time_dim = var.shape[d] < 0l;
            tensor_dim[d] = is_time_dim ? max_prefix : var.shape[d];
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

        Tensorflow::Tensor var_tensor = Tensorflow::Tensor::zeros<f32>(tensor_dim);

        size_t state_offset = 0ul;
        for (size_t b = 0ul; b < prefix_lengths.size(); b++) {
            size_t prefix_length = prefix_lengths[b];
            size_t prefix_offset = original_prefix_lengths[b] - prefix_length;
            for (size_t p = 0ul; p < prefix_length; p++) {
                std::gslice slice(b * batch_stride + (max_prefix - prefix_length + p) * strides[time_dim], sizes, strides);
                ContiguousBlockInfo block_info(slice);
                size_t idx = state_offset;
                if (not alwaysIncludeFirstTokenState_ or p != 0ul) {
                    idx += prefix_offset + p;
                }
                prefix_states[idx]->at(v)->uncompress(var_tensor.data<f32>(), block_info);
            }
            state_offset += original_prefix_lengths[b];
        }

        result.emplace_back(vars[v].initial_value_name, var_tensor);
    }

    return result;
}

std::vector<TransformerStateManager::HistoryState> TransformerStateManager::splitStates(StateVariables const& vars,
                                                                                        std::vector<size_t>& suffix_lengths,
                                                                                        std::vector<Tensorflow::Tensor> const& state_tensors,
                                                                                        CompressedVectorFactory<float> const& vector_factory) {
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

        std::valarray<size_t> sizes(var.shape.size()-1);  // ignore first dimension as that is the batch dimension
        std::valarray<size_t> strides(var.shape.size()-1);
        for (size_t d = 1ul; d < var.shape.size(); d++) {
            bool is_time_dim = var.shape[d] < 0l;
            if (is_time_dim) {
                time_dim = d - 1;
            }
            else {
                require_eq(var.shape[d], tensor.dimSize(d));
            }
            sizes[d-1] = is_time_dim ? 1ul : var.shape[d];
            batch_stride *= tensor.dimSize(d);
        }
        require_lt(time_dim, static_cast<size_t>(-1));
        size_t max_prefix = tensor.dimSize(time_dim + 1) - max_suffix;

        strides[strides.size() - 1ul] = 1ul;
        for (size_t d = strides.size() - 1ul; d > 0ul; d--) {
            strides[d-1ul] = tensor.dimSize(d+1) * strides[d];
        }

        size_t output_idx = 0ul;
        for (size_t b = 0ul; b < suffix_lengths.size(); b++) {
            for (size_t p = 0ul; p < suffix_lengths[b]; p++) {
                std::gslice slice(b * batch_stride + (max_prefix + p) * strides[time_dim], sizes, strides);
                ContiguousBlockInfo block_info(slice);
                auto compression_param_estimator = vector_factory.getEstimator();
                compression_param_estimator->accumulate(tensor.data<f32>(), block_info);
                auto compression_params = compression_param_estimator->estimate();
                result[output_idx].emplace_back(vector_factory.compress(tensor.data<f32>(), block_info, compression_params.get()));
                output_idx += 1ul;
            }
        }
    }

    for (auto const& r : result) {
        require_eq(r.size(), 6);
    }

    return result;
}

bool TransformerStateManager::requiresAllParentStates() const {
    return true;
}

}  // namespace Lm
