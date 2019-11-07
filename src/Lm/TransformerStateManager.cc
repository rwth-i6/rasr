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

CompressedVectorPtr<float> TransformerStateManager::initialState(Tensorflow::Variable const& var, CompressedVectorFactory<float> const& vector_factory) {
    std::vector<float> vec(0, 0.0f);
    auto               compression_param_estimator = vector_factory.getEstimator();
    compression_param_estimator->accumulate(vec.data(), vec.size());
    auto compression_params = compression_param_estimator->estimate();
    return vector_factory.compress(vec.data(), vec.size(), compression_params.get());
}

Tensorflow::Tensor TransformerStateManager::mergeStates(Tensorflow::Variable const& var, std::vector<StateInfo>& states) {
    require_ge(var.shape.size(), 2);

    size_t max_prefix = 0ul;
    for (auto& info : states) {
        info.prefixLength = std::min(info.prefixLength, maxHistory_);
        max_prefix = std::max(max_prefix, info.prefixLength);
    }

    std::vector<Tensorflow::int64> tensor_dim(var.shape.size());
    tensor_dim[0] = states.size();
    size_t batch_stride = 1ul;
    size_t time_dim = static_cast<size_t>(-1);
    std::valarray<size_t> sizes(var.shape.size()-1);
    std::valarray<size_t> strides(var.shape.size()-1);
    for (size_t d = 1ul; d < var.shape.size(); d++) {
        bool is_time_dim = var.shape[d] < 0l;
        tensor_dim[d] = is_time_dim ? max_prefix : var.shape[d];
        if (is_time_dim) {
            time_dim = d-1;
        }
        else {
            require_eq(var.shape[d], var.shape[d]);
        }
        sizes[d-1ul] = is_time_dim ? std::min(max_prefix, 1ul) : var.shape[d];
        batch_stride *= tensor_dim[d];
    }
    require_lt(time_dim, static_cast<size_t>(-1));

    strides[strides.size() - 1ul] = 1ul;
    for (size_t d = strides.size() - 1ul; d > 0ul; d--) {
        strides[d-1ul] = tensor_dim[d+1] * strides[d];
    }

    Tensorflow::Tensor result = Tensorflow::Tensor::zeros<f32>(tensor_dim);

    for (size_t s = 0ul; s < states.size(); s++) {
        size_t prefix_length = states[s].prefixLength;
        size_t state_offset = states[s].state.size() - prefix_length;
        for (size_t p = 0ul; p < prefix_length; p++) {
            std::gslice slice(s * batch_stride + (max_prefix - prefix_length + p) * strides[time_dim], sizes, strides);
            ContiguousBlockInfo block_info(slice);
            if (alwaysIncludeFirstTokenState_ and p == 0ul) {
                states[s].state[0]->uncompress(result.data<f32>(), block_info);
            }
            else {
                states[s].state[state_offset + p]->uncompress(result.data<f32>(), block_info);
            }
        }
    }

    return result;
}

void TransformerStateManager::splitStates(Tensorflow::Variable const& var,
                                          Tensorflow::Tensor const& tensor,
                                          CompressedVectorFactory<float> const& vector_factory,
                                          std::vector<StateInfo>& states) {
    require_ge(var.shape.size(), 2);

    size_t max_prefix = 0ul;
    size_t max_suffix = 0ul;
    for (auto const& info : states) {
        max_prefix = std::max(max_prefix, info.prefixLength);
        max_suffix = std::max(max_suffix, info.suffixLength);
    }
    max_prefix = std::min<size_t>(max_prefix, maxHistory_);

    size_t batch_stride = 1ul;
    size_t time_dim = static_cast<size_t>(-1);
    std::valarray<size_t> sizes(var.shape.size()-1);  // ignore first dimension as that is the batch dimension
    std::valarray<size_t> strides(var.shape.size()-1);
    for (size_t d = 1ul; d < var.shape.size(); d++) {
        bool is_time_dim = var.shape[d] < 0l;
        if (is_time_dim) {
            time_dim = d-1;
            require_eq(max_prefix + max_suffix, tensor.dimSize(d));
        }
        else {
            require_eq(var.shape[d], tensor.dimSize(d));
        }
        sizes[d-1] = is_time_dim ? 1ul : var.shape[d];
        batch_stride *= tensor.dimSize(d);
    }
    require_lt(time_dim, static_cast<size_t>(-1));

    strides[strides.size() - 1ul] = 1ul;
    for (size_t d = strides.size() - 1ul; d > 0ul; d--) {
        strides[d-1ul] = tensor.dimSize(d+1) * strides[d];
    }

    for (size_t s = 0ul; s < tensor.dimSize(0); s++) {
        for (size_t p = 0ul; p < states[s].suffixLength; p++) {
            require_eq(states[s].suffixLength, states[s].state.size());
            std::gslice slice(s * batch_stride + (max_prefix + p) * strides[time_dim], sizes, strides);
            ContiguousBlockInfo block_info(slice);
            auto compression_param_estimator = vector_factory.getEstimator();
            compression_param_estimator->accumulate(tensor.data<f32>(), block_info);
            auto compression_params = compression_param_estimator->estimate();
            states[s].state[p] = vector_factory.compress(tensor.data<f32>(), block_info, compression_params.get()).release();
        }
    }
}

bool TransformerStateManager::requiresAllParentStates() const {
    return true;
}

}  // namespace Lm
