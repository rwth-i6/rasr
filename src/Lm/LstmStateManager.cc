#include "LstmStateManager.hh"

namespace Lm {

LstmStateManager::HistoryState LstmStateManager::initialState(StateVariables const& vars, CompressedVectorFactory<float> const& vector_factory) {
    HistoryState result;
    result.reserve(vars.size());
    for (auto const& var : vars) {
        require_gt(var.shape.size(), 0ul);
        s64 state_size = var.shape.back();
        require_ge(state_size, 0);  // variable must not be of unknown size
        std::vector<float> vec(state_size, 0.0f);
        auto               compression_param_estimator = vector_factory.getEstimator();
        compression_param_estimator->accumulate(vec.data(), vec.size());
        auto compression_params = compression_param_estimator->estimate();
        result.emplace_back(vector_factory.compress(vec.data(), vec.size(), compression_params.get()));
    }
    return result;
}

void LstmStateManager::mergeStates(StateVariables const& vars,
                                   std::vector<size_t>& prefix_lengths,
                                   std::vector<HistoryState const*> const& prefix_states,
                                   FeedDict& feed_dict,
                                   TargetList& targets) {
    require_eq(prefix_states.size(), prefix_lengths.size());
    feed_dict.reserve(vars.size());
    targets.reserve(vars.size());
    Tensorflow::int64 batch_size = prefix_lengths.size();
    for (size_t v = 0ul; v < vars.size(); v++) {
        Tensorflow::int64 state_size = prefix_states.front()->at(v)->size();
        Tensorflow::Tensor var_tensor = Tensorflow::Tensor::zeros<float>({batch_size, state_size});
        float* data = var_tensor.data<f32>();
        for (size_t b = 0ul; b < static_cast<size_t>(batch_size); b++) {
            auto const& compressed_state = prefix_states[b]->at(v);
            require_eq(compressed_state->size(), static_cast<size_t>(state_size));
            compressed_state->uncompress(data + b * state_size, state_size);
        }
        feed_dict.emplace_back(vars[v].initial_value_name, var_tensor);
        targets.emplace_back(vars[v].initializer_name);
    }
}

std::vector<LstmStateManager::HistoryState> LstmStateManager::splitStates(StateVariables const& vars,
                                                                          std::vector<size_t>& suffix_lengths,
                                                                          std::vector<Tensorflow::Tensor> const& state_tensors,
                                                                          CompressedVectorFactory<float> const& vector_factory) {
    require_eq(vars.size(), state_tensors.size());

    std::vector<HistoryState> result(suffix_lengths.size());
    for (size_t r = 0ul; r < suffix_lengths.size(); r++) {
        result[r].reserve(vars.size());
        suffix_lengths[r] = 1ul;
    }

    for (size_t v = 0ul; v < vars.size(); v++) {
        Tensorflow::Tensor const& tensor = state_tensors[v];
        require_eq(tensor.numDims(), 2);
        size_t batch_size = tensor.dimSize(0);
        size_t state_size = tensor.dimSize(1);
        require_eq(batch_size, suffix_lengths.size());

        for (size_t b = 0ul; b < batch_size; b++) {
            auto         compression_param_estimator = vector_factory.getEstimator();
            float const* data                        = tensor.data<f32>(b, 0);
            compression_param_estimator->accumulate(data, state_size);
            auto compression_params = compression_param_estimator->estimate();
            result[b].emplace_back(vector_factory.compress(data, state_size, compression_params.get()).release());
        }
    }

    return result;
}

}  // namespace Lm
