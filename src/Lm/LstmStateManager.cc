#include "LstmStateManager.hh"

namespace Lm {

CompressedVectorPtr<float> LstmStateManager::initialState(Tensorflow::Variable const& var, CompressedVectorFactory<float> const& vector_factory) {
    require_gt(var.shape.size(), 0ul);
    s64 state_size = var.shape.back();
    require_ge(state_size, 0);  // variable must not be of unknown size
    std::vector<float> vec(state_size, 0.0f);
    auto               compression_param_estimator = vector_factory.getEstimator();
    compression_param_estimator->accumulate(vec.data(), vec.size());
    auto compression_params = compression_param_estimator->estimate();
    return vector_factory.compress(vec.data(), vec.size(), compression_params.get());
}

Tensorflow::Tensor LstmStateManager::mergeStates(Tensorflow::Variable const& var, std::vector<StateInfo> const& states) {
    require_gt(states.size(), 0ul);
    require_eq(states.front().state.size(), 1ul);
    Tensorflow::int64 num_states = states.size();
    Tensorflow::int64 state_size = states.front().state.front()->size();
    Tensorflow::Tensor result = Tensorflow::Tensor::zeros<float>({num_states, state_size});
    float* data = result.data<f32>();
    for (size_t s = 0ul; s < static_cast<size_t>(num_states); s++) {
        require_eq(states[s].state.size(), 1ul);
        states[s].state.front()->uncompress(data + s * state_size, state_size);
    }
    return result;
}

void LstmStateManager::splitStates(Tensorflow::Variable const& var,
                                   Tensorflow::Tensor const& tensor,
                                   CompressedVectorFactory<float> const& vector_factory,
                                   std::vector<StateInfo>& states) {
    require_eq(tensor.numDims(), 2);
    size_t state_size = tensor.dimSize(1);

    for (size_t s = 0ul; s < states.size(); s++) {
        auto         compression_param_estimator = vector_factory.getEstimator();
        float const* data                        = tensor.data<f32>(s, 0);
        compression_param_estimator->accumulate(data, state_size);
        auto compression_params = compression_param_estimator->estimate();
        states[s].state.back() = vector_factory.compress(data, state_size, compression_params.get()).release();
    }
}

}  // namespace Lm
