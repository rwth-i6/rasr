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
#ifndef _LM_LSTM_STATE_MANAGER_HH
#define _LM_LSTM_STATE_MANAGER_HH

#include "AbstractStateManager.hh"
#include "CompressedVector.hh"

namespace Lm {

template<typename value_t, typename state_variable_t>
class LstmStateManager : public AbstractStateManager<value_t, state_variable_t> {
public:
    using Precursor = AbstractStateManager<value_t, state_variable_t>;

    LstmStateManager(Core::Configuration const& config);
    virtual ~LstmStateManager() = default;

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
};

template<typename value_t, typename state_variable_t>
LstmStateManager<value_t, state_variable_t>::LstmStateManager(Core::Configuration const& config)
        : Precursor(config) {
}

template<typename value_t, typename state_variable_t>
typename LstmStateManager<value_t, state_variable_t>::Precursor::HistoryState LstmStateManager<value_t, state_variable_t>::initialState(
        typename Precursor::StateVariables const& vars,
        CompressedVectorFactory<float> const&     vector_factory) {
    typename Precursor::HistoryState result;
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

template<typename value_t, typename state_variable_t>
void LstmStateManager<value_t, state_variable_t>::mergeStates(
        typename LstmStateManager<value_t, state_variable_t>::Precursor::StateVariables const&                   vars,
        std::vector<size_t>&                                                                                     prefix_lengths,
        std::vector<typename LstmStateManager<value_t, state_variable_t>::Precursor::HistoryState const*> const& prefix_states,
        typename LstmStateManager<value_t, state_variable_t>::Precursor::FeedDict&                               feed_dict,
        typename LstmStateManager<value_t, state_variable_t>::Precursor::TargetList&                             targets) {
    require_eq(prefix_states.size(), prefix_lengths.size());
    feed_dict.reserve(vars.size());
    targets.reserve(vars.size());

    s64 batch_size = prefix_lengths.size();

    for (size_t v = 0ul; v < vars.size(); v++) {
        s64     state_size = prefix_states.front()->at(v)->size();
        value_t var_tensor = value_t::template zeros<float>({batch_size, state_size});
        float*  data       = var_tensor.template data<f32>();

        for (size_t b = 0ul; b < static_cast<size_t>(batch_size); b++) {
            auto const& compressed_state = prefix_states[b]->at(v);
            require_eq(compressed_state->size(), static_cast<size_t>(state_size));
            compressed_state->uncompress(data + b * state_size, state_size);
        }

        extendFeedDict(feed_dict, vars[v], var_tensor);
        extendTargets(targets, vars[v]);
    }
}

template<typename value_t, typename state_variable_t>
std::vector<typename LstmStateManager<value_t, state_variable_t>::Precursor::HistoryState> LstmStateManager<value_t, state_variable_t>::splitStates(
        typename LstmStateManager<value_t, state_variable_t>::Precursor::StateVariables const& vars,
        std::vector<size_t>&                                                                   suffix_lengths,
        std::vector<value_t> const&                                                            state_tensors,
        CompressedVectorFactory<float> const&                                                  vector_factory) {
    require_eq(vars.size(), state_tensors.size());

    std::vector<typename Precursor::HistoryState> result(suffix_lengths.size());

    for (size_t r = 0ul; r < suffix_lengths.size(); r++) {
        result[r].reserve(vars.size());
        suffix_lengths[r] = 1ul;
    }

    for (size_t v = 0ul; v < vars.size(); v++) {
        value_t const& tensor = state_tensors[v];
        require_eq(tensor.numDims(), 2);
        size_t batch_size = tensor.dimSize(0);
        size_t state_size = tensor.dimSize(1);
        require_eq(batch_size, suffix_lengths.size());

        for (size_t b = 0ul; b < batch_size; b++) {
            auto         compression_param_estimator = vector_factory.getEstimator();
            float const* data                        = tensor.template data<f32>(b, 0);
            compression_param_estimator->accumulate(data, state_size);
            auto compression_params = compression_param_estimator->estimate();
            result[b].emplace_back(vector_factory.compress(data, state_size, compression_params.get()).release());
        }
    }

    return result;
}

}  // namespace Lm

#endif  // _LM_LSTM_STATE_MANAGER_HH
