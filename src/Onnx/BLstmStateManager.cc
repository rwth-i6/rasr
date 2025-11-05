#include "BLstmStateManager.hh"

namespace Onnx {

BLstmStateManager::BLstmStateManager(Core::Configuration const& config)
        : Precursor(config) {
}

void BLstmStateManager::setInitialStates(std::vector<OnnxStateVariable> const& state_vars) {
    state_values_.clear();

    for (auto state_var : state_vars) {
        state_values_.emplace_back(Value::zeros<f32>({1, state_var.shape.back()}));  // batch size fixed to be 1
    }
}

void BLstmStateManager::extendFeedDict(FeedDict& feed_dict, std::vector<OnnxStateVariable> const& state_vars) {
    for (size_t i = 0; i < state_vars.size(); i++) {
        feed_dict.emplace_back(state_vars[i].input_state_key, state_values_[i]);
    }
}

void BLstmStateManager::extendTargets(TargetList& targets, std::vector<OnnxStateVariable> const& state_vars) {
    for (auto state_var : state_vars) {
        targets.push_back(state_var.output_state_key);
    }
}

void BLstmStateManager::updateStates(std::vector<Value>& states) {
    require_eq(states.size(), state_values_.size());

    state_values_.clear();

    for (size_t i = 0ul; i < states.size(); i++) {
        state_values_.push_back(Value(std::move(states[i])));
    }
}

}  // namespace Onnx
