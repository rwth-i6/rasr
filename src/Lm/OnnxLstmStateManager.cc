#include "OnnxLstmStateManager.hh"

namespace Lm {

void OnnxLstmStateManager::extendFeedDict(FeedDict& feed_dict, Onnx::OnnxStateVariable const& state_var, Onnx::Value& var) {
    feed_dict.emplace_back(state_var.input_state_key, std::move(var));
}

void OnnxLstmStateManager::extendTargets(TargetList& targets, Onnx::OnnxStateVariable const& state_var) {
    targets.emplace_back(state_var.output_state_key);
}

}  // namespace Lm
