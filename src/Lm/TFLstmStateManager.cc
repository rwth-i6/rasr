
#include "TFLstmStateManager.hh"

namespace Lm {

void TFLstmStateManager::extendFeedDict(FeedDict& feed_dict, Tensorflow::Variable const& state_var, Tensorflow::Tensor& var) {
    feed_dict.emplace_back(state_var.initial_value_name, var);
}

void TFLstmStateManager::extendTargets(TargetList& targets, Tensorflow::Variable const& state_var) {
    targets.emplace_back(state_var.initializer_name);
}

}  // namespace Lm
