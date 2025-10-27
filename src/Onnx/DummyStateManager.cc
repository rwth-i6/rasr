#include "DummyStateManager.hh"

namespace Onnx {

DummyStateManager::DummyStateManager(Core::Configuration const& config)
        : Precursor(config) {
}

void DummyStateManager::setInitialStates(std::vector<OnnxStateVariable> const& state_vars) {
}
void DummyStateManager::extendFeedDict(FeedDict& feed_dict, std::vector<OnnxStateVariable> const& state_vars) {
}
void DummyStateManager::extendTargets(TargetList& targets, std::vector<OnnxStateVariable> const& state_vars) {
}
void DummyStateManager::updateStates(std::vector<Value>& states) {
}

}  // namespace Onnx
