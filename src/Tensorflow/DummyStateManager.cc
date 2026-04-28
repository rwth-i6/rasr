#include "DummyStateManager.hh"

namespace Tensorflow {

DummyStateManager::DummyStateManager(Core::Configuration const& config, Graph const& graph, Session& session)
        : Precursor(config, graph, session) {
}

void DummyStateManager::setInitialState() {
}

std::vector<std::string> DummyStateManager::getOutputs() const {
    return std::vector<std::string>();
}

std::vector<std::string> DummyStateManager::getTargets() const {
    return std::vector<std::string>();
}

void DummyStateManager::updateState(std::vector<Tensor> const& state_tensors) {
}

}  // namespace Tensorflow
