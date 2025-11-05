#include "StateManager.hh"

#include "DummyStateManager.hh"
#include "TransformerStateManager.hh"

namespace {
enum StateManagerType : int {
    Dummy,
    Transformer
};

const Core::Choice stateManagerTypeChoice(
        "dummy", StateManagerType::Dummy,
        "transformer", StateManagerType::Transformer,
        Core::Choice::endMark());

const Core::ParameterChoice stateManagerTypeParam(
        "type", &stateManagerTypeChoice, "type of stateManager", StateManagerType::Dummy);

}  // namespace

namespace Tensorflow {

std::unique_ptr<StateManager> StateManager::create(Core::Configuration const& config, Graph const& graph, Session& session) {
    switch (stateManagerTypeParam(config)) {
        case Transformer:
            return std::unique_ptr<StateManager>(new TransformerStateManager(config, graph, session));
        case Dummy:
        default:
            return std::unique_ptr<StateManager>(new DummyStateManager(config, graph, session));
    }

    return std::unique_ptr<StateManager>();
}

StateManager::StateManager(Core::Configuration const& config, Graph const& graph, Session& session)
        : Precursor(config),
          graph_(graph),
          session_(session) {
}

}  // namespace Tensorflow
