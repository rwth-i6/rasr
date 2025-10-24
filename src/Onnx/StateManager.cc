#include "StateManager.hh"
#include "BLstmStateManager.hh"
#include "ConformerStateManager.hh"
#include "DummyStateManager.hh"

namespace {
enum StateManagerType : int {
    Dummy,
    Conformer,
    BLstm,
};

const Core::Choice stateManagerTypeChoice(
        "dummy", StateManagerType::Dummy,
        "conformer", StateManagerType::Conformer,
        "blstm", StateManagerType::BLstm,
        Core::Choice::endMark());

const Core::ParameterChoice stateManagerTypeParam(
        "type", &stateManagerTypeChoice, "type of stateManager", StateManagerType::Dummy);

}  // namespace

namespace Onnx {

std::unique_ptr<StateManager> StateManager::create(Core::Configuration const& config) {
    switch (stateManagerTypeParam(config)) {
        case Conformer:
            return std::unique_ptr<StateManager>(new ConformerStateManager(config));
        case BLstm:
            return std::unique_ptr<StateManager>(new BLstmStateManager(config));
        case Dummy:
        default:
            return std::unique_ptr<StateManager>(new DummyStateManager(config));
    }
    return std::unique_ptr<StateManager>();
}

StateManager::StateManager(Core::Configuration const& config)
        : Precursor(config) {}

}  // namespace Onnx
