#ifndef _TENSORFLOW_DUMMYSTATEMANAGER_HH
#define _TENSORFLOW_DUMMYSTATEMANAGER_HH

#include "StateManager.hh"

namespace Tensorflow {

class DummyStateManager : public StateManager {
public:
    using Precursor = StateManager;

    DummyStateManager(Core::Configuration const& config, Graph const& graph, Session& session);
    virtual ~DummyStateManager() = default;

    virtual void                     setInitialState();
    virtual std::vector<std::string> getOutputs() const;
    virtual std::vector<std::string> getTargets() const;
    virtual void                     updateState(std::vector<Tensor> const& state_tensors);
};

}  // namespace Tensorflow

#endif  // _TENSORFLOW_DUMMYSTATEMANAGER_HH
