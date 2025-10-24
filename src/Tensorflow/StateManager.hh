#ifndef _TENSORFLOW_STATEMANAGER_HH
#define _TENSORFLOW_STATEMANAGER_HH

#include "Session.hh"
#include "Tensor.hh"
#include "TensorMap.hh"

namespace Tensorflow {

class StateManager : public Core::Component {
public:
    using Precursor = Core::Component;

    static std::unique_ptr<StateManager> create(Core::Configuration const& config, Graph const& graph, Session& session);

    StateManager(Core::Configuration const& config, Graph const& graph, Session& session);
    virtual ~StateManager() = default;

    virtual void                     setInitialState()                                     = 0;
    virtual std::vector<std::string> getOutputs() const                                    = 0;
    virtual std::vector<std::string> getTargets() const                                    = 0;
    virtual void                     updateState(std::vector<Tensor> const& state_tensors) = 0;

protected:
    Graph const& graph_;
    Session&     session_;
};

}  // namespace Tensorflow

#endif  // _TENSORFLOW_STATEMANAGER_HH
