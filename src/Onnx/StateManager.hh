#ifndef _ONNX_STATEMANAGER_HH
#define _ONNX_STATEMANAGER_HH

#include <Core/Component.hh>

#include "OnnxStateVariable.hh"
#include "Value.hh"

namespace Onnx {

class StateManager : public Core::Component {
public:
    using Precursor  = Core::Component;
    using FeedDict   = std::vector<std::pair<std::string, Value>>;
    using TargetList = std::vector<std::string>;

    static std::unique_ptr<StateManager> create(Core::Configuration const& config);

    StateManager(Core::Configuration const& config);
    virtual ~StateManager() = default;

    virtual void setInitialStates(std::vector<OnnxStateVariable> const& state_vars)                    = 0;
    virtual void extendFeedDict(FeedDict& feed_dict, std::vector<OnnxStateVariable> const& state_vars) = 0;
    virtual void extendTargets(TargetList& targets, std::vector<OnnxStateVariable> const& state_vars)  = 0;
    virtual void updateStates(std::vector<Value>& states)                                              = 0;
};

}  // namespace Onnx

#endif  // _ONNX_STATEMANAGER_HH
