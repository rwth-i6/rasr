#ifndef _ONNX_DUMMYSTATEMANAGER_HH
#define _ONNX_DUMMYSTATEMANAGER_HH

#include "StateManager.hh"

namespace Onnx {

class DummyStateManager : public StateManager {
public:
    using Precursor  = StateManager;
    using FeedDict   = Precursor::FeedDict;
    using TargetList = Precursor::TargetList;

    DummyStateManager(Core::Configuration const& config);
    virtual ~DummyStateManager() = default;

    virtual void setInitialStates(std::vector<OnnxStateVariable> const& state_vars);
    virtual void extendFeedDict(FeedDict& feed_dict, std::vector<OnnxStateVariable> const& state_vars);
    virtual void extendTargets(TargetList& targets, std::vector<OnnxStateVariable> const& state_vars);
    virtual void updateStates(std::vector<Value>& states);
};

}  // namespace Onnx

#endif  // _ONNX_DUMMYSTATEMANAGER_HH
