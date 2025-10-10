#ifndef _ONNX_BLSTM_CARRYOVER_STATE_MANAGER_HH
#define _ONNX_BLSTM_CARRYOVER_STATE_MANAGER_HH

#include "StateManager.hh"

namespace Onnx {

class BLstmStateManager : public StateManager {
public:
    using Precursor  = StateManager;
    using FeedDict   = StateManager::FeedDict;
    using TargetList = StateManager::TargetList;

public:
    BLstmStateManager(Core::Configuration const& config);

    virtual ~BLstmStateManager() = default;

public:
    void setInitialStates(std::vector<OnnxStateVariable> const& state_var) override;

    void extendFeedDict(FeedDict& feed_dict, std::vector<OnnxStateVariable> const& state_vars) override;

    void extendTargets(TargetList& targets, std::vector<OnnxStateVariable> const& state_vars) override;

    void updateStates(std::vector<Value>& states) override;

private:
    std::vector<Value> state_values_;
};

}  // namespace Onnx

#endif
