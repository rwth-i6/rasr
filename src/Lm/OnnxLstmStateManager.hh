#ifndef _LM_ONNX_LSTM_STATE_MANAGER_HH
#define _LM_ONNX_LSTM_STATE_MANAGER_HH

#include <Onnx/OnnxStateVariable.hh>
#include <Onnx/Value.hh>

#include "LstmStateManager.hh"

namespace Lm {

class OnnxLstmStateManager : public LstmStateManager<Onnx::Value, Onnx::OnnxStateVariable> {
public:
    using Precursor = LstmStateManager<Onnx::Value, Onnx::OnnxStateVariable>;

    OnnxLstmStateManager(Core::Configuration const& config);
    virtual ~OnnxLstmStateManager() = default;

protected:
    virtual void extendFeedDict(FeedDict& feed_dict, Onnx::OnnxStateVariable const& state_var, Onnx::Value& var);
    virtual void extendTargets(TargetList& targets, Onnx::OnnxStateVariable const& state_var);
};

// inline implementations

inline OnnxLstmStateManager::OnnxLstmStateManager(Core::Configuration const& config)
        : Precursor(config) {
}

}  // namespace Lm

#endif  // _LM_ONNX_LSTM_STATE_MANAGER_HH
