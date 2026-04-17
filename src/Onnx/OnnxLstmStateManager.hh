#ifndef _ONNX_LSTM_STATE_MANAGER_HH
#define _ONNX_LSTM_STATE_MANAGER_HH

#include <Onnx/OnnxStateVariable.hh>
#include <Onnx/Value.hh>

#include <Nn/LstmStateManager.hh>

namespace Onnx {

class OnnxLstmStateManager : public Nn::LstmStateManager<Onnx::Value, Onnx::OnnxStateVariable> {
public:
    using Precursor = Nn::LstmStateManager<Onnx::Value, Onnx::OnnxStateVariable>;

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

}  // namespace Onnx

#endif  // _ONNX_LSTM_STATE_MANAGER_HH
