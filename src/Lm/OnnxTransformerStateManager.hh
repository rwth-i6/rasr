#ifndef _LM_ONNX_TRANSFORMER_STATE_MANAGER_HH
#define _LM_ONNX_TRANSFORMER_STATE_MANAGER_HH

#include <Onnx/Value.hh>

#include <Onnx/OnnxStateVariable.hh>
#include "TransformerStateManager.hh"

namespace Lm {

template<typename T>
class OnnxTransformerStateManager : public TransformerStateManager<T, Onnx::Value, Onnx::OnnxStateVariable> {
public:
    using Precursor = TransformerStateManager<T, Onnx::Value, Onnx::OnnxStateVariable>;

    OnnxTransformerStateManager(Core::Configuration const& config);
    virtual ~OnnxTransformerStateManager() = default;

protected:
    virtual void extendFeedDict(typename Precursor::FeedDict& feed_dict, Onnx::OnnxStateVariable const& state_var, Onnx::Value& var);
    virtual void extendTargets(typename Precursor::TargetList& targets, Onnx::OnnxStateVariable const& state_var);
};

template<typename T>
inline OnnxTransformerStateManager<T>::OnnxTransformerStateManager(Core::Configuration const& config)
        : Precursor(config) {
}

template<typename T>
void OnnxTransformerStateManager<T>::extendFeedDict(typename Precursor::FeedDict& feed_dict, Onnx::OnnxStateVariable const& state_var, Onnx::Value& var) {
    feed_dict.emplace_back(state_var.input_state_key, std::move(var));
}

template<typename T>
void OnnxTransformerStateManager<T>::extendTargets(typename Precursor::TargetList& targets, Onnx::OnnxStateVariable const& state_var) {
    targets.emplace_back(state_var.output_state_key);
}

}  // namespace Lm

#endif  // _LM_ONNX_TRANSFORMER_STATE_MANAGER_HH
