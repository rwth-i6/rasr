#ifndef _ONNX_CONFORMERSTATEMANAGER_HH
#define _ONNX_CONFORMERSTATEMANAGER_HH

#include "StateManager.hh"

namespace Onnx {

class ConformerStateManager : public StateManager {
public:
    using Precursor  = StateManager;
    using FeedDict   = std::vector<std::pair<std::string, Value>>;
    using TargetList = std::vector<std::string>;

    static const Core::ParameterInt paramAttentionContextSize;
    static const Core::ParameterInt paramConvContextSize;
    static const Core::ParameterInt paramPrefixLength;
    static const Core::ParameterInt paramDiscardSuffixLength;

    ConformerStateManager(Core::Configuration const& config);
    virtual ~ConformerStateManager() = default;

    virtual void setInitialStates(std::vector<OnnxStateVariable> const& state_vars);
    virtual void extendFeedDict(FeedDict& feed_dict, std::vector<OnnxStateVariable> const& state_vars);
    virtual void extendTargets(TargetList& targets, std::vector<OnnxStateVariable> const& state_vars);
    virtual void updateStates(std::vector<Value>& states);

private:
    const unsigned att_context_size_;
    const unsigned conv_context_size_;
    const unsigned discard_suffix_length_;

    std::vector<Value> states_;
    std::vector<int>   time_axes_;
};

}  // namespace Onnx

#endif  // _ONNX_CONFORMERSTATEMANAGER_HHw
