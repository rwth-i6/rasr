#ifndef _TENSORFLOW_TRANSFORMERSTATEMANAGER_HH
#define _TENSORFLOW_TRANSFORMERSTATEMANAGER_HH

#include "StateManager.hh"

namespace Tensorflow {

class TransformerStateManager : public StateManager {
public:
    using Precursor  = StateManager;
    using FeedDict   = std::vector<std::pair<std::string, Tensor>>;
    using TargetList = std::vector<std::string>;

    static const Core::ParameterInt paramContextSize;
    static const Core::ParameterInt paramPrefixLength;
    static const Core::ParameterInt paramDiscardSuffixLength;

    TransformerStateManager(Core::Configuration const& config, Graph const& graph, Session& session);
    virtual ~TransformerStateManager() = default;

    virtual void                     setInitialState();
    virtual std::vector<std::string> getOutputs() const;
    virtual std::vector<std::string> getTargets() const;
    virtual void                     updateState(std::vector<Tensor> const& state_tensors);

private:
    const int context_size_;
    const int prefix_length_;
    const int discard_suffix_length_;

    std::vector<std::string> state_fetches_;
    std::vector<std::string> state_setters_;
    std::vector<std::string> state_setter_values_;
    std::vector<int>         time_axis_;
    std::vector<Tensor>      state_;
};

}  // namespace Tensorflow

#endif  // _TENSORFLOW_TRANSFORMERSTATEMANAGER_HH
