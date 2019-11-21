#ifndef _LM_TRANSFORMER_STATE_MANAGER_HH
#define _LM_TRANSFORMER_STATE_MANAGER_HH

#include "StateManager.hh"

namespace Lm {

class TransformerStateManager : public StateManager {
public:
    using Precursor = StateManager;

    static const Core::ParameterInt  paramMaxHistoryLength;
    static const Core::ParameterBool paramAlwaysIncludeFirstTokenState;

    TransformerStateManager(Core::Configuration const& config);
    virtual ~TransformerStateManager() = default;

    virtual bool requiresAllParentStates() const;

    virtual HistoryState              initialState(StateVariables const& vars, CompressedVectorFactory<float> const& vector_factory);
    virtual void                      mergeStates (StateVariables const& vars,
                                                   std::vector<size_t>& prefix_lengths,
                                                   std::vector<HistoryState const*> const& prefix_states,
                                                   FeedDict& feed_dict,
                                                   TargetList& targets);
    virtual std::vector<HistoryState> splitStates (StateVariables const& vars,
                                                   std::vector<size_t>& suffix_lengths,
                                                   std::vector<Tensorflow::Tensor> const& state_tensors,
                                                   CompressedVectorFactory<float> const& vector_factory);
private:
    size_t maxHistory_;
    bool   alwaysIncludeFirstTokenState_;
};

// inline implementations

inline TransformerStateManager::TransformerStateManager(Core::Configuration const& config) : Precursor(config),
                                                                                             maxHistory_(paramMaxHistoryLength(config)),
                                                                                             alwaysIncludeFirstTokenState_(paramAlwaysIncludeFirstTokenState(config)) {
}

}  // namespace Lm

#endif  // _LM_TRANSFORMER_STATE_MANAGER_HH
