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

    virtual CompressedVectorPtr<float> initialState(Tensorflow::Variable const& var, CompressedVectorFactory<float> const& vector_factory);
    virtual Tensorflow::Tensor         mergeStates(Tensorflow::Variable const& var, std::vector<StateInfo>& states);
    virtual void                       splitStates(Tensorflow::Variable const& var,
                                                   Tensorflow::Tensor const& tensor,
                                                   CompressedVectorFactory<float> const& vector_factory,
                                                   std::vector<StateInfo>& states);
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
