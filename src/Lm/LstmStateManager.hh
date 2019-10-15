#ifndef _LM_LSTM_STATE_MANAGER_HH
#define _LM_LSTM_STATE_MANAGER_HH

#include "StateManager.hh"

namespace Lm {

class LstmStateManager : public StateManager {
public:
    using Precursor = StateManager;

    LstmStateManager(Core::Configuration const& config);
    virtual ~LstmStateManager() = default;

    virtual CompressedVectorPtr<float> initialState(Tensorflow::Variable const& var, CompressedVectorFactory<float> const& vector_factory);
    virtual Tensorflow::Tensor         mergeStates(Tensorflow::Variable const& var, std::vector<StateInfo> const& states);
    virtual void                       splitStates(Tensorflow::Variable const& var,
                                                   Tensorflow::Tensor const& tensor,
                                                   CompressedVectorFactory<float> const& vector_factory,
                                                   std::vector<StateInfo>& states);
};

// inline implementations

inline LstmStateManager::LstmStateManager(Core::Configuration const& config) : Precursor(config) {
}

}  // namespace Lm

#endif   // _LM_LSTM_STATE_MANAGER_HH
