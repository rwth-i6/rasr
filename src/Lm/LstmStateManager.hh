#ifndef _LM_LSTM_STATE_MANAGER_HH
#define _LM_LSTM_STATE_MANAGER_HH

#include "StateManager.hh"

namespace Lm {

class LstmStateManager : public StateManager {
public:
    using Precursor = StateManager;

    LstmStateManager(Core::Configuration const& config);
    virtual ~LstmStateManager() = default;

    virtual HistoryState              initialState(StateVariables const& vars, CompressedVectorFactory<float> const& vector_factory);
    virtual FeedDict                  mergeStates (StateVariables const& vars,
                                                   std::vector<size_t>& prefix_lengths,
                                                   std::vector<HistoryState const*> const& prefix_states);
    virtual std::vector<HistoryState> splitStates (StateVariables const& vars,
                                                   std::vector<size_t>& suffix_lengths,
                                                   std::vector<Tensorflow::Tensor> const& state_tensors,
                                                   CompressedVectorFactory<float> const& vector_factory);
};

// inline implementations

inline LstmStateManager::LstmStateManager(Core::Configuration const& config) : Precursor(config) {
}

}  // namespace Lm

#endif   // _LM_LSTM_STATE_MANAGER_HH
