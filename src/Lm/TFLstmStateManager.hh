#ifndef _LM_TF_LSTM_STATE_MANAGER_HH
#define _LM_TF_LSTM_STATE_MANAGER_HH

#include <Tensorflow/Graph.hh>
#include <Tensorflow/Tensor.hh>

#include "LstmStateManager.hh"

namespace Lm {

class TFLstmStateManager : public LstmStateManager<Tensorflow::Tensor, Tensorflow::Variable> {
public:
    using Precursor = LstmStateManager<Tensorflow::Tensor, Tensorflow::Variable>;

    TFLstmStateManager(Core::Configuration const& config);
    virtual ~TFLstmStateManager() = default;

protected:
    virtual void extendFeedDict(FeedDict& feed_dict, Tensorflow::Variable const& state_var, Tensorflow::Tensor& var);
    virtual void extendTargets(TargetList& targets, Tensorflow::Variable const& state_var);
};

// inline implementations

inline TFLstmStateManager::TFLstmStateManager(Core::Configuration const& config)
        : Precursor(config) {
}

}  // namespace Lm

#endif  // _LM_TF_LSTM_STATE_MANAGER_HH
