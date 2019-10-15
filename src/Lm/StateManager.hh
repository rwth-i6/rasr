#ifndef _LM_STATE_MANAGER_HH
#define _LM_STATE_MANAGER_HH

#include <Core/Component.hh>
#include <Tensorflow/Graph.hh>
#include <Tensorflow/Tensor.hh>

#include "CompressedVector.hh"

namespace Lm {

struct StateInfo {
    std::vector<CompressedVector<float>*> state;

    size_t prefixLength;
    size_t suffixLength;
};

class StateManager : public Core::Component {
public:
    using Precursor = Core::Component;

    StateManager(Core::Configuration const& config);
    virtual ~StateManager() = default;

    virtual bool requiresAllParentStates() const;

    virtual CompressedVectorPtr<float> initialState(Tensorflow::Variable const& var, CompressedVectorFactory<float> const& vector_factory) = 0;
    virtual Tensorflow::Tensor         mergeStates(Tensorflow::Variable const& var, std::vector<StateInfo> const& states) = 0;
    virtual void                       splitStates(Tensorflow::Variable const& var,
                                                   Tensorflow::Tensor const& tensor,
                                                   CompressedVectorFactory<float> const& vector_factory,
                                                   std::vector<StateInfo>& states) = 0;
};

// inline implementations

inline bool StateManager::requiresAllParentStates() const {
    return false;
}

inline StateManager::StateManager(Core::Configuration const& config) : Precursor(config) {
}

}  // namespace Lm

#endif  // _LM_STATE_MANAGER_HH
