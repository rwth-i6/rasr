#ifndef _NN_ABSTRACT_STATE_MANAGER_HH
#define _NN_ABSTRACT_STATE_MANAGER_HH

#include <string>
#include <utility>
#include <vector>

#include <Core/Component.hh>

#include "CompressedVector.hh"

namespace Nn {

template<typename value_t, typename state_variable_t>
class AbstractStateManager : public Core::Component {
public:
    using Precursor      = Core::Component;
    using FeedDict       = std::vector<std::pair<std::string, value_t>>;
    using TargetList     = std::vector<std::string>;
    using StateVariables = std::vector<state_variable_t>;
    using HistoryState   = std::vector<CompressedVectorPtr<float>>;

    AbstractStateManager(Core::Configuration const& config);
    virtual ~AbstractStateManager() = default;

    virtual bool requiresAllParentStates() const;

    virtual HistoryState initialState(StateVariables const& vars, CompressedVectorFactory<float> const& vector_factory) = 0;

    virtual void mergeStates(StateVariables const&                   vars,
                             std::vector<size_t>&                    prefix_lengths,
                             std::vector<HistoryState const*> const& prefix_states,
                             FeedDict&                               feed_dict,
                             TargetList&                             targets) = 0;

    virtual std::vector<HistoryState> splitStates(StateVariables const&                 vars,
                                                  std::vector<size_t>&                  suffix_lengths,
                                                  std::vector<value_t> const&           state_tensors,
                                                  CompressedVectorFactory<float> const& vector_factory) = 0;
};

template<typename value_t, typename state_variable_t>
inline bool AbstractStateManager<value_t, state_variable_t>::requiresAllParentStates() const {
    return false;
}

template<typename value_t, typename state_variable_t>
inline AbstractStateManager<value_t, state_variable_t>::AbstractStateManager(Core::Configuration const& config)
        : Precursor(config) {
}

}  // namespace Nn

#endif  // _NN_ABSTRACT_STATE_MANAGER_HH
