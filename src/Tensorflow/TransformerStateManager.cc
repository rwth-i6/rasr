#include "TransformerStateManager.hh"

namespace Tensorflow {

const Core::ParameterInt TransformerStateManager::paramContextSize("context-size", "left-context size (in frames)", 100, 0);
const Core::ParameterInt TransformerStateManager::paramPrefixLength("prefix-length", "left-context to always keep", 0, 0);
const Core::ParameterInt TransformerStateManager::paramDiscardSuffixLength("discard-suffix-length", "how many frames to drop from the end of the new state (usefull for overlapping chunks)", 0, 0);

TransformerStateManager::TransformerStateManager(Core::Configuration const& config, Graph const& graph, Session& session)
        : Precursor(config, graph, session),
          context_size_(paramContextSize(config)),
          prefix_length_(paramPrefixLength(config)),
          discard_suffix_length_(paramDiscardSuffixLength(config)) {
    auto const& var_map = graph.variables();
    for (std::string const& name : graph.state_vars()) {
        auto const& var = var_map.find(name)->second;
        state_fetches_.push_back(var.snapshot_name);
        state_setters_.push_back(var.initializer_name);
        state_setter_values_.push_back(var.initial_value_name);
        for (size_t i = 1; i < var.shape.size(); i++) {
            if (var.shape[i] == -1) {
                time_axis_.push_back(i);
                break;
            }
        }
    }
}

void TransformerStateManager::setInitialState() {
    FeedDict   feed_dict;
    TargetList targets;

    state_.clear();

    auto const& var_map = graph_.variables();
    for (std::string const& name : graph_.state_vars()) {
        auto const&        var = var_map.find(name)->second;
        std::vector<int64> shape(var.shape.begin(), var.shape.end());
        shape[0] = 1l;  // set batch dim to 1 (assuming dim 0 is batch-dim)
        for (size_t i = 1ul; i < shape.size(); i++) {
            if (shape[i] < 0) {
                shape[i] = 0l;
            }
        }
        feed_dict.emplace_back(std::make_pair(var.initial_value_name, Tensor::zeros<float>(shape)));
        state_.push_back(feed_dict.back().second);
        targets.push_back(var.initializer_name);
    }

    session_.run(feed_dict, targets);
}

std::vector<std::string> TransformerStateManager::getOutputs() const {
    return std::vector<std::string>();
}

std::vector<std::string> TransformerStateManager::getTargets() const {
    return graph_.update_ops();
}

void TransformerStateManager::updateState(std::vector<Tensor> const& state_tensors) {
    // state_tensors is empty as we need to fetch the vars after the first session.run
    std::vector<Tensor> new_state;
    session_.run({}, state_fetches_, {}, new_state);

    for (size_t i = 0ul; i < new_state.size(); i++) {
        auto&     s                = new_state[i];
        int       time_axis        = time_axis_[i];
        tf::int64 new_state_length = s.dimSize(time_axis);
        // prefix_length_-long prefix is part of the context_size_-long context
        if (new_state_length > context_size_ || discard_suffix_length_ > 0) {
            std::vector<int> first_start;
            std::vector<int> first_end;
            std::vector<int> main_start;
            std::vector<int> main_end;
            for (tf::int64 dim = 0; dim < state_[i].numDims(); dim++) {
                first_start.push_back(0);
                if (dim == time_axis) {
                    first_end.push_back((new_state_length - discard_suffix_length_ > context_size_) ? prefix_length_ : 0);
                    main_start.push_back((new_state_length - discard_suffix_length_ > context_size_) ? new_state_length - discard_suffix_length_ - context_size_ + prefix_length_ : 0);
                }
                else {
                    first_end.push_back(s.dimSize(dim));
                    main_start.push_back(0);
                }
                main_end.push_back(dim == time_axis ? std::max(static_cast<int>(new_state_length) - discard_suffix_length_, 0) : s.dimSize(dim));
            }
            Tensor first;
            if (prefix_length_ > 0) {
                first = s.slice(first_start, first_end);
            }
            s = s.slice(main_start, main_end);
            if (prefix_length_ > 0) {
                s = Tensor::concat(first, s, time_axis);
            }
        }

        state_[i] = s;
    }

    require_eq(state_setter_values_.size(), state_.size());
    FeedDict feed_dict;
    for (size_t i = 0ul; i < state_setter_values_.size(); i++) {
        feed_dict.emplace_back(std::make_pair(state_setter_values_[i], state_[i]));
    }
    session_.run(feed_dict, state_setters_);
}

}  // namespace Tensorflow
