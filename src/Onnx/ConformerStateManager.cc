#include "ConformerStateManager.hh"

namespace Onnx {

const Core::ParameterInt ConformerStateManager::paramAttentionContextSize("attention-context-size", "left-context size (in frames)", 100, 0);
const Core::ParameterInt ConformerStateManager::paramConvContextSize("conv-context-size", "left-context size (in frames)", 100, 0);
const Core::ParameterInt ConformerStateManager::paramDiscardSuffixLength("discard-suffix-length", "how many frames to drop from the end of the new state (usefull for overlapping chunks)", 0, 0);

ConformerStateManager::ConformerStateManager(Core::Configuration const& config)
        : Precursor(config),
          att_context_size_(paramAttentionContextSize(config)),
          conv_context_size_(paramConvContextSize(config)),
          discard_suffix_length_(paramDiscardSuffixLength(config)) {
}

void ConformerStateManager::setInitialStates(std::vector<OnnxStateVariable> const& state_vars) {
    states_.clear();
    for (auto state_var : state_vars) {
        std::vector<int64_t> shape;
        int                  time_axis = -1;
        for (size_t i = 0ul; i < state_var.shape.size(); i++) {
            // we assume the first axis is the batch axis and the next dynamic shape axis is the time axis
            if (i != 0 && (state_var.shape[i] == -1 || state_var.shape[i] == -2)) {
                shape.push_back(0);
                if (time_axis == -1) {
                    time_axes_.push_back(i);
                    time_axis = i;
                }
            }
            else {
                shape.push_back(state_var.shape[i]);
            }
        }
        shape[0] = 1;  // set batch dimension to 1
        states_.emplace_back(Value::zeros<f32>(shape));
    }
}

void ConformerStateManager::extendFeedDict(FeedDict& feed_dict, std::vector<OnnxStateVariable> const& state_vars) {
    require_eq(state_vars.size(), states_.size());

    for (size_t i = 0ul; i < state_vars.size(); i++) {
        OnnxStateVariable state_var = state_vars[i];
        Value&            state     = states_[i];
        Value             input_state;
        int64_t           dim_size = state.dimSize(time_axes_[i]);

        if (state_var.input_state_key.find("mhsa") != std::string::npos) {
            // trim the MHSA module state tensor
            int64_t slice_start = std::max(dim_size - discard_suffix_length_ - att_context_size_, 0l);
            int64_t slice_end   = std::max(dim_size - discard_suffix_length_, 0l);

            input_state = state.slice(slice_start, slice_end, time_axes_[i]);
        }
        else if (state_var.input_state_key.find("conv") != std::string::npos) {
            // trim the convolution module state tensor
            int64_t slice_start = std::max(dim_size - discard_suffix_length_ - conv_context_size_, 0l);
            int64_t slice_end   = std::max(dim_size - discard_suffix_length_, 0l);

            input_state = state.slice(slice_start, slice_end, time_axes_[i]);
        }
        else {
            input_state = Value(std::move(state));
        }
        feed_dict.emplace_back(state_var.input_state_key + ":size1",
                               Value::create(std::vector<s32>{static_cast<s32>(input_state.dimSize((this->time_axes_)[i]))}));
        feed_dict.emplace_back(state_var.input_state_key, std::move(input_state));
    }
}

void ConformerStateManager::extendTargets(TargetList& targets, std::vector<OnnxStateVariable> const& state_vars) {
    for (auto state_var : state_vars) {
        targets.emplace_back(state_var.output_state_key);
    }
}

void ConformerStateManager::updateStates(std::vector<Value>& states) {
    require_eq(states.size(), states_.size());
    states_.clear();
    for (size_t i = 0ul; i < states.size(); i++) {
        states_.push_back(Value(std::move(states[i])));
    }
}

}  // namespace Onnx
