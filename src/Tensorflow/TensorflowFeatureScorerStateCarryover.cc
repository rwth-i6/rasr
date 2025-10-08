/** Copyright 2018 RWTH Aachen University. All rights reserved.
 *
 *  Licensed under the RWTH ASR License (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#include "TensorflowFeatureScorerStateCarryover.hh"
#include "Module.hh"

namespace Tensorflow {

static const Core::ParameterInt paramSaveStatePosition(
        "save-state-position", "save hidden state from this position in time; default = last timestamp", Core::Type<u32>::max);

TensorflowFeatureScorerStateCarryover::TensorflowFeatureScorerStateCarryover(const Core::Configuration& config, Core::Ref<const Mm::MixtureSet> mixtureSet)
        : Core::Component(config),
          TensorflowFeatureScorer(config, mixtureSet),
          save_state_position_(paramSaveStatePosition(config)) {
    log("Initializing tf-feature-scorer-state-carryover. Will save hidden states in position t=%zu for the following variables", save_state_position_);
    for (std::string const& s : graph_->state_vars()) {
        // will only get variables of layers with initial_state = keep_over_epoch_no_init in returnn config
        auto const& var = graph_->variables().find(s)->second;
        initializer_tensor_names_.push_back(var.initializer_name);

        SavedState state(var);
        saved_state_map_[state.initializer_name] = state;
        output_tensor_names_.push_back(state.lstm_output);
        log("SavedState: initial_value_name=%s initializer_name=%s lstm_output=%s dim=%zu\n",
            var.initial_value_name.c_str(), state.initializer_name.c_str(), state.lstm_output.c_str(), state.dim);
    }
}

void TensorflowFeatureScorerStateCarryover::_compute() {
    size_t num_frames = buffer_.size();
    // process buffer if needed
    if (!scoresComputed_) {
        scores_->resize(nClasses_, num_frames);
        std::vector<std::pair<std::string, Tensor>> inputs;
        std::vector<Tensor>                         tf_output;

        // 1. Initialize hidden states from saved values
        for (const auto& it : saved_state_map_) {
            const SavedState& state = it.second;
            inputs.emplace_back(std::make_pair(state.initial_value_name, Tensorflow::Tensor::create(state.state, true)));
            //printf("setting %s from saved state (%d, %d), e.g. %.3f\n", state.initial_value_name.c_str(), state.state.nRows(), state.state.nColumns(), state.state(0,0));
        }
        session_.run(inputs, initializer_tensor_names_);

        // 2. Feed input features through the graph
        inputs.clear();
        auto const& tensor_info = tensor_input_map_.get_info("features");
        inputs.push_back({tensor_info.tensor_name(), _createInputTensor()});
        if (not tensor_info.seq_length_tensor_name().empty()) {
            inputs.push_back({tensor_info.seq_length_tensor_name(),
                              Tensor::create(std::vector<s32>{static_cast<s32>(num_frames)})});
        }

        auto t_start = std::chrono::system_clock::now();
        session_.run(inputs, output_tensor_names_, {}, tf_output);
        tf_output[0].get<>(0, *scores_);

        auto t_end     = std::chrono::system_clock::now();
        auto t_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        printf("num_frames: %zu elapsed: %f AM_RTF: %f\n", num_frames, t_elapsed, t_elapsed / (num_frames / 100.0));

        // mark computed
        scoresComputed_ = true;

        // 3. save hidden states
        if (save_state_position_ < num_frames) {
            for (size_t s = 1 /* 0 = softmax */; s < output_tensor_names_.size(); ++s) {
                if (output_tensor_names_[s].find("NativeLstm2:2") == std::string::npos)
                    continue;

                SavedState& h = saved_state_map_[initializer_tensor_names_[s - 1]];
                SavedState& c = saved_state_map_[initializer_tensor_names_[s - 2]];
                require(c.dim == h.dim);
                Math::FastMatrix<f32> memory_cell(c.dim * 4, 1);
                Math::FastMatrix<f32> output_gate(c.dim, 1);

                tf_output[s - 1].get<>(save_state_position_, c.state, true);
                c.state.resizeRowsAndKeepContent(c.dim);

                tf_output[s - 1].get<>(save_state_position_, h.state, true);
                h.state.resizeRowsAndKeepContent(h.dim);
                h.state.tanh();

                tf_output[s].get<>(save_state_position_, memory_cell, true);
                output_gate.copyBlockFromMatrix(memory_cell, c.dim * 3, 0, 0, 0, c.dim, 1);
                h.state.elementwiseMultiplication(output_gate);
            }
        }
    }
}

void TensorflowFeatureScorerStateCarryover::finalize() const {
    for (auto& it : saved_state_map_) {
        SavedState& state = it.second;
        state.resetState();
    }
}

}  // namespace Tensorflow
