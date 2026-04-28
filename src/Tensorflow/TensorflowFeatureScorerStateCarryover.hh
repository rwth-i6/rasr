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
#ifndef NN_TENSORFLOWFEATURESCORERSTATECARRYOVER_HH
#define NN_TENSORFLOWFEATURESCORERSTATECARRYOVER_HH

#include "TensorflowFeatureScorer.hh"

namespace Tensorflow {

/**
 * This scorer is similar to TensorflowFeatureScorer but additionally keeps track
 * of the hidden states in order to initialize the next round of scoring.
 */
class TensorflowFeatureScorerStateCarryover : public TensorflowFeatureScorer {
private:
    /**
     * SavedState wraps the names of the initializer tensor and NativeLstm2 LSTM
     * cell output tensors corresponding to C and H; see
     * https://returnn.readthedocs.io/en/latest/api/NativeOp.html#NativeOp.NativeLstm2 and
     * https://github.com/rwth-i6/returnn/blob/master/NativeOp.py#L1364-L1769
     *
     * initializer_name   is of the form "my_nativelstm2_fw_layer_name/rec/keep_state_{c,h}/Assign"
     * initial_value_name is of the form "my_nativelstm2_fw_layer_name/rec/zeros:0"
     * lstm_output is the name of the tensor that corresponds to NativeLstm2's C and H, e.g.
     *                                   "my_nativelstm2_fw_layer_name/rec/NativeLstm2:1"
     */
    struct SavedState {
        size_t                dim;
        std::string           initializer_name, initial_value_name;
        std::string           lstm_output;
        Math::FastMatrix<f32> state;

        SavedState() {}
        SavedState(const Tensorflow::Variable& var)
                : dim(var.shape.back()),
                  initializer_name(var.initializer_name),
                  initial_value_name(var.initial_value_name) {
            require_gt(dim, 0);
            size_t pos = initializer_name.find("/keep_state_");
            require(pos != std::string::npos);
            lstm_output = initializer_name.substr(0, pos + 1);

            if (initializer_name.find("keep_state_c") != std::string::npos) {
                lstm_output += "NativeLstm2:1";
            }
            else if (initializer_name.find("keep_state_h") != std::string::npos) {
                lstm_output += "NativeLstm2:2";
            }
            else {
                Core::Application::us()->error("can't parse initializer_name '%s'", initializer_name.c_str());
            }
            state.resize(dim, 1);
            resetState();
        }

        void resetState() {
            state.setToZero();
        }
    };

    virtual void _compute();

    size_t save_state_position_;
    // mutable, because we modify it from getScore() and finalize()
    mutable std::map<std::string, SavedState> saved_state_map_;
    std::vector<std::string>                  initializer_tensor_names_;

public:
    TensorflowFeatureScorerStateCarryover(const Core::Configuration& c, Core::Ref<const Mm::MixtureSet> mixtureSet);
    virtual ~TensorflowFeatureScorerStateCarryover() = default;

    /**
     * This is called at the end of each segment. Will reset saved states back to zero.
     */
    virtual void finalize() const;
};
}  // namespace Tensorflow

#endif  // NN_TENSORFLOWFEATURESCORERSTATECARRYOVER_HH
