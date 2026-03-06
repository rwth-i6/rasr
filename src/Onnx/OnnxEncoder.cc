/** Copyright 2025 RWTH Aachen University. All rights reserved.
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

#include "OnnxEncoder.hh"

namespace Onnx {

const std::vector<IOSpecification> encoderIoSpec = {
        IOSpecification{
                "features",
                IODirection::INPUT,
                false,
                {ValueType::TENSOR},
                {ValueDataType::FLOAT},
                {{-1, -1, -2}, {1, -1, -2}}},
        IOSpecification{
                "features-size",
                IODirection::INPUT,
                true,
                {ValueType::TENSOR},
                {ValueDataType::INT32},
                {{-1}, {1}}},
        IOSpecification{
                "outputs",
                IODirection::OUTPUT,
                false,
                {ValueType::TENSOR},
                {ValueDataType::FLOAT},
                {{-1, -1, -2}, {1, -1, -2}}}};

OnnxEncoder::OnnxEncoder(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          onnxModel_(select("onnx-model"), encoderIoSpec),
          featuresName_(onnxModel_.mapping.getOnnxName("features")),
          featuresSizeName_(onnxModel_.mapping.getOnnxName("features-size")),
          outputName_(onnxModel_.mapping.getOnnxName("outputs")) {
    stateManager_   = StateManager::create(select("state-manager"));
    stateVariables_ = onnxModel_.session.getStateVariablesMetadata();
    stateManager_->setInitialStates(stateVariables_);
}

void OnnxEncoder::reset() {
    Encoder::reset();
    stateManager_->setInitialStates(stateVariables_);
}

void OnnxEncoder::encode() {
    if (inputBuffer_.empty()) {
        return;
    }

    // Create session inputs/outputs
    std::vector<std::pair<std::string, Value>> session_inputs;
    std::vector<std::string>                   output_names{{outputName_}};

    size_t T_in = inputBuffer_.size();
    size_t F    = inputBuffer_.front().size();

    std::vector<int64_t> feature_shape = {1l, static_cast<int64_t>(T_in), static_cast<int64_t>(F)};

    Value value = Value::createEmpty<f32>(feature_shape);

    for (size_t t = 0ul; t < T_in; ++t) {
        std::copy(inputBuffer_[t].data(), inputBuffer_[t].data() + F, value.data<f32>(0, t));
    }
    session_inputs.emplace_back(std::make_pair(featuresName_, std::move(value)));

    // features-size is an optional input
    if (featuresSizeName_ != "") {
        session_inputs.emplace_back(std::make_pair(featuresSizeName_, Value::create(std::vector<s32>{static_cast<int>(T_in)})));
    }

    // input and output states
    stateManager_->extendFeedDict(session_inputs, stateVariables_);
    stateManager_->extendTargets(output_names, stateVariables_);

    // Run session
    std::vector<Value> session_outputs;
    onnxModel_.session.run(std::move(session_inputs), output_names, session_outputs);

    // Put outputs into buffer
    size_t T_out       = session_outputs.front().dimSize(1);
    size_t output_size = session_outputs.front().dimSize(2);

    // Make "global" DataView from output value so that feature slice DataViews can be created from it that ref-count the original value
    Nn::DataView onnx_output_view(std::move(session_outputs.front()));

    size_t outputs_per_input = T_in / T_out + (T_in % T_out != 0);
    for (size_t t = 0ul; t < T_out; ++t) {
        outputBuffer_.push_back(Nn::EncodedSpan{{onnx_output_view, output_size, t * output_size}, t * outputs_per_input, std::min((t+1) * outputs_per_input, T_in)});
    }

    // Get new states
    std::vector<Value> output_states;
    for (size_t i = 1ul; i < session_outputs.size(); i++) {  // other model outputs
        output_states.emplace_back(std::move(session_outputs[i]));
    }
    stateManager_->updateStates(output_states);
}

}  // namespace Onnx
