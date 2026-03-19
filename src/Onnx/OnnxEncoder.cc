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

const Core::ParameterInt OnnxEncoder::paramInputsPerOutput("inputs-per-output", "The number of input features needed to produce one output. Set to 0 to infer at runtime.", 0, 0);
const Core::ParameterInt OnnxEncoder::paramInputStepSize("input-step-size", "The difference in the number of input features between the first features corresponding to two consecutive outputs. Set to 0 to copy value from inputs-per-output.", 0, 0);

OnnxEncoder::OnnxEncoder(Core::Configuration const& config, Nn::EncoderModelCache& cachedModel)
        : Core::Component(config),
          Precursor(config),
          inputsPerOutput_(paramInputsPerOutput(config)),
          inputStepSize_(paramInputStepSize(config)) {
    if (cachedModel.empty()) {
        cachedModel.put(std::make_shared<Model>(select("onnx-model"), encoderIoSpec));
    }
    onnxModel_        = cachedModel.get<Model>();
    featuresName_     = onnxModel_->mapping.getOnnxName("features");
    featuresSizeName_ = onnxModel_->mapping.getOnnxName("features-size");
    outputName_       = onnxModel_->mapping.getOnnxName("outputs");
    stateManager_     = StateManager::create(select("state-manager"));
    stateVariables_   = onnxModel_->session.getStateVariablesMetadata();
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
    onnxModel_->session.run(std::move(session_inputs), output_names, session_outputs);

    // Put outputs into buffer
    size_t T_out       = session_outputs.front().dimSize(1);
    size_t output_size = session_outputs.front().dimSize(2);

    // Make "global" DataView from output value so that feature slice DataViews can be created from it that ref-count the original value
    Nn::DataView onnx_output_view(std::move(session_outputs.front()));

    size_t outputs_per_input = (inputsPerOutput_ != 0ul) ? inputsPerOutput_ : (T_in / T_out + (T_in % T_out != 0));
    size_t input_step        = (inputStepSize_ != 0ul) ? inputStepSize_ : outputs_per_input;
    size_t start_input       = 0ul;
    for (size_t t = 0ul; t < T_out; ++t) {
        size_t end_input = std::min(start_input + outputs_per_input, T_in);
        outputBuffer_.push_back(Nn::EncodedSpan{{onnx_output_view, output_size, t * output_size}, start_input, end_input});
        start_input = std::min(start_input + input_step, std::max(T_in, 1ul) - 1ul);
    }

    // Get new states
    std::vector<Value> output_states;
    for (size_t i = 1ul; i < session_outputs.size(); i++) {  // other model outputs
        output_states.emplace_back(std::move(session_outputs[i]));
    }
    stateManager_->updateStates(output_states);
}

}  // namespace Onnx
