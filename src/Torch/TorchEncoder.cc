/** Copyright 2026 RWTH Aachen University. All rights reserved.
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

#include "TorchEncoder.hh"
#include <Torch/Tensor.hh>

#include <algorithm>
#include <cstring>

#pragma push_macro("ensure")
#undef ensure
#include <torch/torch.h>
#pragma pop_macro("ensure")

namespace Torch {

const Core::ParameterInt TorchEncoder::paramInputsPerOutput("inputs-per-output", "The number of input features needed to produce one output. Set to 0 to infer at runtime.", 0, 0);
const Core::ParameterInt TorchEncoder::paramInputStepSize("input-step-size", "The difference in the number of input features between the first features corresponding to two consecutive outputs. Set to 0 to copy value from inputs-per-output.", 0, 0);

TorchEncoder::TorchEncoder(Core::Configuration const& config, Nn::ModelCache& modelCache)
        : Core::Component(config),
          Precursor(config),
          inputsPerOutput_(paramInputsPerOutput(config)),
          inputStepSize_(paramInputStepSize(config)) {
    Core::Configuration modelConfig(config, "torch-model");
    torchModel_   = modelCache.getOrCreate<Model>(modelConfig.getSelection(), modelConfig);
    stateManager_ = StateManager::create(select("state-manager"));

    if (torchModel_->hasJsonIoSpec()) {
        initializeStatesFromModelSpec();
    }
    stateManager_->setInitialStates(stateVariables_);
}

void TorchEncoder::initializeStatesFromModelSpec() {
    if (!torchModel_->hasParsedIoSpec()) {
        criticalError("JSON I/O spec is configured, but no parsed I/O spec is available.");
    }

    const auto& ioSpec = torchModel_->parsedIoSpec();

    stateVariables_.clear();
    stateVariables_.reserve(ioSpec.states.size());

    for (const auto& stateSpec : ioSpec.states) {
        TorchStateVariable stateVar;
        stateVar.name        = stateSpec.name;
        stateVar.kind        = stateSpec.kind;
        stateVar.inputIndex  = ioSpec.statesInputIndex + stateSpec.inputIndex;
        stateVar.outputIndex = ioSpec.statesOutputIndex + stateSpec.outputIndex;
        stateVar.layer       = stateSpec.layer;
        stateVar.shape       = stateSpec.shape;
        stateVar.dtype       = stateSpec.dtype;

        stateVariables_.push_back(std::move(stateVar));
    }
}

void TorchEncoder::reset() {
    Encoder::reset();
    if (torchModel_->hasJsonIoSpec()) {
        initializeStatesFromModelSpec();
    }
    stateManager_->setInitialStates(stateVariables_);
}

void TorchEncoder::encode() {
    if (inputBuffer_.empty()) {
        return;
    }

    // Create session inputs
    size_t T_in = inputBuffer_.size();
    size_t F    = inputBuffer_.front().size();

    torch::Tensor features = torch::empty({1, static_cast<int64_t>(T_in), static_cast<int64_t>(F)}, torch::kFloat32);
    torch::Tensor lengths  = torch::tensor({static_cast<int64_t>(T_in)}, torch::kInt64);
    for (size_t t = 0; t < T_in; ++t) {
        std::memcpy(features[0][t].data_ptr<float>(), inputBuffer_[t].data(), F * sizeof(float));
    }

    std::vector<torch::Tensor> session_outputs;
    std::vector<torch::Tensor> session_inputs = torchModel_->makeInputs(features, lengths);
    stateManager_->extendInputs(session_inputs, stateVariables_);

    // Run session
    torchModel_->session().run(session_inputs, session_outputs);

    stateManager_->updateStates(session_outputs, stateVariables_);

    // Read model outputs and wrap the output tensor so its memory stays alive while DataView points into it
    torch::Tensor out = torchModel_->outputsFrom(session_outputs).contiguous();

    size_t T_out      = static_cast<size_t>(out.size(1));
    size_t outputSize = static_cast<size_t>(out.size(2));

    Torch::Tensor torchTensor(std::move(out));
    Nn::DataView  outputView(std::move(torchTensor));

    // Either use the configured input/output alignment or infer it from the observed input and output lengths
    size_t inputsPerOutput = (inputsPerOutput_ != 0ul) ? inputsPerOutput_ : (T_in / T_out + (T_in % T_out != 0));
    size_t inputStep       = (inputStepSize_ != 0ul) ? inputStepSize_ : inputsPerOutput;

    // Convert each model output frame into an EncodedSpan with the corresponding input frame range
    size_t startInput = 0ul;
    for (size_t t = 0ul; t < T_out; ++t) {
        size_t endInput = std::min(startInput + inputsPerOutput, T_in);
        outputBuffer_.push_back(
                Nn::EncodedSpan{
                        .encoding    = {outputView, outputSize, t * outputSize},
                        .input_start = startInput,
                        .input_end   = endInput});
        startInput = std::min(startInput + inputStep, std::max(T_in, 1ul) - 1ul);
    }
}

}  // namespace Torch