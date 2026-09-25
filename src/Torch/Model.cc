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

#include "Model.hh"
#include <Core/Application.hh>

namespace {
const Core::ParameterString paramModelPath(
        "file",
        "Path to exported torch model package",
        "");

const Core::ParameterString paramIoSpecJson(
        "io-spec-json",
        "Path to JSON file describing the model input/output specification",
        "");

const Core::ParameterInt paramFeaturesPosition(
        "features-position",
        "Runtime input position of the feature tensor",
        0);

const Core::ParameterInt paramFeaturesLengthsPosition(
        "features-lengths-position",
        "Runtime input position of the input-length tensor",
        1);

const Core::ParameterInt paramOutputsPosition(
        "outputs-position",
        "Runtime output position of the outputs tensor",
        0);

static const Core::ParameterBool paramEnableValidation(
        "enable-validation",
        "Run model I/O validation during initialization",
        false);

std::string checkedModelPath(Core::Configuration const& config) {
    std::string path = paramModelPath(config);
    if (path.empty()) {
        Core::Application::us()->criticalError("Path to torch model not set");
    }
    return path;
}

size_t checkedPosition(Core::Configuration const& config, Core::ParameterInt const& param, char const* name) {
    int value = param(config);
    if (value < 0) {
        Core::Application::us()->criticalError("Position of %s must be >= 0", name);
    }
    return static_cast<size_t>(value);
}

void requireUniquePositions(size_t position_a, size_t position_b, char const* name_a, char const* name_b) {
    if (position_a == position_b) {
        Core::Application::us()->criticalError(
                "%s and %s must not use the same runtime position (%zu)",
                name_a, name_b, position_a);
    }
}
}  // namespace

namespace Torch {

Model::Model(Core::Configuration const& config)
        : Core::Component(config),
          session_(checkedModelPath(config)),
          ioSpec_(),
          featuresPosition_(0),
          featuresLengthsPosition_(1),
          outputsPosition_(0) {
    initializeIoSpecMode(config);

    if (hasJsonIoSpec()) {
        log("Using JSON I/O specification from '%s'", ioSpecJsonPath_.c_str());
        parseIoSpecJson();
        if (!ioSpec_.valid) {
            criticalError("Failed to load valid JSON I/O specification from '%s'", ioSpecJsonPath_.c_str());
        }
        if (ioSpec_.featuresInputIndex < 0 || ioSpec_.featuresLengthInputIndex < 0 || ioSpec_.outputsIndex < 0) {
            criticalError("JSON I/O specification is missing required top-level positions");
        }

        featuresPosition_        = static_cast<size_t>(ioSpec_.featuresInputIndex);
        featuresLengthsPosition_ = static_cast<size_t>(ioSpec_.featuresLengthInputIndex);
        outputsPosition_         = static_cast<size_t>(ioSpec_.outputsIndex);
    }
    else {
        featuresPosition_        = checkedPosition(config, paramFeaturesPosition, "features-position");
        featuresLengthsPosition_ = checkedPosition(config, paramFeaturesLengthsPosition, "features-lengths-position");
        outputsPosition_         = checkedPosition(config, paramOutputsPosition, "outputs-position");
    }

    requireUniquePositions(featuresPosition_, featuresLengthsPosition_, "features", "features-lengths");

    if (paramEnableValidation(config)) {
        IOValidator                  validator(select("io-validator"));
        std::vector<IOSpecification> validationSpec = {
                {"features", featuresPosition_, IODirection::INPUT, false, {torch::kFloat32}, {3}},
                {"lengths", featuresLengthsPosition_, IODirection::INPUT, false, {torch::kInt64}, {1}},
                {"outputs", outputsPosition_, IODirection::OUTPUT, false, {torch::kFloat32}, {3}},
        };
        if (!validator.validate(session_, validationSpec)) {
            criticalError("IO validation for the Torch model failed");
        }
    }
}

void Model::initializeIoSpecMode(Core::Configuration const& config) {
    ioSpecJsonPath_ = paramIoSpecJson(config);

    if (not ioSpecJsonPath_.empty()) {
        ioSpecMode_ = IoSpecMode::JsonSpec;
        log("'io-spec-json' is set; positional config parameters are ignored.");
        return;
    }

    ioSpecMode_ = IoSpecMode::ConfigSpec;
}

void Model::parseIoSpecJson() {
    ioSpec_ = ParsedIoSpec{};
    if (ioSpecJsonPath_.empty()) {
        return;
    }
    ioSpec_ = parseIoSpecJsonFile(ioSpecJsonPath_);
}

std::vector<torch::Tensor> Model::makeInputs(torch::Tensor const& features, torch::Tensor const& lengths) const {
    const size_t               numInputs = std::max(featuresPosition_, featuresLengthsPosition_) + 1;
    std::vector<torch::Tensor> inputs(numInputs);

    inputs[featuresPosition_]        = features;
    inputs[featuresLengthsPosition_] = lengths;
    return inputs;
}

torch::Tensor const& Model::outputsFrom(std::vector<torch::Tensor> const& outputs) const {
    if (outputsPosition_ >= outputs.size()) {
        criticalError("Outputs position %zu out of range for %zu outputs", outputsPosition_, outputs.size());
    }
    return outputs[outputsPosition_];
}

}  // namespace Torch