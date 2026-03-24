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

/*
 * ============================
 * ======= OnnxEncoder ========
 * ============================
 */

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

OnnxEncoder::SessionRunResult OnnxEncoder::runSession(size_t inputStartIndex, size_t nInputs) {
    verify(inputStartIndex + nInputs <= inputBuffer_.size());
    verify(nInputs > 0ul);

    std::vector<std::pair<std::string, Value>> sessionInputs;
    std::vector<std::string>                   outputNames{{outputName_}};

    size_t F = inputBuffer_[inputStartIndex].size();

    // Create session inputs
    std::vector<int64_t> featureShape = {1l, static_cast<int64_t>(nInputs), static_cast<int64_t>(F)};
    Value                value        = Value::createEmpty<f32>(featureShape);

    for (size_t t = 0ul; t < nInputs; ++t) {
        std::copy(inputBuffer_[inputStartIndex + t].data(), inputBuffer_[inputStartIndex + t].data() + F, value.data<f32>(0, t));
    }
    sessionInputs.emplace_back(featuresName_, std::move(value));

    if (featuresSizeName_ != "") {
        sessionInputs.emplace_back(featuresSizeName_, Value::create(std::vector<s32>{static_cast<int>(nInputs)}));
    }

    stateManager_->extendFeedDict(sessionInputs, stateVariables_);
    stateManager_->extendTargets(outputNames, stateVariables_);

    // Run session
    std::vector<Value> sessionOutputs;
    onnxModel_->session.run(std::move(sessionInputs), outputNames, sessionOutputs);

    // Retrieve outputs
    size_t T_out      = sessionOutputs.front().dimSize(1);
    size_t outputSize = sessionOutputs.front().dimSize(2);

    Nn::DataView outputView(std::move(sessionOutputs.front()));

    std::vector<Value> outputStates;
    for (size_t i = 1ul; i < sessionOutputs.size(); ++i) {
        outputStates.emplace_back(std::move(sessionOutputs[i]));
    }
    stateManager_->updateStates(outputStates);

    return {.outputView = std::move(outputView), .nOutputs = T_out, .outputSize = outputSize};
}

void OnnxEncoder::encode() {
    if (inputBuffer_.empty()) {
        return;
    }

    size_t nInputs = inputBuffer_.size();

    auto [outputView, nOutputs, outputSize] = runSession(0ul, nInputs);

    size_t inputsPerOutput = (inputsPerOutput_ != 0ul) ? inputsPerOutput_ : (nInputs / nOutputs + (nInputs % nOutputs != 0ul));
    size_t inputStep       = (inputStepSize_ != 0ul) ? inputStepSize_ : inputsPerOutput;
    size_t startInput      = 0ul;

    for (size_t t = 0ul; t < nOutputs; ++t) {
        size_t endInput = std::min(startInput + inputsPerOutput, nInputs);
        outputBuffer_.push_back(
                Nn::EncodedSpan{
                        .encoding    = {outputView, outputSize, t * outputSize},
                        .input_start = startInput,
                        .input_end   = endInput});
        startInput = std::min(startInput + inputStep, std::max(nInputs, 1ul) - 1ul);
    }
}

/*
 * ============================
 * ==== ChunkedOnnxEncoder ====
 * ============================
 */

const Core::ParameterInt ChunkedOnnxEncoder::paramChunkSize(
        "chunk-size",
        "The number of central input features processed in one chunk.",
        1,
        1);

const Core::ParameterInt ChunkedOnnxEncoder::paramStepSize(
        "step-size",
        "The shift in central input features between two consecutive chunks.",
        1,
        1);

const Core::ParameterInt ChunkedOnnxEncoder::paramLeftPadding(
        "left-padding",
        "The number of input features of left context to prepend to each chunk.",
        0,
        0);

const Core::ParameterInt ChunkedOnnxEncoder::paramRightPadding(
        "right-padding",
        "The number of input features of right context to append to each chunk.",
        0,
        0);

ChunkedOnnxEncoder::ChunkedOnnxEncoder(Core::Configuration const& config, Nn::EncoderModelCache& cachedModel)
        : Core::Component(config),
          Precursor(config, cachedModel),
          chunkSize_(paramChunkSize(config)),
          stepSize_(paramStepSize(config)),
          leftPadding_(paramLeftPadding(config)),
          rightPadding_(paramRightPadding(config)),
          chunkCenterStart_(0ul),
          numDiscardedFeatures_(0ul) {
    if (not stateVariables_.empty()) {
        // With overlap, state variable outputs of the previous chunk don't work as inputs for the next chunk
        error() << "ChunkedOnnxEncoder does not support state variables.";
    }
}

void ChunkedOnnxEncoder::reset() {
    Precursor::reset();
    chunkCenterStart_     = 0ul;
    numDiscardedFeatures_ = 0ul;
}

bool ChunkedOnnxEncoder::canEncode() const {
    size_t availableEnd = numDiscardedFeatures_ + inputBuffer_.size();

    if (not expectMoreFeatures_) {
        return availableEnd > chunkCenterStart_;
    }

    return availableEnd >= chunkCenterStart_ + chunkSize_ + rightPadding_;
}

void ChunkedOnnxEncoder::encode() {
    size_t availableEnd = numDiscardedFeatures_ + inputBuffer_.size();

    size_t chunkStart     = chunkCenterStart_ > leftPadding_ ? chunkCenterStart_ - leftPadding_ : 0ul;
    size_t chunkCenterEnd = std::min(chunkCenterStart_ + chunkSize_, availableEnd);
    size_t chunkEnd       = std::min(chunkCenterEnd + rightPadding_, availableEnd);

    // If we need to access e.g. feature 17 and so far 10 features have been discarded,
    // feature 17 will be in inputBuffer_[7]
    size_t inputStartIndex = chunkStart - numDiscardedFeatures_;
    size_t nInputs         = chunkEnd - chunkStart;

    auto [outputView, nOutputs, outputSize] = runSession(inputStartIndex, nInputs);

    size_t inputsPerOutput = (inputsPerOutput_ != 0ul) ? inputsPerOutput_ : (nInputs / nOutputs + (nInputs % nOutputs != 0ul));
    size_t inputStep       = (inputStepSize_ != 0ul) ? inputStepSize_ : inputsPerOutput;

    // Buffer all outputs for which the start input lies inside the interval [chunkCenterStart_, chunkCenterEnd)
    // The rest corresponds to the padding frames and gets skipped
    size_t startInput = chunkStart;
    for (size_t t = 0ul; t < nOutputs; ++t) {
        size_t endInput = std::min(startInput + inputsPerOutput, availableEnd);
        if (startInput >= chunkCenterStart_) {
            outputBuffer_.push_back(
                    Nn::EncodedSpan{
                            .encoding    = {outputView, outputSize, t * outputSize},
                            .input_start = startInput,
                            .input_end   = endInput});
        }
        startInput += inputStep;

        if (startInput >= chunkCenterEnd) {
            break;
        }
    }

    if (not expectMoreFeatures_ and chunkCenterEnd == availableEnd) {
        // At segment end, stop once all inputs have been covered by a chunk center
        chunkCenterStart_ = availableEnd;
    }
    else {
        // Move to next chunk after encoding
        chunkCenterStart_ += stepSize_;
    }
}

void ChunkedOnnxEncoder::postEncodeCleanup() {
    size_t firstNeededIndex = chunkCenterStart_ > leftPadding_ ? chunkCenterStart_ - leftPadding_ : 0ul;
    size_t numToDiscard     = std::min(firstNeededIndex - numDiscardedFeatures_, inputBuffer_.size());

    inputBuffer_.erase(inputBuffer_.begin(), inputBuffer_.begin() + numToDiscard);
    numDiscardedFeatures_ += numToDiscard;
}

}  // namespace Onnx
