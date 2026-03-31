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

OnnxEncoder::SessionRunResult OnnxEncoder::runSession(size_t inputStartIndex, size_t nInputs, size_t leftZeroPadding, size_t rightZeroPadding) {
    verify(inputStartIndex + nInputs <= inputBuffer_.size());
    verify(nInputs > 0ul);

    std::vector<std::pair<std::string, Value>> sessionInputs;
    std::vector<std::string>                   outputNames{{outputName_}};

    size_t F = inputBuffer_[inputStartIndex].size();

    // Create session inputs
    std::vector<int64_t> featureShape = {1l, static_cast<int64_t>(leftZeroPadding + nInputs + rightZeroPadding), static_cast<int64_t>(F)};
    Value                value        = Value::createEmpty<f32>(featureShape);

    if (leftZeroPadding > 0) {
        std::fill(value.data<f32>(0, 0), value.data<f32>(0, 0) + leftZeroPadding * F, 0.0f);
    }
    for (size_t t = leftZeroPadding; t < leftZeroPadding + nInputs; ++t) {
        std::copy(inputBuffer_[inputStartIndex + t - leftZeroPadding].data(), inputBuffer_[inputStartIndex + t - leftZeroPadding].data() + F, value.data<f32>(0, t));
    }
    if (rightZeroPadding > 0) {
        std::fill(value.data<f32>(0, leftZeroPadding + nInputs), value.data<f32>(0, leftZeroPadding + nInputs) + rightZeroPadding * F, 0.0f);
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

const Core::ParameterBool ChunkedOnnxEncoder::paramZeroPadding(
        "zero-padding",
        "If set, add zero-padding features at beginning and end of segment so that these chunks have the same total size as the others.",
        false);

const Core::Choice ChunkedOnnxEncoder::windowTypeChoice(
        "none", WindowType::None,
        "triangular", WindowType::Triangular,
        "hamming", WindowType::Hamming,
        Core::Choice::endMark());

const Core::ParameterChoice ChunkedOnnxEncoder::paramWindowType(
        "window-type",
        &windowTypeChoice,
        "Window function used to weight overlapping chunk outputs.",
        WindowType::Triangular);

const Core::Choice ChunkedOnnxEncoder::interpolationModeChoice(
        "no-interpolation", InterpolationMode::NoInterpolation,
        "linear", InterpolationMode::Linear,
        "log-linear", InterpolationMode::LogLinear,
        "neglog-linear", InterpolationMode::NegLogLinear,
        Core::Choice::endMark());

const Core::ParameterChoice ChunkedOnnxEncoder::paramInterpolationMode(
        "interpolation-mode",
        &interpolationModeChoice,
        "How overlapping chunk outputs are interpolated.",
        InterpolationMode::NoInterpolation);

ChunkedOnnxEncoder::ChunkedOnnxEncoder(Core::Configuration const& config, Nn::EncoderModelCache& cachedModel)
        : Core::Component(config),
          Precursor(config, cachedModel),
          chunkSize_(paramChunkSize(config)),
          stepSize_(paramStepSize(config)),
          leftPadding_(paramLeftPadding(config)),
          rightPadding_(paramRightPadding(config)),
          zeroPadding_(paramZeroPadding(config)),
          interpolationMode_(static_cast<InterpolationMode>(paramInterpolationMode(config))),
          chunkCenterStart_(0ul),
          numDiscardedFeatures_(0ul),
          pendingOutputs_() {
    if (inputsPerOutput_ > chunkSize_) {
        error("Chunk size must be large enough to produce at least one output");
    }
    initWindow(static_cast<WindowType>(paramWindowType(config)));
}

void ChunkedOnnxEncoder::reset() {
    Precursor::reset();
    chunkCenterStart_     = 0ul;
    numDiscardedFeatures_ = 0ul;
    pendingOutputs_.clear();
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

    // Figure out which inputs need to be supplied to the session and which parts belong to the chunk center
    // If zero-padding is enabled, also count how many zero-features need to be prepended or appended to the session input
    size_t chunkStart     = chunkCenterStart_ > leftPadding_ ? chunkCenterStart_ - leftPadding_ : 0ul;
    size_t chunkCenterEnd = std::min(chunkCenterStart_ + chunkSize_, availableEnd);
    size_t chunkEnd       = std::min(chunkCenterEnd + rightPadding_, availableEnd);

    size_t prependZeroCount = 0ul;
    if (zeroPadding_ and leftPadding_ > chunkCenterStart_ - chunkStart) {
        prependZeroCount = leftPadding_ - (chunkCenterStart_ - chunkStart);
    }
    size_t appendZeroCount = 0ul;
    if (zeroPadding_ and rightPadding_ > chunkEnd - chunkCenterEnd) {
        appendZeroCount = rightPadding_ - (chunkEnd - chunkCenterEnd);
    }

    // If we need to access e.g. feature 17 and so far 10 features have been discarded,
    // feature 17 will be in inputBuffer_[7]
    size_t inputStartIndex = chunkStart - numDiscardedFeatures_;
    size_t nInputs         = chunkEnd - chunkStart;

    auto [outputView, nOutputs, outputSize] = runSession(inputStartIndex, nInputs, prependZeroCount, appendZeroCount);

    size_t totalSessionInputs = prependZeroCount + nInputs + appendZeroCount;
    size_t inputsPerOutput    = (inputsPerOutput_ != 0ul) ? inputsPerOutput_ : (totalSessionInputs / nOutputs + (totalSessionInputs % nOutputs != 0ul));
    size_t inputStep          = (inputStepSize_ != 0ul) ? inputStepSize_ : inputsPerOutput;

    // Buffer all outputs for which the start input lies inside the interval [chunkCenterStart_, chunkCenterEnd)
    // The rest corresponds to the padding frames and gets skipped.
    size_t startInput     = chunkStart;
    auto   weightIterator = window_.begin();
    for (size_t t = 0ul; t < nOutputs; ++t) {
        size_t endInput = std::min(startInput + inputsPerOutput, availableEnd);
        if (startInput >= chunkCenterStart_) {
            Nn::EncodedSpan output{
                    .encoding    = {outputView, outputSize, t * outputSize},
                    .input_start = startInput,
                    .input_end   = endInput};
            if (interpolationMode_ == InterpolationMode::NoInterpolation) {
                outputBuffer_.push_back(output);
            }
            else {
                accumulatePendingOutput(output, *weightIterator);
            }
            ++weightIterator;
        }
        startInput += inputStep;

        if (startInput >= chunkCenterEnd) {
            break;
        }
    }

    bool isLastChunk  = not expectMoreFeatures_ and chunkCenterEnd == availableEnd;
    chunkCenterStart_ = isLastChunk ? availableEnd : chunkCenterStart_ + stepSize_;

    if (interpolationMode_ != InterpolationMode::NoInterpolation) {
        if (isLastChunk) {
            flushPendingOutputsUpTo(Core::Type<size_t>::max);
        }
        else {
            flushPendingOutputsUpTo(chunkCenterStart_);
        }
    }
}

void ChunkedOnnxEncoder::postEncodeCleanup() {
    size_t firstNeededIndex = chunkCenterStart_ > leftPadding_ ? chunkCenterStart_ - leftPadding_ : 0ul;
    size_t numToDiscard     = std::min(firstNeededIndex - numDiscardedFeatures_, inputBuffer_.size());

    inputBuffer_.erase(inputBuffer_.begin(), inputBuffer_.begin() + numToDiscard);
    numDiscardedFeatures_ += numToDiscard;
}

void ChunkedOnnxEncoder::PendingOutput::finalize(ChunkedOnnxEncoder::InterpolationMode mode) {
    switch (mode) {
        case InterpolationMode::Linear:
            std::transform(
                    accumulator.get(),
                    accumulator.get() + accumulatorSize,
                    accumulator.get(),
                    [this](f32 value) { return value / totalWeight; });
            break;
        case InterpolationMode::LogLinear:
        case InterpolationMode::NegLogLinear:
            std::transform(
                    accumulator.get(),
                    accumulator.get() + accumulatorSize,
                    accumulator.get(),
                    [this](f32 value) { return value - totalWeight; });
            break;
        default:
            break;
    }
}

void ChunkedOnnxEncoder::initWindow(WindowType windowType) {
    if (inputStepSize_ == 0ul) {
        // We can't calculate the true window size based on the chunk size if the parameter hasn't been set.
        if (interpolationMode_ == InterpolationMode::NoInterpolation or windowType == WindowType::None) {
            // If the weights don't depend on the window size, we don't care about the true window size and just
            // resize it with an upper bound.
            window_.resize(chunkSize_);
        }
        else {
            error("Input step size must be set so that the encoder can calculate how many outputs are expected per chunk.");
        }
    }
    else {
        // Ceildiv
        window_.resize((chunkSize_ + inputStepSize_ - 1) / inputStepSize_);
    }

    switch (windowType) {
        case WindowType::None:
            std::fill(window_.begin(), window_.end(), 1.0f);
            break;
        case WindowType::Triangular: {
            f32 len = window_.size() + 1;
            for (size_t t = 0ul; t < window_.size() / 2; ++t) {
                window_[t] = static_cast<f32>(t + 1) / len;
            }
            for (size_t t = window_.size() / 2; t < window_.size(); ++t) {
                window_[t] = 1.0 - static_cast<f32>(t + 1) / len;
            }
            break;
        }
        case WindowType::Hamming: {
            if (window_.size() == 1ul) {
                window_.front() = 1.0f;
                break;
            }
            f32 const alpha = 0.54;
            f32 const pi    = std::acos(-1.0);
            for (size_t n = 0ul; n < window_.size(); ++n) {
                window_[n] = alpha - (1.0 - alpha) * std::cos((2.0 * pi * static_cast<f32>(n)) / static_cast<f32>(window_.size() - 1ul));
            }
            break;
        }
    }

    switch (interpolationMode_) {
        case InterpolationMode::LogLinear:
            std::transform(
                    window_.begin(),
                    window_.end(),
                    window_.begin(),
                    [](f32 value) { return std::log(value); });
            break;
        case InterpolationMode::NegLogLinear:
            std::transform(
                    window_.begin(),
                    window_.end(),
                    window_.begin(),
                    [](f32 value) { return -std::log(value); });
            break;
        default:
            break;
    }
}

void ChunkedOnnxEncoder::flushPendingOutputsUpTo(size_t inputStart) {
    auto it = pendingOutputs_.begin();
    while (it != pendingOutputs_.end() and it->first < inputStart) {
        PendingOutput& pendingOutput = it->second;
        pendingOutput.finalize(interpolationMode_);

        outputBuffer_.push_back(
                Nn::EncodedSpan{
                        .encoding    = {pendingOutput.accumulator, pendingOutput.accumulatorSize},
                        .input_start = it->first,
                        .input_end   = pendingOutput.inputEnd,
                });
        it = pendingOutputs_.erase(it);
    }
}

void ChunkedOnnxEncoder::accumulatePendingOutput(Nn::EncodedSpan data, f32 weight) {
    verify(interpolationMode_ != InterpolationMode::NoInterpolation);

    // Check if matching output is already pending
    auto [it, inserted] = pendingOutputs_.emplace(
            data.input_start,
            PendingOutput{
                    .inputEnd        = data.input_end,
                    .accumulator     = std::shared_ptr<f32[]>(new f32[data.encoding.size()], std::default_delete<f32[]>()),
                    .accumulatorSize = data.encoding.size(),
                    .totalWeight     = weight});

    PendingOutput& pendingOutput = it->second;
    if (inserted) {
        // No matching output exists, so initialize the accumulator with the weighted encoding
        pendingOutput.inputEnd = data.input_end;
        std::function<f32(f32)> weightingFunction;
        switch (interpolationMode_) {
            case InterpolationMode::Linear:
                weightingFunction = std::function<f32(f32)>([weight](f32 value) { return weight * value; });
                break;
            case InterpolationMode::LogLinear:
            case InterpolationMode::NegLogLinear:
                weightingFunction = std::function<f32(f32)>([weight](f32 value) { return weight + value; });
                break;
            default:
                error("Encountered unexpected interpolation mode");
                break;
        }
        std::transform(
                data.encoding.data(),
                data.encoding.data() + data.encoding.size(),
                pendingOutput.accumulator.get(),
                weightingFunction);
        pendingOutput.totalWeight = weight;
        return;
    }

    // A matching output exists, so we combine it with the weighted encoding
    std::function<f32(f32, f32)> interpolationFunction;
    switch (interpolationMode_) {
        case InterpolationMode::Linear:
            interpolationFunction = std::function<f32(f32, f32)>([weight](f32 accumValue, f32 encodingValue) { return accumValue + encodingValue * weight; });
            pendingOutput.totalWeight += weight;
            break;
        case InterpolationMode::LogLinear:
            interpolationFunction     = std::function<f32(f32, f32)>([weight](f32 accumValue, f32 encodingValue) { return -Math::scoreSum(-accumValue, -(encodingValue + weight)); });
            pendingOutput.totalWeight = -Math::scoreSum(-pendingOutput.totalWeight, -weight);
            break;
        case InterpolationMode::NegLogLinear:
            interpolationFunction     = std::function<f32(f32, f32)>([weight](f32 accumValue, f32 encodingValue) { return Math::scoreSum(accumValue, encodingValue + weight); });
            pendingOutput.totalWeight = Math::scoreSum(pendingOutput.totalWeight, weight);
            break;
        default:
            error("Encountered unexpected interpolation mode");
            break;
    }

    require(pendingOutput.accumulatorSize == data.encoding.size());
    pendingOutput.inputEnd = std::max(pendingOutput.inputEnd, data.input_end);
    std::transform(
            pendingOutput.accumulator.get(),
            pendingOutput.accumulator.get() + pendingOutput.accumulatorSize,
            data.encoding.data(),
            pendingOutput.accumulator.get(),
            interpolationFunction);
}

}  // namespace Onnx
