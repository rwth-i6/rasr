#include "OnnxEncoder.hh"
#include <memory>

namespace Nn {

/*
 * =============================
 * === OnnxEncoder =============
 * =============================
 */

const std::vector<Onnx::IOSpecification> encoderIoSpec = {
        Onnx::IOSpecification{
                "features",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -1, -2}, {1, -1, -2}}},
        Onnx::IOSpecification{
                "features-size",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{-1}, {1}}},
        Onnx::IOSpecification{
                "outputs",
                Onnx::IODirection::OUTPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -1, -2}, {1, -1, -2}}}};

OnnxEncoder::OnnxEncoder(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          onnxModel_(select("onnx-model"), encoderIoSpec),
          featuresName_(onnxModel_.mapping.getOnnxName("features")),
          featuresSizeName_(onnxModel_.mapping.getOnnxName("features-size")),
          outputName_(onnxModel_.mapping.getOnnxName("outputs")) {
}

std::pair<size_t, size_t> OnnxEncoder::validOutFrameRange(size_t T_in, size_t T_out) {
    return std::make_pair(0ul, T_out);
}

void OnnxEncoder::encode() {
    if (inputBuffer_.empty()) {
        return;
    }

    /*
     * Create session inputs
     */

    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;

    size_t T_in = inputBuffer_.size();
    size_t F    = featureSize_;

    std::vector<int64_t> featuresShape = {1l, static_cast<int64_t>(T_in), static_cast<int64_t>(F)};

    Onnx::Value value = Onnx::Value::createEmpty<f32>(featuresShape);

    for (size_t t = 0ul; t < T_in; ++t) {
        std::copy(inputBuffer_[t].get(), inputBuffer_[t].get() + F, value.data<f32>(0, t));
    }
    sessionInputs.emplace_back(std::make_pair(featuresName_, value));

    // features-size is an optional input
    if (featuresSizeName_ != "") {
        sessionInputs.emplace_back(std::make_pair(featuresSizeName_, Onnx::Value::create(std::vector<s32>{static_cast<int>(T_in)})));
    }

    /*
     * Run session
     */
    std::vector<Onnx::Value> sessionOutputs;
    onnxModel_.session.run(std::move(sessionInputs), {outputName_}, sessionOutputs);

    /*
     * Put outputs into buffer
     */

    auto onnxOutputValue        = sessionOutputs.front();
    auto onnxOutputValueWrapper = std::make_shared<Onnx::Value>(onnxOutputValue);

    size_t T_out;
    if (onnxOutputValue.numDims() == 3) {
        T_out       = onnxOutputValue.dimSize(1);
        outputSize_ = onnxOutputValue.dimSize(2);
    }
    else {
        T_out       = onnxOutputValue.dimSize(0);
        outputSize_ = onnxOutputValue.dimSize(1);
    }

    const auto& [rangeStart, rangeEnd] = validOutFrameRange(T_in, T_out);

    for (size_t t = rangeStart; t < rangeEnd; ++t) {
        // The custom deleter ties the lifetime of `scoreValue` to the lifetime
        // of `scorePtr` by capturing the `scoreValueWrapper` by value.
        auto scorePtr = std::shared_ptr<const f32[]>(
                onnxOutputValueWrapper->data<f32>() + t * outputSize_,
                [onnxOutputValueWrapper](const f32*) mutable {});
        outputBuffer_.push_back(scorePtr);
    }
}

/*
 * =============================
 * === ChunkedOnnxEncoder ======
 * =============================
 */
ChunkedOnnxEncoder::ChunkedOnnxEncoder(Core::Configuration const& config)
        : Core::Component(config),
          Encoder(config),
          ChunkedEncoder(config),
          OnnxEncoder(config) {}

std::pair<size_t, size_t> ChunkedOnnxEncoder::validOutFrameRange(size_t T_in, size_t T_out) {
    // Only outputs that correspond to chunk center are valid
    size_t historyOut = (T_out * currentHistoryFeatures_) / T_in;
    size_t centerOut  = (T_out * currentCenterFeatures_) / T_in;
    return std::make_pair(historyOut, historyOut + centerOut);
}

}  // namespace Nn
