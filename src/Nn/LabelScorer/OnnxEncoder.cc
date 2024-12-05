#include "OnnxEncoder.hh"

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

OnnxEncoder::OnnxEncoder(Core::Configuration config)
        : Core::Component(config),
          Precursor(config),
          onnxModel_(select("onnx-model"), encoderIoSpec),
          featuresName_(onnxModel_.mapping.getOnnxName("features")),
          featuresSizeName_(onnxModel_.mapping.getOnnxName("features-size")),
          outputName_(onnxModel_.mapping.getOnnxName("outputs")) {
}

void OnnxEncoder::encode() {
    if (numNewFeatures_ == 0ul) {
        return;
    }

    /*
     * Create session inputs
     */

    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;

    size_t T_in = inputBuffer_.size();

    if (featuresAreContiguous_) {
        std::vector<int64_t> featuresShape = {1l, static_cast<int64_t>(T_in), static_cast<int64_t>(featureSize_)};
        sessionInputs.emplace_back(std::make_pair(featuresName_, Onnx::Value::create(inputBuffer_.front(), featuresShape)));
    }
    else {
        std::vector<Math::FastMatrix<f32>> batchMat;  // will only contain a single element but packed in a vector for 1 x T x F Onnx value creation

        // Initialize empty matrix of shape F x T.
        // Transposing is done because FastMatrix has col-major storage and this way each column is one feature vector
        batchMat.emplace_back(featureSize_, T_in);

        for (size_t t = 0ul; t < T_in; ++t) {
            // Copy featureVector into next column of matrix and increment column index
            std::copy(inputBufferCopy_[t].begin(), inputBufferCopy_[t].end(), &(batchMat.front().at(0, t)));
        }
        sessionInputs.emplace_back(std::make_pair(featuresName_, Onnx::Value::create(batchMat, true)));  // transpose to 1 x T x F
    }

    // features-size is an optional input
    if (featuresSizeName_ != "") {
        std::vector<s32> seqLengths = {static_cast<int>(T_in)};
        sessionInputs.emplace_back(std::make_pair(featuresSizeName_, Onnx::Value::create(seqLengths)));
    }

    /*
     * Run session
     */

    // auto t_start = std::chrono::steady_clock::now();

    // Run Onnx session with given inputs
    std::vector<Onnx::Value> sessionOutputs;
    onnxModel_.session.run(std::move(sessionInputs), {outputName_}, sessionOutputs);

    // auto t_end     = std::chrono::steady_clock::now();
    // auto t_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();  // in seconds

    // auto inputTime = inputTimestamps.back().endTime() - inputTimestamps.front().startTime();
    // log("Processed %.4f seconds of input in %.4f seconds; AM RTF: %.4f", inputTime, t_elapsed, t_elapsed / inputTime);

    // log("Computed encoder state of shape (%zu x %zu x %zu)", sessionOutputs.front().dimSize(0), sessionOutputs.front().dimSize(1), sessionOutputs.front().dimSize(2));

    /*
     * Put outputs into buffer
     */

    size_t T_out = sessionOutputs.front().dimSize(1);
    outputSize_  = sessionOutputs.front().dimSize(2);

    for (size_t t = 0ul; t < T_out; ++t) {
        outputBuffer_.push_back(sessionOutputs.front().data<f32>(0, t));
    }
}

}  // namespace Nn
