#include "OnnxEncoder.hh"

namespace Nn {

/*
 * =============================
 * === OnnxEncoder =============
 * =============================
 */

const Core::Choice OnnxEncoder::choiceSubsamplingType(
        // Expect 1:1 correspondence, throw away all outputs after the end of the input time axis
        // For example with input length 10 and output length 12, the last 2 outputs are ignored
        // Used for models with no subsampling or subsampling + upsampling of the same factor
        "none", SubsamplingType::None,
        // The last chunk of features is used even when they don't fill the usual size
        // For example with input length 17 and subsampling factor 4 the output length would be 5
        "ceil-division", SubsamplingType::CeilDivision,
        // The last chunk of features is thrown away when they doesn't fill the usual size
        // For example with input length 17 and subsampling factor 4 the output length would be 4
        "floor-division", SubsamplingType::FloorDivision,
        Core::Choice::endMark());

const Core::ParameterChoice OnnxEncoder::paramSubsamplingType(
        "subsampling-type",
        &choiceSubsamplingType,
        "Way that the output time axis is affected if input time is not cleanly divisible by the subsampling factor of the model.",
        FloorDivision);

OnnxEncoder::OnnxEncoder(Core::Configuration config)
        : Core::Component(config),
          Precursor(config),
          session_(select("session")),
          validator_(select("validator")),
          mapping_(select("io-map"), ioSpec_),
          featuresName_(mapping_.getOnnxName("features")),
          featuresSizeName_(mapping_.getOnnxName("features-size")),
          outputName_(mapping_.getOnnxName("outputs")),
          subsamplingType_(static_cast<SubsamplingType>(paramSubsamplingType(config))) {
    validator_.validate(ioSpec_, mapping_, session_);
}

const std::vector<Onnx::IOSpecification> OnnxEncoder::ioSpec_ = {
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

size_t OnnxEncoder::calcInputsPerOutput(size_t T_in, size_t T_out) const {
    switch (subsamplingType_) {
        case SubsamplingType::None:
            return 1ul;
        case SubsamplingType::FloorDivision:
            return T_in / T_out;
        case SubsamplingType::CeilDivision:
            return (T_in + T_out - 1ul) / T_out;
        default:
            error() << "Subsampling type not implemented";
            return 1ul;
    }
}

void OnnxEncoder::encode() {
    /*
     * Create session inputs
     */

    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;
    std::vector<Math::FastMatrix<f32>>               batchMat;  // will only contain a single element but packed in a vector for 1 x T x F Onnx value creation

    // Keep track of timestamps to be able to set them correctly for the outputs
    std::vector<Flow::Timestamp> inputTimestamps;
    inputTimestamps.reserve(std::min(inputBuffer_.size(), maxBufferSize_));

    // Initialize empty matrix of shape F x T.
    // Transposing is done because FastMatrix has col-major storage and this way each column is one feature vector
    batchMat.emplace_back(inputBuffer_.front()->size(), inputBuffer_.size());

    size_t T_in = 0ul;  // Keep track of current column in matrix and end up at total input timesteps
    while (not inputBuffer_.empty() and T_in < maxBufferSize_) {
        const auto& inputVectorRef = inputBuffer_.front();
        // Copy featureVector into next column of matrix and increment column index
        std::copy(inputVectorRef->begin(), inputVectorRef->end(), &(batchMat.front().at(0, T_in++)));
        inputTimestamps.push_back({inputVectorRef->getStartTime(), inputVectorRef->getEndTime()});
    }

    log("Encoder input features of shape (%zu x %u x %u)", batchMat.size(), batchMat.front().nColumns(), batchMat.front().nRows());

    sessionInputs.emplace_back(std::make_pair(featuresName_, Onnx::Value::create(batchMat, true)));  // transpose to 1 x T x F

    // features-size is an optional input
    if (featuresSizeName_ != "") {
        std::vector<s32> seqLengths({static_cast<int>(T_in)});
        sessionInputs.emplace_back(std::make_pair(featuresSizeName_, Onnx::Value::create(seqLengths)));
    }

    /*
     * Run session
     */

    auto t_start = std::chrono::steady_clock::now();

    // Run Onnx session with given inputs
    std::vector<Onnx::Value> sessionOutputs;
    session_.run(std::move(sessionInputs), {outputName_}, sessionOutputs);

    auto t_end     = std::chrono::steady_clock::now();
    auto t_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();  // in seconds

    auto inputTime = inputTimestamps.back().endTime() - inputTimestamps.front().startTime();
    log("Processed %f seconds of input in %f seconds; AM RTF: %f", inputTime, t_elapsed, t_elapsed / inputTime);

    log("Compured encoder state of shape (%zu x %zu x %zu)", sessionOutputs.front().dimSize(0), sessionOutputs.front().dimSize(1), sessionOutputs.front().dimSize(2));

    /*
     * Put outputs into buffer
     */

    // Calculate subsampling factor to set timestamps for output features
    size_t T_out = sessionOutputs.front().dimSize(1);

    size_t inputsPerOutput = calcInputsPerOutput(T_in, T_out);

    // TODO: How do existing Apptek FeatureScorers handle subsampling?
    for (size_t t = 0ul; t < T_out; ++t) {
        std::vector<f32> outputVec;
        sessionOutputs.front().get(0, t, outputVec);
        // TODO: Avoid data copying
        FeatureVectorRef outputVectorRef(new FeatureVector(outputVec));

        // outputs spans feature indices `t * inputsPerOutput` to `(t+1) * inputsPerOutput`
        // so start time is start of feature `t * inputsPerOutput` and end time is end of feature `(t+1) * inputsPerOutput - 1`
        // also make sure to cap off at last index to avoid out-of-bounds access
        outputVectorRef->setStartTime(inputTimestamps.at(std::min(inputTimestamps.size() - 1, t * inputsPerOutput)).startTime());
        outputVectorRef->setEndTime(inputTimestamps.at(std::min(inputTimestamps.size() - 1, (t + 1) * inputsPerOutput - 1)).endTime());
        outputBuffer_.push(outputVectorRef);
    }
}
}  // namespace Nn
