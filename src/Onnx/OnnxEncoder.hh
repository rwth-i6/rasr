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

#ifndef ONNX_ENCODER_HH
#define ONNX_ENCODER_HH

#include <map>

#include <Nn/LabelScorer/Encoder.hh>
#include <Nn/LabelScorer/EncoderFactory.hh>

#include "Model.hh"
#include "StateManager.hh"

namespace Onnx {

// Encoder that runs the input features through an ONNX model
class OnnxEncoder : public Nn::Encoder {
public:
    using Precursor = Nn::Encoder;

    static const Core::ParameterInt paramInputsPerOutput;
    static const Core::ParameterInt paramInputStepSize;

    OnnxEncoder(Core::Configuration const& config, Nn::EncoderModelCache& cachedModel);
    virtual ~OnnxEncoder() = default;

    // Clear buffers and reset segment end flag.
    virtual void reset() override;

protected:
    struct SessionRunResult {
        Nn::DataView outputView;
        size_t       nOutputs;
        size_t       outputSize;
    };

    // Encode features inside the input buffer and put the results into the output buffer
    virtual void encode() override;

    // Runs onnxModel_ on features from inputBuffer_[inputStartIndex : inputStartIndex + nInputs]
    // Potentially add zero-padding features to inputs before running session
    SessionRunResult runSession(size_t inputStartIndex, size_t nInputs, size_t leftZeroPadding = 0ul, size_t rightZeroPadding = 0ul);

    const size_t inputsPerOutput_;
    const size_t inputStepSize_;

    std::shared_ptr<Model> onnxModel_;
    std::string            featuresName_;
    std::string            featuresSizeName_;
    std::string            outputName_;

    std::unique_ptr<StateManager>  stateManager_;
    std::vector<OnnxStateVariable> stateVariables_;
};

/*
 * Encoder that chunks the input features before running them through an ONNX model
 * For example with chunk-size = 50, step-size = 25, left-padding = 10, right-padding = 5
 * and 90 feature inputs in total
 *  - The first chunk consists of features 0 to 55 and the outputs corresponding to
 *    features 0 to 50 are returned
 *  - The second chunk consists of features 15 to 80 and the outputs corresponding to
 *    features 25 to 75 are returned
 *  - The third chunk consists of features 40 to 90 and the outputs corresponding to
 *    features 50 to 90 are returned
 */
class ChunkedOnnxEncoder : public OnnxEncoder {
public:
    using Precursor = OnnxEncoder;

    static const Core::ParameterInt    paramChunkSize;
    static const Core::ParameterInt    paramStepSize;
    static const Core::ParameterInt    paramLeftPadding;
    static const Core::ParameterInt    paramRightPadding;
    static const Core::ParameterBool   paramZeroPadding;
    static const Core::Choice          windowTypeChoice;
    static const Core::ParameterChoice paramWindowType;
    static const Core::Choice          interpolationModeChoice;
    static const Core::ParameterChoice paramInterpolationMode;

    ChunkedOnnxEncoder(Core::Configuration const& config, Nn::EncoderModelCache& cachedModel);

    virtual void reset() override;

protected:
    // Check if enough features are buffered to fill the chunk or segment end has been signaled
    virtual bool canEncode() const override;

    // Encode a single chunk of features
    virtual void encode() override;

    // Discard all features from input buffer that are no longer needed for future chunks
    virtual void postEncodeCleanup() override;

private:
    enum WindowType {
        None,
        Triangular,
        Hamming,
    };

    enum InterpolationMode {
        NoInterpolation,
        Linear,
        LinearRenorm,
        LogLinear,
        LogLinearRenorm,
        NegLogLinear,
        NegLogLinearRenorm,
    };

    struct PendingOutput {
        size_t                 inputEnd;
        std::shared_ptr<f32[]> accumulator;  // Used to build up the interpolation result
        size_t                 accumulatorSize;
        f32                    totalWeight;  // For potential renormalization

        // Optional renormalization based on total weight depending on mode
        void finalize(InterpolationMode mode);
    };

    // Pre-compute window weights for expected chunk output size
    void initWindow(WindowType windowType);

    // Flush all outputs for which the input start time is before `inputStart` as no outputs from later chunks will add to them
    void flushPendingOutputsUpTo(size_t inputStart);

    // Add data from outputView to associated pending output or create a new one if none exists
    void accumulatePendingOutput(Nn::EncodedSpan data, f32 weight);

    size_t            chunkSize_;
    size_t            stepSize_;
    size_t            leftPadding_;
    size_t            rightPadding_;
    bool              zeroPadding_;
    std::vector<f32>  window_;
    InterpolationMode interpolationMode_;

    size_t                          chunkCenterStart_;  // Current absolute chunk start position disregarding how many features have been discarded so far
    size_t                          numDiscardedFeatures_;
    std::map<size_t, PendingOutput> pendingOutputs_;  // Associate outputs with their input start time to interpolate outputs from different chunks. Ordered for flushing
};

}  // namespace Onnx

#endif  // ONNX_ENCODER_HH
