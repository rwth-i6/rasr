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

    OnnxEncoder(Core::Configuration const& config, Nn::ModelCache& modelCache);
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
 * This is the left-aligned chunking, where the center of chunk i spans [i * step-size, i * step-size + chunk-size).
 * With right-aligned chunking, the center of chunk i spans (i * step-size - chunk-size, i * step-size] instead.
 * This requires that output t of a session depends on no input features after t * input-step-size,
 * e.g. a left-padded causal frontend. On the last chunk, outputs starting past the segment end (computed from
 * the model's internal padding) are kept as well.
 * Since the session start is aligned to the output grid, sessions overlap by a few features even without
 * left-padding, so models carrying a state between sessions are not supported.
 */
class ChunkedOnnxEncoder : public OnnxEncoder {
public:
    using Precursor = OnnxEncoder;

    static const Core::ParameterInt    paramChunkSize;
    static const Core::ParameterInt    paramStepSize;
    static const Core::ParameterInt    paramLeftPadding;
    static const Core::ParameterInt    paramRightPadding;
    static const Core::ParameterBool   paramZeroPadding;
    static const Core::Choice          chunkAlignmentChoice;
    static const Core::ParameterChoice paramChunkAlignment;
    static const Core::Choice          windowTypeChoice;
    static const Core::ParameterChoice paramWindowType;
    static const Core::Choice          interpolationModeChoice;
    static const Core::ParameterChoice paramInterpolationMode;

    ChunkedOnnxEncoder(Core::Configuration const& config, Nn::ModelCache& modelCache);

    virtual void reset() override;

protected:
    // Check if enough features are buffered to fill the chunk or segment end has been signaled
    virtual bool canEncode() const override;

    // Encode a single chunk of features
    virtual void encode() override;

    // Discard all features from input buffer that are no longer needed for future chunks
    virtual void postEncodeCleanup() override;

private:
    enum ChunkAlignment {
        LeftAligned,
        RightAligned,
    };

    enum WindowType {
        None,
        Triangular,
        Hamming,
    };

    enum InterpolationMode {
        NoInterpolation,
        Linear,
        LogLinear,
        NegLogLinear,
    };

    struct PendingOutput {
        size_t                 inputEnd;
        std::shared_ptr<f32[]> accumulator;  // Used to build up the interpolation result
        size_t                 accumulatorSize;
        f32                    totalWeight;  // For potential renormalization

        // Optional renormalization based on total weight depending on mode
        void finalize(InterpolationMode mode);
    };

    // Chunk 0's center ends with feature chunk-size - 1 when left-aligned and with feature 0 when right-aligned
    size_t firstChunkCenterEnd() const;

    // Absolute start of the current chunk center, clipped at 0
    size_t chunkCenterStart() const;

    // First input feature of the session for a chunk whose center starts at `centerStart`
    size_t sessionStart(size_t centerStart) const;

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
    ChunkAlignment    chunkAlignment_;
    size_t            outputGridStep_;  // Input features between the starts of consecutive outputs, used to align right-aligned sessions
    std::vector<f32>  window_;
    InterpolationMode interpolationMode_;

    size_t                          chunkCenterEnd_;  // Current absolute chunk end position disregarding discarded features and the segment end
    size_t                          numDiscardedFeatures_;
    std::map<size_t, PendingOutput> pendingOutputs_;  // Associate outputs with their input start time to interpolate outputs from different chunks. Ordered for flushing
};

}  // namespace Onnx

#endif  // ONNX_ENCODER_HH
