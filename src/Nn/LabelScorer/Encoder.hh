/** Copyright 2020 RWTH Aachen University. All rights reserved.
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

#ifndef ENCODER_HH
#define ENCODER_HH

#include <Core/Component.hh>
#include <Nn/Types.hh>
#include <Speech/Feature.hh>
#include <deque>
#include <optional>

namespace Nn {

// Encoder class can take features (e.g. from feature flow) and run them through an encoder model to get encoder states
// Works with input/output buffer logic, i.e. features get added to an input buffer and outputs are retreived from an
// output buffer
class Encoder : public virtual Core::Component,
                public Core::ReferenceCounted {
public:
    static const Core::ParameterInt paramChunkSize;
    static const Core::ParameterInt paramChunkStep;

    Encoder(const Core::Configuration& config);
    virtual ~Encoder() = default;

    // Clear buffers and reset segment end flag
    void reset();

    // Signal that no more features are expected for the current segment
    // At that point, encoder can run regardless of whether the buffer has been filled
    void signalNoMoreFeatures();

    // Add a single input feature to an input buffer
    void addInput(f32 const* input, size_t F);

    // Add input features for multiple time steps to an input buffer
    void addInputs(f32 const* input, size_t T, size_t F);

    // Retrieve a single encoder output
    // Performs encoder forwarding internally if necessary
    // Can return None if not enough input features are available yet
    std::optional<f32 const*> getNextOutput();

    size_t getOutputSize() const;

protected:
    // Consume all features inside input buffer, encode them and put the results into the output buffer
    // By default no-op, i.e. just move from input buffer over to output buffer
    virtual void encode() = 0;

    std::deque<f32 const*>       inputBuffer_;
    std::deque<std::vector<f32>> inputBufferCopy_;
    std::deque<f32 const*>       outputBuffer_;

    const size_t chunkSize_;
    const size_t chunkStep_;
    size_t       numNewFeatures_;

    size_t featureSize_;
    size_t outputSize_;
    bool   featuresAreContiguous_;
    bool   featuresMissing_;

private:
    // Check if encoder is ready to run
    bool canEncode() const;
};

// Simple dummy encoder that just moves features over from input buffer to output buffer
class NoOpEncoder : public Encoder {
    using Precursor = Encoder;

public:
    NoOpEncoder(const Core::Configuration& config);

protected:
    void encode() override;
};

}  // namespace Nn

#endif  // ENCODER_HH
