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

#ifndef ENCODER_HH
#define ENCODER_HH

#include <deque>
#include <optional>

#include <Core/Component.hh>

namespace Nn {

/*
 * Encoder class can take features (e.g. from feature flow) and run them through an encoder model to get encoder states.
 * Works with input/output buffer logic, i.e. features get added to an input buffer and outputs are retreived from an output buffer.
 */
class Encoder : public virtual Core::Component,
                public Core::ReferenceCounted {
public:
    Encoder(Core::Configuration const& config);
    virtual ~Encoder() = default;

    // Clear buffers and reset segment end flag.
    virtual void reset();

    // Signal that no more features are expected for the current segment.
    void signalNoMoreFeatures();

    // Add a single input feature
    virtual void addInput(std::shared_ptr<const f32[]> const& input, size_t featureSize);

    // Add input features for multiple time steps at once
    virtual void addInputs(std::shared_ptr<const f32[]> const& inputs, size_t timeSize, size_t featureSize);

    // Retrieve the next encoder output frame
    // Performs encoder forwarding internally if necessary
    // Can return None if not enough input features are available yet
    std::optional<std::shared_ptr<const f32[]>> getNextOutput();

    // Get dimension of outputs that are fetched via `getNextOutput`.
    size_t getOutputSize() const;

protected:
    std::deque<std::shared_ptr<const f32[]>> inputBuffer_;
    std::deque<std::shared_ptr<const f32[]>> outputBuffer_;

    size_t featureSize_;
    size_t outputSize_;
    bool   expectMoreFeatures_;

    // Encode features inside the input buffer and put the results into the output buffer
    virtual void encode() = 0;

    // Clean up all features from input buffer that have been processed by `encode` and are not needed anymore.
    // By default, this clears out the entire input buffer.
    virtual void postEncodeCleanup();

    // Check if encoder is ready to encode.
    // By default, allow encoding only after segment end has been signaled.
    virtual bool canEncode() const;
};

}  // namespace Nn

#endif  // ENCODER_HH
