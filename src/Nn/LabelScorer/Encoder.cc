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

#include "Encoder.hh"

#include <Flow/Data.hh>
#include <Mm/Module.hh>

namespace Nn {

Encoder::Encoder(Core::Configuration const& config)
        : Core::Component(config),
          inputBuffer_(),
          outputBuffer_(),
          featureSize_(Core::Type<size_t>::max),
          outputSize_(Core::Type<size_t>::max),
          expectMoreFeatures_(true) {}

void Encoder::reset() {
    expectMoreFeatures_ = true;
    inputBuffer_.clear();

    featureSize_ = Core::Type<size_t>::max;
    outputSize_  = Core::Type<size_t>::max;

    outputBuffer_.clear();
}

void Encoder::signalNoMoreFeatures() {
    expectMoreFeatures_ = false;
}

void Encoder::addInput(std::shared_ptr<const f32[]> const& input, size_t featureSize) {
    if (featureSize_ == Core::Type<size_t>::max) {
        featureSize_ = featureSize;
    }
    else if (featureSize_ != featureSize) {
        error() << "Encoder received incompatible feature size " << featureSize << "; was set to " << featureSize_ << " before.";
    }

    inputBuffer_.push_back(input);
}

void Encoder::addInputs(std::shared_ptr<const f32[]> const& input, size_t timeSize, size_t featureSize) {
    for (size_t t = 0ul; t < timeSize; ++t) {
        // Use aliasing constructor to create sub-`shared_ptr`s that share ownership with the original one but point to different memory locations
        addInput(std::shared_ptr<const f32[]>(input, input.get() + t * featureSize), featureSize);
    }
}

bool Encoder::canEncode() const {
    return not inputBuffer_.empty() and not expectMoreFeatures_;
}

std::optional<std::shared_ptr<const f32[]>> Encoder::getNextOutput() {
    // Check if there are still outputs in the buffer to pass
    if (not outputBuffer_.empty()) {
        auto result = outputBuffer_.front();
        outputBuffer_.pop_front();
        return result;
    }

    // Output buffer is empty but encoder is not ready? -> Return none
    if (not canEncode()) {
        return {};
    }

    // Encoder is ready to run, so run it and try fetching an output again.
    encode();
    postEncodeCleanup();

    // If there are still no outputs after encoding, return None to avoid recursive call
    // resulting in infinite loop
    if (outputBuffer_.empty()) {
        return {};
    }

    return getNextOutput();
}

size_t Encoder::getOutputSize() const {
    return outputSize_;
}

void Encoder::postEncodeCleanup() {
    inputBuffer_.clear();
}

}  // namespace Nn
