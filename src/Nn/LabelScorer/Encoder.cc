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

namespace Nn {

Encoder::Encoder(Core::Configuration const& config)
        : Core::Component(config),
          inputBuffer_(),
          outputBuffer_(),
          expectMoreFeatures_(true) {}

void Encoder::reset() {
    expectMoreFeatures_ = true;
    inputBuffer_.clear();

    outputBuffer_.clear();
}

void Encoder::signalNoMoreFeatures() {
    expectMoreFeatures_ = false;
}

void Encoder::addInput(DataView const& input) {
    inputBuffer_.push_back(input);
}

void Encoder::addInputs(DataView const& input, size_t nTimesteps) {
    auto featureSize = input.size() / nTimesteps;
    for (size_t t = 0ul; t < nTimesteps; ++t) {
        addInput({input, featureSize, t * featureSize});
    }
}

bool Encoder::canEncode() const {
    return not inputBuffer_.empty() and not expectMoreFeatures_;
}

std::optional<DataView> Encoder::getNextOutput() {
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

void Encoder::postEncodeCleanup() {
    inputBuffer_.clear();
}

}  // namespace Nn
