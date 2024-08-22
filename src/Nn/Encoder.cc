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

#include "Encoder.hh"
#include <Mm/Module.hh>
#include "Flow/Data.hh"

namespace Nn {

/*
 * =============================
 * === Encoder =================
 * =============================
 */

const Core::ParameterInt Encoder::paramMaxBufferSize(
        "max-buffer-size",
        "Maximum number of features that can be encoded at once.",
        Core::Type<u32>::max);

Encoder::Encoder(const Core::Configuration& config)
        : Core::Component(config), maxBufferSize_() {}

void Encoder::reset() {
    segmentEnd_ = false;
    while (not inputBuffer_.empty()) {
        inputBuffer_.pop();
    }
    while (not outputBuffer_.empty()) {
        outputBuffer_.pop();
    }
}

void Encoder::signalNoMoreFeatures() {
    segmentEnd_ = true;
}

void Encoder::addInput(FeatureVectorRef input) {
    inputBuffer_.push(input);
}

void Encoder::addInput(Core::Ref<const Speech::Feature> input) {
    // TODO: Avoid copying data from one vector to another
    inputBuffer_.push(Flow::dataPtr(new FeatureVector(*input->mainStream(), input->timestamp().startTime(), input->timestamp().endTime())));
}

bool Encoder::canEncode() const {
    return not inputBuffer_.empty() and (inputBuffer_.size() >= maxBufferSize_ or segmentEnd_);
}

std::optional<FeatureVectorRef> Encoder::getNextOutput() {
    // Check if there are still outputs in the buffer to pass
    if (not outputBuffer_.empty()) {
        FeatureVectorRef result(outputBuffer_.front());
        outputBuffer_.pop();
        return result;
    }

    if (not canEncode()) {
        return {};
    }

    // inputBuffer is filled or segment end is reached -> run encoder and try again
    encode();

    if (outputBuffer_.empty()) {
        return {};
    }

    return getNextOutput();
}

/*
 * =============================
 * === NoOpEncoder =============
 * =============================
 */

NoOpEncoder::NoOpEncoder(const Core::Configuration& config)
        : Core::Component(config), Precursor(config) {}

void NoOpEncoder::encode() {
    size_t T_in = 0ul;
    while (not inputBuffer_.empty() and T_in < maxBufferSize_) {
        outputBuffer_.push(inputBuffer_.front());
        inputBuffer_.pop();
        ++T_in;
    }
}

}  // namespace Nn
