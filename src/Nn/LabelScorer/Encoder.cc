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
#include <Flow/Data.hh>
#include <Mm/Module.hh>

namespace Nn {

/*
 * =============================
 * === Encoder =================
 * =============================
 */

const Core::ParameterInt Encoder::paramChunkStep(
        "chunk-step",
        "Number of new features to wait for before allowing next encoding step.",
        Core::Type<u32>::max);

const Core::ParameterInt Encoder::paramChunkSize(
        "chunk-size",
        "Maximum number of features that are encoded at once.",
        Core::Type<u32>::max);

Encoder::Encoder(const Core::Configuration& config)
        : Core::Component(config),
          chunkSize_(paramChunkSize(config)),
          chunkStep_(paramChunkStep(config)),
          numNewFeatures_(0ul),
          featureSize_(Core::Type<size_t>::max),
          outputSize_(Core::Type<size_t>::max),
          featuresAreContiguous_(true),
          featuresMissing_(true) {}

void Encoder::reset() {
    featuresMissing_ = true;
    inputBuffer_.clear();
    inputBufferCopy_.clear();

    featureSize_ = Core::Type<size_t>::max;
    outputSize_  = Core::Type<size_t>::max;

    featuresAreContiguous_ = true;
    outputBuffer_.clear();
    numNewFeatures_ = 0ul;
}

void Encoder::signalNoMoreFeatures() {
    featuresMissing_ = false;
}

void Encoder::addInput(f32 const* input, size_t F) {
    if (featureSize_ == Core::Type<size_t>::max) {
        featureSize_ = F;
    }
    else if (featureSize_ != F) {
        error() << "Label scorer received incompatible feature size " << F << "; was set to " << featureSize_ << " before.";
    }

    if (not inputBuffer_.empty() and input != inputBuffer_.back() + F) {
        featuresAreContiguous_ = false;
    }

    inputBuffer_.push_back(input);
    inputBufferCopy_.push_back(std::vector<f32>(F));
    std::copy(input, input + F, inputBufferCopy_.back().data());
    ++numNewFeatures_;
    while (inputBuffer_.size() > chunkSize_) {
        inputBuffer_.pop_front();
        inputBufferCopy_.pop_front();
    }
}

void Encoder::addInputs(f32 const* input, size_t T, size_t F) {
    for (size_t t = 0ul; t < T; ++t) {
        addInput(input + t * F, F);
    }
}

bool Encoder::canEncode() const {
    return not inputBuffer_.empty() and (not featuresMissing_ or numNewFeatures_ >= chunkStep_);
}

std::optional<f32 const*> Encoder::getNextOutput() {
    // Check if there are still outputs in the buffer to pass
    if (not outputBuffer_.empty()) {
        auto result = outputBuffer_.front();
        outputBuffer_.pop_front();
        return result;
    }

    if (not canEncode()) {
        return {};
    }

    // run encoder and try again
    encode();
    numNewFeatures_ = 0ul;

    if (outputBuffer_.empty()) {
        return {};
    }

    return getNextOutput();
}

size_t Encoder::getOutputSize() const {
    return outputSize_;
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
    while (not inputBuffer_.empty() and T_in < chunkStep_) {
        outputSize_ = featureSize_;
        outputBuffer_.push_back(inputBuffer_.front());
        inputBuffer_.pop_front();
        inputBufferCopy_.pop_front();
        ++T_in;
    }
}

}  // namespace Nn
