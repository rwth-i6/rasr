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
#include <algorithm>

namespace Nn {

/*
 * =============================
 * === Encoder =================
 * =============================
 */

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
        error() << "Label scorer received incompatible feature size " << featureSize << "; was set to " << featureSize_ << " before.";
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

    if (not canEncode()) {
        return {};
    }

    // run encoder and try again
    encode();
    postEncodeCleanup();

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

/*
 * =============================
 * === ChunkedEncoder ==========
 * =============================
 */

const Core::ParameterInt ChunkedEncoder::paramChunkCenter(
        "chunk-center",
        "Max number of features in chunk-center. Only encoder-states corresponding to these are transmitted as outputs. This is also used as step-size.",
        Core::Type<u32>::max);

const Core::ParameterInt ChunkedEncoder::paramChunkHistory(
        "chunk-history",
        "Max number of features used as left-context for the encoder. Encoder states corresponding to these are not transmitted as outputs.",
        Core::Type<u32>::max);

const Core::ParameterInt ChunkedEncoder::paramChunkFuture(
        "chunk-future",
        "Max number of features used as right-context for the encoder. Encoder states corresponding to these are not transmitted as outputs.",
        Core::Type<u32>::max);

ChunkedEncoder::ChunkedEncoder(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          chunkCenter_(paramChunkCenter(config)),
          chunkHistory_(paramChunkHistory(config)),
          chunkFuture_(paramChunkFuture(config)),
          chunkSize_(chunkHistory_ + chunkCenter_ + chunkFuture_),
          currentHistoryFeatures_(0ul),
          currentCenterFeatures_(0ul),
          currentFutureFeatures_(0ul) {}

void ChunkedEncoder::reset() {
    Precursor::reset();
    currentHistoryFeatures_ = 0ul;
    currentCenterFeatures_  = 0ul;
    currentFutureFeatures_  = 0ul;
}

void ChunkedEncoder::addInput(std::shared_ptr<const f32[]> const& input, size_t featureSize) {
    Precursor::addInput(input, featureSize);
    if (currentCenterFeatures_ < chunkCenter_) {
        ++currentCenterFeatures_;
    }
    else if (currentFutureFeatures_ < chunkFuture_) {
        ++currentFutureFeatures_;
    }
    else if (currentHistoryFeatures_ < chunkHistory_) {
        ++currentHistoryFeatures_;
    }
    else {
        warning() << "New feature is added while chunk is already full, thus moving the chunk forward before it has been encoded.";
        inputBuffer_.pop_front();
    }
}

bool ChunkedEncoder::canEncode() const {
    return not inputBuffer_.empty() and (not expectMoreFeatures_ or (currentCenterFeatures_ == chunkCenter_ and currentFutureFeatures_ == chunkFuture_));
}

void ChunkedEncoder::postEncodeCleanup() {
    // Current center is added to history. If history exceeds maximum size, the oldest features are popped from the buffer.
    currentHistoryFeatures_ = currentHistoryFeatures_ + currentCenterFeatures_;
    while (currentHistoryFeatures_ > chunkHistory_) {
        inputBuffer_.pop_front();
        --currentHistoryFeatures_;
    }

    // Previous future features become new center until center chunk size is filled.
    currentCenterFeatures_ = std::min(currentFutureFeatures_, chunkCenter_);

    // New future becomes whatever is left from the previous future after moving features over to new center
    if (currentFutureFeatures_ > currentCenterFeatures_) {
        currentFutureFeatures_ -= currentCenterFeatures_;
    }
    else {
        currentFutureFeatures_ = 0ul;
    }
}

/*
 * =============================
 * === NoOpEncoder =============
 * =============================
 */

NoOpEncoder::NoOpEncoder(Core::Configuration const& config)
        : Core::Component(config), Precursor(config) {}

void NoOpEncoder::addInput(std::shared_ptr<const f32[]> const& input, size_t featureSize) {
    Precursor::addInput(input, featureSize);
    outputSize_ = featureSize_;
}

void NoOpEncoder::encode() {
    while (not inputBuffer_.empty()) {
        outputBuffer_.push_back(inputBuffer_.front());
        inputBuffer_.pop_front();
    }
}

}  // namespace Nn
