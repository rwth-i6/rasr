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

void ChunkedEncoder::addInput(DataView const& input) {
    Precursor::addInput(input);
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

void NoOpEncoder::addInput(DataView const& input) {
    Precursor::addInput(input);
}

void NoOpEncoder::encode() {
    while (not inputBuffer_.empty()) {
        outputBuffer_.push_back(inputBuffer_.front());
        inputBuffer_.pop_front();
    }
}

}  // namespace Nn
