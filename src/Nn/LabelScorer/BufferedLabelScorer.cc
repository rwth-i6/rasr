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

#include "BufferedLabelScorer.hh"

namespace Nn {

BufferedLabelScorer::BufferedLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          expectMoreFeatures_(true),
          inputBuffer_(),
          numDeletedInputs_(0ul) {
}

void BufferedLabelScorer::reset() {
    inputBuffer_.clear();
    numDeletedInputs_   = 0ul;
    expectMoreFeatures_ = true;
}

void BufferedLabelScorer::signalNoMoreFeatures() {
    expectMoreFeatures_ = false;
}

void BufferedLabelScorer::addInput(DataView const& input) {
    inputBuffer_.push_back(input);
}

void BufferedLabelScorer::cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) {
    if (inputBuffer_.empty()) {
        return;
    }

    auto minActiveTime = minActiveTimeIndex(activeContexts);
    if (minActiveTime > numDeletedInputs_) {
        size_t deleteInputs = minActiveTime - numDeletedInputs_;
        deleteInputs        = std::min(deleteInputs, inputBuffer_.size());
        inputBuffer_.erase(inputBuffer_.begin(), inputBuffer_.begin() + deleteInputs);
        numDeletedInputs_ += deleteInputs;
    }
}
std::optional<DataView> BufferedLabelScorer ::getInput(Speech::TimeframeIndex timeIndex) const {
    if (timeIndex < numDeletedInputs_) {
        error("Tried to get input feature that was already cleaned up.");
    }

    size_t bufferPosition = timeIndex - numDeletedInputs_;
    if (bufferPosition >= inputBuffer_.size()) {
        return {};
    }

    return inputBuffer_[bufferPosition];
}

}  // namespace Nn
