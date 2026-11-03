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

#include <algorithm>

namespace Nn {

BufferedLabelScorer::BufferedLabelScorer(Core::Configuration const& config, TransitionPresetType defaultPreset)
        : Core::Component(config),
          Precursor(config, defaultPreset),
          segmentBuffers_() {
    segmentBuffers_.emplace(defaultSegmentId, SegmentBuffer());
}

void BufferedLabelScorer::resetInternal() {
    segmentBuffers_.clear();
    segmentBuffers_.emplace(defaultSegmentId, SegmentBuffer());
}

void BufferedLabelScorer::signalNoMoreFeatures() {
    signalNoMoreSegmentFeatures(defaultSegmentId);
}

void BufferedLabelScorer::addInput(DataView const& input) {
    addSegmentInput(defaultSegmentId, input);
}

ScoringContextRef BufferedLabelScorer::getInitialScoringContext() {
    return getInitialSegmentScoringContext(defaultSegmentId);
}

bool BufferedLabelScorer::supportsMultipleSegments() const {
    return true;
}

void BufferedLabelScorer::addSegment(SegmentId segment) {
    segmentBuffers_.insert_or_assign(segment, SegmentBuffer());
}

void BufferedLabelScorer::removeSegment(SegmentId segment) {
    segmentBuffers_.erase(segment);
}

void BufferedLabelScorer::signalNoMoreSegmentFeatures(SegmentId segment) {
    segmentBuffers_[segment].expectMoreFeatures = false;
}

void BufferedLabelScorer::addSegmentInput(SegmentId segment, DataView const& input) {
    segmentBuffers_[segment].inputBuffer.push_back(input);
}

void BufferedLabelScorer::cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) {
    auto const& contexts = activeContexts.internalData();
    if (contexts.empty()) {
        return;
    }

    // Common case: all active contexts belong to the same segment
    SegmentId firstSegment = contexts.front()->segment;
    if (std::all_of(contexts.begin(), contexts.end(), [firstSegment](auto const& context) { return context->segment == firstSegment; })) {
        auto it = segmentBuffers_.find(firstSegment);
        if (it != segmentBuffers_.end()) {
            cleanupSegmentBuffer(it->second, activeContexts);
        }
        return;
    }

    std::unordered_map<SegmentId, Core::CollapsedVector<ScoringContextRef>> contextsPerSegment;
    for (auto const& context : contexts) {
        contextsPerSegment[context->segment].push_back(context);
    }
    for (auto const& [segment, segmentContexts] : contextsPerSegment) {
        auto it = segmentBuffers_.find(segment);
        if (it != segmentBuffers_.end()) {
            cleanupSegmentBuffer(it->second, segmentContexts);
        }
    }
}

void BufferedLabelScorer::cleanupSegmentBuffer(SegmentBuffer& segmentBuffer, Core::CollapsedVector<ScoringContextRef> const& activeContexts) {
    if (segmentBuffer.inputBuffer.empty()) {
        return;
    }

    auto minActiveInput = getMinActiveInputIndex(activeContexts);
    if (minActiveInput > segmentBuffer.numDeletedInputs) {
        size_t numInputsToDelete = minActiveInput - segmentBuffer.numDeletedInputs;
        numInputsToDelete        = std::min(numInputsToDelete, segmentBuffer.inputBuffer.size());
        segmentBuffer.inputBuffer.erase(segmentBuffer.inputBuffer.begin(), segmentBuffer.inputBuffer.begin() + numInputsToDelete);
        segmentBuffer.numDeletedInputs += numInputsToDelete;
    }
}

BufferedLabelScorer::SegmentBuffer const* BufferedLabelScorer::findSegmentBuffer(SegmentId segment) const {
    auto it = segmentBuffers_.find(segment);
    if (it == segmentBuffers_.end()) {
        return nullptr;
    }
    return &it->second;
}

bool BufferedLabelScorer::expectMoreFeatures(SegmentId segment) const {
    auto const* segmentBuffer = findSegmentBuffer(segment);
    return segmentBuffer == nullptr or segmentBuffer->expectMoreFeatures;
}

std::optional<DataView> BufferedLabelScorer::getInput(SegmentId segment, size_t inputIndex) const {
    auto const* segmentBuffer = findSegmentBuffer(segment);
    if (segmentBuffer == nullptr) {
        return {};
    }

    if (inputIndex < segmentBuffer->numDeletedInputs) {
        error("Tried to get input feature that was already cleaned up.");
    }

    size_t bufferPosition = inputIndex - segmentBuffer->numDeletedInputs;
    if (bufferPosition >= segmentBuffer->inputBuffer.size()) {
        return {};
    }

    return segmentBuffer->inputBuffer[bufferPosition];
}

size_t BufferedLabelScorer::bufferSize(SegmentId segment) const {
    auto const* segmentBuffer = findSegmentBuffer(segment);
    return segmentBuffer == nullptr ? 0ul : segmentBuffer->inputBuffer.size();
}

}  // namespace Nn
