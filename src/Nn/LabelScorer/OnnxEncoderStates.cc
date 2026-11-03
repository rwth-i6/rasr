/** Copyright 2026 RWTH Aachen University. All rights reserved.
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

#include "OnnxEncoderStates.hh"

#include <algorithm>
#include <cstring>
#include <unordered_map>

#include <Core/Assertions.hh>

namespace Nn {

Onnx::Value stackEncoderStates(std::vector<Onnx::Value const*> const& encoderStates) {
    require(not encoderStates.empty());

    int64_t featureDim = encoderStates.front()->dimSize(2);
    int64_t maxT       = 0l;
    for (auto const* states : encoderStates) {
        require_eq(states->numDims(), 3);
        require_eq(states->dimSize(0), 1l);
        require_eq(states->dimSize(2), featureDim);
        maxT = std::max(maxT, states->dimSize(1));
    }

    auto stacked = Onnx::Value::zeros<f32>({static_cast<int64_t>(encoderStates.size()), maxT, featureDim});
    for (size_t b = 0ul; b < encoderStates.size(); ++b) {
        auto const* states = encoderStates[b];
        if (states->dimSize(1) > 0l) {
            std::memcpy(stacked.data<f32>(b), states->data<f32>(), states->dimSize(1) * featureDim * sizeof(f32));
        }
    }
    return stacked;
}

std::string checkStackableEncoderStatesInput(Onnx::Session const& session, std::string const& inputName) {
    auto shape = session.getInputShape(inputName);
    if (shape.size() != 3ul) {
        return "input '" + inputName + "' doesn't have rank 3";
    }
    if (shape.front() == 1l) {
        return "input '" + inputName + "' has a fixed batch size of 1";
    }
    return "";
}

void addEncoderStatesInputs(EncoderStatesInputNames const&                          names,
                            Onnx::Session const&                                    session,
                            std::vector<SegmentId> const&                           segmentsBatch,
                            std::function<Onnx::PersistentInput&(SegmentId)> const& segmentStates,
                            std::function<Onnx::Value const&(SegmentId)> const&     segmentSize,
                            std::function<size_t(SegmentId)> const&                 segmentLength,
                            std::vector<std::pair<std::string, Onnx::Value>>&       inputs) {
    require(not segmentsBatch.empty());

    SegmentId firstSegment  = segmentsBatch.front();
    bool      singleSegment = std::all_of(segmentsBatch.begin(), segmentsBatch.end(), [firstSegment](SegmentId segment) { return segment == firstSegment; });

    if (singleSegment) {
        if (not names.states.empty()) {
            inputs.emplace_back(names.states, segmentStates(firstSegment).view(session));
        }
        if (not names.size.empty()) {
            inputs.emplace_back(names.size, segmentSize(firstSegment).view());
        }
        if (not names.index.empty()) {
            inputs.emplace_back(names.index, Onnx::Value::create(std::vector<s32>(segmentsBatch.size(), 0)));
        }
        return;
    }

    // Rows of the stacked encoder states: one per segment if the model receives an index, otherwise one per batch entry
    std::vector<SegmentId> rowSegments;
    std::vector<s32>       index;
    if (not names.index.empty()) {
        std::unordered_map<SegmentId, s32> segmentToRow;
        index.reserve(segmentsBatch.size());
        for (auto segment : segmentsBatch) {
            auto [it, inserted] = segmentToRow.emplace(segment, static_cast<s32>(rowSegments.size()));
            if (inserted) {
                rowSegments.push_back(segment);
            }
            index.push_back(it->second);
        }
    }
    else {
        rowSegments = segmentsBatch;
    }

    if (not names.states.empty()) {
        std::vector<Onnx::Value const*> encoderStates;
        encoderStates.reserve(rowSegments.size());
        for (auto segment : rowSegments) {
            encoderStates.push_back(&segmentStates(segment).host());
        }
        inputs.emplace_back(names.states, stackEncoderStates(encoderStates));
    }
    if (not names.size.empty()) {
        std::vector<s32> sizes;
        sizes.reserve(rowSegments.size());
        for (auto segment : rowSegments) {
            sizes.push_back(static_cast<s32>(segmentLength(segment)));
        }
        inputs.emplace_back(names.size, Onnx::Value::create(sizes));
    }
    if (not names.index.empty()) {
        inputs.emplace_back(names.index, Onnx::Value::create(index));
    }
}

}  // namespace Nn
