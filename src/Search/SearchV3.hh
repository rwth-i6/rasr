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
#ifndef SEARCH_V3_HH
#define SEARCH_V3_HH

#include <string>
#include <vector>

#include <Am/AcousticModel.hh>
#include <Nn/LabelScorer/DataView.hh>
#include <Search/LatticeAdaptor.hh>
#include <Speech/ModelCombination.hh>

#include "Traceback.hh"

namespace Search {

/*
 * Abstract base class for search algorithms that decode several segments at the same time ("batched recognition").
 *
 * In contrast to `SearchAlgorithmV2`, which works on one segment at a time, segments are identified by handles and
 * each search step advances all active segments together. This allows the label scorers to compute the scores of all
 * segments in one batch (e.g. one ONNX session run per step for the hypotheses of all segments) which increases the
 * throughput for large models where the cost of a model call barely depends on the batch size.
 *
 * Segments can be added at any time, also while other segments are still being decoded ("continuous batching").
 * When a segment is finished, the caller retrieves its result, removes it and adds the next segment so that the
 * batch stays full instead of waiting for the longest segment of a fixed batch.
 *
 * The "workflow" usually happens as follows:
 *  1. Check which `Speech::ModelCombination` (and possibly `Am::AcousticModel`) are needed by the search algorithm via
 *     `requiredModelCombination` (and `requiredAcousticModel`).
 *  2. Create appropriate `Speech::ModelCombination` and provide it to the search algorithm via `setModelCombination`.
 *  3. Add a segment via `addSegment` which returns a handle for the segment. At most `maxActiveSegments` segments may be
 *     unfinished at the same time.
 *  4. Pass the audio features of the segment via `putFeature` or `putFeatures` and call `signalNoMoreFeatures` once
 *     all features of the segment have been passed.
 *  5. Call `decodeStep` or `decodeManySteps` to advance all unfinished segments given their currently available features.
 *  (6. Optionally retrieve intermediate results via `getCurrentBestTraceback` or `getCurrentBestWordLattice`.)
 *  7. Once `isFinished` returns true for a segment, retrieve its final result via `getCurrentBestTraceback` or
 *     `getCurrentBestWordLattice` and release it via `removeSegment`.
 * Steps 3. to 7. can be interleaved arbitrarily for different segments.
 *
 * Note that, unlike `SearchAlgorithmV2::finishSegment`, `signalNoMoreFeatures` does not decode the segment to the end
 * since that would prevent batching with other segments. The segment is decoded by subsequent `decodeStep` calls.
 *
 * All functions must be called from the same thread.
 */
class SearchAlgorithmV3 : public virtual Core::Component {
public:
    typedef size_t SegmentHandle;

    SearchAlgorithmV3(Core::Configuration const& config)
            : Core::Component(config) {}
    virtual ~SearchAlgorithmV3() = default;

    // Check which parts of the `Speech::ModelCombination` are required for the search and need to be set via `setModelCombination`
    virtual Speech::ModelCombination::Mode requiredModelCombination() const = 0;

    // Check which parts of the `Am::AcousticModel` are required for the search (only in the case that an `AcousticModel` is needed in the ModelCombination).
    virtual Am::AcousticModel::Mode requiredAcousticModel() const {
        return Am::AcousticModel::noEmissions | Am::AcousticModel::noStateTying | Am::AcousticModel::noStateTransition;
    }

    // Pass a `Speech::ModelCombination` that matches the requirements set by `requiredModelCombination` (and `requiredAcousticModel`) to the search.
    virtual bool setModelCombination(Speech::ModelCombination const& modelCombination) = 0;

    // Maximum number of segments that can be unfinished at the same time.
    virtual size_t maxActiveSegments() const = 0;

    // Number of segments that have been added and are not finished yet.
    virtual size_t numActiveSegments() const = 0;

    // Signal the beginning of a new audio segment and return the handle that identifies the segment in all further calls.
    // The optional name is only used for logging, if it is empty the handle is used instead.
    virtual SegmentHandle addSegment(std::string const& name = "") = 0;

    // Signal that all features of the given segment have been passed.
    virtual void signalNoMoreFeatures(SegmentHandle segment) = 0;

    // Pass a single feature vector of the given segment.
    virtual void putFeature(SegmentHandle segment, Nn::DataView const& feature) = 0;

    // Pass feature vectors for multiple time steps of the given segment.
    virtual void putFeatures(SegmentHandle segment, Nn::DataView const& features, size_t nTimesteps) = 0;

    // Whether the search of the given segment is finished, i.e. the result is final.
    virtual bool isFinished(SegmentHandle segment) const = 0;

    // Release all data of the given segment. The handle becomes invalid afterwards.
    virtual void removeSegment(SegmentHandle segment) = 0;

    // Handles of all segments that have been added and not removed yet (finished or not), in the order they were added.
    virtual std::vector<SegmentHandle> segments() const = 0;

    // Return the current best traceback of the given segment. May contain unstable results if the segment isn't finished.
    virtual Core::Ref<const Traceback> getCurrentBestTraceback(SegmentHandle segment) const = 0;

    // Similar to `getCurrentBestTraceback` but return the lattice instead of just single-best traceback.
    virtual Core::Ref<const LatticeAdaptor> getCurrentBestWordLattice(SegmentHandle segment) const = 0;

    // Return the current best LatticeTrace of the given segment.
    virtual Core::Ref<const LatticeTrace> getCurrentBestLatticeTrace(SegmentHandle segment) const = 0;

    // Return common prefix of all active traces of the given segment.
    virtual Core::Ref<const LatticeTrace> getCommonPrefix(SegmentHandle segment) const = 0;

    // Try to decode one more step for unfinished segments given their currently available features.
    // One call may only advance a subset of them (e.g. one pipeline group).
    // Returns whether anything changed, i.e. some segment could make a step or got finished.
    virtual bool decodeStep() = 0;

    // Decode as much as possible given the currently available features of all segments.
    // Return number of successful `decodeStep` calls.
    virtual unsigned decodeManySteps() {
        unsigned count = 0u;
        while (decodeStep()) {
            count++;
        }

        return count;
    }
};

}  // namespace Search

#endif  // SEARCH_V3_HH
