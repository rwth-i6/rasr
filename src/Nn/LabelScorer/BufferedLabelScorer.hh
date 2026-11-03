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

#ifndef BUFFERED_LABEL_SCORER_HH
#define BUFFERED_LABEL_SCORER_HH

#include <deque>
#include <unordered_map>

#include "DataView.hh"
#include "LabelScorer.hh"

namespace Nn {

/*
 * Extension of `LabelScorer` that implements some commonly used buffering logic for input features
 * and timeframes as well as a flag that indicates whether more features are expected to be added to the buffer.
 * This serves as a base class for other LabelScorers.
 *
 * Buffers and flags are kept separately per segment, so all derived LabelScorers can hold the inputs of several
 * segments at the same time (see the multi-segment interface of `LabelScorer`). The single-segment functions
 * act on the default segment.
 */
class BufferedLabelScorer : public StatTrackingLabelScorer {
public:
    using Precursor = StatTrackingLabelScorer;

    BufferedLabelScorer(Core::Configuration const& config, TransitionPresetType defaultPreset);

    // Tells the LabelScorer that there will be no more input features coming in the default segment
    virtual void signalNoMoreFeatures() override;

    // Add a single input feature to the buffer of the default segment
    virtual void addInput(DataView const& input) override;

    // Initial scoring context of the default segment. Derived LabelScorers implement `getInitialSegmentScoringContext`
    virtual ScoringContextRef getInitialScoringContext() override;

    // Buffers are kept per segment
    virtual bool supportsMultipleSegments() const override;

    // Register a new segment with an empty buffer (replaces existing data of that segment)
    virtual void addSegment(SegmentId segment) override;

    // Delete the buffer of the segment
    virtual void removeSegment(SegmentId segment) override;

    // Tells the LabelScorer that there will be no more input features coming in the given segment
    virtual void signalNoMoreSegmentFeatures(SegmentId segment) override;

    // Add a single input feature to the buffer of the given segment
    virtual void addSegmentInput(SegmentId segment, DataView const& input) override;

    // Clean up input buffers of all segments that appear in the active contexts
    virtual void cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) override;

protected:
    // Prepares the LabelScorer to receive new inputs by removing all segments and their buffers
    // and registering an empty default segment
    void resetInternal() override;

    // Whether more input features are expected for the given segment, i.e. no segment end was signaled yet
    bool expectMoreFeatures(SegmentId segment) const;

    // Get the minimum index of previously buffered inputs that is still needed to process the given active contexts.
    // All given contexts belong to the same segment.
    // Does not take previously deleted inputs into account, i.e. if 5 inputs were added and the active contexts
    // require at least the input of index 3, this will return 3 regardless of how many inputs have been deleted from the buffer so far.
    virtual size_t getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const = 0;

    // Get the input that has been buffered at the specific input index in the given segment.
    // The input index corresponds to when that input has been added to the buffer, disregarding previously deleted inputs.
    // For example if 3 inputs were added and 1 was deleted, then
    //  - getInput(0) will raise an error since it was deleted
    //  - getInput(1) will return the second added input
    //  - getInput(2) will return the third added input
    //  - getInput(3) will return None since no fourth input was added yet
    std::optional<DataView> getInput(SegmentId segment, size_t inputIndex) const;

    // Get number of currently buffered elements of the given segment
    size_t bufferSize(SegmentId segment) const;

private:
    struct SegmentBuffer {
        std::deque<DataView> inputBuffer;                // Buffer that contains all the feature data for the segment
        size_t               numDeletedInputs   = 0ul;   // Count deleted inputs in order to address the correct index in inputBuffer
        bool                 expectMoreFeatures = true;  // Flag to record segment end signal
    };

    std::unordered_map<SegmentId, SegmentBuffer> segmentBuffers_;

    SegmentBuffer const* findSegmentBuffer(SegmentId segment) const;

    void cleanupSegmentBuffer(SegmentBuffer& segmentBuffer, Core::CollapsedVector<ScoringContextRef> const& activeContexts);
};

}  // namespace Nn

#endif  // BUFFERED_LABEL_SCORER_HH
