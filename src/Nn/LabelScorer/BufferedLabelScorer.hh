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
#include "DataView.hh"
#include "LabelScorer.hh"
#include "Speech/Types.hh"

namespace Nn {

/*
 * Extension of `LabelScorer` that implements some commonly used buffering logic for input features
 * and timeframes as well as a flag that indicates whether more features are expected to be added to the buffer.
 * This serves as a base class for other LabelScorers.
 */
class BufferedLabelScorer : public LabelScorer {
public:
    using Precursor = LabelScorer;

    BufferedLabelScorer(Core::Configuration const& config);

    // Prepares the LabelScorer to receive new inputs by resetting input buffer, timeframe buffer
    // and segment end flag
    virtual void reset() override;

    // Tells the LabelScorer that there will be no more input features coming in the current segment
    virtual void signalNoMoreFeatures() override;

    // Add a single input feature to the buffer
    virtual void addInput(DataView const& input) override;

    // Clean up input buffer
    virtual void cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) override;

protected:
    std::deque<DataView> inputBuffer_;         // Buffer that contains all the feature data for the current segment
    size_t               numDeletedInputs_;    // Count delted inputs in order to adress the correct index in inputBuffer_
    bool                 expectMoreFeatures_;  // Flag to record segment end signal

    virtual Speech::TimeframeIndex minActiveTimeIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const = 0;
};

}  // namespace Nn

#endif  // BUFFERED_LABEL_SCORER_HH
