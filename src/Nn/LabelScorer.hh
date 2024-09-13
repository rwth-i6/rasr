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

#ifndef LABEL_SCORER_HH
#define LABEL_SCORER_HH

#include <Core/Component.hh>
#include <Core/Parameter.hh>
#include <Core/Types.hh>
#include <Flow/Timestamp.hh>
#include <Speech/Feature.hh>
#include <optional>
#include "LabelHistory.hh"
#include "Types.hh"

namespace Nn {

// TODO: Change mechanism from `LabelIndex` to general token from tokenInventory similar to LM -> Fsa::Alphabet

class LabelScorer : public virtual Core::Component,
                    public Core::ReferenceCounted {
public:
    enum TransitionType {
        FORWARD,
        LOOP
    };

    struct Request {
        Core::Ref<LabelHistory> history;
        LabelIndex              nextToken;
        TransitionType          transitionType;
    };

    struct ScoreWithTime {
        NegLogScore     score;
        Flow::Timestamp timestamp;
    };

    LabelScorer(const Core::Configuration& config);
    virtual ~LabelScorer() = default;

    // Prepares the LabelScorer to receive new inputs
    // e.g. by resetting input buffers and segmentEnd flags
    virtual void reset() = 0;

    // Tells the LabelScorer that there will be no more input features coming in the current segment
    virtual void signalNoMoreFeatures() = 0;

    // Get start history for decoder
    virtual Core::Ref<LabelHistory> getStartHistory() = 0;

    // Logic for extending the history in the request by the given labelIndex
    virtual void extendHistory(Request request) = 0;

    // Add a single input feature
    virtual void addInput(FeatureVectorRef input)                 = 0;
    virtual void addInput(Core::Ref<const Speech::Feature> input) = 0;

    // Perform scoring computation for a single request
    virtual std::optional<ScoreWithTime> getScoreWithTime(const Request request) = 0;

    // Perform scoring computation for a vector of requests
    // Loops over `getScore` by default but may also implement more efficient batched logic
    virtual std::vector<std::optional<ScoreWithTime>> getScoresWithTime(const std::vector<Request>& requests);
};

}  // namespace Nn

#endif  // LABEL_SCORER_HH
