/** Copyright 2024 RWTH Aachen University. All rights reserved.
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

#include <optional>

#include <Core/CollapsedVector.hh>
#include <Core/Component.hh>
#include <Core/Configuration.hh>
#include <Core/Parameter.hh>
#include <Core/ReferenceCounting.hh>
#include <Core/Types.hh>
#include <Flow/Timestamp.hh>
#include <Flow/Vector.hh>
#include <Mm/FeatureScorer.hh>
#include <Nn/Types.hh>
#include <Search/Types.hh>
#include <Speech/Feature.hh>
#include <Speech/Types.hh>

#include "ScoringContext.hh"

namespace Nn {

/*
 * Abstract base class for scoring tokens within an ASR search algorithm.
 *
 * This class provides an interface for different types of label scorers in an ASR system.
 * Label Scorers compute the scores of tokens based on input features and a scoring context.
 * Children of this base class should represent various ASR model architectures and cover a
 * wide range of possibilities such as CTC, transducer, AED or other models.
 *
 * The usage is intended as follows:
 *  - Before or during the search, features can be added
 *  - At the beginning of search, `getInitialScoringContext` should be called
 *    and used for the first hypotheses
 *  - For a given hypothesis in search, its search context together with a successor token and
 *    transition type are packed into a request and scored via `getScoreWithTime`. This also returns
 *    the timestamp of the successor.
 *    Note: The scoring function may return no value, in this case it is not ready yet
 *    and needs more input features.
 *    Note: There is also the function `getScoresWithTimes` which can handle an entire batch of
 *    requests at once and might be implemented more efficiently (e.g. using batched model forwarding).
 *  - For all hypotheses that survive pruning, the LabelScorer can compute a new scoring context
 *    that extends the previous scoring context of that hypothesis with a given successor token. This new
 *    scoring context can then be used as context in subsequent search steps.
 *  - After all features have been passed, the `signalNoMoreFeatures` function is called to inform
 *    the label scorer that it doesn't need to wait for more features and can score as much as possible.
 *    This is especially important when the label scorer internally uses an encoder or window with right
 *    context.
 *  - When all necessary scores for the current segment have been computed, the `reset` function is called
 *    to clean up any internal data (e.g. feature buffer) or reset flags of the LabelScorer. Afterwards
 *    it is ready to receive features for the next segment.
 *
 * Each concrete subclass internally implements a concrete type of scoring context which the outside
 * search algorithm is agnostic to. Depending on the model, this scoring context can consist of things like
 * the current timestep, a label history, a hidden state or other values.
 */
class LabelScorer : public virtual Core::Component,
                    public Core::ReferenceCounted {
public:
    typedef Search::Score                Score;
    typedef Flow::Vector<f32>            FeatureVector;
    typedef Flow::DataPtr<FeatureVector> FeatureVectorRef;

    enum TransitionType {
        LABEL_TO_LABEL,
        LABEL_LOOP,
        LABEL_TO_BLANK,
        BLANK_TO_LABEL,
        BLANK_LOOP,
    };

    // Request for scoring or context extension
    struct Request {
        ScoringContextRef context;
        LabelIndex        nextToken;
        TransitionType    transitionType;
    };

    // Return value of scoring function
    struct ScoreWithTime {
        Score                  score;
        Speech::TimeframeIndex timeframe;
    };

    // Return value of batched scoring function
    struct ScoresWithTimes {
        std::vector<Score>                            scores;
        Core::CollapsedVector<Speech::TimeframeIndex> timeframes;  // Timeframes vector is internally collapsed  if all timeframes are the same (e.g. time-sync decoding)
    };

    LabelScorer(Core::Configuration const& config);
    virtual ~LabelScorer() = default;

    // Prepares the LabelScorer to receive new inputs
    // e.g. by resetting input buffers and segmentEnd flags
    virtual void reset() = 0;

    // Tells the LabelScorer that there will be no more input features coming in the current segment
    virtual void signalNoMoreFeatures() = 0;

    // Gets initial scoring context to use for the hypotheses in the first search step
    virtual ScoringContextRef getInitialScoringContext() = 0;

    // Creates a copy of the context in the request that is extended using the given token and transition type
    virtual ScoringContextRef extendedScoringContext(Request const& request) = 0;

    // Add a single input feature
    virtual void addInput(std::shared_ptr<const f32[]> const& input, size_t featureSize) = 0;
    virtual void addInput(std::vector<f32> const& input);

    // Add input features for multiple time steps at once
    virtual void addInputs(std::shared_ptr<const f32[]> const& input, size_t timeSize, size_t featureSize);

    // Perform scoring computation for a single request
    // Return score and timeframe index of the corresponding output
    // May not return a value if the LabelScorer is not ready to score the request yet
    // (e.g. not enough features received)
    virtual std::optional<ScoreWithTime> computeScoreWithTime(Request const& request) = 0;

    // Perform scoring computation for a batch of requests
    // May be implemented more efficiently than iterated calls of `getScoreWithTime`
    // Return two vectors: one vector with scores and one vector with times
    // By default loops over the single-request version
    virtual std::optional<ScoresWithTimes> computeScoresWithTimes(std::vector<Request> const& requests);
};

}  // namespace Nn

#endif  // LABEL_SCORER_HH
