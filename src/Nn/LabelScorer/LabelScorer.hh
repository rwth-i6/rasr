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
#include <Nn/Types.hh>

#include "DataView.hh"
#include "ScoreAccessor.hh"
#include "ScoringContext.hh"
#include "TransitionTypes.hh"

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
 *  - For all scoring contexts of active hypotheses in search, call `getScoreAccessor` on each or
 *    `getScoresAccessors` on the list (possibly utilizing batched computation) to construct
 *    ScoreAccessor objects.
 *    Note: If the LabelScorer is not able to produce scores for a given context (e.g. when
 *    not enough input features are available) it returns std::nullopt instead.
 *  - For each possible extension of the active hypotheses, check whether the LabelScorer
 *    handles the associated transiton type via `scoresTransition`. If this is true, get the
 *    label- and transition-score from the ScoreAccessor and use them to rank and prune the
 *    extensions.
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
    LabelScorer(Core::Configuration const& config, TransitionPresetType defaultPreset = TransitionPresetType::NONE);
    virtual ~LabelScorer() = default;

    // Prepares the LabelScorer to receive new inputs
    // e.g. by resetting input buffers and segmentEnd flags
    virtual void reset() = 0;

    // Tells the LabelScorer that there will be no more input features coming in the current segment
    virtual void signalNoMoreFeatures() = 0;

    // Gets initial scoring context to use for the hypotheses in the first search step
    virtual ScoringContextRef getInitialScoringContext() = 0;

    // Creates a copy of the context in the request that is extended using the given token and transition type
    virtual ScoringContextRef extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) = 0;

    // Given a collection of currently active contexts, this function can clean up values in any internal caches
    // or buffers that are saved for scoring contexts which no longer are active.
    virtual void cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) {};

    // Add a single input feature
    virtual void addInput(DataView const& input) = 0;

    // Add input features for multiple time steps at once
    virtual void addInputs(DataView const& input, size_t nTimesteps);

    // Perform scoring computation for a single context and return a score accessor
    // that allows retrieving the scores of specific labels or transition types as well
    // as the associated timeframe.
    // Returns std::nullopt if the LabelScorer is not ready to score the context yet
    virtual std::optional<ScoreAccessorRef> getScoreAccessor(ScoringContextRef scoringContext) = 0;

    // Perform scoring computation for a batch of contexts and return score accessors for each
    // that allow retrieving the scores of specific labels or transition types as well
    // as the associated timeframes.
    // By default loops over the single-context version if not overridden in concrete LabelScorer
    // Returns std::nullopt for a context if the LabelScorer is not ready to score it yet
    virtual std::vector<std::optional<ScoreAccessorRef>> getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts);

    // Check whether the given transition type can be scored by this LabelScorer
    inline bool scoresTransition(TransitionType transitionType) const {
        return enabledTransitions_.contains(transitionType);
    }

    // Return the set of all transition types that can get scored by this label scorer
    TransitionSet enabledTransitions() const;

protected:
    TransitionSet enabledTransitions_;
};

}  // namespace Nn

#endif  // LABEL_SCORER_HH
