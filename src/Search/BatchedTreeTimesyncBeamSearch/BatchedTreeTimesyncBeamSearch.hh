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

#ifndef BATCHED_TREE_TIMESYNC_BEAM_SEARCH_HH
#define BATCHED_TREE_TIMESYNC_BEAM_SEARCH_HH

#include <mutex>

#include <Bliss/Lexicon.hh>
#include <Core/Channel.hh>
#include <Core/Parameter.hh>
#include <Lm/ScaledLanguageModel.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/BatchedSearchBase.hh>
#include <Search/Histogram.hh>
#include <Search/LanguageModelLookahead.hh>
#include <Search/PersistentStateTree.hh>
#include <Search/Traceback.hh>

namespace Search {

/*
 * Multi-segment version of `TreeTimesyncBeamSearch` (see `SearchAlgorithmV3`).
 *
 * Time synchronous beam search on a search tree built by a TreeBuilder with a word-level language model and optional
 * LM lookahead. Decodes several segments in lockstep: each search step advances the hypotheses of all unfinished
 * segments by one timestep and the scoring contexts of all segments are passed to the label scorers in one
 * `getScoreAccessors` call per step (and label scorer). The search tree, the language models and the LM lookahead tables
 * are shared by all segments. Segments drop out once their frames are used up and all features have been passed; their
 * hypotheses are then finalized with the sentence-end scores (batched over all segments that are finalized in the same step).
 * Pruning, word-end handling, recombination and maximum-stable-delay pruning are applied per segment exactly as in
 * `TreeTimesyncBeamSearch` and the search uses the same parameters, so for each segment the result is the same as with
 * the single-segment search.
 *
 * Differences to `TreeTimesyncBeamSearch`:
 *  - `signalNoMoreFeatures` replaces `finishSegment`; the segment is decoded by the following `decodeStep` calls.
 *  - The language models are not informed about the segments (`Lm::LanguageModel::setSegment`), so segment-dependent
 *    language models are not supported.
 *  - Label scorer caches are cleaned up based on the global step count instead of the step count of a segment.
 *  - Search statistics are logged per segment when its search is finished; timing statistics, the statistics of the label
 *    scorers and the LM lookahead statistics are logged for all segments together whenever no unfinished segments are
 *    left. Timers of the work that runs in the search threads are summed over the threads.
 */
class BatchedTreeTimesyncBeamSearch : public BatchedSearchBase {
public:
    static const Core::ParameterIntVector   paramMaxBeamSizes;
    static const Core::ParameterInt         paramMaxWordEndBeamSize;
    static const Core::ParameterFloatVector paramScoreThresholds;
    static const Core::ParameterFloat       paramWordEndScoreThreshold;
    static const Core::ParameterInt         paramNumHistogramBins;
    static const Core::ParameterBool        paramEnableHistogramPruning;
    static const Core::ParameterBool        paramCollapseRepeatedLabels;
    static const Core::ParameterBool        paramLmLookahead;
    static const Core::ParameterBool        paramSeparateLookaheadLm;
    static const Core::ParameterBool        paramSparseLmLookAhead;
    static const Core::ParameterBool        paramSentenceEndFallBack;
    static const Core::ParameterInt         paramCacheCleanupInterval;
    static const Core::ParameterInt         paramMaximumStableDelay;
    static const Core::ParameterInt         paramMaximumStableDelayPruningInterval;
    static const Core::Choice               choiceRecombinationMode;
    static const Core::ParameterChoice      paramRecombinationMode;

    BatchedTreeTimesyncBeamSearch(Core::Configuration const&);

    // Inherited methods from `SearchAlgorithmV3`

    Speech::ModelCombination::Mode  requiredModelCombination() const override;
    Am::AcousticModel::Mode         requiredAcousticModel() const override;
    bool                            setModelCombination(Speech::ModelCombination const& modelCombination) override;
    Core::Ref<const Traceback>      getCurrentBestTraceback(SegmentHandle segment) const override;
    Core::Ref<const LatticeAdaptor> getCurrentBestWordLattice(SegmentHandle segment) const override;
    Core::Ref<const LatticeTrace>   getCurrentBestLatticeTrace(SegmentHandle segment) const override;
    Core::Ref<const LatticeTrace>   getCommonPrefix(SegmentHandle segment) const override;

protected:
    /*
     * Possible extension for some label hypothesis in the beam
     */
    struct WithinWordExtensionCandidate {
        Nn::LabelIndex                                    nextToken;         // Proposed token to extend the hypothesis with
        StateId                                           nextState;         // State in the search tree of this extension
        Search::TimeframeIndex                            timeframe;         // Timestamp of `nextToken` for traceback
        Score                                             score;             // Would-be total score of the full hypothesis after extension
        Score                                             lookaheadScore;    // LM-lookahead score, i.e. `lookahead`'s entry for `nextState` plus `lookaheadBackOff`
        LanguageModelLookahead::ContextLookaheadReference lookahead;         // LM-lookahead table `nextState` was scored from
        Score                                             lookaheadBackOff;  // Accumulated back-off score paid to descend to `lookahead`
        Nn::TransitionType                                transitionType;    // Type of transition toward `nextToken`
        size_t                                            baseHypIndex;      // Index of base hypothesis in beam

        bool operator<(WithinWordExtensionCandidate const& other) {
            return score < other.score;
        }
    };

    struct WordEndExtensionCandidate {
        Bliss::LemmaPronunciation const* pron;            // Proposed lemma pronunciation
        StateId                          rootState;       // Proposed root-state to transition to
        Score                            score;           // Would-be total score of the full hypothesis after LM score contribution
        Nn::TransitionType               transitionType;  // Type of transition towward `rootState`
        size_t                           baseHypIndex;    // Index of base hypothesis in beam

        bool operator<(WordEndExtensionCandidate const& other) {
            return score < other.score;
        }
    };

    /*
     * Struct containing all information about a single hypothesis in the beam
     */
    struct LabelHypothesis {
        std::vector<Nn::ScoringContextRef>                scoringContexts;   // Context to compute scores based on this hypothesis
        Nn::LabelIndex                                    currentToken;      // Most recent token in associated label sequence (useful to infer transition type)
        StateId                                           currentState;      // Current state in the search tree
        LanguageModelLookahead::ContextLookaheadReference lookahead;         // LM-lookahead table in use
        Lm::History                                       lmHistory;         // Language model history
        Lm::History                                       lookaheadHistory;  // LM history of this hypothesis for the lookahead; changes only at word ends, never within a word
        Speech::TimeframeIndex                            timeframe;         // Timeframe of current token
        Score                                             score;             // Full score of the hypothesis
        Score                                             lookaheadScore;    // LM-lookahead score
        Score                                             lookaheadBackOff;  // Accumulated back-off score paid to descend from `lookaheadHistory` to `lookahead`
        Core::Ref<LatticeTrace>                           trace;             // Associated trace for traceback or lattice building of hypothesis

        LabelHypothesis();

        // Within-word constructor from base and within-word extension
        LabelHypothesis(LabelHypothesis const& base, WithinWordExtensionCandidate const& extension, std::vector<Nn::ScoringContextRef> const& newScoringContexts);

        // Word-end constructor from base and word-end extension
        LabelHypothesis(LabelHypothesis const& base, WordEndExtensionCandidate const& extension, Lm::History const& newLmHistory, LanguageModelLookahead::ContextLookaheadReference const newLookahead, Lm::History const& newLookaheadHistory, Score newLookaheadBackOff);

        bool operator<(LabelHypothesis const& other) const {
            return score < other.score;
        }

        /*
         * Get string representation for debugging
         */
        std::string toString() const;
    };

    /*
     * All search data of a single segment
     */
    struct SegmentState : public SegmentStateBase {
        std::vector<LabelHypothesis> beam;
        size_t                       currentSearchStep;

        // Intermediate data of the current search step
        std::vector<int>                          hypIndexToContextIndexMap;  // Maps beam index to position of its context in the global context list of the step
        std::vector<WithinWordExtensionCandidate> withinWordExtensions;
        std::vector<WordEndExtensionCandidate>    wordEndExtensions;
        std::vector<LabelHypothesis>              newBeam;
        std::vector<LabelHypothesis>              wordEndHypotheses;
        std::vector<LabelHypothesis>              finalHypotheses;           // Hypotheses that are finalized at the segment end
        bool                                      stepFailed;                // No extensions could be created in the current step
        bool                                      stableDelayLimitExceeded;  // Maximum-stable-delay-pruning had to keep a too old trace in this step

        Core::Statistics<u32>              numInputHyps;
        Core::Statistics<u32>              numWithinWordExtensionsBeforeFirstPruning;
        std::vector<Core::Statistics<u32>> numWithinWordHypsAfterIntermediatePruning;
        Core::Statistics<u32>              numWithinWordHypsAfterRecombination;
        Core::Statistics<u32>              numWithinWordHypsAfterPruning;
        Core::Statistics<u32>              numWordEndExtensionsBeforePruning;
        Core::Statistics<u32>              numWordEndHypsAfterScorePruning;
        Core::Statistics<u32>              numWordEndHypsAfterRecombination;
        Core::Statistics<u32>              numWordEndHypsAfterBeamPruning;
        Core::Statistics<u32>              numActiveHyps;
        Core::Statistics<u32>              numActiveTrees;

        SegmentState(SegmentHandle handle, std::string const& name, size_t numLabelScorers);
    };

private:
    /*
     * Intermediate buffers of one search thread (see `BatchedSearchBase::parallelForSegments`)
     */
    struct ThreadBuffers {
        std::vector<LabelHypothesis>      tempHypotheses;         // Used by recombination and pruning
        std::vector<Nn::ScoreAccessorRef> wordEndScoreAccessors;  // Of the word-end hypothesis being expanded, per label scorer
        Histogram                         scoreHistogram;

        // Phases of the search step done by this thread (see `logSearchStepBreakdown`)
        std::vector<Core::StopWatch> scoreReadoutTimes;         // Per label scorer
        std::vector<Core::StopWatch> intermediatePruningTimes;  // Per label scorer
        Core::StopWatch              buildWithinWordHypsTime;
        Core::StopWatch              wordEndExpansionTime;
        Core::StopWatch              wordEndExtensionTime;
        Core::StopWatch              lmTime;
        Core::StopWatch              wordEndScorePruningTime;
        Core::StopWatch              wordEndHypBuildingTime;
        Core::StopWatch              lmLookaheadTime;
        Core::StopWatch              wordEndRecombinationTime;
        Core::StopWatch              wordEndBeamPruningTime;
        Core::StopWatch              recombinationTime;
        Core::StopWatch              beamPruningTime;

        ThreadBuffers(size_t numHistogramBins);
    };

    std::vector<size_t> maxBeamSizes_;
    size_t              maxWordEndBeamSize_;
    std::vector<Score>  scoreThresholds_;
    bool                histogramPruningEnabled_;
    Score               wordEndScoreThreshold_;
    Nn::LabelIndex      blankLabelIndex_;
    Nn::LabelIndex      silenceLabelIndex_;
    Bliss::Lemma const* sentenceEndLemma_;
    Nn::LabelIndex      sentenceEndLabelIndex_;
    size_t              cacheCleanupInterval_;
    size_t              maximumStableDelay_;
    size_t              maximumStableDelayPruningInterval_;

    bool useBlank_;
    bool useSilence_;
    bool collapseRepeatedLabels_;
    bool sentenceEndFallback_;
    bool recombinationEnabled_;

    Bliss::LexiconRef                              lexicon_;
    robin_hood::unordered_set<const Bliss::Lemma*> nonWordLemmas_;
    Core::Ref<PersistentStateTree>                 network_;
    Core::Ref<const Am::AcousticModel>             acousticModel_;
    Core::Ref<Lm::ScaledLanguageModel>             languageModel_;
    Core::Ref<Lm::ScaledLanguageModel>             lookaheadLm_;
    Core::Channel                                  debugChannel_;

    bool                                    enableLmLookahead_;
    bool                                    separateLookaheadLm_;
    bool                                    sparseLmLookahead_;
    std::unique_ptr<LanguageModelLookahead> lmLookahead_;

    // Pre-allocated intermediate vectors
    std::vector<SegmentState*>         stepSegments_;
    std::vector<SegmentState*>         segmentsToFinalize_;
    std::vector<Nn::ScoringContextRef> scoringContexts_;
    std::vector<ThreadBuffers>         threadBuffers_;  // One per search thread

    /*
     * Phases of the search step per label scorer in the main thread (wall-clock time): scoring and pruning the extensions
     * in total, the label scorer call (or waiting for it) and the parts of score readout and intermediate pruning that
     * are done in the main thread. The rest of score readout and pruning is timed per search thread.
     */
    std::vector<Core::StopWatch> scoreAndPruneExtensionsTimes_;
    std::vector<Core::StopWatch> scoringTimes_;
    std::vector<Core::StopWatch> scoreReadoutTimes_;
    std::vector<Core::StopWatch> intermediatePruningTimes_;
    Core::StopWatch              finalizeTime_;         // Finalizing the hypotheses of finished segments
    Core::StopWatch              finalizeScoringTime_;  // Label scorer calls during finalization

    /*
     * With several search threads, the segments use the shared LM-lookahead cache in parallel: tables are retrieved
     * (and computed) under `lookaheadMutex_`. Releasing the last reference to a table modifies the cache as well, so
     * all tables that are in use during a step of a pipeline group are additionally referenced in
     * `lookaheadTablesInUse_[group]` and released at the end of the step of the group in the main thread.
     */
    bool                                                                        keepLookaheadTables_;
    std::mutex                                                                  lookaheadMutex_;
    std::vector<std::vector<LanguageModelLookahead::ContextLookaheadReference>> lookaheadTablesInUse_;  // One per pipeline group
    size_t                                                                      currentGroup_;          // Group of the current step

    // Precomputed successor/exit lookups (offset tables + contiguous data).
    std::vector<size_t>                    stateSuccessorsOffset_;
    std::vector<StateId>                   stateSuccessors_;
    std::vector<size_t>                    stateExitsOffset_;
    std::vector<PersistentStateTree::Exit> stateExits_;

    // Inherited methods from `BatchedSearchBase`
    std::unique_ptr<SegmentStateBase> createSegmentState(SegmentHandle handle, std::string const& name) override;
    bool                              decodeGroupStep(size_t group) override;
    void                              logSearchStepBreakdown() const override;
    void                              resetSearchStepBreakdown() override;

    // Prepare the next step of the segments and collect the contexts of the first label scorer
    void collectScoringContexts(std::vector<SegmentState*> const& segments, std::vector<Nn::ScoringContextRef>& contexts);

    // Pipelining: start the label scorer call for the next step of the group in the background
    void startNextScoring(size_t group);
    void logSegmentStatistics(SegmentStateBase const& segment) const override;
    void releaseSegmentState(SegmentStateBase& segment) override;
    void logAdditionalStatistics() const override;

    LabelHypothesis const& getBestHypothesis(std::vector<LabelHypothesis> const& beam) const;
    LabelHypothesis const& getWorstHypothesis(std::vector<LabelHypothesis> const& beam) const;

    /*
     * Infer type of transition between two tokens based on whether each of them is blank or silence,
     * and/or whether the state in the search tree changed
     */
    Nn::TransitionType inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel, bool isSameState) const;

    /*
     * Extend the hypotheses of the segment with the scores of the first label scorer (with score-based pre-pruning
     * and LM lookahead) or update the scores of its extensions with the scores of a later label scorer.
     */
    void scoreExtensions(SegmentState&                                           segment,
                         size_t                                                  scorerIdx,
                         std::vector<std::optional<Nn::ScoreAccessorRef>> const& scoreAccessors,
                         std::vector<std::optional<Nn::DenseScoreSpan>> const&   denseScoreSpans,
                         std::vector<Nn::TimeframeIndex> const&                  scoreTimes);

    // Create the new beam of the segment from its surviving within-word extensions (including word ends and LM) and prune it
    void createNewBeam(SegmentState& segment, ThreadBuffers& buffers);

    /*
     * Helper function for pruning. Calculates an absolute threshold based on best score + relative threshold and
     * score histogram. Removes all hypotheses with a score > absolute threshold.
     */
    template<typename Element>
    void scorePruning(std::vector<Element>& hypotheses, Score relativeThreshold, size_t maxBeamSize, Histogram& scoreHistogram);

    /*
     * Helper function for recombination of hypotheses at the same point in the tree with the same scoring context and LM history.
     * With `createTraceSiblings` the traces of the recombined hypotheses will be added as siblings (for word-end recombination).
     */
    void recombination(std::vector<LabelHypothesis>& hypotheses, bool createTraceSiblings, std::vector<LabelHypothesis>& tempHypotheses);

    /*
     * Retrieve or compute the LM lookahead for the given history
     */
    void getLmLookahead(LanguageModelLookahead::ContextLookaheadReference& lookahead, Lm::History history);

    /*
     * Compute the sparse or non-sparse LM lookahead score for an extension's state and history, with back-off if needed
     */
    Score getLmLookaheadScore(std::vector<LabelHypothesis> const& beam, WithinWordExtensionCandidate& extension);

    /*
     * Precompute successor and exit lookups for each state to avoid traversing the network structure during decoding.
     */
    void createSuccessorLookups();

    /*
     * Finalize the hypotheses of the given segments (see `TreeTimesyncBeamSearch::finalizeHypotheses`), scoring sentence-end
     * for all of them in one call per label scorer, and finish their search.
     */
    void finalizeSegments(std::vector<SegmentState*> const& segments);

    /*
     * Apply maximum-stable-delay-pruning to the beam of the segment
     */
    // Returns false if the trace of the best hypothesis had to be kept although it is before the cutoff point
    bool maximumStableDelayPruning(SegmentState& segment, std::vector<LabelHypothesis>& tempHypotheses);
};

}  // namespace Search

#endif  // BATCHED_TREE_TIMESYNC_BEAM_SEARCH_HH
