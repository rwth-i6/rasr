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

#ifndef BATCHED_TREE_LABELSYNC_BEAM_SEARCH_HH
#define BATCHED_TREE_LABELSYNC_BEAM_SEARCH_HH

#include <optional>

#include <Bliss/Lexicon.hh>
#include <Core/Channel.hh>
#include <Core/Parameter.hh>
#include <Lm/ScaledLanguageModel.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/BatchedSearchBase.hh>
#include <Search/Histogram.hh>
#include <Search/PersistentStateTree.hh>
#include <Search/Traceback.hh>

namespace Search {

/*
 * Multi-segment version of `TreeLabelsyncBeamSearch` (see `SearchAlgorithmV3`).
 *
 * Label synchronous beam search on a search tree built by the AedTreeBuilder with a word-level language model.
 * Decodes several segments in lockstep: each search step extends the hypotheses of all unfinished segments by one label
 * and the scoring contexts of all segments are passed to the label scorers in one `getScoreAccessors` call per step (and
 * label scorer). The search tree, the language model and its caches are shared by all segments.
 * Pruning, word-end handling, recombination, length limits, termination and the final fallback hypotheses are handled
 * per segment exactly as in `TreeLabelsyncBeamSearch` and the search uses the same parameters, so for each segment the
 * result is the same as with the single-segment search.
 *
 * Differences to `TreeLabelsyncBeamSearch`:
 *  - `signalNoMoreFeatures` replaces `finishSegment`; the segment is decoded by the following `decodeStep` calls.
 *  - The output length limit (max-labels-per-timestep with lower and upper bound) only terminates a segment once all of
 *    its features have been passed (or the upper bound is reached); before that, the segment just waits for more
 *    features.
 *  - The language model is not informed about the segments (`Lm::LanguageModel::setSegment`), so segment-dependent
 *    language models are not supported.
 *  - Label scorer caches are cleaned up based on the global step count instead of the step count of a segment.
 *  - Search statistics are logged per segment when its search is finished; timing statistics and the statistics of the
 *    label scorers are logged for all segments together whenever no unfinished segments are left. Timers of the work
 *    that runs in the search threads are summed over the threads.
 */
class BatchedTreeLabelsyncBeamSearch : public BatchedSearchBase {
public:
    static const Core::ParameterIntVector   paramMaxBeamSizes;
    static const Core::ParameterInt         paramMaxWordEndBeamSize;
    static const Core::ParameterFloatVector paramScoreThresholds;
    static const Core::ParameterFloat       paramWordEndScoreThreshold;
    static const Core::ParameterInt         paramNumHistogramBins;
    static const Core::ParameterBool        paramEnableHistogramPruning;
    static const Core::ParameterInt         paramCacheCleanupInterval;
    static const Core::ParameterFloat       paramLengthNormScale;
    static const Core::ParameterFloat       paramMaxLabelsPerTimestep;
    static const Core::ParameterInt         paramMaxLabelsLowerBound;
    static const Core::ParameterInt         paramMaxLabelsUpperBound;
    static const Core::Choice               choicePruningStrategyType;
    static const Core::ParameterChoice      paramPruningStrategyType;
    static const Core::Choice               choiceRecombinationMode;
    static const Core::ParameterChoice      paramRecombinationMode;
    static const Core::ParameterBool        paramSentenceEndFallBack;

    BatchedTreeLabelsyncBeamSearch(Core::Configuration const&);

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
        Nn::LabelIndex         nextToken;       // Proposed token to extend the hypothesis with
        StateId                nextState;       // State in the search tree of this extension
        Search::TimeframeIndex timeframe;       // Timestamp of `nextToken` for traceback
        Score                  score;           // Would-be total score of the full hypothesis after extension
        Nn::TransitionType     transitionType;  // Type of transition toward `nextToken`
        size_t                 baseHypIndex;    // Index of base hypothesis in beam

        inline Score pruningScore() const {
            return score;
        }

        bool operator<(WithinWordExtensionCandidate const& other) const {
            return score < other.score;
        }
    };

    struct WordEndExtensionCandidate {
        Bliss::LemmaPronunciation const* pron;            // Proposed lemma pronunciation
        StateId                          rootState;       // Proposed root-state to transition to
        Score                            score;           // Would-be total score of the full hypothesis after LM score contribution
        Search::TimeframeIndex           timeframe;       // Timestamp of `nextToken` for traceback
        Nn::TransitionType               transitionType;  // Type of transition towward `rootState`
        size_t                           baseHypIndex;    // Index of base hypothesis in beam
        bool                             isActive;        // Indicates whether the base hypothesis has not produced a sentence-end label yet

        inline Score pruningScore() const {
            return score;
        }

        bool operator<(WordEndExtensionCandidate const& other) const {
            return score < other.score;
        }
    };

    /*
     * Struct containing all information about a single hypothesis in the beam
     */
    struct LabelHypothesis {
        std::vector<Nn::ScoringContextRef> scoringContexts;  // Context to compute scores based on this hypothesis
        Nn::LabelIndex                     currentToken;     // Most recent token in associated label sequence (useful to infer transition type)
        StateId                            currentState;     // Current state in the search tree
        Lm::History                        lmHistory;        // Language model history
        Speech::TimeframeIndex             timeframe;        // Timeframe of current token
        size_t                             length;           // Number of tokens in hypothesis for length normalization
        Score                              score;            // Full score of the hypothesis
        Score                              scaledScore;      // Length-normalized score of hypothesis
        Core::Ref<LatticeTrace>            trace;            // Associated trace for traceback or lattice building of hypothesis
        bool                               isActive;         // Indicates whether the hypothesis has not produced a sentence-end label yet

        LabelHypothesis();

        // Within-word constructor from base and within-word extension
        LabelHypothesis(LabelHypothesis const& base, WithinWordExtensionCandidate const& extension, std::vector<Nn::ScoringContextRef> const& newScoringContexts, float lengthNormScale);

        // Word-end constructor from base and word-end extension
        LabelHypothesis(LabelHypothesis const& base, WordEndExtensionCandidate const& extension, Lm::History const& newLmHistory, float lengthNormScale);

        inline Score pruningScore() const {
            return scaledScore;
        }

        bool operator<(LabelHypothesis const& other) const {
            return scaledScore < other.scaledScore;
        }
        bool operator>(LabelHypothesis const& other) const {
            return scaledScore > other.scaledScore;
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
        bool                                      stepFailed;  // No extensions could be created in the current step

        Core::Statistics<u32>              numInputHyps;
        Core::Statistics<u32>              numWithinWordExtensionsBeforeFirstPruning;
        std::vector<Core::Statistics<u32>> numWithinWordHypsAfterIntermediatePruning;
        Core::Statistics<u32>              numTerminatedHypsAfterScorePruning;
        Core::Statistics<u32>              numTerminatedHypsAfterRecombination;
        Core::Statistics<u32>              numTerminatedHypsAfterBeamPruning;
        Core::Statistics<u32>              numActiveHypsAfterScorePruning;
        Core::Statistics<u32>              numActiveHypsAfterRecombination;
        Core::Statistics<u32>              numActiveHypsAfterBeamPruning;
        Core::Statistics<u32>              numWordEndExtensionsBeforePruning;
        Core::Statistics<u32>              numActiveWordEndHypsAfterPruning;
        Core::Statistics<u32>              numActiveWordEndHypsAfterScorePruning;
        Core::Statistics<u32>              numActiveWordEndHypsAfterRecombination;
        Core::Statistics<u32>              numActiveWordEndHypsAfterBeamPruning;
        Core::Statistics<u32>              numActiveTrees;

        SegmentState(SegmentHandle handle, std::string const& name, size_t numLabelScorers);
    };

private:
    enum class HypothesisFilter {
        Active,
        Terminated,
        Any,
    };

    /*
     * `referenceScore` is the score that `relativeThreshold` is applied relative to.
     * If unset, the best score within the pruned set is used. In `separateScorePruning`
     * this is the best score within the respective pool.
     */
    struct PruningParams {
        Score                relativeThreshold;
        size_t               maxBeamSize;
        std::optional<Score> referenceScore = std::nullopt;
    };

    /*
     * Intermediate buffers and timers of one search thread (see `BatchedSearchBase::parallelForSegments`)
     */
    struct ThreadBuffers {
        std::vector<LabelHypothesis>      tempHypotheses;         // Used by recombination
        std::vector<Nn::ScoreAccessorRef> wordEndScoreAccessors;  // Of the word-end hypothesis being expanded, per label scorer
        Histogram                         scoreHistogram;
        Histogram                         activeScoreHistogram;
        Histogram                         terminatedScoreHistogram;

        // Phases of the search step done by this thread (see `logSearchStepBreakdown`)
        std::vector<Core::StopWatch> scoreReadoutTimes;         // Per label scorer
        std::vector<Core::StopWatch> intermediatePruningTimes;  // Per label scorer
        Core::StopWatch              buildWithinWordHypsTime;
        Core::StopWatch              wordEndExpansionTime;
        Core::StopWatch              wordEndExtensionTime;
        Core::StopWatch              wordEndScorePruningTime;
        Core::StopWatch              wordEndHypBuildingTime;
        Core::StopWatch              lmTime;
        Core::StopWatch              recombinationTime;
        Core::StopWatch              beamPruningTime;

        ThreadBuffers(size_t numHistogramBins);
    };

    std::vector<size_t> maxBeamSizes_;
    size_t              maxWordEndBeamSize_;
    std::vector<bool>   useScorePruning_;
    std::vector<Score>  scoreThresholds_;
    bool                histogramPruningEnabled_;
    Score               wordEndScoreThresholdFactor_;  // Configured factor, relative to the final within-word threshold
    Score               wordEndScoreThreshold_;        // Absolute threshold derived from the factor in `setModelCombination`
    float               lengthNormScale_;
    float               maxLabelsPerTimestep_;
    size_t              maxLabelsLowerBound_;
    size_t              maxLabelsUpperBound_;
    Bliss::Lemma const* sentenceEndLemma_;
    Nn::LabelIndex      sentenceEndLabelIndex_;
    size_t              cacheCleanupInterval_;

    bool                sentenceEndFallback_;
    Core::Choice::Value pruningStrategyType_;
    bool                recombinationEnabled_;

    Bliss::LexiconRef                              lexicon_;
    robin_hood::unordered_set<const Bliss::Lemma*> nonWordLemmas_;
    Core::Ref<PersistentStateTree>                 network_;
    Core::Ref<const Am::AcousticModel>             acousticModel_;
    Core::Ref<Lm::ScaledLanguageModel>             languageModel_;
    Core::Channel                                  debugChannel_;

    // Pre-allocated intermediate vectors
    std::vector<SegmentState*>         stepSegments_;
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

    // Maximum number of output labels of the segment given its number of features so far
    size_t maxOutputLength(SegmentState const& segment) const;

    // Whether the segment can take part in a search step (unfinished, below the length limit, active hypotheses)
    bool canMakeStep(SegmentState const& segment) const;

    // Prepare the next step of the segments and collect the contexts of the first label scorer
    void collectScoringContexts(std::vector<SegmentState*> const& segments, std::vector<Nn::ScoringContextRef>& contexts);

    // Pipelining: start the label scorer call for the next step of the group in the background
    void startNextScoring(size_t group);
    void logSegmentStatistics(SegmentStateBase const& segment) const override;
    void releaseSegmentState(SegmentStateBase& segment) override;

    template<typename Element>
    bool matchesHypothesisFilter(Element const& element, HypothesisFilter filter) const;

    template<typename Element>
    Element const* getBestHypothesis(std::vector<Element> const& elements, HypothesisFilter filter) const;

    LabelHypothesis const* getWorstHypothesis(std::vector<LabelHypothesis> const& hypotheses, HypothesisFilter filter) const;

    // Overall best while preferring terminated over active hypotheses if any terminated ones exist
    LabelHypothesis const& getOutputHypothesis(std::vector<LabelHypothesis> const& hypotheses) const;

    /*
     * Extend the hypotheses of the segment with the scores of the first label scorer (with score-based pre-pruning)
     * or update the scores of its extensions with the scores of a later label scorer.
     */
    void scoreExtensions(SegmentState&                                           segment,
                         size_t                                                  scorerIdx,
                         std::vector<std::optional<Nn::ScoreAccessorRef>> const& scoreAccessors,
                         std::vector<std::optional<Nn::DenseScoreSpan>> const&   denseScoreSpans,
                         std::vector<Nn::TimeframeIndex> const&                  scoreTimes);

    // Create the new beam of the segment from its surviving within-word extensions (including word ends and LM) and prune it
    void createNewBeam(SegmentState& segment, ThreadBuffers& buffers);

    /*
     * Helper function for joint pruning of extensions/hypotheses by a relative score threshold
     * and by max beam size (see `TreeLabelsyncBeamSearch`).
     */
    template<typename Element>
    void scorePruning(std::vector<Element>& hypotheses, PruningParams const& pruningParams, Histogram& scoreHistogram);

    /*
     * Score pruning applied separately for active/terminated extensions/hypotheses.
     */
    template<typename Element>
    void separateScorePruning(
            std::vector<Element>& hypotheses,
            PruningParams const&  activePruningParams,
            PruningParams const&  terminatedPruningParams,
            ThreadBuffers&        buffers);

    /*
     * Helper function for recombination of hypotheses at the same point in the tree with the same
     * scoring context, LM history and activity status (see `TreeLabelsyncBeamSearch`).
     */
    void recombination(std::vector<LabelHypothesis>& hypotheses, std::vector<LabelHypothesis>& tempHypotheses);

    /*
     * Count hyps with `isActive` flag
     */
    size_t numActiveHyps(std::vector<LabelHypothesis> const& hypotheses) const;

    /*
     * Count hyps in root state with `isActive` flag
     */
    size_t numActiveWordEndHyps(std::vector<LabelHypothesis> const& hypotheses) const;

    /*
     * Precompute successor and exit lookups for each state to avoid traversing the network structure during decoding.
     */
    void createSuccessorLookups();

    /*
     * After reaching the segment end, only keep terminated hypotheses of the segment (with the fallbacks of
     * `TreeLabelsyncBeamSearch::finalizeHypotheses`).
     */
    void finalizeHypotheses(SegmentState& segment);

    // Finalize the hypotheses of the segment and finish its search
    void finalizeAndFinishSegment(SegmentState& segment);
};

}  // namespace Search

#endif  // BATCHED_TREE_LABELSYNC_BEAM_SEARCH_HH
