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

#ifndef BATCHED_LEXICONFREE_TIMESYNC_BEAM_SEARCH_HH
#define BATCHED_LEXICONFREE_TIMESYNC_BEAM_SEARCH_HH

#include <Bliss/Lexicon.hh>
#include <Core/Channel.hh>
#include <Core/Parameter.hh>
#include <Core/StopWatch.hh>
#include <Nn/LabelScorer/DataView.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/BatchedSearchBase.hh>
#include <Search/Histogram.hh>
#include <Search/SearchV3.hh>
#include <Search/Traceback.hh>

namespace Search {

/*
 * Multi-segment version of `LexiconfreeTimesyncBeamSearch` (see `SearchAlgorithmV3`).
 *
 * Decodes several segments in lockstep: each search step advances the hypotheses of all unfinished segments by one
 * timestep. The scoring contexts of all segments are passed to the label scorers in one `getScoreAccessors` call per step
 * (and label scorer) so that the label scorers can batch the computations of all segments (e.g. the per-step calls of a
 * transducer prediction network or a label-level LM). Segments drop out once their frames are used up and all features
 * have been passed; their hypotheses are then finalized with the sentence-end scores (batched over all segments that are
 * finalized in the same step).
 * Pruning, recombination and maximum-stable-delay pruning are applied per segment exactly as in
 * `LexiconfreeTimesyncBeamSearch` and the search uses the same parameters, so for each segment the result is the same as
 * with the single-segment search.
 *
 * Differences to `LexiconfreeTimesyncBeamSearch`:
 *  - `signalNoMoreFeatures` replaces `finishSegment`; the segment is decoded by the following `decodeStep` calls.
 *  - Label scorer caches are cleaned up based on the global step count instead of the step count of a segment.
 *  - Search statistics are logged per segment when its search is finished; timing statistics and the statistics of the
 *    label scorers are logged for all segments together whenever no unfinished segments are left. Timers of the work
 *    that runs in the search threads are summed over the threads.
 */
class BatchedLexiconfreeTimesyncBeamSearch : public BatchedSearchBase {
public:
    static const Core::ParameterIntVector   paramMaxBeamSizes;
    static const Core::ParameterFloatVector paramScoreThresholds;
    static const Core::ParameterInt         paramNumHistogramBins;
    static const Core::ParameterBool        paramEnableHistogramPruning;
    static const Core::ParameterInt         paramBlankLabelIndex;
    static const Core::ParameterInt         paramSilenceLabelIndex;
    static const Core::ParameterInt         paramSentenceEndLabelIndex;
    static const Core::ParameterBool        paramCollapseRepeatedLabels;
    static const Core::ParameterInt         paramMaxExtensionsPerHyp;
    static const Core::ParameterInt         paramCacheCleanupInterval;
    static const Core::ParameterInt         paramMaximumStableDelay;
    static const Core::ParameterInt         paramMaximumStableDelayPruningInterval;
    static const Core::Choice               choiceRecombinationMode;
    static const Core::ParameterChoice      paramRecombinationMode;

    BatchedLexiconfreeTimesyncBeamSearch(Core::Configuration const&);

    // Inherited methods from `SearchAlgorithmV3`

    Speech::ModelCombination::Mode  requiredModelCombination() const override;
    bool                            setModelCombination(Speech::ModelCombination const& modelCombination) override;
    Core::Ref<const Traceback>      getCurrentBestTraceback(SegmentHandle segment) const override;
    Core::Ref<const LatticeAdaptor> getCurrentBestWordLattice(SegmentHandle segment) const override;
    Core::Ref<const LatticeTrace>   getCurrentBestLatticeTrace(SegmentHandle segment) const override;
    Core::Ref<const LatticeTrace>   getCommonPrefix(SegmentHandle segment) const override;

protected:
    /*
     * Possible extension for some label hypothesis in the beam
     */
    struct ExtensionCandidate {
        Nn::LabelIndex                   nextToken;       // Proposed token to extend the hypothesis with
        const Bliss::LemmaPronunciation* pron;            // Pronunciation of lemma corresponding to `nextToken` for traceback
        Score                            score;           // Would-be score of full hypothesis after extension
        Search::TimeframeIndex           timeframe;       // Timestamp of `nextToken` for traceback
        Nn::TransitionType               transitionType;  // Type of transition toward `nextToken`
        size_t                           baseHypIndex;    // Index of base hypothesis in the beam of the segment

        bool operator<(ExtensionCandidate const& other) const {
            return score < other.score;
        }
    };

    /*
     * Struct containing all information about a single hypothesis in the beam
     */
    struct LabelHypothesis {
        std::vector<Nn::ScoringContextRef> scoringContexts;  // Context to compute scores based on this hypothesis
        Nn::LabelIndex                     currentToken;     // Most recent token in associated label sequence (useful to infer transition type)
        Score                              score;            // Full score of hypothesis
        Core::Ref<LatticeTrace>            trace;            // Associated trace for traceback or lattice building off of hypothesis

        LabelHypothesis();
        LabelHypothesis(LabelHypothesis const& base, ExtensionCandidate const& extension, std::vector<Nn::ScoringContextRef> const& newScoringContexts);

        bool operator<(LabelHypothesis const& other) const {
            return score < other.score;
        }

        /*
         * Get string representation for debugging.
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
        std::vector<int>                hypIndexToContextIndexMap;  // Maps beam index to position of its context in the global context list of the step
        std::vector<ExtensionCandidate> extensions;
        std::vector<LabelHypothesis>    newBeam;
        bool                            stepFailed;                // No extensions could be created in the current step
        bool                            stableDelayLimitExceeded;  // Maximum-stable-delay-pruning had to keep a too old trace in this step

        Core::Statistics<u32>              numInputHyps;
        Core::Statistics<u32>              numExtensionsBeforeFirstPruning;
        std::vector<Core::Statistics<u32>> numHypsAfterIntermediatePruning;
        Core::Statistics<u32>              numHypsAfterRecombination;
        Core::Statistics<u32>              numHypsAfterPruning;
        Core::Statistics<u32>              numActiveHyps;

        SegmentState(SegmentHandle handle, std::string const& name, size_t numLabelScorers);
    };

private:
    /*
     * Intermediate buffers and timers of one search thread (see `BatchedSearchBase::parallelForSegments`)
     */
    struct ThreadBuffers {
        std::vector<LabelHypothesis>                  tempHypotheses;  // Used by recombination and pruning
        std::vector<Nn::LabelIndex>                   selectedTokens;  // Tokens selected by `selectTokens`
        std::vector<std::pair<Score, Nn::LabelIndex>> tokenHeap;       // Used by `selectTokens`
        Histogram                                     scoreHistogram;

        // Phases of the search step done by this thread (see `logSearchStepBreakdown`)
        std::vector<Core::StopWatch> scoreReadoutTimes;         // Per label scorer
        std::vector<Core::StopWatch> intermediatePruningTimes;  // Per label scorer
        Core::StopWatch              buildNewBeamTime;
        Core::StopWatch              recombinationTime;
        Core::StopWatch              beamPruningTime;

        ThreadBuffers(size_t numHistogramBins);
    };

    std::vector<size_t> maxBeamSizes_;
    std::vector<bool>   useScorePruning_;
    std::vector<Score>  scoreThresholds_;
    bool                histogramPruningEnabled_;
    bool                useBlank_;
    Nn::LabelIndex      blankLabelIndex_;
    bool                useSilence_;
    Nn::LabelIndex      silenceLabelIndex_;
    bool                useSentenceEnd_;
    Bliss::Lemma const* sentenceEndLemma_;
    Nn::LabelIndex      sentenceEndLabelIndex_;
    bool                collapseRepeatedLabels_;
    size_t              cacheCleanupInterval_;
    size_t              maximumStableDelay_;
    size_t              maximumStableDelayPruningInterval_;
    bool                recombinationEnabled_;
    size_t              maxExtensionsPerHyp_;

    Core::Channel debugChannel_;

    Bliss::LexiconRef lexicon_;

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
    Core::StopWatch              finalizeTime_;         // Finalizing the segments with sentence-end scores
    Core::StopWatch              finalizeScoringTime_;  // Label scorer calls during finalization

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

    LabelHypothesis const& getBestHypothesis(std::vector<LabelHypothesis> const& beam) const;
    LabelHypothesis const& getWorstHypothesis(std::vector<LabelHypothesis> const& beam) const;

    /*
     * Infer type of transition between two tokens based on whether each of them is blank or silence
     * and/or whether they are the same
     */
    Nn::TransitionType inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const;

    /*
     * Limit the tokens to extend a hypothesis with to the `maxExtensionsPerHyp_` tokens with the best dense score,
     * plus the tokens that can't be ranked by it (no dense score or transition not scored). The tokens are stored
     * in `buffers.selectedTokens`. Returns false if the tokens can't or need not be limited, i.e. all tokens should be extended.
     */
    bool selectTokens(std::optional<Nn::DenseScoreSpan> const& denseScores, Nn::LabelIndex currentToken, ThreadBuffers& buffers) const;

    /*
     * Extend the hypotheses of the segment with the scores of the first label scorer (with score-based pre-pruning)
     * or update the scores of its extensions with the scores of a later label scorer.
     */
    void scoreExtensions(SegmentState&                                           segment,
                         size_t                                                  scorerIdx,
                         std::vector<std::optional<Nn::ScoreAccessorRef>> const& scoreAccessors,
                         std::vector<std::optional<Nn::DenseScoreSpan>> const&   denseScoreSpans,
                         std::vector<Nn::TimeframeIndex> const&                  scoreTimes,
                         ThreadBuffers&                                          buffers);

    // Create the new beam of the segment from its surviving extensions and prune it
    void createNewBeam(SegmentState& segment, ThreadBuffers& buffers);

    /*
     * Helper function for acoustic pruning. Calculates an absolute threshold based on best score + relative threshold and
     * score histogram. Removes all hypotheses with a score > absolute threshold.
     */
    template<typename Element>
    void scorePruning(std::vector<Element>& hypotheses, Score relativeThreshold, size_t maxBeamSize, Histogram& scoreHistogram);

    /*
     * Helper function for recombination of hypotheses with the same scoring context
     */
    void recombination(std::vector<LabelHypothesis>& hypotheses, std::vector<LabelHypothesis>& tempHypotheses);

    /*
     * Score sentence-end with all label scores for all hypotheses in the beams of the given segments
     * (in one scoring call per label scorer) and finish their search.
     */
    void finalizeSegments(std::vector<SegmentState*> const& segments);

    /*
     * Apply maximum-stable-delay-pruning to the beam of the segment
     */
    // Returns false if the trace of the best hypothesis had to be kept although it is before the cutoff point
    bool maximumStableDelayPruning(SegmentState& segment, std::vector<LabelHypothesis>& tempHypotheses);
};

}  // namespace Search

#endif  // BATCHED_LEXICONFREE_TIMESYNC_BEAM_SEARCH_HH
