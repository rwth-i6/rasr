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

#ifndef TREE_TIMESYNC_BEAM_SEARCH_HH
#define TREE_TIMESYNC_BEAM_SEARCH_HH

#include <Bliss/Lexicon.hh>
#include <Core/Channel.hh>
#include <Core/Parameter.hh>
#include <Core/StopWatch.hh>
#include <Nn/LabelScorer/DataView.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/Histogram.hh>
#include <Search/PersistentStateTree.hh>
#include <Search/SearchV2.hh>
#include <Search/Traceback.hh>
#include <Search/UnknownWordFallback.hh>

namespace Search {

/*
 * Simple time synchronous beam search algorithm on a search tree built by a TreeBuilder.
 * At a word end, a language model score is added to the hypothesis score,
 * if no language model should be used, the LM-scale has to be set to 0.0.
 * Performs separate pruning of within-word and word-end hypotheses
 * by max beam-size and by score difference to the best hypothesis.
 * Uses one or more LabelScorers for context initialization/extension and scoring.
 * The LabelScorers are applied one after another with intermediate pruning in-between.
 *
 * The (optional) blank label index is retrieved from the lexicon to ensure consistency with the blank index used for the search tree.
 * If the search tree contains label-loops, one will most likely want to set "collapse-repeated-labels" to true so
 * the label loops are also considered when inferring the transtion type as scoring context.
 */
class TreeTimesyncBeamSearch : public SearchAlgorithmV2 {
public:
    static const Core::ParameterIntVector   paramMaxBeamSizes;
    static const Core::ParameterInt         paramMaxWordEndBeamSize;
    static const Core::ParameterFloatVector paramScoreThresholds;
    static const Core::ParameterFloat       paramWordEndScoreThreshold;
    static const Core::ParameterInt         paramNumHistogramBins;
    static const Core::ParameterBool        paramCollapseRepeatedLabels;
    static const Core::ParameterBool        paramSentenceEndFallBack;
    static const Core::ParameterBool        paramLogStepwiseStatistics;
    static const Core::ParameterInt         paramCacheCleanupInterval;
    static const Core::ParameterInt         paramMaximumStableDelay;
    static const Core::ParameterInt         paramMaximumStableDelayPruningInterval;
    static const Core::Choice               choiceRecombinationMode;
    static const Core::ParameterChoice      paramRecombinationMode;

    TreeTimesyncBeamSearch(Core::Configuration const&);

    // Inherited methods from `SearchAlgorithmV2`

    Speech::ModelCombination::Mode requiredModelCombination() const override;
    Am::AcousticModel::Mode        requiredAcousticModel() const override;
    bool                           setModelCombination(Speech::ModelCombination const& modelCombination) override;
    void                           enterSegment(Bliss::SpeechSegment const* = nullptr) override;
    void                           finishSegment() override;
    void                           putFeature(Nn::DataView const& feature) override;
    void                           putFeatures(Nn::DataView const& features, size_t nTimesteps) override;

    Core::Ref<const Traceback>      getCurrentBestTraceback() const override;
    Core::Ref<const LatticeAdaptor> getCurrentBestWordLattice() const override;
    Core::Ref<const LatticeTrace>   getCurrentBestLatticeTrace() const override;
    Core::Ref<const LatticeTrace>   getCommonPrefix() const override;

    bool decodeStep() override;

protected:
    /*
     * State of the open-vocabulary fallback for one hypothesis.
     *
     * `prefixNodes` are the nodes of `pronunciationTrie_` which spell exactly the
     * pieces emitted for the currently pending fallback word. They decide whether that
     * piece sequence is an exact known pronunciation, and they are tracked
     * independently of where the hypothesis actually sits in the search tree.
     *
     * Under the exact-token-sequence rule the trie is deterministic, so the set holds
     * at most one node today. It is kept as a set because a lexicon with optional
     * neutral pieces or pronunciations spanning a surface word boundary would need
     * several alternatives to stay alive at once.
     *
     * `diverged` records that some emitted piece already left every known
     * pronunciation; once set it stays set until the pending word is closed.
     *
     * Instances are immutable and shared between hypotheses, so copying a hypothesis
     * only copies a reference.
     */
    struct OovState : public Core::ReferenceCounted {
        std::vector<u32> prefixNodes;
        u32              numPieces;
        bool             diverged;

        OovState()
                : prefixNodes(), numPieces(0u), diverged(false) {}

        bool wordPending() const {
            return numPieces > 0u;
        }

        bool operator==(OovState const& other) const {
            return numPieces == other.numPieces and diverged == other.diverged and prefixNodes == other.prefixNodes;
        }
    };
    using OovStateRef = Core::Ref<const OovState>;

    /*
     * Word-LM event a word-end extension performs. At most one such event happens per
     * completed lexical word; piece exits and blanks carry none.
     */
    struct WordLmEvent {
        // Token the word LM is advanced with, or null for no event at all.
        Bliss::SyntacticToken const* token = nullptr;
        // Additive unknown-word cost `beta`, charged once per completed unknown word.
        Score unknownBias = 0.0;
        // Whether this event took the unknown route rather than a known lexical one.
        bool isUnknown = false;
    };

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
        WordLmEvent                      lmEvent;         // Word-LM event this exit performs, if any
        OovStateRef                      oov;             // Fallback state after this exit; null if the fallback is inactive

        bool operator<(WordEndExtensionCandidate const& other) {
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
        Score                              score;            // Full score of the hypothesis
        Core::Ref<LatticeTrace>            trace;            // Associated trace for traceback or lattice building of hypothesis
        OovStateRef                        oov;              // Open-vocabulary fallback state; null unless the known-excluding fallback is active

#ifdef SEARCHV2_DEBUG
        std::vector<Nn::LabelIndex>         tokenSequence;     // Full sequence of predicted tokens for debugging purposes
        std::vector<Score>                  tokenScoreDeltas;  // Score contribution of each token in `tokenSequence` for debugging purposes
        std::vector<Speech::TimeframeIndex> tokenTimeframes;   // Timeframe of each token in `tokenSequence` for debugging purposes
#endif

        LabelHypothesis();

        // Within-word constructor from base and within-word extension
        LabelHypothesis(LabelHypothesis const& base, WithinWordExtensionCandidate const& extension, std::vector<Nn::ScoringContextRef> const& newScoringContexts);

        // Word-end constructor from base and word-end extension
        LabelHypothesis(LabelHypothesis const& base, WordEndExtensionCandidate const& extension, Lm::History const& newLmHistory);

        bool operator<(LabelHypothesis const& other) const {
            return score < other.score;
        }

        /*
         * Get string representation for debugging
         */
        std::string toString() const;
    };

private:
    std::vector<size_t> maxBeamSizes_;
    size_t              maxWordEndBeamSize_;
    std::vector<Score>  scoreThresholds_;
    Score               wordEndScoreThreshold_;
    Histogram           scoreHistogram_;
    Nn::LabelIndex      blankLabelIndex_;
    Nn::LabelIndex      silenceLabelIndex_;
    Bliss::Lemma const* blankLemma_;
    Bliss::Lemma const* silenceLemma_;
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
    bool logStepwiseStatistics_;

    std::vector<Core::Ref<Nn::LabelScorer>>        labelScorers_;
    Bliss::LexiconRef                              lexicon_;
    robin_hood::unordered_set<const Bliss::Lemma*> nonWordLemmas_;
    Core::Ref<PersistentStateTree>                 network_;
    Core::Ref<const Am::AcousticModel>             acousticModel_;
    Core::Ref<Lm::ScaledLanguageModel>             languageModel_;
    Core::Channel                                  debugChannel_;

    // Open-vocabulary fallback. `unknownWordFallback_` is only constructed once the
    // lexicon is known; the remaining members are only used in known-excluding mode.
    std::unique_ptr<UnknownWordFallback> unknownWordFallback_;
    bool                                 excludeKnownWordsFromFallback_;
    StateId                              unknownWordRoot_;
    Bliss::SyntacticToken const*         unknownSyntacticToken_;
    Score                                unknownWordPenalty_;
    Score                                unknownPiecePenalty_;
    OovStateRef                          initialOovState_;

    // Pre-allocated intermediate vectors
    std::vector<int>                          hypIndexToContextIndexMap_;
    std::vector<WithinWordExtensionCandidate> withinWordExtensions_;
    std::vector<WordEndExtensionCandidate>    wordEndExtensions_;
    std::vector<LabelHypothesis>              beam_;
    std::vector<LabelHypothesis>              newBeam_;
    std::vector<LabelHypothesis>              wordEndHypotheses_;
    std::vector<Nn::ScoringContextRef>        scoringContexts_;
    std::vector<LabelHypothesis>              tempHypotheses_;

    // Precomputed successor/exit lookups (offset tables + contiguous data).
    std::vector<size_t>                    stateSuccessorsOffset_;
    std::vector<StateId>                   stateSuccessors_;
    std::vector<size_t>                    stateExitsOffset_;
    std::vector<PersistentStateTree::Exit> stateExits_;

    /*
     * Prefix trie over the pronunciations of all ordinary lexical entries, used to
     * decide whether the pieces of a pending fallback word spell an exact known
     * pronunciation. It is built from the lexicon rather than read off the search
     * tree, so it does not depend on how state tying maps a label to an emission
     * index -- in particular not on the word-boundary flags of the allophone, which
     * differ between a one-piece word and the first piece of a longer one.
     * Node 0 is the root. Only filled in known-excluding mode.
     */
    struct PronunciationTrieNode {
        // Children by acoustic-label phoneme, sorted by phoneme id.
        std::vector<std::pair<Bliss::Phoneme::Id, u32>> children;
        // Ordinary lexical entries whose pronunciation ends exactly here.
        std::vector<Bliss::Lemma const*> completedLemmas;
    };
    std::vector<PronunciationTrieNode> pronunciationTrie_;

    size_t currentSearchStep_;
    bool   finishedSegment_;

    Core::StopWatch initializationTime_;
    Core::StopWatch featureProcessingTime_;
    Core::StopWatch scoringTime_;

    std::vector<Core::Statistics<u32>> numHypsAfterIntermediatePruning_;
    Core::Statistics<u32>              numHypsAfterRecombination_;
    Core::Statistics<u32>              numHypsAfterPruning_;
    Core::Statistics<u32>              numWordEndHypsAfterScorePruning_;
    Core::Statistics<u32>              numWordEndHypsAfterRecombination_;
    Core::Statistics<u32>              numWordEndHypsAfterBeamPruning_;
    Core::Statistics<u32>              numActiveHyps_;
    Core::Statistics<u32>              numActiveTrees_;

    // Open-vocabulary fallback accounting for the current segment. Plain counts
    // rather than `Core::Statistics`, which reports min/avg/max over its samples and
    // would only ever say "1" for an event counter.
    u32 numUnknownWordEvents_;
    u32 numKnownResolvedFallbackWords_;

    LabelHypothesis const& getBestHypothesis() const;
    LabelHypothesis const& getWorstHypothesis() const;

    void logStatistics() const;

    /*
     * Infer type of transition between two tokens based on whether each of them is blank or silence,
     * and/or whether the state in the search tree changed
     */
    Nn::TransitionType inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel, bool isSameState) const;

    /*
     * Helper function for pruning. Calculates an absolute threshold based on best score + relative threshold and
     * score histogram. Removes all hypotheses with a score > absolute threshold.
     */
    template<typename Element>
    void scorePruning(std::vector<Element>& hypotheses, Score relativeThreshold, size_t maxBeamSize);

    /*
     * Helper function for recombination of hypotheses at the same point in the tree with the same scoring context and LM history.
     * With `createTraceSiblings` the traces of the recombined hypotheses will be added as siblings (for word-end recombination).
     */
    void recombination(std::vector<LabelHypothesis>& hypotheses, bool createTraceSiblings);

    /*
     * Precompute successor and exit lookups for each state to avoid traversing the network structure during decoding.
     * Successors and exits are stored in the contiguous vectors stateSuccessors_ and stateExits_.
     * for a state `s`, the corresponding ranges are indexed by
     * (stateSuccessorsOffset_[s], stateSuccessorsOffset_[s+1]) and (stateExitsOffset_[s], stateExitsOffset_[s+1])
     */
    void createSuccessorLookups();

    /*
     * Build `pronunciationTrie_` from every ordinary lexical entry of the lexicon.
     */
    void createPronunciationTrie();

    /*
     * The fallback state a hypothesis has when no fallback word is pending.
     */
    OovStateRef emptyOovState() const {
        return initialOovState_;
    }

    /*
     * Advance the known-prefix tracking of `base` by one emitted fallback piece.
     */
    OovStateRef advanceOovState(OovStateRef const& base, Bliss::Pronunciation const& piece) const;

    /*
     * Ordinary lexical entries which spell exactly the pieces of the pending fallback
     * word described by `oov`. A non-empty result means the pending word is an exact
     * known pronunciation, in which case the unknown route is disallowed and the
     * fallback hypothesis is resolved into these known interpretations instead.
     */
    void collectKnownLemmas(OovState const& oov, std::vector<Bliss::Lemma const*>& knownLemmas) const;

    /*
     * Word-LM events which close the pending fallback word described by `oov`:
     * either one event per known lexical interpretation, or a single unknown event.
     */
    void resolveWordLmEvents(OovState const& oov, std::vector<WordLmEvent>& events) const;

    /*
     * Append the word-end extension candidates produced by one exit of a hypothesis
     * whose state carries the open-vocabulary fallback.
     */
    // Scratch buffers for the fallback bookkeeping, kept as members to avoid
    // reallocating them once per exit.
    mutable std::vector<Bliss::Lemma const*> knownLemmaBuffer_;
    mutable std::vector<WordLmEvent>         wordLmEventBuffer_;

    void expandFallbackExit(LabelHypothesis const&           hyp,
                            size_t                           hypIndex,
                            PersistentStateTree::Exit const& exit,
                            Bliss::LemmaPronunciation const* lemmaPron,
                            UnknownWordFallback::PieceRole   role);

    /*
     * Score contribution of all label scorers for a word-end transition of `hyp`.
     */
    Score wordEndTransitionScore(LabelHypothesis const& hyp, Nn::TransitionType transitionType) const;

    /*
     * After reaching the segment end, go through the active hypotheses, only keep those
     * which are final states of the search tree.
     * If no such hypotheses exist, use sentence-end fallback or construct an empty hypothesis.
     * Score sentence-end with all label scorers for all final hypotheses and add the LM's sentence-end score
     */
    void finalizeHypotheses();

    /*
     * Apply maximum-stable-delay-pruning to beam_
     */
    void maximumStableDelayPruning();
};

}  // namespace Search

#endif  // TREE_TIMESYNC_BEAM_SEARCH_HH
