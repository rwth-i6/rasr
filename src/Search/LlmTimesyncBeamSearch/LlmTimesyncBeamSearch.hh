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

#ifndef LLM_TIMESYNC_BEAM_SEARCH_HH
#define LLM_TIMESYNC_BEAM_SEARCH_HH

#include <string>
#include <vector>

#include <Bliss/Lexicon.hh>
#include <Core/Channel.hh>
#include <Core/Parameter.hh>
#include <Core/StopWatch.hh>
#include <Nn/LabelScorer/DataView.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/Histogram.hh>
#include <Search/SearchV2.hh>
#include <Search/Traceback.hh>

#include "LlmScoreCache.hh"
#include "LlmScorer.hh"

namespace Search {

/*
 * Time synchronous beam search without pronunciation lexicon which integrates an external token-level
 * language model with its own tokenizer, e.g. an LLM such as Qwen, at the word level.
 *
 * The output labels are word pieces in SentencePiece convention: a piece whose orthography starts with
 * the word-start marker (`word-start-marker`, "▁" by default) begins a new word, every other piece continues
 * the current one. Each hypothesis accumulates the spelling of its pending word. When a piece begins a new
 * word, the pending word is finished: its surface spelling is tokenized with the LLM's tokenizer, every
 * resulting LLM token is scored, and the LLM history of the hypothesis is advanced by them. If the LLM scorer
 * tokenizes a spelling into several variants (e.g. of different casing), all of them are scored and the
 * hypothesis greedily continues with the cheapest one, both for its score and for its LLM history. At the segment end
 * the pending word is finished in the same way and the LLM's sentence-end tokens are scored.
 *
 * Tokenizations are cached by spelling and LLM scores by history plus token sequence, see `LlmScoreCache`.
 * All LLM requests of one search step are sent to the LLM as one batch.
 *
 * Within a step, the label scorers are applied first, with intermediate pruning as in the lexicon-free
 * timesync search. The surviving extensions are pruned once more (`pre-llm-max-beam-size`,
 * `pre-llm-score-threshold`) before the LLM scores the words they finish, so that the LLM is only asked
 * about hypotheses that may survive. Hypotheses are recombined if their label scorer contexts, last label,
 * LLM history and pending word are equal.
 *
 * The LLM costs are scaled by `llm-scale`, and every finished word additionally gets `word-penalty`. Both are
 * reported as LM score in the traceback.
 *
 * The search requires a lexicon that represents the vocabulary. Each lemma is viewed as a token with its index
 * in the lexicon corresponding to the associated output index of the label scorer, and its preferred
 * orthography is the text of the word piece.
 */
class LlmTimesyncBeamSearch : public SearchAlgorithmV2 {
public:
    static const Core::ParameterIntVector   paramMaxBeamSizes;
    static const Core::ParameterFloatVector paramScoreThresholds;
    static const Core::ParameterInt         paramPreLlmMaxBeamSize;
    static const Core::ParameterFloat       paramPreLlmScoreThreshold;
    static const Core::ParameterInt         paramNumHistogramBins;
    static const Core::ParameterInt         paramBlankLabelIndex;
    static const Core::ParameterInt         paramSilenceLabelIndex;
    static const Core::ParameterInt         paramSentenceEndLabelIndex;
    static const Core::ParameterBool        paramCollapseRepeatedLabels;
    static const Core::ParameterString      paramWordStartMarker;
    static const Core::ParameterString      paramWordSeparator;
    static const Core::ParameterFloat       paramLlmScale;
    static const Core::ParameterFloat       paramWordPenalty;
    static const Core::ParameterInt         paramCacheCleanupInterval;
    static const Core::ParameterInt         paramMaximumStableDelay;
    static const Core::ParameterInt         paramMaximumStableDelayPruningInterval;
    static const Core::Choice               choiceRecombinationMode;
    static const Core::ParameterChoice      paramRecombinationMode;
    static const Core::ParameterBool        paramLogStepwiseStatistics;

    LlmTimesyncBeamSearch(Core::Configuration const&);

    // Inherited methods from `SearchAlgorithmV2`

    Speech::ModelCombination::Mode requiredModelCombination() const override;
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
     * Possible extension for some label hypothesis in the beam
     */
    struct ExtensionCandidate {
        Nn::LabelIndex                   nextToken;       // Proposed token to extend the hypothesis with
        const Bliss::LemmaPronunciation* pron;            // Pronunciation of lemma corresponding to `nextToken` for traceback
        Score                            score;           // Would-be score of full hypothesis after extension
        Search::TimeframeIndex           timeframe;       // Timestamp of `nextToken` for traceback
        Nn::TransitionType               transitionType;  // Type of transition toward `nextToken`
        size_t                           baseHypIndex;    // Index of base hypothesis in global beam

        // Only set for the extensions that survive until the LLM is applied
        Score       lmScore;      // Would-be scaled LLM score (plus word penalties) of the full hypothesis
        LlmHistory  lmHistory;    // LLM history after the extension
        std::string pendingWord;  // Spelling of the pending word after the extension

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
        Score                              lmScore;          // Part of `score` that is due to the LLM (scaled, plus word penalties)
        LlmHistory                         lmHistory;        // LLM history covering all finished words
        std::string                        pendingWord;      // Spelling of the word which is not finished yet
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

private:
    std::vector<size_t> maxBeamSizes_;
    std::vector<bool>   useScorePruning_;
    std::vector<Score>  scoreThresholds_;
    size_t              preLlmMaxBeamSize_;
    Score               preLlmScoreThreshold_;
    Histogram           scoreHistogram_;
    bool                useBlank_;
    Nn::LabelIndex      blankLabelIndex_;
    bool                useSilence_;
    Nn::LabelIndex      silenceLabelIndex_;
    bool                useSentenceEnd_;
    Bliss::Lemma const* sentenceEndLemma_;
    Nn::LabelIndex      sentenceEndLabelIndex_;
    bool                collapseRepeatedLabels_;
    std::string         wordStartMarker_;
    std::string         wordSeparator_;
    Score               llmScale_;
    Score               wordPenalty_;
    size_t              cacheCleanupInterval_;
    size_t              maximumStableDelay_;
    size_t              maximumStableDelayPruningInterval_;
    bool                recombinationEnabled_;
    bool                logStepwiseStatistics_;

    Core::Channel debugChannel_;

    std::vector<Core::Ref<Nn::LabelScorer>> labelScorers_;
    Bliss::LexiconRef                       lexicon_;
    std::vector<LabelHypothesis>            beam_;

    // Word piece of each label: text without the word-start marker, and whether it begins a word
    std::vector<std::string> pieceTexts_;
    std::vector<bool>        pieceStartsWord_;

    LlmScoreCache llmCache_;

    // Pre-allocated intermediate vectors
    std::vector<int>                             hypIndexToContextIndexMap_;
    std::vector<ExtensionCandidate>              extensions_;
    std::vector<LabelHypothesis>                 newBeam_;
    std::vector<Nn::ScoringContextRef>           scoringContexts_;
    std::vector<LabelHypothesis>                 tempHypotheses_;
    std::vector<std::string>                     wordTexts_;
    std::vector<LlmTokenSequenceVariants const*> wordTokenizations_;
    std::vector<size_t>                          wordOwners_;
    std::vector<LlmHistory>                      wordHistories_;
    std::vector<LlmScoreCache::Request>          llmRequests_;
    std::vector<LlmScoreCache::Result>           llmResults_;
    std::vector<size_t>                          llmRequestOffsets_;
    std::vector<LlmScoreCache::Result>           bestLlmResults_;

    Core::StopWatch initializationTime_;
    Core::StopWatch featureProcessingTime_;
    Core::StopWatch scoringTime_;
    Core::StopWatch llmTime_;

    std::vector<Core::Statistics<u32>> numHypsAfterIntermediatePruning_;
    Core::Statistics<u32>              numHypsBeforeLlm_;
    Core::Statistics<u32>              numFinishedWords_;
    Core::Statistics<u32>              numHypsAfterRecombination_;
    Core::Statistics<u32>              numHypsAfterPruning_;
    Core::Statistics<u32>              numActiveHyps_;

    size_t currentSearchStep_;
    bool   finishedSegment_;

    LabelHypothesis const& getBestHypothesis() const;
    LabelHypothesis const& getWorstHypothesis() const;

    void logStatistics() const;

    /*
     * Infer type of transition between two tokens based on whether each of them is blank or silence
     * and/or whether they are the same
     */
    Nn::TransitionType inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const;

    /*
     * Whether a transition of this type emits a new word piece
     */
    static bool emitsPiece(Nn::TransitionType transitionType);

    /*
     * Text that is handed to the LLM tokenizer for a finished word after the given history
     */
    std::string wordText(LlmHistory history, std::string const& word) const;

    /*
     * For every `i`, score all variants of `*variants[i]` (a single empty one if it is null), each followed by
     * `suffix`, after `histories[i]` and store the result of the cheapest one in `bestLlmResults_[i]`.
     * All requests are sent to the LLM in one batch.
     */
    void scoreCheapestVariants(std::vector<LlmHistory> const&                      histories,
                               std::vector<LlmTokenSequenceVariants const*> const& variants,
                               LlmTokenSequence const&                             suffix);

    /*
     * Apply the LLM to the surviving extensions: update their pending word, score the words they
     * finish and advance their LLM history.
     */
    void applyLlm();

    /*
     * Helper function for acoustic pruning. Calculates an absolute threshold based on best score + relative threshold and
     * score histogram. Removes all hypotheses with a score > absolute threshold.
     */
    template<typename Element>
    void scorePruning(std::vector<Element>& hypotheses, Score relativeThreshold, size_t maxBeamSize);

    /*
     * Helper function for recombination of hypotheses with the same scoring context, LLM history and pending word
     */
    void recombination(std::vector<LabelHypothesis>& hypotheses);

    /*
     * Finish the pending words, score the LLM's sentence end and score sentence-end with all label scorers
     * for all hypotheses in the beam
     */
    void finalizeHypotheses();

    /*
     * Apply maximum-stable-delay-pruning to beam_
     */
    void maximumStableDelayPruning();
};

}  // namespace Search

#endif  // LLM_TIMESYNC_BEAM_SEARCH_HH
