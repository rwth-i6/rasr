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

#ifndef LLM_RNNT_TIMESYNC_BEAM_SEARCH_HH
#define LLM_RNNT_TIMESYNC_BEAM_SEARCH_HH

#include <optional>
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
#include <Search/Llm/LlmWordScorer.hh>
#include <Search/Llm/WordAssembler.hh>
#include <Search/SearchV2.hh>
#include <Search/Traceback.hh>

namespace Search {

/*
 * Time synchronous beam search for RNN-T/transducer models without pronunciation lexicon which scores the words
 * assembled from the word-piece labels with an LLM as soon as they are finished, see `WordAssembler` and
 * `LlmWordScorer`. Within a timestep, hypotheses emit labels until they emit blank or reach
 * `max-labels-per-timeframe`. Each lemma of the lexicon is a label whose index is the output index of the label
 * scorers.
 */
class LlmRnntTimesyncBeamSearch : public SearchAlgorithmV2 {
public:
    static const Core::ParameterIntVector   paramMaxBeamSizes;
    static const Core::ParameterFloatVector paramScoreThresholds;
    static const Core::ParameterInt         paramPreLlmMaxBeamSize;
    static const Core::ParameterFloat       paramPreLlmScoreThreshold;
    static const Core::ParameterInt         paramNumHistogramBins;
    static const Core::ParameterFloat       paramLengthNormScale;
    static const Core::ParameterInt         paramMaxLabelsPerFrame;
    static const Core::ParameterInt         paramBlankLabelIndex;
    static const Core::ParameterInt         paramSentenceEndLabelIndex;
    static const Core::ParameterBool        paramSentenceEndFallBack;
    static const Core::ParameterBool        paramCollapseRepeatedLabels;
    static const Core::ParameterFloat       paramLlmScale;
    static const Core::ParameterFloat       paramWordPenalty;
    static const Core::ParameterInt         paramCacheCleanupInterval;
    static const Core::ParameterInt         paramMaximumStableDelay;
    static const Core::ParameterInt         paramMaximumStableDelayPruningInterval;
    static const Core::Choice               choiceRecombinationMode;
    static const Core::ParameterChoice      paramRecombinationMode;

    LlmRnntTimesyncBeamSearch(Core::Configuration const&);

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
    struct ExtensionCandidate {
        Nn::LabelIndex                   nextToken;       // Proposed token to extend the hypothesis with
        const Bliss::LemmaPronunciation* pron;            // Pronunciation of lemma corresponding to `nextToken` for traceback
        Score                            score;           // Would-be score of full hypothesis after extension
        Search::TimeframeIndex           timeframe;       // Timestamp of `nextToken` for traceback
        Nn::TransitionType               transitionType;  // Type of transition toward `nextToken`
        size_t                           baseHypIndex;    // Index of base hypothesis in its vector

        bool operator<(ExtensionCandidate const& other) const {
            return score < other.score;
        }
    };

    // Word and LLM state of a hypothesis after an extension
    struct WordState {
        WordAssembler::WordId pendingWord;
        LlmHistory            llmHistory;
        Score                 llmScore;  // Scaled LLM costs and word penalties included in the score
    };

    struct LabelHypothesis {
        std::vector<Nn::ScoringContextRef> scoringContexts;     // Contexts to compute scores based on this hypothesis, one per label scorer
        Nn::LabelIndex                     currentToken;        // Most recent token in associated label sequence (useful to infer transition type)
        size_t                             length;              // Number of tokens for length normalization
        Score                              score;               // Full score of hypothesis
        Score                              scaledScore;         // Length-normalized score of hypothesis
        std::vector<int>                   outputTokens;        // Non-blank tokens emitted so far
        size_t                             outputTokensHash;    // Hash of `outputTokens`, updated incrementally
        WordState                          words;               // Words finished and pending
        Core::Ref<LatticeTrace>            trace;               // Associated trace for traceback or lattice building off of hypothesis
        bool                               reachedSentenceEnd;  // Whether the sentence-end label was emitted

        LabelHypothesis();
        LabelHypothesis(LabelHypothesis const& base, ExtensionCandidate const& extension, WordState const& words, std::vector<Nn::ScoringContextRef> const& newScoringContexts, float lengthNormScale);

        bool operator<(LabelHypothesis const& other) const {
            return scaledScore < other.scaledScore;
        }

        std::string toString(WordAssembler const& wordAssembler) const;
    };

private:
    std::vector<size_t> maxBeamSizes_;
    std::vector<bool>   useScorePruning_;
    std::vector<Score>  scoreThresholds_;
    size_t              preLlmMaxBeamSize_;
    Score               preLlmScoreThreshold_;
    Histogram           scoreHistogram_;
    float               lengthNormScale_;
    size_t              maxLabelsPerFrame_;
    Nn::LabelIndex      blankLabelIndex_;
    // Blank items in the traceback carry this, so that tracebacks which skip items without pronunciation still end
    // with the final scores
    Bliss::LemmaPronunciation const* blankPron_;
    bool                             useSentenceEnd_;
    Bliss::Lemma const*              sentenceEndLemma_;
    Nn::LabelIndex                   sentenceEndLabelIndex_;
    bool                             sentenceEndFallback_;
    bool                             collapseRepeatedLabels_;
    Score                            llmScale_;
    Score                            wordPenalty_;
    size_t                           cacheCleanupInterval_;
    size_t                           maximumStableDelay_;
    size_t                           maximumStableDelayPruningInterval_;
    Core::Choice::Value              recombinationMode_;

    mutable Core::XmlChannel statisticsChannel_;
    Core::XmlChannel         stepwiseStatisticsChannel_;
    Core::Channel            debugChannel_;

    std::vector<Core::Ref<Nn::LabelScorer>> labelScorers_;
    Bliss::LexiconRef                       lexicon_;
    WordAssembler                           wordAssembler_;
    LlmWordScorer                           llmWordScorer_;
    std::vector<LabelHypothesis>            beam_;

    std::vector<LabelHypothesis> innerHyps_;  // Hypotheses that can still emit labels in the current timestep
    std::vector<LabelHypothesis> outerHyps_;  // Hypotheses that emitted blank and wait for the next timestep

    // Pre-allocated intermediate vectors
    std::vector<int>                                 hypIndexToContextIndexMap_;
    std::vector<ExtensionCandidate>                  extensions_;
    std::vector<WordState>                           extensionWords_;  // Word state of each of `extensions_` after the LLM phase
    std::vector<LabelHypothesis>                     newHyps_;
    std::vector<Nn::ScoringContextRef>               scoringContexts_;
    std::vector<LabelHypothesis>                     tempHypotheses_;
    std::vector<std::optional<Nn::ScoreAccessorRef>> scoreAccessors_;
    std::vector<LlmWordScorer::Request>              wordRequests_;
    std::vector<LlmWordScorer::Result>               wordResults_;
    std::vector<size_t>                              wordRequestExtensions_;  // Index into `extensions_` of each word request
    std::vector<LlmHistory>                          activeHistories_;

    Core::StopWatch initializationTime_;
    Core::StopWatch featureProcessingTime_;
    Core::StopWatch recognitionTime_;
    Core::StopWatch scoringTime_;  // First label scorer on the inner hypotheses
    Core::StopWatch blankExtensionTime_;
    Core::StopWatch labelExtensionTime_;
    Core::StopWatch preLlmPruningTime_;
    Core::StopWatch llmTime_;
    Core::StopWatch wordAssemblyTime_;
    Core::StopWatch wordScoringTime_;
    Core::StopWatch buildInnerHypsTime_;
    Core::StopWatch recombinationTime_;
    Core::StopWatch beamPruningTime_;
    Core::StopWatch cleanupTime_;
    Core::StopWatch finalizeTime_;
    Core::StopWatch finalizeLlmTime_;

    Core::Statistics<u32> numSymbolSteps_;
    Core::Statistics<u32> numInnerHyps_;
    Core::Statistics<u32> numOuterHyps_;
    Core::Statistics<u32> numHypsBeforeLlm_;
    Core::Statistics<u32> numFinishedWords_;
    Core::Statistics<u32> numActiveHyps_;

    size_t currentSearchStep_;
    bool   finishedSegment_;

    LabelHypothesis const& getBestHypothesis() const;
    LabelHypothesis const& getWorstHypothesis() const;

    void resolveSpecialLabels();
    void resetStatistics();
    void logStatistics() const;
    void logOwnStatistics() const;
    void logTimingStatistics() const;
    void logSearchStatistics() const;
    void logBeamStatistics();

    Nn::TransitionType inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const;

    static bool emitsPiece(Nn::TransitionType transitionType);

    template<typename Element, typename ScoreFn>
    void scorePruning(std::vector<Element>& elements, Score relativeThreshold, size_t maxBeamSize, ScoreFn scoreFn);

    void recombination(std::vector<LabelHypothesis>& hypotheses);

    // Phases of a timestep; `advanceTimestep` returns false if the label scorers need more input
    bool advanceTimestep();
    bool advanceSymbolStep(size_t symbolStep);
    bool scoreInnerHyps();
    void extendWithBlank();
    void extendWithLabels();
    void scoreWithRemainingLabelScorers();
    bool pruneBeforeLlm();
    void applyLlm();
    void assembleWords();
    void scoreFinishedWords();
    void buildInnerHyps();
    void pruneOuterHyps();
    void cleanupCaches();
    void maximumStableDelayPruning();

    std::vector<Nn::ScoringContextRef> extendedScoringContexts(LabelHypothesis const& baseHyp, ExtensionCandidate const& extension) const;

    // Phases of the segment end
    void finalizeHypotheses();
    void keepSentenceEndHypotheses();
    void finishWordsAtSegmentEnd();
};

}  // namespace Search

#endif  // LLM_RNNT_TIMESYNC_BEAM_SEARCH_HH
