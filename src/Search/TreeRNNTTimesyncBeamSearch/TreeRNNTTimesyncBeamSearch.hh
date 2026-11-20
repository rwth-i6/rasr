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

#ifndef TREE_RNNT_TIMESYNC_BEAM_SEARCH_HH
#define TREE_RNNT_TIMESYNC_BEAM_SEARCH_HH

#include <optional>
#include <unordered_set>
#include <vector>

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

namespace Search {

/*
 * Time synchronous beam search for standard (non-monotonic) RNN-T/Transducer models on a search tree built by a
 * TreeBuilder (usually `rna`), with a word-level language model scored at word ends.
 * Like LexiconfreeRNNTTimesyncBeamSearch, a hypothesis can emit several non-blank labels per timestep and is
 * finished for the timestep once it emits blank.
 * Uses one or more LabelScorers, applied one after another with intermediate pruning in-between.
 */
class TreeRNNTTimesyncBeamSearch : public SearchAlgorithmV2 {
public:
    static const Core::ParameterIntVector   paramMaxBeamSizes;
    static const Core::ParameterInt         paramMaxWordEndBeamSize;
    static const Core::ParameterFloatVector paramScoreThresholds;
    static const Core::ParameterFloat       paramWordEndScoreThreshold;
    static const Core::ParameterInt         paramNumHistogramBins;
    static const Core::ParameterFloat       paramLengthNormScale;
    static const Core::ParameterInt         paramMaxLabelsPerFrame;
    static const Core::ParameterBool        paramSentenceEndFallBack;
    static const Core::ParameterInt         paramCacheCleanupInterval;
    static const Core::ParameterInt         paramMaximumStableDelay;
    static const Core::ParameterInt         paramMaximumStableDelayPruningInterval;
    static const Core::Choice               choiceRecombinationMode;
    static const Core::ParameterChoice      paramRecombinationMode;

    TreeRNNTTimesyncBeamSearch(Core::Configuration const&);

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
     * Possible extension of some label hypothesis with a blank or non-blank token
     */
    struct WithinWordExtensionCandidate {
        Nn::LabelIndex         nextToken;       // Proposed token to extend the hypothesis with
        StateId                nextState;       // State in the search tree of this extension
        Search::TimeframeIndex timeframe;       // Timeframe at which `nextToken` is scored
        Score                  score;           // Would-be total score of the full hypothesis after extension
        Nn::TransitionType     transitionType;  // Type of transition toward `nextToken`
        size_t                 baseHypIndex;    // Index of base hypothesis

        bool operator<(WithinWordExtensionCandidate const& other) const {
            return score < other.score;
        }
    };

    struct WordEndExtensionCandidate {
        Bliss::LemmaPronunciation const* pron;            // Proposed lemma pronunciation
        StateId                          rootState;       // Proposed root-state to transition to
        Score                            score;           // Would-be total score of the full hypothesis after LM score contribution
        Nn::TransitionType               transitionType;  // Type of transition toward `rootState`
        size_t                           baseHypIndex;    // Index of base hypothesis

        bool operator<(WordEndExtensionCandidate const& other) const {
            return score < other.score;
        }
    };

    /*
     * Struct containing all information about a single hypothesis in the beam
     */
    struct LabelHypothesis {
        std::vector<Nn::ScoringContextRef> scoringContexts;  // Contexts to compute scores based on this hypothesis, one per label scorer
        Nn::LabelIndex                     currentToken;     // Most recent token in associated label sequence (useful to infer transition type)
        StateId                            currentState;     // Current state in the search tree
        Lm::History                        lmHistory;        // Language model history
        Speech::TimeframeIndex             timeframe;        // End time of the current token (its own frame for a label, the next one for blank)
        size_t                             length;           // Number of non-blank tokens for length normalization
        Score                              score;            // Full score of the hypothesis
        Score                              scaledScore;      // Length-normalized score of the hypothesis
        std::vector<Bliss::Lemma const*>   outputWords;      // Previously finished words, without blank
        size_t                             outputWordsHash;  // Hash of `outputWords`, updated incrementally
        Core::Ref<LatticeTrace>            trace;            // Associated trace for traceback or lattice building of hypothesis

        LabelHypothesis();

        // Within-word constructor from base and within-word extension
        LabelHypothesis(LabelHypothesis const& base, WithinWordExtensionCandidate const& extension, std::vector<Nn::ScoringContextRef> const& newScoringContexts, float lengthNormScale);

        // Word-end constructor from base and word-end extension
        LabelHypothesis(LabelHypothesis const& base, WordEndExtensionCandidate const& extension, Lm::History const& newLmHistory, Bliss::Lemma const* blankLemma, float lengthNormScale);

        bool operator<(LabelHypothesis const& other) const {
            return scaledScore < other.scaledScore;
        }

        /*
         * Get string representation for debugging
         */
        std::string toString() const;
    };

private:
    std::vector<size_t> maxBeamSizes_;
    size_t              maxWordEndBeamSize_;
    std::vector<bool>   useScorePruning_;
    std::vector<Score>  scoreThresholds_;
    Score               wordEndScoreThreshold_;
    Histogram           scoreHistogram_;
    float               lengthNormScale_;
    size_t              maxLabelsPerFrame_;
    Nn::LabelIndex      blankLabelIndex_;
    Bliss::Lemma const* blankLemma_;
    Bliss::Lemma const* sentenceEndLemma_;
    Nn::LabelIndex      sentenceEndLabelIndex_;
    bool                sentenceEndFallback_;
    size_t              cacheCleanupInterval_;
    size_t              maximumStableDelay_;
    size_t              maximumStableDelayPruningInterval_;
    Core::Choice::Value recombinationMode_;
    // Timing and beam statistics, per segment and per search step. Both are disabled unless a
    // target is configured.
    mutable Core::XmlChannel statisticsChannel_;
    Core::XmlChannel         stepwiseStatisticsChannel_;
    bool                     stepStatisticsOpen_ = false;

    Core::Channel debugChannel_;

    std::vector<Core::Ref<Nn::LabelScorer>> labelScorers_;
    Bliss::LexiconRef                       lexicon_;
    Core::Ref<PersistentStateTree>          network_;
    Core::Ref<const Am::AcousticModel>      acousticModel_;
    Core::Ref<Lm::ScaledLanguageModel>      languageModel_;
    std::vector<LabelHypothesis>            beam_;

    std::vector<LabelHypothesis> innerHyps_;  // Hyps that are active at the current timestep, so which can still be extended
    std::vector<LabelHypothesis> outerHyps_;  // Hyps that are finished for this timestep are waiting for the next timestep (ended with blank)

    // Pre-allocated intermediate vectors
    std::vector<int>                          hypIndexToContextIndexMap_;
    std::vector<WithinWordExtensionCandidate> extensions_;
    std::vector<WordEndExtensionCandidate>    wordEndExtensions_;
    std::vector<LabelHypothesis>              newBeam_;
    std::vector<LabelHypothesis>              wordEndHypotheses_;
    std::vector<Nn::ScoringContextRef>        scoringContexts_;
    std::vector<LabelHypothesis>              tempHypotheses_;

    // Score accessors of the hypothesis whose word ends are currently being expanded, indexed by label scorer
    std::vector<Nn::ScoreAccessorRef> wordEndScoreAccessors_;

    // Precomputed successor/exit lookups (offset tables + contiguous data)
    std::vector<size_t>                    stateSuccessorsOffset_;
    std::vector<StateId>                   stateSuccessors_;
    std::vector<size_t>                    stateExitsOffset_;
    std::vector<PersistentStateTree::Exit> stateExits_;

    // States with an exit of the blank lemma
    std::unordered_set<StateId> blankLemmaStates_;

    Core::StopWatch initializationTime_;
    Core::StopWatch featureProcessingTime_;
    Core::StopWatch recognitionTime_;
    /*
     * Phases of scoring and pruning the blank and non-blank extensions per label scorer, adding up to
     * the matching `scoreAndPruneExtensionsTimes_` entry. Scorers that compute scores lazily do that
     * work during the score readout.
     */
    std::vector<Core::StopWatch> scoreAndPruneExtensionsTimes_;
    std::vector<Core::StopWatch> scoringTimes_;
    std::vector<Core::StopWatch> scoreReadoutTimes_;
    std::vector<Core::StopWatch> intermediatePruningTimes_;
    Core::StopWatch              buildNewBeamTime_;
    Core::StopWatch              wordEndExpansionTime_;
    Core::StopWatch              recombinationTime_;
    Core::StopWatch              beamPruningTime_;
    Core::StopWatch              finalizeTime_;

    Core::Statistics<u32>              numInputHyps_;
    std::vector<Core::Statistics<u32>> numHypsAfterIntermediatePruning_;  // Non-blank extensions only
    Core::Statistics<u32>              numWordEndHyps_;
    Core::Statistics<u32>              numOuterHyps_;
    Core::Statistics<u32>              numInnerHyps_;
    Core::Statistics<u32>              numInnerAndOuterHyps_;
    Core::Statistics<u32>              numHypsAfterPruning_;
    Core::Statistics<u32>              numActiveHyps_;

    size_t currentSearchStep_;
    bool   finishedSegment_;

    LabelHypothesis const& getBestHypothesis() const;
    LabelHypothesis const& getWorstHypothesis() const;

    void logStatistics() const;

    /*
     * Log the timing and statistics of the search itself, without the label scorers.
     */
    void logOwnStatistics() const;

    /*
     * Infer type of transition between two tokens based on whether each of them is blank
     */
    Nn::TransitionType inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const;

    /*
     * Create the blank extension candidates in `extensions_` for all inner hyps from the scores of the first label scorer
     */
    void createBlankExtensions(std::vector<std::optional<Nn::ScoreAccessorRef>> const& scoreAccessors);

    /*
     * Create the non-blank extension candidates in `extensions_` for all inner hyps from the scores of the first label scorer
     */
    void createExtensions(std::vector<std::optional<Nn::ScoreAccessorRef>> const& scoreAccessors);

    /*
     * Score `extensions_` (raw score) with labelScorers_[1..], applying intermediate score-threshold
     * and max-beam-size pruning after each of them except the last.
     * `countStatistics` enables the intermediate pruning statistics.
     * Returns false if one of these label scorers can't score any of the needed contexts yet, e.g. because it
     * hasn't received enough input. Then the timestep has to be redone later.
     */
    bool scoreAndPruneWithRemainingLabelScorers(bool countStatistics);

    /*
     * Append a new hypothesis to `hyps` for every extension in `extensions_` on top of its base hyp in `innerHyps_`
     */
    void buildHypsFromExtensions(std::vector<LabelHypothesis>& hyps);

    /*
     * Expand the hyps in `hyps` that are at a word end to word-end hypotheses by applying the language model,
     * prune them and append them to `hyps` (without recombination)
     */
    void expandWordEnds(std::vector<LabelHypothesis>& hyps);

    /*
     * Helper function for pruning to a given score-threshold/max-beam-size, approximated via
     * scoreHistogram_. Compares elements via `scoreFn`, so the caller states explicitly whether that
     * means raw or length-normalized (scaled) score for the given Element type
     */
    template<typename Element, typename ScoreFn>
    void scorePruning(std::vector<Element>& elements, Score relativeThreshold, size_t maxBeamSize, ScoreFn scoreFn);

    /*
     * Helper function for recombination of hypotheses at the same point in the tree with the same scoring
     * contexts and LM history. Traces of recombined hypotheses at a root become siblings.
     */
    void recombination(std::vector<LabelHypothesis>& hypotheses);

    /*
     * Open the per-step statistics element before the first statistic of a step is written, and
     * close it again when the step ends. A step that can't be scored yet writes nothing, so it
     * neither opens an element nor consumes a step number.
     */
    void openStepStatistics();
    void closeStepStatistics();

    /*
     * Log the per-step statistics and debug output for the current beam.
     */
    void logStepStatistics();

    /*
     * Precompute successor and exit lookups for each state, see TreeTimesyncBeamSearch::createSuccessorLookups
     */
    void createSuccessorLookups();

    /*
     * After reaching the segment end, only keep hypotheses in final states of the search tree.
     * If no such hypotheses exist, use sentence-end fallback or construct an empty hypothesis.
     * Score sentence-end with all label scorers and the LM for all final hypotheses.
     */
    void finalizeHypotheses();

    /*
     * Apply maximum-stable-delay-pruning to beam_
     */
    void maximumStableDelayPruning();
};

}  // namespace Search

#endif  // TREE_RNNT_TIMESYNC_BEAM_SEARCH_HH
