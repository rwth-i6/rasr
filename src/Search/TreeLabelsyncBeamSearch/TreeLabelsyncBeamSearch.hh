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

#ifndef TREE_LABELSYNC_BEAM_SEARCH_HH
#define TREE_LABELSYNC_BEAM_SEARCH_HH

#include <Bliss/Lexicon.hh>
#include <Core/Channel.hh>
#include <Core/Hash.hh>
#include <Core/Parameter.hh>
#include <Core/StopWatch.hh>
#include <Lm/LanguageModel.hh>
#include <Nn/LabelScorer/DataView.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/PersistentStateTree.hh>
#include <Search/SearchV2.hh>
#include <Search/Traceback.hh>
#include <Speech/Types.hh>

namespace Search {

/*
 * Simple label synchronous beam search algorithm on a search tree built by the AedTreeBuilder.
 * Uses a sentence-end symbol to terminate hypotheses.
 * At a word end, a language model score is added to the hypothesis score,
 * if no language model should be used, the LM-scale has to be set to 0.0.
 * Supports global or separate pruning of within-word and word-end hypotheses
 * by max beam-size and by score difference to the best hypothesis.
 * Uses a LabelScorer to context initialization/extension and scoring.
 * The sentence-end label index is retrieved from the lexicon to ensure consistency with the label index used for the search tree.
 */
class TreeLabelsyncBeamSearch : public SearchAlgorithmV2 {
protected:
    /*
     * Possible extension for some label hypothesis in the beam
     */
    struct ExtensionCandidate {
        Nn::LabelIndex                  nextToken;       // Proposed token to extend the hypothesis with
        StateId                         nextState;       // State in the search tree of this extension
        Score                           score;           // Would-be score of full hypothesis after extension
        Search::TimeframeIndex          timeframe;       // Timestamp of `nextToken` for traceback
        Nn::LabelScorer::TransitionType transitionType;  // Type of transition toward `nextToken`
        size_t                          baseHypIndex;    // Index of base hypothesis in global beam

        bool operator<(ExtensionCandidate const& other) const {
            return score < other.score;
        }
    };

    /*
     * Struct containing all information about a single hypothesis in the beam
     */
    struct LabelHypothesis {
        Nn::ScoringContextRef   scoringContext;  // Context to compute scores based on this hypothesis
        Nn::LabelIndex          currentToken;    // Most recent token in associated label sequence (useful to infer transition type)
        StateId                 currentState;    // Current state in the search tree
        Lm::History             lmHistory;       // Language model history
        size_t                  length;          // Number of tokens in hypothesis for length normalization
        Speech::TimeframeIndex  time;            // Timeframe of current token
        Score                   score;           // Full score of hypothesis
        Score                   scaledScore;     // Length-normalized score of hypothesis
        Core::Ref<LatticeTrace> trace;           // Associated trace for traceback or lattice building off of hypothesis
        bool                    isActive;        // Indicates whether the hypothesis has not produced a sentence-end label yet

        LabelHypothesis();

        // Within-word constructor from base and extension
        LabelHypothesis(LabelHypothesis const& base, ExtensionCandidate const& extension, Nn::ScoringContextRef const& newScoringContext, float lengthNormScale);

        // Word-end constructor from base and lemma pronunciation
        LabelHypothesis(LabelHypothesis const& base, StateId rootState, Bliss::LemmaPronunciation const& pron, Core::Ref<Lm::ScaledLanguageModel const> const& lm, float lengthNormScale);

        bool operator<(LabelHypothesis const& other) const {
            return scaledScore < other.scaledScore;
        }

        /*
         * Get string representation for debugging.
         */
        std::string toString() const;
    };

    // Label hypotheses that share this recombination context are recombined
    struct RecombinationContext {
        StateId               state;
        Nn::ScoringContextRef scoringContext;
        Lm::History           lmHistory;

        RecombinationContext(LabelHypothesis const& hyp) : state(hyp.currentState), scoringContext(hyp.scoringContext), lmHistory(hyp.lmHistory) {}

        bool operator==(const RecombinationContext& other) const {
            return state == other.state and Nn::ScoringContextEq{}(scoringContext, other.scoringContext) and lmHistory == other.lmHistory;
        }
    };

    struct RecombinationContextHash {
        size_t operator()(RecombinationContext const& context) const {
            size_t h1 = context.state;
            size_t h2 = Nn::ScoringContextHash{}(context.scoringContext);
            size_t h3 = Lm::History::Hash{}(context.lmHistory);
            return Core::combineHashes(Core::combineHashes(h1, h2), h3);
        }
    };

    struct RecombinationContextEq {
        bool operator()(RecombinationContext const& lhs, RecombinationContext const& rhs) const {
            return lhs.state == rhs.state and Nn::ScoringContextEq{}(lhs.scoringContext, rhs.scoringContext) and lhs.lmHistory == rhs.lmHistory;
        }
    };

public:
    static const Core::ParameterInt   paramMaxBeamSize;
    static const Core::ParameterInt   paramMaxWordEndBeamSize;
    static const Core::ParameterFloat paramScoreThreshold;
    static const Core::ParameterFloat paramWordEndScoreThreshold;
    static const Core::ParameterInt   paramGlobalMaxBeamSize;
    static const Core::ParameterFloat paramGlobalScoreThreshold;
    static const Core::ParameterBool  paramPruneActiveAgainstTerminated;

    static const Core::ParameterFloat paramLengthNormScale;
    static const Core::ParameterFloat paramMaxLabelsPerTimestep;
    static const Core::ParameterBool  paramSentenceEndFallBack;
    static const Core::ParameterBool  paramLogStepwiseStatistics;
    static const Core::ParameterInt   paramCacheCleanupInterval;

    TreeLabelsyncBeamSearch(Core::Configuration const&);

    // Inherited methods from `SearchAlgorithmV2`

    Speech::ModelCombination::Mode  requiredModelCombination() const override;
    Speech::ModelCombination::Mode  requiredAcousticModel() const override;
    bool                            setModelCombination(Speech::ModelCombination const& modelCombination) override;
    void                            reset() override;
    void                            enterSegment(Bliss::SpeechSegment const* = nullptr) override;
    void                            finishSegment() override;
    void                            putFeature(Nn::DataView const& feature) override;
    void                            putFeatures(Nn::DataView const& features, size_t nTimesteps) override;
    Core::Ref<Traceback const>      getCurrentBestTraceback() const override;
    Core::Ref<LatticeAdaptor const> getCurrentBestWordLattice() const override;
    bool                            decodeStep() override;

private:
    size_t maxBeamSize_;
    size_t maxWordEndBeamSize_;

    bool  useScorePruning_;
    Score scoreThreshold_;
    Score wordEndScoreThreshold_;

    size_t globalMaxBeamSize_;
    Score  globalScoreThreshold_;
    bool   pruneActiveAgainstTerminated_;

    float lengthNormScale_;

    float maxLabelsPerTimestep_;

    Bliss::Lemma const* sentenceEndLemma_;
    Nn::LabelIndex      sentenceEndLabelIndex_;

    bool sentenceEndFallback_;

    bool logStepwiseStatistics_;

    size_t cacheCleanupInterval_;

    Core::Channel debugChannel_;

    Core::Ref<Nn::LabelScorer>               labelScorer_;
    Bliss::LexiconRef                        lexicon_;
    Core::Ref<PersistentStateTree>           network_;
    Core::Ref<Am::AcousticModel const>       acousticModel_;
    Core::Ref<Lm::ScaledLanguageModel const> languageModel_;
    std::vector<LabelHypothesis>             beamActive_;
    std::vector<LabelHypothesis>             beamTerminated_;

    // Pre-allocated intermediate vectors
    std::vector<ExtensionCandidate>       extensions_;
    std::vector<Nn::LabelScorer::Request> requests_;
    std::vector<LabelHypothesis>          withinWordHypotheses_;
    std::vector<LabelHypothesis>          wordEndHypotheses_;
    std::vector<LabelHypothesis>          recombinedHypotheses_;

    int maxNumberOfExits_;

    std::vector<std::vector<StateId>>                   stateSuccessorLookup_;
    std::vector<std::vector<PersistentStateTree::Exit>> exitLookup_;

    Core::StopWatch initializationTime_;
    Core::StopWatch featureProcessingTime_;
    Core::StopWatch scoringTime_;
    Core::StopWatch contextExtensionTime_;

    Core::Statistics<u32> numHypsAfterScorePruning_;
    Core::Statistics<u32> numHypsAfterRecombination_;
    Core::Statistics<u32> numHypsAfterBeamPruning_;
    Core::Statistics<u32> numWordEndHypsAfterScorePruning_;
    Core::Statistics<u32> numWordEndHypsAfterRecombination_;
    Core::Statistics<u32> numWordEndHypsAfterBeamPruning_;
    Core::Statistics<u32> numActiveTrees_;
    Core::Statistics<u32> numActiveHyps_;
    Core::Statistics<u32> numTerminatedHyps_;

    size_t currentSearchStep_;
    size_t totalTimesteps_;
    bool   finishedSegment_;

    LabelHypothesis const* getBestTerminatedHypothesis() const;
    LabelHypothesis const* getWorstTerminatedHypothesis() const;

    LabelHypothesis const* getBestActiveHypothesis() const;
    LabelHypothesis const* getWorstActiveHypothesis() const;

    LabelHypothesis const& getBestHypothesis() const;
    LabelHypothesis const& getWorstHypothesis() const;

    void resetStatistics();
    void logStatistics() const;

    /*
     * Collect all possible within-word extensions for all active hypotheses in the beam.
     * Also create scoring requests for the label scorer.
     * Each extension candidate makes up a request.
     */
    void createExtensions();

    /*
     * Perform scoring of all the requests with the label scorer.
     * Return true if scoring was possible
     */
    bool scoreExtensions();

    /*
     * Expand `extensions_` to fully fledged `withinWordHypotheses_` with updated scoring context
     */
    void createWithinWordHypothesesFromExtensions();

    /*
     * Create set of word-end hypotheses from `withinWordHypotheses_` and `terminatedHypotheses_` and also add the LM score for each
     */
    void createWordEndHypotheses();

    /*
     * Perform recombination on hypotheses in `hyps`. If `createTraceSiblings` is true, the traces of the hypotheses that are pruned
     * are added as siblings to the hypotheses that are kept.
     */
    void recombination(std::vector<LabelHypothesis>& hyps, bool createTraceSiblings);

    /*
     * Helper function for pruning of hyps to maxBeamSize
     */
    void beamSizePruning(std::vector<LabelHypothesis>& hyps, size_t maxBeamSize);

    /*
     * Helper function for pruning of `extensions_` to `scoreThreshold_`
     */
    void scorePruningExtensions();

    /*
     * Helper function for pruning of `wordEndHypotheses_` to `wordEndScoreThreshold_`
     */
    void scorePruningWordEnds();

    /*
     * Fill `beamActive_` and `beamTerminated_` from `withinWordHypotheses_` and `wordEndHypotheses_`
     */
    void createNewBeam();

    /*
     * Precompute information about the successor structure of each state in the search tree
     * to avoid repeated computation during the decode steps
     * stateSuccessorLookup_: contains a list of all state successors for the state at the corresponding index
     * exitLookup_: contains a list of all exits for the state at the corresponding index
     */
    // TODO make this more efficient, especially for states with only one exit (cf. AdvancedTreeSearch)
    void createSuccessorLookups();

    /*
     * All active hypotheses that worse than the best terminated one plus a threshold are pruned.
     */
    void pruneActiveAgainstTerminatedByScore();

    /*
     * All active hypotheses that are not within the overall top-k across both active and terminated
     * hypotheses are pruned.
     */
    void pruneActiveAgainstTerminatedByLimit();

    /*
     * Return true if no active hypothesis is within a score-limit of the best terminated one plus a threshold
     * or no active hypothesis is within the overall top-k across both active and terminated hypotheses.
     */
    bool stopCriterion();

    /*
     * After reaching the segment end, if no terminated hypotheses exist, use sentence-end fallback
     * or construct an empty terminated hypothesis.
     */
    void finalize();
};

}  // namespace Search

#endif  // TREE_LABELSYNC_BEAM_SEARCH_HH
