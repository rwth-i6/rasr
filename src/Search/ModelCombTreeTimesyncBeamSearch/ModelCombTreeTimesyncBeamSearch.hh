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

#ifndef MODEL_COMB_TREE_TIMESYNC_BEAM_SEARCH_HH
#define MODEL_COMB_TREE_TIMESYNC_BEAM_SEARCH_HH

#include <Bliss/Lexicon.hh>
#include <Core/Channel.hh>
#include <Core/FIFOCache.hh>
#include <Core/Parameter.hh>
#include <Core/StopWatch.hh>
#include <Nn/LabelScorer/DataView.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/PersistentStateTree.hh>
#include <Search/SearchV2.hh>
#include <Search/Traceback.hh>
#include "Search/LanguageModelLookahead.hh"

namespace Search {

// Algorithm to combine multiple models working on different subword tokenizations.
// The within-word search of all models is done in parallel and independently, the combination only takes place at word ends.
// Therefore, all models have their own label scorer, lexicon, AM, search tree, beam, etc.
// The models' scores are scaled and added at a common word-end, so a word-end hypothesis is only yielded if all models have predicted this word end.
// Such a shared word-end hypothesis is referred to as "global word end".
// If one model predicts a word end, it is allowed to loop blank while waiting for the other models to predict the same word end.
// These hypotheses are called "previous word ends". They are handled and pruned separately with separate pruning parameters.
// The maximum number of timeframes they are kept can be controlled with the parameter "max-previous-word-end-lifetime".
// The parameter "keep-global-word-ends-as-previous" allows to keep word ends which are combined to a common word end additionally
// as previous word ends in their corresponding model, so they be combined again in a later timestep.
// When setting the parameter "allow-non-global-word-ends", word ends which only occur in one model (non-global) are also added to the other models' beam
// with a penalty that is set via "non-global-word-end-penalty". These hypotheses are called "individual word ends" and are also pruned separately.
// All pruning parameters can be defined specific to each model or "globally", so they are equal for all models. These global settings are only used if
// they are not defined in the model-specific configurations.

// Important limitations:
// only models using blank can be used
// no support for models using a word-boundary token
// only one (global) language model (future work: each model can have its own LM?)
// non-global word ends works only for models without scoring context that depends on previous words/tokens
// TODO: adding the sentence-end scoring in finalizeHypotheses() gave a slight degradation in WER

class ModelCombTreeTimesyncBeamSearch : public SearchAlgorithmV2 {
public:
    static const Core::ParameterInt   paramGlobalMaxBeamSize;
    static const Core::ParameterInt   paramGlobalMaxWordEndBeamSize;
    static const Core::ParameterInt   paramGlobalMaxPreviousWordEndBeamSize;
    static const Core::ParameterFloat paramGlobalScoreThreshold;
    static const Core::ParameterFloat paramGlobalWordEndScoreThreshold;
    static const Core::ParameterFloat paramGlobalPreviousWordEndScoreThreshold;
    static const Core::ParameterInt   paramGlobalMaxPreviousWordEndLifetime;
    static const Core::ParameterBool  paramAllowNonGlobalWordEnds;
    static const Core::ParameterFloat paramNonGlobalWordEndPenalty;
    static const Core::ParameterFloat paramNonGlobalWordEndScoreThreshold;
    static const Core::ParameterInt   paramMaxNonGlobalWordEndBeamSize;
    static const Core::ParameterBool  paramPruneWordEndHypsBeforeSplit;
    static const Core::ParameterBool  paramKeepGlobalWordEndHyps;
    static const Core::ParameterBool  paramSentenceEndFallBack;
    static const Core::ParameterBool  paramLogStepwiseStatistics;
    static const Core::ParameterBool  paramCacheCleanupInterval;
    static const Core::ParameterInt   paramNumModels;

    ModelCombTreeTimesyncBeamSearch(Core::Configuration const&);

    // Inherited methods from `SearchAlgorithmV2`

    Speech::ModelCombination::Mode  requiredModelCombination() const override;
    Am::AcousticModel::Mode         requiredAcousticModel() const override;
    bool                            setModelCombination(Speech::ModelCombination const& modelCombination) override;
    void                            reset() override;
    void                            enterSegment(Bliss::SpeechSegment const* = nullptr) override;
    void                            finishSegment() override;
    void                            putFeature(Nn::DataView const& feature) override;
    void                            putFeatures(Nn::DataView const& features, size_t nTimesteps) override;
    Core::Ref<const Traceback>      getCurrentBestTraceback() const override;
    Core::Ref<const LatticeAdaptor> getCurrentBestWordLattice() const override;
    Core::Ref<const LatticeTrace>   getCurrentBestLatticeTrace() const override;
    Core::Ref<const LatticeTrace>   getCommonPrefix() const override;

    bool decodeStep() override;

protected:
    /*
     * Possible extension for some label hypothesis in the beam
     */
    struct WithinWordExtensionCandidate {
        Nn::LabelIndex                  nextToken;       // Proposed token to extend the hypothesis with
        StateId                         state;           // State in the search tree of this extension
        Score                           score;           // Would-be total score of the full hypothesis after extension
        Search::TimeframeIndex          timeframe;       // Timestamp of `nextToken` for traceback
        Nn::LabelScorer::TransitionType transitionType;  // Type of transition toward `nextToken`
        size_t                          baseHypIndex;    // Index of base hypothesis in global beam

        bool operator<(WithinWordExtensionCandidate const& other) const {
            return score < other.score;
        }
    };

    struct WordEndExtensionCandidate {
        Nn::LabelIndex                   nextToken;           // Proposed token to extend the hypothesis with
        const Bliss::LemmaPronunciation* pron;                // Pronunciation of the lemma
        StateId                          state;               // State in the search tree of this extension
        bool                             atGlobalWordEnd;     // Whether the word end is global (shared by all models) and not blank
        bool                             atNonGlobalWordEnd;  // Whether the word end is not global and not blank
        Lm::History                      lmHistory;           // LM history of the hypothesis, possibly extended at a word end
        Score                            score;               // Would-be total score of the full hypothesis after extension (incl. LM score)
        Score                            lmScore;             // Would-be LM score of a word-end hypothesis after extension
        Search::TimeframeIndex           timeframe;           // Timestamp of `nextToken` for traceback
        Nn::LabelScorer::TransitionType  transitionType;      // Type of transition toward `nextToken`
        size_t                           lifetime;            // Number of timeframes the (previous) word end already exists uncombined
        bool                             previousWordEndHyp;  // Whether this word end is a previous word-end hypothesis which waits for combination
        Score                            lastWordEndScore;    // For previous word-end hypotheses, the actual score of the word end
        size_t                           baseHypIndex;        // Index of base hypothesis in global beam

        bool operator<(WordEndExtensionCandidate const& other) const {
            return score < other.score;
        }

        bool operator==(WordEndExtensionCandidate const& other) const {
            return std::string{pron->lemma()->symbol().str()} == std::string{other.pron->lemma()->symbol().str()};
        }
    };

    /*
     * Struct containing all information about a single hypothesis in the beam
     */
    struct LabelHypothesis {
        Nn::ScoringContextRef            scoringContext;     // Context to compute scores based on this hypothesis
        Nn::LabelIndex                   currentToken;       // Most recent token in associated label sequence (useful to infer transition type)
        StateId                          currentState;       // Current state in the search tree
        const Bliss::LemmaPronunciation* pron;               // Pronunciation of the lemma (required for previous word-end hypotheses)
        bool                             atWordEnd;          // Whether the hypothesis is at a word end
        Lm::History                      lmHistory;          // Language model history
        Speech::TimeframeIndex           timeframe;          // Timeframe of current token
        Score                            score;              // Full score of the hypothesis
        Score                            lastWordEndScore;   // For previous word-end hypotheses, the actual score of the word end
        size_t                           lifetime;           // Number of timeframes the word end already exists uncombined
        Core::Ref<LatticeTrace>          trace;              // Associated trace for traceback or lattice building of hypothesis
        size_t                           globalWordEnds;     // Total count of non-blank global word ends in this hypothesis
        size_t                           nonGlobalWordEnds;  // Total count of non-blank non-global word ends in this hypothesis

        LabelHypothesis();
        LabelHypothesis(LabelHypothesis const& base, WordEndExtensionCandidate const& extension, Nn::ScoringContextRef const& newScoringContext);
        LabelHypothesis(LabelHypothesis const& base, WithinWordExtensionCandidate const& extension, Nn::ScoringContextRef const& newScoringContext);

        bool operator<(LabelHypothesis const& other) const {
            return score < other.score;
        }
        /*
         * Get string representation for debugging
         */
        std::string toString() const;
    };

    struct Model {
        static const Core::ParameterFloat paramModelScale;
        static const Core::ParameterInt   paramMaxBeamSize;
        static const Core::ParameterInt   paramMaxWordEndBeamSize;
        static const Core::ParameterInt   paramMaxPreviousWordEndBeamSize;
        static const Core::ParameterFloat paramScoreThreshold;
        static const Core::ParameterFloat paramWordEndScoreThreshold;
        static const Core::ParameterFloat paramPreviousWordEndScoreThreshold;
        static const Core::ParameterInt   paramMaxPreviousWordEndLifetime;
        static const Core::ParameterBool  paramCollapseRepeatedLabels;

        Core::Ref<Nn::LabelScorer>                     labelScorer;
        Bliss::LexiconRef                              lexicon;
        robin_hood::unordered_set<const Bliss::Lemma*> nonWordLemmas;
        Core::Ref<PersistentStateTree>                 network;
        Core::Ref<const Am::AcousticModel>             acousticModel;

        Score scale;

        bool collapseRepeatedLabels;

        Nn::LabelIndex      blankLabelIndex;
        Bliss::Lemma const* sentenceEndLemma;
        Nn::LabelIndex      sentenceEndLabelIndex;

        size_t maxBeamSize;
        size_t maxWordEndBeamSize;
        size_t maxPreviousWordEndBeamSize;
        Score  scoreThreshold;
        Score  wordEndScoreThreshold;
        Score  previousWordEndScoreThreshold;
        size_t maxLifetime;

        std::vector<WithinWordExtensionCandidate> withinWordExtensions;
        std::vector<WordEndExtensionCandidate>    wordEndExtensions;
        std::vector<LabelHypothesis>              beam;
        std::vector<LabelHypothesis>              newBeam;
        std::vector<Nn::LabelScorer::Request>     requests;
        std::vector<WordEndExtensionCandidate>    previousWordEndExtensions;
        std::vector<LabelHypothesis>              previousWordEndHyps;
        std::vector<Nn::LabelScorer::Request>     requestsForPreviousExtensions;
        std::vector<LabelHypothesis>              withinWordHyps;
        std::vector<LabelHypothesis>              wordEndHyps;
        std::vector<LabelHypothesis>              tempHypotheses;

        std::vector<size_t>                    stateSuccessorsOffset;
        std::vector<StateId>                   stateSuccessors;
        std::vector<size_t>                    stateExitsOffset;
        std::vector<PersistentStateTree::Exit> stateExits;

        Core::Statistics<u32> numWithinWordHyps;
        Core::Statistics<u32> numIndividualWordEndHyps;
        Core::Statistics<u32> numGlobalWordEndHyps;
        Core::Statistics<u32> numPreviousWordEndHyps;

        Model(Core::Configuration const& config);
    };

private:
    size_t             numModels_;
    std::vector<Model> models_;

    size_t globalMaxBeamSize_;
    size_t globalMaxWordEndBeamSize_;
    size_t globalMaxPreviousWordEndBeamSize_;

    Score  globalScoreThreshold_;
    Score  globalWordEndScoreThreshold_;
    Score  globalPreviousWordEndScoreThreshold_;
    size_t globalMaxPreviousWordEndLifetime_;

    bool   allowNonGlobalWordEnds_;
    Score  nonGlobalWordEndPenalty_;
    Score  nonGlobalWordEndScoreThreshold_;
    size_t maxNonGlobalWordEndBeamSize_;

    bool pruneWordEndHypsBeforeSplit_;
    bool keepGlobalWordEndHyps_;

    bool   sentenceEndFallback_;
    size_t cacheCleanupInterval_;

    bool          logStepwiseStatistics_;
    Core::Channel debugChannel_;

    Bliss::LexiconRef                  globalLexicon_;
    Core::Ref<Lm::ScaledLanguageModel> globalLanguageModel_;

    std::vector<LabelHypothesis>           recombinedHypotheses_;  // only temporary storage for one model
    std::vector<LabelHypothesis>           individualWordEndHypotheses_;
    std::vector<WordEndExtensionCandidate> globalWordEndExtensions_;
    std::vector<LabelHypothesis>           finalBeam_;

    size_t currentSearchStep_;

    Core::StopWatch initializationTime_;
    Core::StopWatch featureProcessingTime_;
    Core::StopWatch scoringTime_;

    bool finishedSegment_;

    LabelHypothesis const& getBestHypothesis() const;
    LabelHypothesis const& getWorstHypothesis() const;

    void resetStatistics();
    void logStatistics() const;

    void wordEndExtensionHandling();

    /*
     * Infer type of transition between two tokens based on whether each of them is blank
     * and/or whether they are the same
     */
    Nn::LabelScorer::TransitionType inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel, bool collapseRepeatedLabels, Nn::LabelIndex blankLabelIndex) const;

    /*
     * Helper function for pruning to maxBeamSize
     */
    template<typename Element>
    void beamSizePruning(std::vector<Element>& extensions, size_t maxBeamSize) const;

    /*
     * Helper function for pruning to scoreThreshold
     */
    template<typename Element>
    void scorePruning(std::vector<Element>& extensions, Score scoreThreshold) const;

    /*
     * Helper function for recombination of hypotheses at the same point in the tree with the same scoring context and LM history
     */
    void recombination(std::vector<LabelHypothesis>& hypotheses);
    void recombinationPrevious(std::vector<LabelHypothesis>& hypotheses);  // also take into account the pron (for previous word-end hyps where the LM history is not updated yet)

    /*
     * Precompute successor and exit lookups for each state to avoid traversing the network structure during decoding.
     * Successors and exits are stored in the contiguous vectors stateSuccessors_ and stateExits_.
     * for a state `s`, the corresponding ranges are indexed by
     * (stateSuccessorsOffset_[s], stateSuccessorsOffset_[s+1]) and (stateExitsOffset_[s], stateExitsOffset_[s+1])
     */
    void createSuccessorLookups();

    /*
     * After reaching the segment end, go through the active hypotheses, only keep those
     * which are at a word end (in the root state) and add the sentence end LM score.
     * If no word-end hypotheses exist, use sentence-end fallback or construct an empty hypothesis
     */
    void finalizeHypotheses();
};

}  // namespace Search

#endif  // MODEL_COMB_TREE_TIMESYNC_BEAM_SEARCH_HH
