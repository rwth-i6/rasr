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
#include <Core/Parameter.hh>
#include <Core/StopWatch.hh>
#include <Nn/LabelScorer/DataView.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/PersistentStateTree.hh>
#include <Search/SearchV2.hh>
#include <Search/Traceback.hh>

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
        Nn::LabelIndex                   nextToken;       // Proposed token to extend the hypothesis with
        const Bliss::LemmaPronunciation* pron;            // Pronunciation of lemma corresponding to `nextToken` for traceback
        StateId                          state;           // State in the search tree of this extension
        Lm::History                      lmHistory;       // LM history of the hypothesis, possibly extended at a word end
        Score                            score;           // Would-be score of full hypothesis after extension
        Score                            lmScore;         // Would-be LM score of a word-end hypothesis after extension
        Search::TimeframeIndex           timeframe;       // Timestamp of `nextToken` for traceback
        Nn::LabelScorer::TransitionType  transitionType;  // Type of transition toward `nextToken`
        size_t                           baseHypIndex;    // Index of base hypothesis in global beam

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
        Score                   score;           // Full score of hypothesis
        Score                   scaledScore;     // Length-normalized score of hypothesis
        Core::Ref<LatticeTrace> trace;           // Associated trace for traceback or lattice building off of hypothesis
        bool                    isActive;        // Indicates whether the hypothesis has not produced a sentence-end label yet

        LabelHypothesis();
        LabelHypothesis(LabelHypothesis const& base, ExtensionCandidate const& extension, Nn::ScoringContextRef const& newScoringContext, float lengthNormScale);

        bool operator<(LabelHypothesis const& other) const {
            return scaledScore < other.scaledScore;
        }

        bool operator>(LabelHypothesis const& other) const {
            return scaledScore > other.scaledScore;
        }

        /*
         * Get string representation for debugging.
         */
        std::string toString() const;
    };

public:
    static const Core::ParameterInt   paramMaxBeamSize;
    static const Core::ParameterInt   paramMaxWordEndBeamSize;
    static const Core::ParameterFloat paramScoreThreshold;
    static const Core::ParameterFloat paramWordEndScoreThreshold;

    static const Core::ParameterFloat paramLengthNormScale;
    static const Core::ParameterFloat paramMaxLabelsPerTimestep;
    static const Core::ParameterBool  paramSentenceEndFallBack;
    static const Core::ParameterBool  paramLogStepwiseStatistics;

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
    Core::Ref<const Traceback>      getCurrentBestTraceback() const override;
    Core::Ref<const LatticeAdaptor> getCurrentBestWordLattice() const override;
    bool                            decodeStep() override;

private:
    size_t maxBeamSize_;
    size_t maxWordEndBeamSize_;

    bool  useScorePruning_;
    Score scoreThreshold_;
    Score wordEndScoreThreshold_;

    float lengthNormScale_;

    float maxLabelsPerTimestep_;

    Nn::LabelIndex sentenceEndLabelIndex_;

    bool sentenceEndFallback_;

    bool logStepwiseStatistics_;

    Core::Channel debugChannel_;

    Core::Ref<Nn::LabelScorer>               labelScorer_;
    Bliss::LexiconRef                        lexicon_;
    Core::Ref<PersistentStateTree>           network_;
    Core::Ref<const Am::AcousticModel>       acousticModel_;
    Core::Ref<const Lm::ScaledLanguageModel> languageModel_;
    std::vector<LabelHypothesis>             beam_;

    // Pre-allocated intermediate vectors
    std::vector<ExtensionCandidate>       extensions_;
    std::vector<ExtensionCandidate>       withinWordExtensions_;
    std::vector<ExtensionCandidate>       wordEndExtensions_;
    std::vector<LabelHypothesis>          newBeam_;
    std::vector<Nn::LabelScorer::Request> requests_;
    std::vector<LabelHypothesis>          recombinedHypotheses_;

    int maxNumberOfExits_;

    std::vector<std::vector<StateId>>                   stateSuccessorLookup_;
    std::vector<std::vector<PersistentStateTree::Exit>> exitLookup_;

    Core::StopWatch initializationTime_;
    Core::StopWatch featureProcessingTime_;
    Core::StopWatch scoringTime_;
    Core::StopWatch contextExtensionTime_;

    Core::Statistics<u32> numTerminatedHypsAfterScorePruning_;
    Core::Statistics<u32> numTerminatedHypsAfterBeamPruning_;
    Core::Statistics<u32> numActiveHypsAfterScorePruning_;
    Core::Statistics<u32> numActiveHypsAfterBeamPruning_;
    Core::Statistics<u32> numActiveWordEndHypsAfterScorePruning_;
    Core::Statistics<u32> numActiveWordEndHypsAfterBeamPruning_;

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
     * Helper function for pruning of extensions to maxBeamSize
     */
    void beamSizePruningExtensions(std::vector<TreeLabelsyncBeamSearch::ExtensionCandidate>& extensions, size_t maxBeamSize);

    /*
     * Helper function for pruning of hyps to maxBeamSize_
     */
    void beamSizePruning();

    /*
     * Helper function for pruning of extensions to scoreThreshold
     */
    void scorePruningExtensions(std::vector<TreeLabelsyncBeamSearch::ExtensionCandidate>& extensions, Score scoreThreshold);

    /*
     * Helper function for pruning of hyps to scoreThreshold_
     */
    void scorePruning();

    /*
     * Helper function for recombination of hypotheses at the same point in the tree with the same scoring context and LM history
     */
    void recombination();

    /*
     * Precompute information about the successor structure of each state in the search tree
     * to avoid repeated computation during the decode steps
     * stateSuccessorLookup_: contains a list of all state successors for the state at the corresponding index
     * exitLookup_: contains a list of all exits for the state at the corresponding index
     */
    // TODO make this more efficient, especially for states with only one exit (cf. AdvancedTreeSearch)
    void createSuccessorLookups();

    /*
     * After reaching the segment end, go through the hypotheses, only keep those
     * which are terminated or at a word end (in the root state) and add the sentence end LM score.
     * If no terminated or word-end hypotheses exist, use sentence-end fallback or construct an empty hypothesis
     */
    void finalizeLmScoring();
};

}  // namespace Search

#endif  // TREE_LABELSYNC_BEAM_SEARCH_HH