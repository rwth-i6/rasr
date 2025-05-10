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
#include <Search/PersistentStateTree.hh>
#include <Search/SearchV2.hh>
#include <Search/Traceback.hh>

namespace Search {

/*
 * Simple time synchronous beam search algorithm on a search tree built by the CtcTreeBuilder oder RnaTreeBuilder.
 * At a word end, a language model score is added to the hypothesis score,
 * if no language model should be used, the LM-scale has to be set to 0.0.
 * Supports global or separate pruning of within-word and word-end hypotheses
 * by max beam-size and by score difference to the best hypothesis.
 * Uses a LabelScorer to context initialization/extension and scoring.
 *
 * The blank label index is retrieved from the lexicon to ensure consistency with the blank index used for the search tree.
 * If the search tree contains label-loops, one will most likely want to set "collapse-repeated-labels" to true so
 * the label loops are also considered when inferring the transtion type as scoring context.
 * Similarly, if the search tree forces blank between two repeated labels (and if repeated labels are collapsed),
 * blank should also be forced across words if the new word starts with the same label as the previous word ended,
 * so "force-blank-between-repeated-labels-across-words" has to be set to true in this case.
 */
class TreeTimesyncBeamSearch : public SearchAlgorithmV2 {
protected:
    /*
     * Possible extension for some label hypothesis in the beam
     */
    struct ExtensionCandidate {
        Nn::LabelIndex                   nextToken;       // Proposed token to extend the hypothesis with
        const Bliss::LemmaPronunciation* pron;            // Pronunciation of the lemma if we are at a word end
        StateId                          state;           // State in the search tree of this extension
        Lm::History                      lmHistory;       // LM history of the hypothesis, possibly extended at a word end
        Score                            score;           // Would-be total score of the full hypothesis after extension (incl. LM score)
        Score                            lmScore;         // Would-be LM score of a word-end hypothesis after extension
        Search::TimeframeIndex           timeframe;       // Timestamp of `nextToken` for traceback
        Nn::LabelScorer::TransitionType  transitionType;  // Type of transition toward `nextToken`
        size_t                           baseHypIndex;    // Index of base hypothesis in global beam

        bool operator<(ExtensionCandidate const& other) {
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
        Score                   score;           // Full score of the hypothesis
        Core::Ref<LatticeTrace> trace;           // Associated trace for traceback or lattice building of hypothesis

        LabelHypothesis();
        LabelHypothesis(LabelHypothesis const& base, ExtensionCandidate const& extension, Nn::ScoringContextRef const& newScoringContext);

        bool operator<(LabelHypothesis const& other) const {
            return score < other.score;
        }

        /*
         * Get string representation for debugging
         */
        std::string toString() const;
    };

public:
    static const Core::ParameterInt   paramMaxBeamSize;
    static const Core::ParameterInt   paramMaxWordEndBeamSize;
    static const Core::ParameterFloat paramScoreThreshold;
    static const Core::ParameterFloat paramWordEndScoreThreshold;
    static const Core::ParameterBool  paramCollapseRepeatedLabels;
    static const Core::ParameterBool  paramForceBlankAcrossWords;
    static const Core::ParameterBool  paramSentenceEndFallBack;
    static const Core::ParameterBool  paramLogStepwiseStatistics;

    TreeTimesyncBeamSearch(Core::Configuration const&);

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

    Score scoreThreshold_;
    Score wordEndScoreThreshold_;

    Nn::LabelIndex blankLabelIndex_;

    bool collapseRepeatedLabels_;
    bool forceBlankAcrossWords_;

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

    Core::Statistics<u32> numHypsAfterScorePruning_;
    Core::Statistics<u32> numHypsAfterBeamPruning_;
    Core::Statistics<u32> numWordEndHypsAfterScorePruning_;
    Core::Statistics<u32> numWordEndHypsAfterBeamPruning_;
    Core::Statistics<u32> numActiveHyps_;

    bool finishedSegment_;

    LabelHypothesis const& getBestHypothesis() const;
    LabelHypothesis const& getWorstHypothesis() const;

    void resetStatistics();
    void logStatistics() const;

    /*
     * Infer type of transition between two tokens based on whether each of them is blank
     * and/or whether they are the same
     */
    Nn::LabelScorer::TransitionType inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel, bool inRoot = false) const;

    /*
     * Helper function for pruning to maxBeamSize
     */
    void beamSizePruning(std::vector<TreeTimesyncBeamSearch::ExtensionCandidate>& extensions, size_t maxBeamSize) const;

    /*
     * Helper function for pruning to scoreThreshold
     */
    void scorePruning(std::vector<TreeTimesyncBeamSearch::ExtensionCandidate>& extensions, Score scoreThreshold) const;

    /*
     * Helper function for recombination of hypotheses at the same point in the tree with the same scoring context and LM history
     */
    void recombination(std::vector<LabelHypothesis>& hypotheses);

    /*
     * Precompute information about the successor structure of each state in the search tree
     * to avoid repeated computation during the decode steps
     * stateSuccessorLookup_: contains a list of all state successors for the state at the corresponding index
     * exitLookup_: contains a list of all exits for the state at the corresponding index
     */
    // TODO make this more efficient, especially for states with only one exit (cf. AdvancedTreeSearch)
    void createSuccessorLookups();

    /*
     * After reaching the segment end, go through the active hypotheses, only keep those
     * which are at a word end (in the root state) and add the sentence end LM score.
     * If no word-end hypotheses exist, use sentence-end fallback or construct an empty hypothesis
     */
    void finalizeLmScoring();
};

}  // namespace Search

#endif  // TREE_TIMESYNC_BEAM_SEARCH_HH
