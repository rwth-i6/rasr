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

#ifndef LEXICONFREE_TIMESYNC_BEAM_SEARCH_HH
#define LEXICONFREE_TIMESYNC_BEAM_SEARCH_HH

#include <Bliss/Lexicon.hh>
#include <Core/Channel.hh>
#include <Core/Parameter.hh>
#include <Core/StopWatch.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/SearchV2.hh>
#include <Search/Traceback.hh>

namespace Search {

/*
 * Simple time synchronous beam search algorithm without pronunciation lexicon, word-level LM or transition model.
 * Can handle a blank symbol if a blank index is set.
 * Main purpose is open vocabulary search with CTC/Neural Transducer (or similar) models.
 * Supports global pruning by max beam-size and by score difference to the best hypothesis.
 * Uses a LabelScorer to context initialization/extension and scoring.
 *
 * The search requires a lexicon that represents the vocabulary. Each lemma is viewed as a token with its index
 * in the lexicon corresponding to the associated output index of the label scorer.
 */
class LexiconfreeTimesyncBeamSearch : public SearchAlgorithmV2 {
protected:
    /*
     * Possible extension for some label hypothesis in the beam
     */
    struct ExtensionCandidate {
        Nn::LabelIndex                   nextToken;       // Proposed token to extend the hypothesis with
        const Bliss::LemmaPronunciation* pron;            // Pronunciation of lemma corresponding to `nextToken` for traceback
        Score                            score;           // Would-be score of full hypothesis after extension
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
        Score                   score;           // Full score of hypothesis
        Core::Ref<LatticeTrace> trace;           // Associated trace for traceback or lattice building off of hypothesis

        LabelHypothesis();
        LabelHypothesis(LabelHypothesis const& base, ExtensionCandidate const& extension, Nn::ScoringContextRef const& newScoringContext);

        bool operator<(LabelHypothesis const& other) const {
            return score < other.score;
        }

        /*
         * Get string representation for debugging.
         */
        std::string toString() const;
    };

public:
    static const Core::ParameterInt   paramMaxBeamSize;
    static const Core::ParameterFloat paramScoreThreshold;
    static const Core::ParameterInt   paramBlankLabelIndex;
    static const Core::ParameterBool  paramCollapseRepeatedLabels;
    static const Core::ParameterBool  paramLogStepwiseStatistics;

    LexiconfreeTimesyncBeamSearch(Core::Configuration const&);

    // Inherited methods from `SearchAlgorithmV2`

    Speech::ModelCombination::Mode  requiredModelCombination() const override;
    bool                            setModelCombination(Speech::ModelCombination const& modelCombination) override;
    void                            reset() override;
    void                            enterSegment(Bliss::SpeechSegment const* = nullptr) override;
    void                            finishSegment() override;
    void                            putFeature(std::shared_ptr<const f32[]> const& data, size_t featureSize) override;
    void                            putFeature(std::vector<f32> const& data) override;
    void                            putFeatures(std::shared_ptr<const f32[]> const& data, size_t timeSize, size_t featureSize) override;
    Core::Ref<const Traceback>      getCurrentBestTraceback() const override;
    Core::Ref<const LatticeAdaptor> getCurrentBestWordLattice() const override;
    bool                            decodeStep() override;

private:
    size_t maxBeamSize_;

    bool  useScorePruning_;
    Score scoreThreshold_;

    bool           useBlank_;
    Nn::LabelIndex blankLabelIndex_;

    bool collapseRepeatedLabels_;

    bool logStepwiseStatistics_;

    Core::Channel debugChannel_;

    Core::Ref<Nn::LabelScorer>   labelScorer_;
    Bliss::LexiconRef            lexicon_;
    std::vector<LabelHypothesis> beam_;

    // Pre-allocated intermediate vectors
    std::vector<ExtensionCandidate>       extensions_;
    std::vector<LabelHypothesis>          newBeam_;
    std::vector<Nn::LabelScorer::Request> requests_;
    std::vector<LabelHypothesis>          recombinedHypotheses_;

    Core::StopWatch initializationTime_;
    Core::StopWatch featureProcessingTime_;
    Core::StopWatch scoringTime_;
    Core::StopWatch contextExtensionTime_;

    Core::Statistics<u32> numHypsAfterScorePruning_;
    Core::Statistics<u32> numHypsAfterBeamPruning_;
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
    Nn::LabelScorer::TransitionType inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const;

    /*
     * Helper function for pruning to maxBeamSize_
     */
    void beamSizePruning(std::vector<LexiconfreeTimesyncBeamSearch::ExtensionCandidate>& extensions) const;

    /*
     * Helper function for pruning to scoreThreshold_
     */
    void scorePruning(std::vector<LexiconfreeTimesyncBeamSearch::ExtensionCandidate>& extensions) const;

    /*
     * Helper function for recombination of hypotheses with the same scoring context
     */
    void recombination(std::vector<LabelHypothesis>& hypotheses);
};

}  // namespace Search

#endif  // LEXICONFREE_TIMESYNC_BEAM_SEARCH_HH
