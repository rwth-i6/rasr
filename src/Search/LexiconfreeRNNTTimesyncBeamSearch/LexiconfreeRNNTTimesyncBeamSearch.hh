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

#ifndef LEXICONFREE_RNNT_TIMESYNC_BEAM_SEARCH_HH
#define LEXICONFREE_RNNT_TIMESYNC_BEAM_SEARCH_HH

#include <Bliss/Lexicon.hh>
#include <Core/Channel.hh>
#include <Core/Parameter.hh>
#include <Core/StopWatch.hh>
#include <Nn/LabelScorer/DataView.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/SearchV2.hh>
#include <Search/Traceback.hh>

namespace Search {

/*
 * Time synchronous beam search algorithm without pronunciation lexicon, word-level LM or transition model
 * for an open vocabulary search with standard (non-monotonic) RNN-T/Transducer models.
 * At each timestep, multiple non-blank labels can be predicted.
 * A hypothesis is finished in the current timestep if it has emitted a blank label.
 * Supports global pruning by max beam-size and by score difference to the best hypothesis.
 * Uses a LabelScorer to context initialization/extension and scoring.
 *
 * The search requires a lexicon that represents the vocabulary. Each lemma is viewed as a token with its index
 * in the lexicon corresponding to the associated output index of the label scorer.
 */
class LexiconfreeRNNTTimesyncBeamSearch : public SearchAlgorithmV2 {
public:
    static const Core::ParameterInt   paramMaxBeamSize;
    static const Core::ParameterFloat paramScoreThreshold;
    static const Core::ParameterFloat paramLengthNormScale;
    static const Core::ParameterInt   paramMaxLabelsPerFrame;
    static const Core::ParameterInt   paramBlankLabelIndex;
    static const Core::ParameterInt   paramSentenceEndLabelIndex;
    static const Core::ParameterBool  paramAllowBlankAfterSentenceEnd;
    static const Core::ParameterBool  paramSentenceEndFallBack;
    static const Core::ParameterBool  paramCollapseRepeatedLabels;
    static const Core::ParameterBool  paramCacheCleanupInterval;
    static const Core::ParameterInt   paramMaximumStableDelay;
    static const Core::ParameterInt   paramMaximumStableDelayPruningInterval;
    static const Core::ParameterBool  paramLogStepwiseStatistics;

    LexiconfreeRNNTTimesyncBeamSearch(Core::Configuration const&);

    // Inherited methods from `SearchAlgorithmV2`

    Speech::ModelCombination::Mode requiredModelCombination() const override;
    bool                           setModelCombination(Speech::ModelCombination const& modelCombination) override;
    void                           reset() override;
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
        Nn::ScoringContextRef   scoringContext;      // Context to compute scores based on this hypothesis
        Nn::LabelIndex          currentToken;        // Most recent token in associated label sequence (useful to infer transition type)
        size_t                  length;              // Number of tokens in hypothesis for length normalization
        Score                   score;               // Full score of hypothesis
        Score                   scaledScore;         // Length-normalized score of hypothesis
        std::vector<int>        outputTokens;        // Previously predicted non-blank output tokens of hypothesis
        Core::Ref<LatticeTrace> trace;               // Associated trace for traceback or lattice building off of hypothesis
        bool                    reachedSentenceEnd;  // Flag whether hypothesis trace contains a sentence end emission

        LabelHypothesis();
        LabelHypothesis(LabelHypothesis const& base, ExtensionCandidate const& extension, Nn::ScoringContextRef const& newScoringContext, float lengthNormScale);

        bool operator<(LabelHypothesis const& other) const {
            return scaledScore < other.scaledScore;
        }

        /*
         * Get string representation for debugging.
         */
        std::string toString() const;
    };

private:
    size_t              maxBeamSize_;
    bool                useScorePruning_;
    Score               scoreThreshold_;
    float               lengthNormScale_;
    size_t              maxLabelsPerFrame_;
    Nn::LabelIndex      blankLabelIndex_;
    bool                allowBlankAfterSentenceEnd_;
    bool                useSentenceEnd_;
    Bliss::Lemma const* sentenceEndLemma_;
    Nn::LabelIndex      sentenceEndLabelIndex_;
    bool                sentenceEndFallback_;
    bool                collapseRepeatedLabels_;
    size_t              cacheCleanupInterval_;
    size_t              maximumStableDelay_;
    size_t              maximumStableDelayPruningInterval_;
    bool                logStepwiseStatistics_;

    Core::Channel debugChannel_;

    Core::Ref<Nn::LabelScorer>   labelScorer_;
    Bliss::LexiconRef            lexicon_;
    std::vector<LabelHypothesis> beam_;

    std::vector<LabelHypothesis> innerHyps_;  // Hyps that are active at the current timestep, so which can still be extended
    std::vector<LabelHypothesis> outerHyps_;  // Hyps that are finished for this timestep are waiting for the next timestep (ended with blank)

    // Pre-allocated intermediate vectors
    std::vector<ExtensionCandidate>       extensions_;
    std::vector<LabelHypothesis>          newBeam_;
    std::vector<Nn::LabelScorer::Request> requests_;
    std::vector<LabelHypothesis>          tempHypotheses_;

    Core::StopWatch initializationTime_;
    Core::StopWatch featureProcessingTime_;
    Core::StopWatch scoringTime_;
    Core::StopWatch contextExtensionTime_;

    Core::Statistics<u32> numActiveHyps_;
    Core::Statistics<u32> numOuterHyps_;
    Core::Statistics<u32> numInnerHyps_;
    Core::Statistics<u32> numInnerAndOuterHyps_;

    size_t currentSearchStep_;
    bool   finishedSegment_;

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
     * Helper functions for pruning to maxBeamSize_
     */
    void beamSizePruning(std::vector<LabelHypothesis>& hypotheses) const;
    void beamSizePruningLengthnormalized(std::vector<LabelHypothesis>& hypotheses) const;

    /*
     * Helper functions for pruning to scoreThreshold_
     */
    void scorePruning(std::vector<ExtensionCandidate>& extensions) const;
    void scorePruningLengthnormalized(std::vector<LabelHypothesis>& hypotheses) const;

    /*
     * Helper function for recombination of hypotheses with the same scoring context
     */
    void recombination(std::vector<LabelHypothesis>& hypotheses);

    /*
     * Prune away all hypotheses that have not reached sentence end.
     * If no hypotheses would survive this, either construct an empty one or keep the beam intact if sentence-end fallback is enabled.
     */
    void finalizeHypotheses();

    /*
     * Apply maximum-stable-delay-pruning to beam_
     */
    void maximumStableDelayPruning();
};

}  // namespace Search

#endif  // LEXICONFREE_RNNT_TIMESYNC_BEAM_SEARCH_HH
