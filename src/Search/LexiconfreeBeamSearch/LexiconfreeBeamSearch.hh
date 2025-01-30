/** Copyright 2020 RWTH Aachen University. All rights reserved.
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

#ifndef LEXICONFREE_BEAM_SEARCH_HH
#define LEXICONFREE_BEAM_SEARCH_HH

#include <Search/SearchV2.hh>
#include <chrono>
#include <ratio>
#include "Bliss/Lexicon.hh"
#include "Core/Parameter.hh"
#include "Nn/LabelScorer/SharedDataHolder.hh"

namespace Search {

// Bare-bones beam search algorithm without pronunciation lexicon, LM, transition model.
// Given a lexicon only containing labels (without lemmas), pick the label index with
// maximum probability at each decoding step.
// Supports pruning of the top-k successor of each hypothesis, max-beam-size-pruning and score-based pruning.
class LexiconfreeBeamSearch : public SearchAlgorithmV2 {
    struct HypothesisExtension {
        const Bliss::LemmaPronunciation* pron;
        Nn::ScoringContextRef            scoringContext;
        Nn::LabelIndex                   label;
        Score                            score;
        Search::TimeframeIndex           timestep;
        Nn::LabelScorer::TransitionType  transitionType;

        HypothesisExtension()
                : pron(), scoringContext(), label(), score(Core::Type<Score>::max), timestep(), transitionType() {}

        HypothesisExtension(const Bliss::LemmaPronunciation* pron, Core::Ref<const Nn::ScoringContext> scoringContext, Nn::LabelIndex label, Score score, Search::TimeframeIndex timestep, Nn::LabelScorer::TransitionType transitionType)
                : pron(pron), scoringContext(scoringContext), label(label), score(score), timestep(timestep), transitionType(transitionType) {}
    };

    struct LabelHypothesis {
        Nn::ScoringContextRef scoringContext;
        Nn::LabelIndex        currentLabel;
        Score                 score;
        unsigned int          length;
        Traceback             traceback;

        LabelHypothesis()
                : scoringContext(), currentLabel(Core::Type<Nn::LabelIndex>::max), score(0.0), length(0), traceback() {}

        LabelHypothesis(LabelHypothesis const& base);
        LabelHypothesis(LabelHypothesis const& base, HypothesisExtension const& extension);

        std::string toString() const;
    };

    struct TimeStatistic {
        double total = 0.0;

        void reset() {
            total = 0.0;
        }

        void tic() {
            startTime = std::chrono::steady_clock::now();
        }

        void toc() {
            auto endTime = std::chrono::steady_clock::now();
            // Duration in milliseconds
            total += std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(endTime - startTime).count();
        }

    private:
        std::chrono::time_point<std::chrono::steady_clock> startTime;
    };

public:
    static const Core::ParameterInt   paramMaxBeamSize;
    static const Core::ParameterInt   paramTopKTokens;
    static const Core::ParameterFloat paramScoreThreshold;
    static const Core::ParameterFloat paramLengthNormScale;
    static const Core::ParameterBool  paramUseBlank;
    static const Core::ParameterInt   paramBlankLabelIndex;
    static const Core::ParameterBool  paramAllowLabelLoop;
    static const Core::ParameterBool  paramUseSentenceEnd;
    static const Core::ParameterBool  paramSentenceEndIndex;

    LexiconfreeBeamSearch(Core::Configuration const&);

    // Inherited methods

    Speech::ModelCombination::Mode  requiredModelCombination() const override;
    bool                            setModelCombination(Speech::ModelCombination const& modelCombination) override;
    void                            reset() override;
    void                            enterSegment(Bliss::SpeechSegment const* = nullptr) override;
    void                            finishSegment() override;
    void                            passFeature(Nn::SharedDataHolder const& data, size_t featureSize) override;
    void                            passFeatures(Nn::SharedDataHolder const& data, size_t timeSize, size_t featureSize) override;
    Core::Ref<const Traceback>      getCurrentBestTraceback() const override;
    Core::Ref<const LatticeAdaptor> getCurrentBestWordLattice() const override;
    bool                            decodeStep() override;

private:
    void resetStatistics();
    void logStatistics() const;

    Nn::LabelScorer::TransitionType inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const;

    // Helper function for top-k pruning of the successor tokens
    void topKTokenPruning(std::vector<size_t>& indices, std::vector<Score> const& extensionScores, size_t numUnfinishedHyps);

    /* Helper function for pruning to maxBeamSize_
     * @tparam hypotheses A container type (e.g. std::vector) that holds the hypotheses (or their inidces) to be sorted and pruned
     * @tparam compare A callable (e.g. lambda or function pointer) that takes two elements of the hypotheses and returns true if the first element should precede the second element
     */
    template<typename T>
    void beamPruning(std::vector<T>& hypotheses, std::function<bool(T const&, T const&)>&& compare);

    /* Helper function for score-based pruning
     * @tparam hypotheses A container type (e.g. std::vector) that holds the hypotheses (or their indices) to be pruned sorted by their score
     * @tparam getScore A callable (e.g. lambda or function pointer) that takes a single element from the hypotheses and returns its score
     */
    template<typename T>
    void scorePruning(std::vector<T>& hypotheses, std::function<Score(T const&)>&& getScore);

    size_t maxBeamSize_;

    bool   useTokenPruning_;
    size_t topKTokens_;

    bool  useScorePruning_;
    Score scoreThreshold_;

    f32 lengthNormScale_;

    bool useBlank_;
    bool useSentenceEnd_;
    bool allowLabelLoop_;

    Nn::LabelIndex blankLabelIndex_;
    Nn::LabelIndex sentenceEndIndex_;

    Core::Ref<Nn::LabelScorer>   labelScorer_;
    Nn::LabelIndex               numClasses_;
    Bliss::LexiconRef            lexicon_;
    std::vector<LabelHypothesis> beam_;

    TimeStatistic initializationTime_;
    TimeStatistic featureProcessingTime_;
    TimeStatistic scoringTime_;
    TimeStatistic contextExtensionTime_;
};

}  // namespace Search
#endif  // LEXICONFREE_BEAM_SEARCH_HH
