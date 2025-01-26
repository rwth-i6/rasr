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
#include "Core/Parameter.hh"

namespace Search {

// Bare-bones search algorithm without pronunciation lexicon, LM, transition model, beam or pruning.
// Given a lexicon only containing labels (without lemmas), pick the label index with
// maximum probability at each decoding step.
class LexiconfreeBeamSearch : public SearchAlgorithmV2 {
    struct HypothesisExtension {
        const Bliss::Lemma*             lemma;
        Nn::ScoringContextRef           scoringContext;
        Nn::LabelIndex                  label;
        Score                           score;
        int                             length;
        Search::TimeframeIndex          timestep;
        Nn::LabelScorer::TransitionType transitionType;

        HypothesisExtension()
                : lemma(), scoringContext(), label(), score(Core::Type<Score>::max), length(), timestep(), transitionType() {}

        HypothesisExtension(const Bliss::Lemma* lemma, Core::Ref<const Nn::ScoringContext> scoringContext, Nn::LabelIndex label, Score score, int length, Search::TimeframeIndex timestep, Nn::LabelScorer::TransitionType transitionType)
                : lemma(lemma), scoringContext(scoringContext), label(label), score(score), length(length), timestep(timestep), transitionType(transitionType) {}
    };

    struct LabelHypothesis {
        Nn::ScoringContextRef scoringContext;
        Nn::LabelIndex        currentLabel;
        Score                 score;
        int                   length;
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

    Speech::ModelCombination::Mode  modelCombinationNeeded() const override;
    bool                            setModelCombination(Speech::ModelCombination const& modelCombination) override;
    void                            reset() override;
    void                            enterSegment() override;
    void                            enterSegment(Bliss::SpeechSegment const*) override;
    void                            finishSegment() override;
    void                            addFeature(std::shared_ptr<const f32[]> const& data, size_t featureSize) override;
    void                            addFeature(std::vector<f32> const& data) override;
    void                            addFeatures(std::shared_ptr<const f32[]> const& data, size_t timeSize, size_t featureSize) override;
    Core::Ref<const Traceback>      getCurrentBestTraceback() const override;
    Core::Ref<const LatticeAdaptor> getCurrentBestWordLattice() const override;
    void                            resetStatistics() override;
    void                            logStatistics() const override;
    bool                            decodeStep() override;

private:
    Nn::LabelScorer::TransitionType inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const;

    size_t maxBeamSize_;

    bool   useTokenPruning_;
    size_t topKTokens_;

    bool  useScorePruning_;
    Score scoreThreshold_;

    bool useLengthNormalization_;
    float lengthNormScale_;

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
