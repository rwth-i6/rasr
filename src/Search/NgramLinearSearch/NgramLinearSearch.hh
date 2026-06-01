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

#ifndef NGRAM_LINEAR_SEARCH_HH
#define NGRAM_LINEAR_SEARCH_HH

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include <Am/AcousticModel.hh>
#include <Bliss/CorpusDescription.hh>
#include <Bliss/Lexicon.hh>
#include <Core/Channel.hh>
#include <Core/Parameter.hh>
#include <Core/Statistics.hh>
#include <Core/StopWatch.hh>
#include <Lm/ScaledLanguageModel.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/Types.hh>
#include <Search/Histogram.hh>
#include <Search/LatticeAdaptor.hh>
#include <Search/SearchV2.hh>
#include <Search/Traceback.hh>
#include <Search/Types.hh>
#include <Speech/ModelCombination.hh>

namespace Search {

/**
 * Linear pronunciation decoder for SearchV2, optimized for bigram LMs.
 *
 * The main difference to TreeTimesyncBeamSearch is:
 *
 *   TreeTimesyncBeamSearch:
 *     - search state is a tree state
 *     - LM score is applied at word end
 *     - active recombination key includes LM history
 *
 *   NgramLinearSearch:
 *     - search state is pronunciation + position
 *     - LM score is applied at word start
 *     - within-word recombination key does not need previous LM history
 *
 * After paying LM(prev_word, current_word), the reduced bigram history is just
 * current_word. Thus, inside the current pronunciation, hypotheses with
 * different predecessors but the same current pronunciation/state/scoring
 * context are equivalent.
 */
class NgramLinearSearch : public SearchAlgorithmV2 {
public:
    static const Core::ParameterInt   paramMaxBeamSize;
    static const Core::ParameterFloat paramScoreThreshold;
    static const Core::ParameterInt         paramNumHistogramBins;
    static const Core::ParameterInt         paramBlankLabelIndex;
    static const Core::ParameterBool        paramLogStepwiseStatistics;

    NgramLinearSearch(Core::Configuration const&);

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
    static constexpr size_t invalidPronunciation = static_cast<size_t>(-1);
    static constexpr size_t invalidState         = static_cast<size_t>(-1);

    struct Pronunciation {
        Bliss::LemmaPronunciation const* lemmaPronunciation;
        Nn::LabelIndex     label;
        const Bliss::SyntacticToken* st;

        Pronunciation()
                : lemmaPronunciation(nullptr),
                  label(Nn::invalidLabelIndex) {}

        bool empty() const {
            return label == Nn::invalidLabelIndex;
        }
    };

    struct LabelHypothesis {
        Nn::ScoringContextRef scoringContext;
        Nn::LabelIndex                     currentToken;
        Lm::History                        lmHistory;
        Speech::TimeframeIndex             timeframe;
        Score                              score;
        Score                              acousticScore;
        Core::Ref<LatticeTrace>            trace;

        LabelHypothesis();

        bool operator<(LabelHypothesis const& other) const {
            return score < other.score;
        }

        std::string toString() const;
    };

private:
    size_t maxBeamSize_;
    Score  scoreThreshold_;
    Histogram           scoreHistogram_;

    Nn::LabelIndex      blankLabelIndex_;

    bool logStepwiseStatistics_;

    Bliss::LexiconRef                       lexicon_;
    Core::Ref<Am::AcousticModel>            acousticModel_;
    Core::Ref<Lm::ScaledLanguageModel>      languageModel_;
    Core::Ref<Nn::LabelScorer> labelScorer_;

    Core::Channel debugChannel_;

    std::vector<Pronunciation> pronunciations_;

    std::vector<LabelHypothesis> beam_;
    std::vector<LabelHypothesis> newBeam_;
    std::unordered_map<Lm::History, size_t, Lm::History::Hash> seenHistories_;

    std::vector<Nn::ScoringContextRef> scoringContexts_;

    size_t currentSearchStep_;
    bool   finishedSegment_;

    Core::StopWatch initializationTime_;
    Core::StopWatch featureProcessingTime_;
    Core::StopWatch scoringTime_;

    Core::Statistics<u32>              numHypsBeforeRecombination_;
    Core::Statistics<u32>              numHypsAfterRecombination_;
    Core::Statistics<u32>              numHypsAfterPruning_;

    LabelHypothesis const& getBestHypothesis() const;
    LabelHypothesis const& getWorstHypothesis() const;

    void logStatistics() const;

    Nn::TransitionType inferTransitionType(Nn::LabelIndex previous, Nn::LabelIndex next) const;

    template<class Element>
    void scorePruning(std::vector<Element>& hypotheses, Score relativeThreshold, size_t maxBeamSize);

    void          initializePronunciations();
    Pronunciation createPronunciation(Bliss::LemmaPronunciation const* lemmaPronunciation) const;
};

}  // namespace Search

#endif  // NGRAM_LINEAR_SEARCH_HH