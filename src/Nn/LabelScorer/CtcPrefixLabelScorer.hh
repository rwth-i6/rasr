/** Copyright 2026 RWTH Aachen University. All rights reserved.
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

#ifndef CTC_PREFIX_LABEL_SCORER_HH
#define CTC_PREFIX_LABEL_SCORER_HH

#include "LabelScorer.hh"

namespace Nn {

class CtcPrefixScoreAccessor : public ScoreAccessor {
public:
    CtcPrefixScoreAccessor(CtcPrefixScoringContextRef const& scoringContext, std::shared_ptr<Math::FastMatrix<Score>> const& ctcScores);

    // Compute score of extended prefix with labelIndex on-demand
    Score getScore(TransitionType transitionType, LabelIndex labelIndex = invalidLabelIndex) const override;

    // Label-sequence length of extended prefix
    TimeframeIndex getTime() const override;

private:
    CtcPrefixScoringContextRef               scoringContext_;
    std::shared_ptr<Math::FastMatrix<Score>> ctcScores_;
};

/*
 * Compute prefix scores with a CTC model in order to decode label-synchronously.
 * Prefix scores are computed like in algorithm 2 of "Hybrid CTC/Attention Architecture
 * for End-to-End Speech Recognition" (Watanabe et al., 2017)
 */
class CtcPrefixLabelScorer : public LabelScorer {
    using Precursor = LabelScorer;

    static const Core::ParameterInt paramBlankIndex;
    static const Core::ParameterInt paramVocabSize;

public:
    CtcPrefixLabelScorer(Core::Configuration const& config);
    virtual ~CtcPrefixLabelScorer() = default;

    void reset() override;
    void signalNoMoreFeatures() override;
    void addInput(DataView const& input) override;
    void addInputs(DataView const& inputs, size_t nTimesteps) override;

    ScoringContextRef               getInitialScoringContext() override;
    ScoringContextRef               extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) override;
    std::optional<ScoreAccessorRef> getScoreAccessor(ScoringContextRef scoringContext) override;

private:
    LabelIndex             blankIndex_;
    size_t                 vocabSize_;
    Core::Ref<LabelScorer> ctcScorer_;
    bool                   expectMoreFeatures_;

    std::shared_ptr<Math::FastMatrix<Score>> ctcScores_;  // Cached T x V matrix of scores

    // Retrieve matrix of CTC scores from sub-scorer. Assumes that these scores only depend on timestep and label index, not history or transition type.
    void setupCTCScores();

    // Update prefix scores in scoringContext
    void finalizeScoringContext(CtcPrefixScoringContextRef const& scoringContext) const;
};

}  // namespace Nn

#endif  // CTC_PREFIX_LABEL_SCORER_HH
