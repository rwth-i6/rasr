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
#include "ModelCache.hh"
#include "ScaledLabelScorer.hh"

namespace Nn {

/*
 * Scoring context for computation of CTC prefix scores.
 * Contains the label sequence, a parent pointer for lazy computation of the prefix score,
 * time-wise prefix scores and a cache for the score of extended prefixes.
 * Hash and equality operators are based on the label sequence.
 */
struct CtcPrefixScoringContext : public ScoringContext {
    struct PrefixScore {
        Score blankEndingScore    = std::numeric_limits<Score>::infinity();
        Score nonBlankEndingScore = std::numeric_limits<Score>::infinity();

        Score totalScore() const;
    };

    std::vector<LabelIndex>                           labelSeq;
    ScoringContextRef                                 parent;            // Parent prefix without the last label, used to finalize lazily
    mutable std::shared_ptr<std::vector<PrefixScore>> timePrefixScores;  // Represents neg-log-probabilities of emitting `labelSeq` ending in blank or nonblank after each real CTC frame t = 0, ..., T - 1
    mutable std::optional<Score>                      prefixScore;       // Cached score of the prefix -log P(prefix), computed on demand from the parent; needed to compute score-delta for next token
    mutable std::unordered_map<LabelIndex, Score>     extScores;         // Cache for -log P(prefix + token, ...) to avoid repeated computation
    mutable bool                                      requiresFinalize;  // Check whether the mutable members have been set

    CtcPrefixScoringContext();
    CtcPrefixScoringContext(std::vector<LabelIndex>&& seq, ScoringContextRef const& parent);

    bool   isEqual(ScoringContextRef const& other) const override;
    size_t hash() const override;
};

typedef Core::Ref<const CtcPrefixScoringContext> CtcPrefixScoringContextRef;

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
 *
 * A time-synchronous CTC scorer has to be configured via `ctc-scorer` sub-config
 * in order to fetch a matrix of CTC scores for each label at each timestep.
 */
class CtcPrefixLabelScorer : public LabelScorer {
    using Precursor = LabelScorer;

    static const Core::ParameterInt paramBlankIndex;
    static const Core::ParameterInt paramVocabSize;

public:
    CtcPrefixLabelScorer(Core::Configuration const& config, ModelCache& modelCache);
    virtual ~CtcPrefixLabelScorer() = default;

    // Return the CTC label scorer
    Core::Ref<ScaledLabelScorer> getCtcLabelScorer() const;

    void reset() override;
    void signalNoMoreFeatures() override;
    void addInput(DataView const& input) override;
    void addInputs(DataView const& inputs, size_t nTimesteps) override;

    ScoringContextRef               getInitialScoringContext() override;
    ScoringContextRef               extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) override;
    std::optional<ScoreAccessorRef> getScoreAccessor(ScoringContextRef scoringContext) override;

private:
    LabelIndex                   blankIndex_;
    size_t                       vocabSize_;
    Core::Ref<ScaledLabelScorer> ctcScorer_;
    bool                         expectMoreFeatures_;

    std::shared_ptr<Math::FastMatrix<Score>> ctcScores_;  // Cached T x V matrix of scores

    // Retrieve matrix of CTC scores from sub-scorer. Assumes that these scores only depend on timestep and label index, not history or transition type.
    void setupCTCScores();

    // Update prefix scores in scoringContext
    void finalizeScoringContext(CtcPrefixScoringContextRef const& scoringContext) const;
};

}  // namespace Nn

#endif  // CTC_PREFIX_LABEL_SCORER_HH
