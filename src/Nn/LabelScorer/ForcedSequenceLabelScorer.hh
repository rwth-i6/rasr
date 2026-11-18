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
#ifndef FORCED_SEQUENCE_LABEL_SCORER_HH
#define FORCED_SEQUENCE_LABEL_SCORER_HH

#include <memory>
#include <optional>
#include <vector>

#include "LabelScorer.hh"
#include "ScoreAccessor.hh"

namespace Nn {

/*
 * Score accessor for one position in the target sequence
 */
class ForcedSequenceScoreAccessor : public ScoreAccessor {
public:
    ForcedSequenceScoreAccessor(std::shared_ptr<std::vector<LabelIndex> const> targetSequence, size_t position, Score infinityScore);

    // 0 for the next label of the target sequence, the sentence end after the whole sequence and
    // transitions that don't emit a label, and `infinityScore` otherwise
    Score getScore(TransitionType transitionType, LabelIndex labelIndex = invalidLabelIndex) const override;

    // Only the next label of the target sequence can follow via a label transition
    std::optional<SparseScoreSpan> getSparseScores() const override;

private:
    std::shared_ptr<std::vector<LabelIndex> const> targetSequence_;
    size_t                                         position_;
    Score                                          infinityScore_;
};

/*
 * Label scorer that only allows one given label sequence, the label scorer counterpart of the "cheating" word LM
 * `Lm::CheatingSegmentLm`:
 *  - Transitions that emit a label (label-to-label, blank-to-label, silence-to-label, initial-label) get score 0 if
 *    the label is the next label of the target sequence and `infinity-score` otherwise.
 *  - The sentence end gets score 0 after the whole target sequence and `infinity-score` before.
 *  - All other transitions (blank, loops, silence, word exits) don't emit a label of the target sequence and get
 *    score 0.
 *
 * Together with other label scorers, the search then can only follow the target sequence, and the score of the
 * resulting hypothesis is the score that the other label scorers assign to this sequence ("forced scoring"), e.g.
 * to rescore given hypotheses with exactly the models and computations of the search. The score accessors provide
 * sparse scores (only the next label of the target sequence), so the searches that support them only create the
 * extensions with this label (plus the transitions without a new label) instead of iterating over all labels. Most
 * efficient as the first of several label scorer stages (`num-label-scorers`): then the following label scorers only
 * have to read out the scores of these extensions.
 * In time-synchronous searches, the result is the best alignment of the target sequence within the beam, and only the
 * sentence end enforces the complete target sequence, so a sentence end must be configured.
 *
 * The target sequence must be set via `setTargetSequence` (e.g. from Python) before scoring the segment, it is kept
 * until it is replaced. The scoring context is the number of target labels emitted so far.
 */
class ForcedSequenceLabelScorer : public StatTrackingLabelScorer {
public:
    using Precursor = StatTrackingLabelScorer;

    static const Core::ParameterFloat paramInfinityScore;

    ForcedSequenceLabelScorer(Core::Configuration const& config);
    virtual ~ForcedSequenceLabelScorer() = default;

    // Set the label sequence that the following segments are forced to (without sentence end)
    void setTargetSequence(std::vector<LabelIndex> const& targetSequence);

    // No inputs needed, the scores only depend on the target sequence
    void signalNoMoreFeatures() override {}
    void addInput(DataView const& input) override {}
    void addInputs(DataView const& input, size_t nTimesteps) override {}

    ScoringContextRef getInitialScoringContext() override;

    // Scoring context with the number of emitted target labels increased by one if the transition emits a label
    ScoringContextRef extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) override;

protected:
    void                            resetInternal() override {}
    std::optional<ScoreAccessorRef> computeScoreAccessor(ScoringContextRef scoringContext) override;

private:
    Score                                          infinityScore_;
    std::shared_ptr<std::vector<LabelIndex> const> targetSequence_;
};

}  // namespace Nn

#endif  // FORCED_SEQUENCE_LABEL_SCORER_HH
