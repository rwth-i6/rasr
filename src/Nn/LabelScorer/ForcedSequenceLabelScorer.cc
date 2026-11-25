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
#include "ForcedSequenceLabelScorer.hh"

#include "ScoringContext.hh"

namespace Nn {

namespace {

// Whether the transition emits a label, i.e. advances in the target sequence
bool emitsLabel(TransitionType transitionType) {
    switch (transitionType) {
        case TransitionType::LABEL_TO_LABEL:
        case TransitionType::BLANK_TO_LABEL:
        case TransitionType::SILENCE_TO_LABEL:
        case TransitionType::INITIAL_LABEL:
            return true;
        default:
            return false;
    }
}

}  // namespace

ForcedSequenceScoreAccessor::ForcedSequenceScoreAccessor(std::shared_ptr<std::vector<LabelIndex> const> targetSequence, size_t position, Score infinityScore)
        : targetSequence_(std::move(targetSequence)),
          position_(position),
          infinityScore_(infinityScore) {}

Score ForcedSequenceScoreAccessor::getScore(TransitionType transitionType, LabelIndex labelIndex) const {
    if (transitionType == TransitionType::SENTENCE_END) {
        return position_ == targetSequence_->size() ? 0.0 : infinityScore_;
    }
    if (emitsLabel(transitionType)) {
        return (position_ < targetSequence_->size() and (*targetSequence_)[position_] == labelIndex) ? 0.0 : infinityScore_;
    }
    return 0.0;
}

std::optional<SparseScoreSpan> ForcedSequenceScoreAccessor::getSparseScores() const {
    static constexpr Score zeroScore = 0.0;
    size_t                 numLabels = position_ < targetSequence_->size() ? 1ul : 0ul;
    return SparseScoreSpan{.labels = std::span<LabelIndex const>(targetSequence_->data() + position_, numLabels),
                           .scores = std::span<Score const>(&zeroScore, numLabels)};
}

const Core::ParameterFloat ForcedSequenceLabelScorer::paramInfinityScore(
        "infinity-score",
        "Score of labels that deviate from the target sequence (and of a sentence end before its end). The default, the "
        "maximal score value, counts as infinite: hypotheses with such a score are pruned and cannot become the result.",
        Core::Type<Score>::max);

ForcedSequenceLabelScorer::ForcedSequenceLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::ALL),
          infinityScore_(paramInfinityScore(config)),
          targetSequences_() {}

void ForcedSequenceLabelScorer::setTargetSequence(std::vector<LabelIndex> const& targetSequence, SegmentId segment) {
    targetSequences_[segment] = std::make_shared<std::vector<LabelIndex> const>(targetSequence);
}

bool ForcedSequenceLabelScorer::supportsMultipleSegments() const {
    return true;
}

void ForcedSequenceLabelScorer::removeSegment(SegmentId segment) {
    targetSequences_.erase(segment);
}

ScoringContextRef ForcedSequenceLabelScorer::getInitialSegmentScoringContext(SegmentId segment) {
    auto context     = Core::ref(new StepScoringContext());
    context->segment = segment;
    return context;
}

ScoringContextRef ForcedSequenceLabelScorer::getInitialScoringContext() {
    return getInitialSegmentScoringContext(defaultSegmentId);
}

ScoringContextRef ForcedSequenceLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) {
    if (not emitsLabel(transitionType)) {
        return scoringContext;
    }
    StepScoringContextRef stepScoringContext(dynamic_cast<StepScoringContext const*>(scoringContext.get()));
    auto                  context = Core::ref(new StepScoringContext(stepScoringContext->currentStep + 1));
    context->segment              = stepScoringContext->segment;
    return context;
}

std::optional<ScoreAccessorRef> ForcedSequenceLabelScorer::computeScoreAccessor(ScoringContextRef scoringContext) {
    StepScoringContextRef stepScoringContext(dynamic_cast<StepScoringContext const*>(scoringContext.get()));
    auto                  targetSequence = targetSequences_.find(stepScoringContext->segment);
    if (targetSequence == targetSequences_.end()) {
        criticalError() << "No target sequence set for segment " << stepScoringContext->segment << ", call `setTargetSequence` before scoring";
    }
    return Core::ref(new ForcedSequenceScoreAccessor(targetSequence->second, stepScoringContext->currentStep, infinityScore_));
}

}  // namespace Nn
