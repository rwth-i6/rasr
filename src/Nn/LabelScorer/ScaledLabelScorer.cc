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

#include "ScaledLabelScorer.hh"

namespace {

using namespace Nn;

/*
 * Score accessor that wraps a sub-accessor and scales all its scores by a given factor
 */
class ScaledScoreAccessor : public ScoreAccessor {
public:
    ScaledScoreAccessor(ScoreAccessorRef base, Score scale)
        : base_(base),
          scale_(scale) {}

    Score getScore(TransitionType transitionType, LabelIndex labelIndex = invalidLabelIndex) const override {
        return base_->getScore(transitionType, labelIndex) * scale_;
    }

    TimeframeIndex getTime() const override {
        return base_->getTime();
    }

private:
    ScoreAccessorRef base_;
    Score            scale_;
};

}  // namespace

namespace Nn {

const Core::ParameterFloat ScaledLabelScorer::paramScale(
        "scale",
        "Scale used to multiply the sub-scorer scores.",
        1.0);

ScaledLabelScorer::ScaledLabelScorer(Core::Configuration const& config, Core::Ref<LabelScorer> const& scorer)
        : Core::Component(config),
          LabelScorer(config),
          scorer_(scorer),
          scale_(paramScale(config)) {
    enabledTransitions_ = scorer->enabledTransitions();
}

void ScaledLabelScorer::reset() {
    scorer_->reset();
}

void ScaledLabelScorer::signalNoMoreFeatures() {
    scorer_->signalNoMoreFeatures();
}

ScoringContextRef ScaledLabelScorer::getInitialScoringContext() {
    return scorer_->getInitialScoringContext();
}

ScoringContextRef ScaledLabelScorer::extendedScoringContext(ScoringContextRef context, LabelIndex nextToken, TransitionType transitionType) {
    return scorer_->extendedScoringContext(context, nextToken, transitionType);
}

void ScaledLabelScorer::cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) {
    scorer_->cleanupCaches(activeContexts);
}

void ScaledLabelScorer::addInput(DataView const& input) {
    scorer_->addInput(input);
}

void ScaledLabelScorer::addInputs(DataView const& input, size_t nTimesteps) {
    scorer_->addInputs(input, nTimesteps);
}

std::optional<ScoreAccessorRef> ScaledLabelScorer::getScoreAccessor(ScoringContextRef scoringContext) {
    auto subAccessor = scorer_->getScoreAccessor(scoringContext);
    if (subAccessor) {
        return Core::ref(new ScaledScoreAccessor(*subAccessor, scale_));
    }
    return {};
}

std::vector<std::optional<ScoreAccessorRef>> ScaledLabelScorer::getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) {
    auto                                         subAccessors = scorer_->getScoreAccessors(scoringContexts);
    std::vector<std::optional<ScoreAccessorRef>> result(subAccessors.size(), std::nullopt);
    for (size_t i = 0ul; i < subAccessors.size(); ++i) {
        if (subAccessors[i]) {
            result[i] = Core::ref(new ScaledScoreAccessor(*subAccessors[i], scale_));
        }
    }
    return result;
}

}  // namespace Nn
