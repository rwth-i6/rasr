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

    std::optional<DenseScoreSpan> getDenseScores() const override {
        auto denseScores = base_->getDenseScores();
        if (scale_ == 1.0 or not denseScores) {
            return denseScores;
        }

        for (auto& term : denseScores->terms) {
            term.scale *= scale_;
        }
        return denseScores;
    }

    std::optional<SparseScoreSpan> getSparseScores() const override {
        auto sparseScores = base_->getSparseScores();
        if (sparseScores) {
            sparseScores->scale *= scale_;
        }
        return sparseScores;
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

Core::Ref<LabelScorer> ScaledLabelScorer::labelScorer() const {
    return scorer_;
}

Score ScaledLabelScorer::scale() const {
    return scale_;
}

void ScaledLabelScorer::setScale(Score scale) {
    scale_ = scale;
}

void ScaledLabelScorer::logStatistics() const {
    scorer_->logStatistics();
}

Core::Status ScaledLabelScorer::signalNoMoreFeatures() {
    return scorer_->signalNoMoreFeatures();
}

Core::Status ScaledLabelScorer::getInitialScoringContext(ScoringContextRef& initialScoringContext) {
    return scorer_->getInitialScoringContext(initialScoringContext);
}

Core::Status ScaledLabelScorer::extendedScoringContext(ScoringContextRef context, LabelIndex nextToken, TransitionType transitionType, ScoringContextRef& extendedContext) {
    return scorer_->extendedScoringContext(context, nextToken, transitionType, extendedContext);
}

void ScaledLabelScorer::cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) {
    scorer_->cleanupCaches(activeContexts);
}

Core::Status ScaledLabelScorer::addInput(DataView const& input) {
    return scorer_->addInput(input);
}

Core::Status ScaledLabelScorer::addInputs(DataView const& input, size_t nTimesteps) {
    return scorer_->addInputs(input, nTimesteps);
}

void ScaledLabelScorer::reset() {
    scorer_->reset();
}

Core::Status ScaledLabelScorer::getScoreAccessor(ScoringContextRef scoringContext, std::optional<ScoreAccessorRef>& scoreAccessor) {
    auto status = scorer_->getScoreAccessor(scoringContext, scoreAccessor);
    if (status.ok() and scoreAccessor) {
        scoreAccessor = Core::ref(new ScaledScoreAccessor(*scoreAccessor, scale_));
    }
    return status;
}

Core::Status ScaledLabelScorer::getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts, std::vector<std::optional<ScoreAccessorRef>>& scoreAccessors) {
    auto status = scorer_->getScoreAccessors(scoringContexts, scoreAccessors);
    if (not status.ok()) {
        return status;
    }
    for (auto& scoreAccessor : scoreAccessors) {
        if (scoreAccessor) {
            scoreAccessor = Core::ref(new ScaledScoreAccessor(*scoreAccessor, scale_));
        }
    }
    return status;
}

}  // namespace Nn
