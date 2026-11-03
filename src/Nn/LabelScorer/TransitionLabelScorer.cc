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

#include "TransitionLabelScorer.hh"

#include <Nn/Module.hh>

namespace Nn {

/*
 * ==================================
 * == FixedTransitionScoreAccessor ==
 * ==================================
 */

FixedTransitionScoreAccessor::FixedTransitionScoreAccessor()
        : transitionScores_() {
    for (auto const& [stringIdentifier, enumValue] : TransitionTypeArray) {
        setScore(enumValue, 0.0);
    }
}

void FixedTransitionScoreAccessor::setScore(TransitionType transitionType, Score score) {
    transitionScores_[static_cast<size_t>(transitionType)] = score;
}

Score FixedTransitionScoreAccessor::getScore(TransitionType transitionType, LabelIndex labelIndex) const {
    return transitionScores_.at(transitionType);
}

TimeframeIndex FixedTransitionScoreAccessor::getTime() const {
    return 0;
}

/*
 * ===========================
 * == TransitionLabelScorer ==
 * ===========================
 */

TransitionLabelScorer::TransitionLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::ALL),
          transitionScoreAccessor_(new FixedTransitionScoreAccessor()) {
    for (auto const& [stringIdentifier, enumValue] : TransitionTypeArray) {
        auto paramName = std::string(stringIdentifier) + "-score";
        transitionScoreAccessor_->setScore(enumValue, Core::ParameterFloat(paramName.c_str(), "", 0.0)(config));
    }
}

void TransitionLabelScorer::signalNoMoreFeatures() {}

ScoringContextRef TransitionLabelScorer::getInitialScoringContext() {
    return Core::ref(new ScoringContext());
}

ScoringContextRef TransitionLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) {
    auto context     = Core::ref(new ScoringContext());
    context->segment = scoringContext->segment;
    return context;
}

bool TransitionLabelScorer::supportsMultipleSegments() const {
    return true;
}

void TransitionLabelScorer::addSegment(SegmentId segment) {}

void TransitionLabelScorer::signalNoMoreSegmentFeatures(SegmentId segment) {}

void TransitionLabelScorer::addSegmentInput(SegmentId segment, DataView const& input) {}

ScoringContextRef TransitionLabelScorer::getInitialSegmentScoringContext(SegmentId segment) {
    auto context     = Core::ref(new ScoringContext());
    context->segment = segment;
    return context;
}

void TransitionLabelScorer::addInput(DataView const& input) {}

void TransitionLabelScorer::resetInternal() {}

std::optional<ScoreAccessorRef> TransitionLabelScorer::computeScoreAccessor(ScoringContextRef scoringContext) {
    return transitionScoreAccessor_;
}

}  // namespace Nn
