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

#include "LabelScorer.hh"
#include "ScoringContext.hh"

namespace Nn {

const Core::ParameterFloat TransitionLabelScorer::paramLabelToLabelScore(
        "label-to-label-score",
        "Score for label-to-label transitions",
        0.0);

const Core::ParameterFloat TransitionLabelScorer::paramLabelLoopScore(
        "label-loop-score",
        "Score for label-loop transitions",
        0.0);

const Core::ParameterFloat TransitionLabelScorer::paramLabelToBlankScore(
        "label-to-blank-score",
        "Score for label-to-blank transitions",
        0.0);

const Core::ParameterFloat TransitionLabelScorer::paramBlankToLabelScore(
        "blank-to-label-score",
        "Score for blank-to-label transitions",
        0.0);

const Core::ParameterFloat TransitionLabelScorer::paramBlankLoopScore(
        "blank-loop-score",
        "Score for blank-loop transitions",
        0.0);

const Core::ParameterFloat TransitionLabelScorer::paramInitialLabelScore(
        "initial-label-score",
        "Score for initial-label transitions",
        0.0);

const Core::ParameterFloat TransitionLabelScorer::paramInitialBlankScore(
        "initial-blank-score",
        "Score for initial-blank transitions",
        0.0);

TransitionLabelScorer::TransitionLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          labelToLabelScore_(paramLabelToLabelScore(config)),
          labelLoopScore_(paramLabelLoopScore(config)),
          labelToBlankScore_(paramLabelToBlankScore(config)),
          blankToLabelScore_(paramBlankToLabelScore(config)),
          blankLoopScore_(paramBlankLoopScore(config)),
          initialLabelScore_(paramInitialLabelScore(config)),
          initialBlankScore_(paramInitialBlankScore(config)),
          baseLabelScorer_(Nn::Module::instance().labelScorerFactory().createLabelScorer(select("base-scorer"))) {}

void TransitionLabelScorer::reset() {
    baseLabelScorer_->reset();
}

void TransitionLabelScorer::signalNoMoreFeatures() {
    baseLabelScorer_->signalNoMoreFeatures();
}

ScoringContextRef TransitionLabelScorer::getInitialScoringContext() {
    return baseLabelScorer_->getInitialScoringContext();
}

ScoringContextRef TransitionLabelScorer::extendedScoringContext(LabelScorer::Request const& request) {
    return baseLabelScorer_->extendedScoringContext(request);
}

void TransitionLabelScorer::cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) {
    baseLabelScorer_->cleanupCaches(activeContexts);
}

void TransitionLabelScorer::addInput(DataView const& input) {
    baseLabelScorer_->addInput(input);
}

void TransitionLabelScorer::addInputs(DataView const& input, size_t nTimesteps) {
    baseLabelScorer_->addInputs(input, nTimesteps);
}

std::optional<LabelScorer::ScoreWithTime> TransitionLabelScorer::computeScoreWithTime(LabelScorer::Request const& request) {
    auto result = baseLabelScorer_->computeScoreWithTime(request);
    if (result) {
        result->score += getTransitionScore(request.transitionType);
    }
    return result;
}

std::optional<LabelScorer::ScoresWithTimes> TransitionLabelScorer::computeScoresWithTimes(std::vector<LabelScorer::Request> const& requests) {
    auto results = baseLabelScorer_->computeScoresWithTimes(requests);
    if (results) {
        for (size_t i = 0ul; i < requests.size(); ++i) {
            results->scores[i] += getTransitionScore(requests[i].transitionType);
        }
    }
    return results;
}

LabelScorer::Score TransitionLabelScorer::getTransitionScore(LabelScorer::TransitionType transitionType) const {
    switch (transitionType) {
        case TransitionType::LABEL_TO_LABEL:
            return labelToLabelScore_;
        case TransitionType::LABEL_LOOP:
            return labelLoopScore_;
        case TransitionType::LABEL_TO_BLANK:
            return labelToBlankScore_;
        case TransitionType::BLANK_TO_LABEL:
            return blankToLabelScore_;
        case TransitionType::BLANK_LOOP:
            return blankLoopScore_;
        case TransitionType::INITIAL_LABEL:
            return initialLabelScore_;
        case TransitionType::INITIAL_BLANK:
            return initialBlankScore_;
        default:
            error() << "Unknown transition type " << transitionType;
    }
    return 0;
}

}  // namespace Nn
