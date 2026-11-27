/** Copyright 2024 RWTH Aachen University. All rights reserved.
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

#include "LabelScorer.hh"

#include <Core/XmlStream.hh>

namespace Nn {

/*
 * =============================
 * === LabelScorer =============
 * =============================
 */
LabelScorer::LabelScorer(Core::Configuration const& config, TransitionPresetType defaultPreset)
        : Core::Component(config),
          enabledTransitions_(config, defaultPreset) {
}

Core::Status LabelScorer::addInputs(DataView const& input, size_t nTimesteps) {
    auto featureSize = input.size() / nTimesteps;
    for (size_t t = 0ul; t < nTimesteps; ++t) {
        auto status = addInput({input, featureSize, t * featureSize});
        if (not status.ok()) {
            return status;
        }
    }
    return Core::Status();
}

TransitionSet LabelScorer::enabledTransitions() const {
    return enabledTransitions_;
}

/*
 * =============================
 * === StatTrackingLabelScorer =
 * =============================
 */
StatTrackingLabelScorer::StatTrackingLabelScorer(Core::Configuration const& config, TransitionPresetType defaultPreset)
        : Core::Component(config),
          LabelScorer(config, defaultPreset),
          statisticsChannel_(config, "statistics") {
}

void StatTrackingLabelScorer::reset() {
    scoringTime_.reset();
    numScoreAccessorsRequested_ = 0ul;
    numScoreAccessorsComputed_  = 0ul;
    resetInternal();
}

void StatTrackingLabelScorer::logStatistics() const {
    if (not statisticsChannel_.isOpen()) {
        return;
    }

    statisticsChannel_ << Core::XmlOpen("label-scorer-statistics") + Core::XmlAttribute("component", fullName());
    statisticsChannel_ << Core::XmlOpen("scoring-time") + Core::XmlAttribute("unit", "milliseconds") + Core::XmlAttribute("total", scoringTime_.elapsedMilliseconds());
    logScoringBreakdown();
    statisticsChannel_ << Core::XmlClose("scoring-time");
    logAdditionalStatistics();
    statisticsChannel_ << Core::XmlFull("num-score-accessors-requested", numScoreAccessorsRequested_);
    statisticsChannel_ << Core::XmlFull("num-score-accessors-computed", numScoreAccessorsComputed_);
    statisticsChannel_ << Core::XmlClose("label-scorer-statistics");
}

Core::Status StatTrackingLabelScorer::getScoreAccessor(ScoringContextRef scoringContext, std::optional<ScoreAccessorRef>& scoreAccessor) {
    ++numScoreAccessorsRequested_;
    ++numScoreAccessorsComputed_;
    Core::StopWatch::Scope timer(scoringTime_);
    return computeScoreAccessor(scoringContext, scoreAccessor);
}

Core::Status StatTrackingLabelScorer::getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts, std::vector<std::optional<ScoreAccessorRef>>& scoreAccessors) {
    numScoreAccessorsRequested_ += scoringContexts.size();
    numScoreAccessorsComputed_ += scoringContexts.size();
    Core::StopWatch::Scope timer(scoringTime_);
    return computeScoreAccessors(scoringContexts, scoreAccessors);
}

Core::Status StatTrackingLabelScorer::computeScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts, std::vector<std::optional<ScoreAccessorRef>>& scoreAccessors) {
    scoreAccessors.clear();
    scoreAccessors.reserve(scoringContexts.size());
    for (auto const& scoringContext : scoringContexts) {
        auto status = computeScoreAccessor(scoringContext, scoreAccessors.emplace_back());
        if (not status.ok()) {
            return status;
        }
    }
    return Core::Status();
}

void StatTrackingLabelScorer::countUncomputedAccessors(size_t num) const {
    verify(num <= numScoreAccessorsComputed_);
    numScoreAccessorsComputed_ -= num;
}

}  // namespace Nn
