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

LabelScorer::~LabelScorer() {
    if (scoringTime_.elapsedMilliseconds() > 0 or numScoreAccessorsRequested_ > 0) {
        clog() << Core::XmlOpen("label-scorer-timing") + Core::XmlAttribute("unit", "milliseconds") + Core::XmlAttribute("component", fullName());
        clog() << Core::XmlOpen("scoring-time") << scoringTime_.elapsedMilliseconds() << Core::XmlClose("scoring-time");
        clog() << Core::XmlFull("num-score-accessors-requested", numScoreAccessorsRequested_);
        clog() << Core::XmlFull("num-score-accessors-computed", numScoreAccessorsComputed_);
        clog() << Core::XmlClose("label-scorer-timing");
    }
}

void LabelScorer::addInputs(DataView const& input, size_t nTimesteps) {
    auto featureSize = input.size() / nTimesteps;
    for (size_t t = 0ul; t < nTimesteps; ++t) {
        addInput({input, featureSize, t * featureSize});
    }
}

std::vector<std::optional<ScoreAccessorRef>> LabelScorer::getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) {
    std::vector<std::optional<ScoreAccessorRef>> result;
    result.reserve(scoringContexts.size());
    for (auto const& scoringContext : scoringContexts) {
        result.push_back(getScoreAccessor(scoringContext));
    }
    return result;
}

TransitionSet LabelScorer::enabledTransitions() const {
    return enabledTransitions_;
}

}  // namespace Nn
