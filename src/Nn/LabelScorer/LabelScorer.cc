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

namespace Nn {

/*
 * =============================
 * === LabelScorer =============
 * =============================
 */

const Core::Choice LabelScorer::choiceTransitionPreset(
        "default", TransitionPresetType::DEFAULT,
        "none", TransitionPresetType::NONE,
        "ctc", TransitionPresetType::CTC,
        "ctc-prefix", TransitionPresetType::CTC_PREFIX,
        "transducer", TransitionPresetType::TRANSDUCER,
        "lm", TransitionPresetType::LM,
        Core::Choice::endMark());

const Core::ParameterChoice LabelScorer::paramTransitionPreset(
        "transition-preset",
        &LabelScorer::choiceTransitionPreset,
        "Preset for which transition types should be enabled for the label scorer. Disabled transition types get assigned score 0 and do not affect the ScoringContext.",
        TransitionPresetType::DEFAULT);

const Core::ParameterStringVector LabelScorer::paramExtraTransitionTypes(
        "extra-transition-types",
        "Transition types that should be enabled in addition to the ones given by the preset.",
        ",");

LabelScorer::LabelScorer(const Core::Configuration& config, TransitionPresetType defaultPreset)
        : Core::Component(config),
          enabledTransitionTypes_() {
    enableTransitionTypes(config, defaultPreset);
}

void LabelScorer::addInputs(DataView const& input, size_t nTimesteps) {
    auto featureSize = input.size() / nTimesteps;
    for (size_t t = 0ul; t < nTimesteps; ++t) {
        addInput({input, featureSize, t * featureSize});
    }
}

ScoringContextRef LabelScorer::extendedScoringContext(Request const& request) {
    if (enabledTransitionTypes_.contains(request.transitionType)) {
        return extendedScoringContextInternal(request);
    }
    return request.context;
}

std::optional<LabelScorer::ScoreWithTime> LabelScorer::computeScoreWithTime(Request const& request, std::optional<size_t> scorerIndex) {
    verify(not scorerIndex.has_value() or scorerIndex.value() < numSubScorers());

    if (enabledTransitionTypes_.contains(request.transitionType)) {
        return computeScoreWithTimeInternal(request, scorerIndex);
    }
    return ScoreWithTime{0.0, 0};
}

std::optional<LabelScorer::ScoresWithTimes> LabelScorer::computeScoresWithTimes(std::vector<LabelScorer::Request> const& requests, std::optional<size_t> scorerIndex) {
    verify(not scorerIndex.has_value() or scorerIndex.value() < numSubScorers());

    // First, collect all requests for which the transition type is not ignored
    std::vector<Request> nonIgnoredRequests;
    nonIgnoredRequests.reserve(requests.size());

    std::vector<size_t> nonIgnoredRequestIndices;
    nonIgnoredRequestIndices.reserve(requests.size());

    for (size_t requestIndex = 0ul; requestIndex < requests.size(); ++requestIndex) {
        auto const& request = requests[requestIndex];
        if (enabledTransitionTypes_.contains(request.transitionType)) {
            nonIgnoredRequests.push_back(request);
            nonIgnoredRequestIndices.push_back(requestIndex);
        }
    }

    // Compute scores for non-ignored requests
    auto nonIgnoredResults = computeScoresWithTimesInternal(nonIgnoredRequests, scorerIndex);
    if (not nonIgnoredResults) {
        return {};
    }

    // Interleave actual results with 0 scores for requests with ignored transition types
    ScoresWithTimes result{
            .scores = std::vector<Score>(requests.size(), 0.0),
            .timeframes{requests.size(), 0}};
    for (size_t i = 0ul; i < nonIgnoredRequestIndices.size(); ++i) {
        auto requestIndex           = nonIgnoredRequestIndices[i];
        result.scores[requestIndex] = nonIgnoredResults->scores[i];
        result.timeframes.set(requestIndex, nonIgnoredResults->timeframes[i]);
    }

    return result;
}

std::optional<LabelScorer::ScoresWithTimes> LabelScorer::computeScoresWithTimesInternal(std::vector<LabelScorer::Request> const& requests, std::optional<size_t> scorerIndex) {
    if (requests.empty()) {
        return ScoresWithTimes{};
    }

    // By default, just loop over the non-batched `computeScoreWithTime` and collect the results
    ScoresWithTimes result;

    result.scores.reserve(requests.size());
    result.timeframes.reserve(requests.size());
    for (auto& request : requests) {
        auto singleResult = computeScoreWithTime(request, scorerIndex);
        if (not singleResult.has_value()) {
            return {};
        }
        result.scores.push_back(singleResult->score);
        result.timeframes.push_back(singleResult->timeframe);
    }

    return result;
}

size_t LabelScorer::numSubScorers() const {
    return 1ul;
}

void LabelScorer::enableTransitionTypes(Core::Configuration const& config, TransitionPresetType defaultPreset) {
    auto preset = paramTransitionPreset(config);
    if (preset == TransitionPresetType::DEFAULT) {
        preset = defaultPreset;
    }
    verify(preset != TransitionPresetType::DEFAULT);

    switch (preset) {
        case TransitionPresetType::NONE:
            break;
        case TransitionPresetType::ALL:
            for (auto const& [_, transitionType] : transitionTypeArray_) {
                enabledTransitionTypes_.insert(transitionType);
            }
            break;
        case TransitionPresetType::CTC:
            enabledTransitionTypes_ = {
                    LABEL_TO_LABEL,
                    LABEL_LOOP,
                    LABEL_TO_BLANK,
                    BLANK_TO_LABEL,
                    BLANK_LOOP,
                    INITIAL_LABEL,
                    INITIAL_BLANK,
            };
            break;
        case TransitionPresetType::CTC_PREFIX:
            enabledTransitionTypes_ = {
                    LABEL_TO_LABEL,
                    BLANK_TO_LABEL,
                    INITIAL_LABEL,
            };
            break;
        case TransitionPresetType::TRANSDUCER:
            enabledTransitionTypes_ = {
                    LABEL_TO_LABEL,
                    LABEL_TO_BLANK,
                    BLANK_TO_LABEL,
                    BLANK_LOOP,
                    INITIAL_LABEL,
                    INITIAL_BLANK,
            };
            break;
        case TransitionPresetType::LM:
            enabledTransitionTypes_ = {
                    LABEL_TO_LABEL,
                    BLANK_TO_LABEL,
                    INITIAL_LABEL,
                    SENTENCE_END,
            };
            break;
    }

    auto extraTransitionTypeStrings = paramExtraTransitionTypes(config);
    for (auto const& transitionTypeString : extraTransitionTypeStrings) {
        auto it = std::find_if(transitionTypeArray_.begin(),
                               transitionTypeArray_.end(),
                               [&](auto const& entry) { return entry.first == transitionTypeString; });
        if (it != transitionTypeArray_.end()) {
            enabledTransitionTypes_.insert(it->second);
        }
        else {
            error() << "Extra transition type name '" << transitionTypeString << "' is not a valid identifier";
        }
    }

    if (enabledTransitionTypes_.empty()) {
        error() << "Label scorer has no enabled transition types. Activate a preset and/or add extra transition types that should be considered.";
    }
}

}  // namespace Nn
