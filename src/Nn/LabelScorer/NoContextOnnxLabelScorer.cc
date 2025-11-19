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

#include "NoContextOnnxLabelScorer.hh"

#include <unordered_set>

namespace Nn {

static const std::vector<Onnx::IOSpecification> ioSpec = {
        Onnx::IOSpecification{
                "input-feature",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -2}, {1, -2}}},
        Onnx::IOSpecification{
                "scores",
                Onnx::IODirection::OUTPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -2}, {1, -2}}}};

NoContextOnnxLabelScorer::NoContextOnnxLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::CTC),
          onnxModel_(select("onnx-model"), ioSpec),
          inputFeatureName_(onnxModel_.mapping.getOnnxName("input-feature")),
          scoresName_(onnxModel_.mapping.getOnnxName("scores")),
          scoreCache_() {
}

void NoContextOnnxLabelScorer::reset() {
    Precursor::reset();
    scoreCache_.clear();
}

ScoringContextRef NoContextOnnxLabelScorer::getInitialScoringContext() {
    return Core::ref(new StepScoringContext());
}

void NoContextOnnxLabelScorer::cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) {
    Precursor::cleanupCaches(activeContexts);

    std::unordered_set<ScoringContextRef, ScoringContextHash, ScoringContextEq> activeContextSet(activeContexts.internalData().begin(), activeContexts.internalData().end());

    for (auto it = scoreCache_.begin(); it != scoreCache_.end();) {
        if (activeContextSet.find(it->first) == activeContextSet.end()) {
            it = scoreCache_.erase(it);
        }
        else {
            ++it;
        }
    }
}

size_t NoContextOnnxLabelScorer::getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const {
    auto minTimeIndex = Core::Type<Speech::TimeframeIndex>::max;
    for (auto const& context : activeContexts.internalData()) {
        StepScoringContextRef stepHistory(dynamic_cast<const StepScoringContext*>(context.get()));
        minTimeIndex = std::min(minTimeIndex, stepHistory->currentStep);
    }

    return minTimeIndex;
}

ScoringContextRef NoContextOnnxLabelScorer::extendedScoringContextInternal(LabelScorer::Request const& request) {
    StepScoringContextRef context(dynamic_cast<const StepScoringContext*>(request.context.get()));
    return Core::ref(new StepScoringContext(context->currentStep + 1));
}

std::optional<LabelScorer::ScoresWithTimes> NoContextOnnxLabelScorer::computeScoresWithTimesInternal(std::vector<LabelScorer::Request> const& requests, std::optional<size_t> scorerIdx) {
    require(not scorerIdx.has_value() or scorerIdx.value() == 0ul);
    if (requests.empty()) {
        return ScoresWithTimes{};
    }

    ScoresWithTimes result;
    result.scores.reserve(requests.size());

    /*
     * Collect all requests that are based on the same timestep (-> same input) and
     * group them together
     */
    std::unordered_set<StepScoringContextRef, ScoringContextHash, ScoringContextEq> requestedContexts;

    for (size_t b = 0ul; b < requests.size(); ++b) {
        StepScoringContextRef context(dynamic_cast<const StepScoringContext*>(requests[b].context.get()));
        auto                  step = context->currentStep;

        auto input = getInput(step);
        if (not input) {
            // Early exit if at least one of the histories is not scorable yet
            return {};
        }
        result.timeframes.push_back(step);

        requestedContexts.emplace(context);
    }

    /*
     * Iterate over distinct contexts
     */
    for (auto const& context : requestedContexts) {
        forwardContext(context);
    }

    /*
     * Assign from cache map to result vector
     */
    for (const auto& request : requests) {
        StepScoringContextRef context(dynamic_cast<const StepScoringContext*>(request.context.get()));
        result.scores.push_back(scoreCache_.at(context)[request.nextToken]);
    }

    return result;
}

std::optional<LabelScorer::ScoreWithTime> NoContextOnnxLabelScorer::computeScoreWithTimeInternal(LabelScorer::Request const& request, std::optional<size_t> scorerIdx) {
    auto result = computeScoresWithTimes({request}, scorerIdx);
    if (not result.has_value()) {
        return {};
    }
    return ScoreWithTime{result->scores.front(), result->timeframes.front()};
}

void NoContextOnnxLabelScorer::forwardContext(StepScoringContextRef const& context) {
    /*
     * Create session inputs
     */
    // All requests in this iteration share the same input which is set up here
    auto                 inputDataView = getInput(context->currentStep);
    f32 const*           inputData     = inputDataView->data();
    std::vector<int64_t> inputShape    = {1ul, static_cast<int64_t>(inputDataView->size())};

    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;
    sessionInputs.emplace_back(inputFeatureName_, Onnx::Value::create(inputData, inputShape));

    /*
     * Run session
     */
    std::vector<Onnx::Value> sessionOutputs;
    onnxModel_.session.run(std::move(sessionInputs), {scoresName_}, sessionOutputs);

    /*
     * Put resulting scores into cache map
     */
    std::vector<f32> scoreVec;
    sessionOutputs.front().get(0, scoreVec);
    scoreCache_.emplace(context, std::move(scoreVec));
}
}  // namespace Nn
