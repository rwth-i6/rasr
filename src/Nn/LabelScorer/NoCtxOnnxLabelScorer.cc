/** Copyright 2020 RWTH Aachen University. All rights reserved.
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

#include "NoCtxOnnxLabelScorer.hh"
#include <Core/Assertions.hh>
#include <Core/ReferenceCounting.hh>
#include <Flow/Timestamp.hh>
#include <Math/FastMatrix.hh>
#include <Mm/Module.hh>
#include <Speech/Types.hh>
#include <cstddef>
#include <utility>
#include "LabelScorer.hh"
#include "ScoringContext.hh"

namespace Nn {

/*
 * =============================
 * === NoCtxOnnxDecoder ========
 * =============================
 */

const Core::ParameterBool NoCtxOnnxLabelScorer::paramVerticalLabelTransition(
        "vertical-label-transition",
        "Whether (non-blank) label transitions should be vertical, i.e. not increase the time step.",
        false);

const Core::ParameterInt NoCtxOnnxLabelScorer::paramMaxCachedScores(
        "max-cached-scores",
        "Maximum size of cache that maps histories to scores. This prevents memory overflow in case of very long audio segments.",
        1000);

static const std::vector<Onnx::IOSpecification> ioSpec = {
        Onnx::IOSpecification{
                "encoder-state",
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

NoCtxOnnxLabelScorer::NoCtxOnnxLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          verticalLabelTransition_(paramVerticalLabelTransition(config)),
          onnxModel_(select("onnx-model"), ioSpec),
          encoderStateName_(onnxModel_.mapping.getOnnxName("encoder-state")),
          scoresName_(onnxModel_.mapping.getOnnxName("scores")),
          scoreCache_(paramMaxCachedScores(config)) {
}

void NoCtxOnnxLabelScorer::reset() {
    Precursor::reset();
    scoreCache_.clear();
}

ScoringContextRef NoCtxOnnxLabelScorer::getInitialScoringContext() {
    return Core::ref(new StepScoringContext());
}

ScoringContextRef NoCtxOnnxLabelScorer::extendedScoringContext(LabelScorer::Request const& request) {
    StepScoringContextRef context(dynamic_cast<const StepScoringContext*>(request.context.get()));

    bool incrementTime = false;
    switch (request.transitionType) {
        case TransitionType::BLANK_LOOP:
        case TransitionType::LABEL_TO_BLANK:
            incrementTime = true;
            break;
        case TransitionType::LABEL_LOOP:
        case TransitionType::BLANK_TO_LABEL:
        case TransitionType::LABEL_TO_LABEL:
            incrementTime = not verticalLabelTransition_;
            break;
        default:
            error() << "Unknown transition type " << request.transitionType;
    }

    // If context is not going to be modified, return the original one to avoid copying
    if (not incrementTime) {
        return request.context;
    }

    Core::Ref<StepScoringContext> newContext(new StepScoringContext(context->currentStep + 1));
    return newContext;
}

std::optional<LabelScorer::ScoresWithTimes> NoCtxOnnxLabelScorer::computeScoresWithTimes(std::vector<LabelScorer::Request> const& requests) {
    ScoresWithTimes result;
    result.scores.reserve(requests.size());

    /*
     * Collect all requests that are based on the same timestep (-> same encoder state) and
     * group them together
     */
    std::unordered_set<StepScoringContextRef, ScoringContextHash, ScoringContextEq> requestedContexts;

    for (size_t b = 0ul; b < requests.size(); ++b) {
        StepScoringContextRef context(dynamic_cast<const StepScoringContext*>(requests[b].context.get()));
        auto                  step = context->currentStep;
        if (step >= inputBuffer_.size()) {
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

        auto scores = scoreCache_.get(context);
        if (request.nextToken < scores->get().size()) {
            result.scores.push_back(scores->get()[request.nextToken]);
        }
        else {
            result.scores.push_back(0);
        }
    }

    return result;
}

std::optional<LabelScorer::ScoreWithTime> NoCtxOnnxLabelScorer::computeScoreWithTime(LabelScorer::Request const& request) {
    auto result = computeScoresWithTimes({request});
    if (not result.has_value()) {
        return {};
    }
    return ScoreWithTime{result->scores.front(), result->timeframes.front()};
}

void NoCtxOnnxLabelScorer::forwardContext(StepScoringContextRef const& context) {
    /*
     * Create session inputs
     */

    // All requests in this iteration share the same encoder state which is set up here
    f32 const*           encoderState = inputBuffer_[context->currentStep].get();
    std::vector<int64_t> encoderShape = {1, static_cast<int64_t>(featureSize_)};

    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;
    sessionInputs.emplace_back(encoderStateName_, Onnx::Value::create(encoderState, encoderShape));

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
    scoreCache_.put(context, std::move(scoreVec));
}
}  // namespace Nn
