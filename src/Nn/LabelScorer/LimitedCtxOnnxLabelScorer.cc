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

#include "LimitedCtxOnnxLabelScorer.hh"

namespace Nn {

const Core::ParameterInt LimitedCtxOnnxLabelScorer::paramStartLabelIndex(
        "start-label-index",
        "Initial history in the first step is filled with this label index.",
        0);

const Core::ParameterInt LimitedCtxOnnxLabelScorer::paramHistoryLength(
        "history-length",
        "Number of previous labels that are passed as history.",
        1);

const Core::ParameterBool LimitedCtxOnnxLabelScorer::paramBlankUpdatesHistory(
        "blank-updates-history",
        "Whether previously emitted blank labels should be included in the history.",
        false);

const Core::ParameterBool LimitedCtxOnnxLabelScorer::paramLoopUpdatesHistory(
        "loop-updates-history",
        "Whether in the case of loop transitions every repeated emission should be separately included in the history.",
        false);

const Core::ParameterInt LimitedCtxOnnxLabelScorer::paramMaxBatchSize(
        "max-batch-size",
        "Max number of histories that can be fed into the ONNX model at once.",
        Core::Type<int>::max);

const Core::ParameterBool LimitedCtxOnnxLabelScorer::paramVerticalLabelTransition(
        "vertical-label-transition",
        "Whether (non-blank) label transitions should be vertical, i.e. not increase the time step.",
        false);

static const std::vector<Onnx::IOSpecification> ioSpec = {
        Onnx::IOSpecification{
                "input-feature",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -2}, {1, -2}}},
        Onnx::IOSpecification{
                "history",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{-1, -2}}},
        Onnx::IOSpecification{
                "scores",
                Onnx::IODirection::OUTPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -2}}}};

LimitedCtxOnnxLabelScorer::LimitedCtxOnnxLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          startLabelIndex_(paramStartLabelIndex(config)),
          historyLength_(paramHistoryLength(config)),
          blankUpdatesHistory_(paramBlankUpdatesHistory(config)),
          loopUpdatesHistory_(paramLoopUpdatesHistory(config)),
          verticalLabelTransition_(paramVerticalLabelTransition(config)),
          maxBatchSize_(paramMaxBatchSize(config)),
          onnxModel_(select("onnx-model"), ioSpec),
          inputFeatureName_(onnxModel_.mapping.getOnnxName("input-feature")),
          historyName_(onnxModel_.mapping.getOnnxName("history")),
          scoresName_(onnxModel_.mapping.getOnnxName("scores")),
          scoreCache_() {
}

void LimitedCtxOnnxLabelScorer::reset() {
    Precursor::reset();
    scoreCache_.clear();
}

ScoringContextRef LimitedCtxOnnxLabelScorer::getInitialScoringContext() {
    auto hist = Core::ref(new SeqStepScoringContext());
    hist->labelSeq.resize(historyLength_, startLabelIndex_);
    return hist;
}

ScoringContextRef LimitedCtxOnnxLabelScorer::extendedScoringContext(LabelScorer::Request const& request) {
    SeqStepScoringContextRef context(dynamic_cast<const SeqStepScoringContext*>(request.context.get()));

    bool   pushToken     = false;
    size_t timeIncrement = 0ul;
    switch (request.transitionType) {
        case TransitionType::BLANK_LOOP:
            pushToken     = blankUpdatesHistory_ and loopUpdatesHistory_;
            timeIncrement = 1ul;
            break;
        case TransitionType::LABEL_TO_BLANK:
            pushToken     = blankUpdatesHistory_;
            timeIncrement = 1ul;
            break;
        case TransitionType::LABEL_LOOP:
            pushToken     = loopUpdatesHistory_;
            timeIncrement = not verticalLabelTransition_;
            break;
        case TransitionType::BLANK_TO_LABEL:
        case TransitionType::LABEL_TO_LABEL:
            pushToken     = true;
            timeIncrement = not verticalLabelTransition_;
            break;
        default:
            error() << "Unknown transition type " << request.transitionType;
    }

    // If context is not going to be modified, return the original one to avoid copying
    if (not pushToken and timeIncrement == 0ul) {
        return request.context;
    }

    std::vector<LabelIndex> newLabelSeq;
    newLabelSeq.reserve(context->labelSeq.size());
    if (pushToken) {
        newLabelSeq.insert(newLabelSeq.end(), context->labelSeq.begin() + 1, context->labelSeq.end());
        newLabelSeq.push_back(request.nextToken);
    }
    else {
        newLabelSeq.insert(newLabelSeq.end(), context->labelSeq.begin(), context->labelSeq.end());
    }

    return Core::ref(new SeqStepScoringContext(newLabelSeq, context->currentStep + timeIncrement));
}

void LimitedCtxOnnxLabelScorer::cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) {
    Precursor::cleanupCaches(activeContexts);

    std::unordered_set<ScoringContextRef, ScoringContextHash, ScoringContextEq> activeContextSet(activeContexts.begin(), activeContexts.end());

    for (auto it = scoreCache_.begin(); it != scoreCache_.end();) {
        if (activeContextSet.find(it->first) == activeContextSet.end()) {
            it = scoreCache_.erase(it);
        }
        else {
            ++it;
        }
    }
}

std::optional<LabelScorer::ScoresWithTimes> LimitedCtxOnnxLabelScorer::computeScoresWithTimes(std::vector<LabelScorer::Request> const& requests) {
    ScoresWithTimes result;
    result.scores.reserve(requests.size());

    /*
     * Collect all requests that are based on the same timestep (-> same input feature) and
     * group them together
     */
    std::unordered_map<size_t, std::vector<size_t>> requestsWithTimestep;  // Maps timestep to list of all indices of requests with that timestep

    for (size_t b = 0ul; b < requests.size(); ++b) {
        SeqStepScoringContextRef context(dynamic_cast<const SeqStepScoringContext*>(requests[b].context.get()));
        auto                     step = context->currentStep;

        auto input = getInput(step);
        if (not input) {
            // Early exit if at least one of the histories is not scorable yet
            return {};
        }
        result.timeframes.push_back(step);

        // Create new vector if step value isn't present in map yet
        auto [it, inserted] = requestsWithTimestep.emplace(step, std::vector<size_t>());
        it->second.push_back(b);
    }

    /*
     * Iterate over distinct timesteps
     */
    for (const auto& [timestep, requestIndices] : requestsWithTimestep) {
        /*
         * Identify unique histories that still need session runs
         */
        std::unordered_set<SeqStepScoringContextRef, ScoringContextHash, ScoringContextEq> uniqueUncachedContexts;

        for (auto requestIndex : requestIndices) {
            SeqStepScoringContextRef contextPtr(dynamic_cast<const SeqStepScoringContext*>(requests[requestIndex].context.get()));
            if (scoreCache_.find(contextPtr) == scoreCache_.end()) {
                // Group by unique context
                uniqueUncachedContexts.emplace(contextPtr);
            }
        }

        if (uniqueUncachedContexts.empty()) {
            continue;
        }

        std::vector<SeqStepScoringContextRef> contextBatch;
        contextBatch.reserve(std::min(uniqueUncachedContexts.size(), maxBatchSize_));
        for (auto context : uniqueUncachedContexts) {
            contextBatch.push_back(context);
            if (contextBatch.size() == maxBatchSize_) {  // Batch is full -> forward now
                forwardBatch(contextBatch);
                contextBatch.clear();
            }
        }

        forwardBatch(contextBatch);  // Forward remaining histories
    }

    /*
     * Assign from cache map to result vector
     */
    for (const auto& request : requests) {
        SeqStepScoringContextRef context(dynamic_cast<const SeqStepScoringContext*>(request.context.get()));

        auto const& scores = scoreCache_.at(context);
        result.scores.push_back(scores[request.nextToken]);
    }

    return result;
}

std::optional<LabelScorer::ScoreWithTime> LimitedCtxOnnxLabelScorer::computeScoreWithTime(LabelScorer::Request const& request) {
    auto result = computeScoresWithTimes({request});
    if (not result.has_value()) {
        return {};
    }
    return ScoreWithTime{result->scores.front(), result->timeframes.front()};
}

Speech::TimeframeIndex LimitedCtxOnnxLabelScorer::minActiveTimeIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const {
    auto minTimeIndex = Core::Type<Speech::TimeframeIndex>::max;
    for (auto const& context : activeContexts) {
        StepScoringContextRef stepHistory(dynamic_cast<const StepScoringContext*>(context.get()));
        minTimeIndex = std::min(minTimeIndex, stepHistory->currentStep);
    }

    return minTimeIndex;
}

void LimitedCtxOnnxLabelScorer::forwardBatch(std::vector<SeqStepScoringContextRef> const& contextBatch) {
    if (contextBatch.empty()) {
        return;
    }

    /*
     * Create session inputs
     */

    // All requests in this iteration share the same input feature which is set up here
    auto                 inputFeatureDataView = getInput(contextBatch.front()->currentStep);
    f32 const*           inputFeatureData     = inputFeatureDataView->data();
    std::vector<int64_t> inputFeatureShape    = {1ul, static_cast<int64_t>(inputFeatureDataView->size())};

    // Create batched context input
    Math::FastMatrix<s32> historyMat(historyLength_, contextBatch.size());
    for (size_t b = 0ul; b < contextBatch.size(); ++b) {
        auto context = contextBatch[b];
        std::copy(context->labelSeq.begin(), context->labelSeq.end(), &(historyMat.at(0, b)));  // Pointer to first element in column b
    }

    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;
    sessionInputs.emplace_back(inputFeatureName_, Onnx::Value::create(inputFeatureData, inputFeatureShape));
    sessionInputs.emplace_back(historyName_, Onnx::Value::create(historyMat, true));

    /*
     * Run session
     */

    std::vector<Onnx::Value> sessionOutputs;
    onnxModel_.session.run(std::move(sessionInputs), {scoresName_}, sessionOutputs);

    /*
     * Put resulting scores into cache map
     */
    for (size_t b = 0ul; b < contextBatch.size(); ++b) {
        std::vector<f32> scoreVec;
        sessionOutputs.front().get(b, scoreVec);
        scoreCache_.emplace(contextBatch[b], std::move(scoreVec));
    }
}

}  // namespace Nn
