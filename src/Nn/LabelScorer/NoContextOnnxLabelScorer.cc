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
#include "ScoreAccessor.hh"

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
        StepScoringContextRef stepHistory(dynamic_cast<StepScoringContext const*>(context.get()));
        minTimeIndex = std::min(minTimeIndex, stepHistory->currentStep);
    }

    return minTimeIndex;
}

ScoringContextRef NoContextOnnxLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) {
    StepScoringContextRef stepScoringContext(dynamic_cast<StepScoringContext const*>(scoringContext.get()));
    return Core::ref(new StepScoringContext(stepScoringContext->currentStep + 1));
}

std::vector<std::optional<ScoreAccessorRef>> NoContextOnnxLabelScorer::getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) {
    if (scoringContexts.empty()) {
        return {};
    }

    std::vector<std::optional<ScoreAccessorRef>> scoreAccessors(scoringContexts.size(), std::nullopt);

    for (size_t contextIndex = 0ul; contextIndex < scoringContexts.size(); ++contextIndex) {
        StepScoringContextRef stepScoringContext(dynamic_cast<StepScoringContext const*>(scoringContexts[contextIndex].get()));
        if (not getInput(stepScoringContext->currentStep)) {
            // If input is not available, this context can't be forwarded
            continue;
        }
        forwardContext(stepScoringContext);

        scoreAccessors[contextIndex] = Core::ref(new VectorScoreAccessor(scoreCache_.at(stepScoringContext), stepScoringContext->currentStep));
    }

    return scoreAccessors;
}

std::optional<ScoreAccessorRef> NoContextOnnxLabelScorer::getScoreAccessor(ScoringContextRef scoringContext) {
    return getScoreAccessors({scoringContext})[0];
}

void NoContextOnnxLabelScorer::forwardContext(StepScoringContextRef const& scoringContext) {
    /*
     * Create session inputs
     */
    auto                 inputDataView = getInput(scoringContext->currentStep);
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
    auto scoreVec = std::make_shared<std::vector<Score>>();
    sessionOutputs.front().get(0, *scoreVec);
    scoreCache_.emplace(scoringContext, scoreVec);
}
}  // namespace Nn
