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

#include <Core/XmlStream.hh>

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

const Core::ParameterInt NoContextOnnxLabelScorer::paramMaxBatchSize(
        "max-batch-size",
        "Max number of contexts whose input features are stacked and forwarded through the ONNX model at once."
        " Values > 1 require a dynamic batch dimension of the input feature.",
        1,
        1);

NoContextOnnxLabelScorer::NoContextOnnxLabelScorer(Core::Configuration const& config, ModelCache& modelCache)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::CTC),
          maxBatchSize_(paramMaxBatchSize(config)),
          scoreCache_() {
    Core::Configuration modelConfig(config, "onnx-model");
    auto                key = modelConfig.getSelection();
    onnxModel_              = modelCache.getOrCreate<Onnx::Model>(key, modelConfig, ioSpec);
    inputFeatureName_       = onnxModel_->mapping.getOnnxName("input-feature");
    scoresName_             = onnxModel_->mapping.getOnnxName("scores");

    if (maxBatchSize_ > 1ul) {
        auto inputShape = onnxModel_->session.getInputShape(inputFeatureName_);
        if (not inputShape.empty() and inputShape.front() == 1l) {
            warning() << "ONNX model has a fixed batch size of 1 for the input feature; using max-batch-size 1";
            maxBatchSize_ = 1ul;
        }
    }
}

ScoringContextRef NoContextOnnxLabelScorer::getInitialSegmentScoringContext(SegmentId segment) {
    auto context     = Core::ref(new StepScoringContext());
    context->segment = segment;
    return context;
}

void NoContextOnnxLabelScorer::removeSegment(SegmentId segment) {
    Precursor::removeSegment(segment);
    for (auto it = scoreCache_.begin(); it != scoreCache_.end();) {
        if (it->first->segment == segment) {
            it = scoreCache_.erase(it);
        }
        else {
            ++it;
        }
    }
}

ScoringContextRef NoContextOnnxLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) {
    StepScoringContextRef stepScoringContext(dynamic_cast<StepScoringContext const*>(scoringContext.get()));
    auto                  context = Core::ref(new StepScoringContext(stepScoringContext->currentStep + 1));
    context->segment              = stepScoringContext->segment;
    return context;
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

void NoContextOnnxLabelScorer::resetInternal() {
    Precursor::resetInternal();
    scoreCache_.clear();
}

void NoContextOnnxLabelScorer::resetStatisticsInternal() {
    Precursor::resetStatisticsInternal();
    onnxSessionTime_.reset();
}

std::vector<std::optional<ScoreAccessorRef>> NoContextOnnxLabelScorer::computeScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) {
    if (scoringContexts.empty()) {
        return {};
    }

    std::vector<std::optional<ScoreAccessorRef>> scoreAccessors(scoringContexts.size(), std::nullopt);

    size_t numComputed = 0ul;

    if (maxBatchSize_ > 1ul) {
        // Collect all unique contexts that still need to be forwarded and forward them in batches
        std::unordered_set<StepScoringContextRef, ScoringContextHash, ScoringContextEq> uniqueUncachedContexts;
        std::vector<StepScoringContextRef>                                              contextBatch;
        for (auto const& scoringContext : scoringContexts) {
            StepScoringContextRef stepScoringContext(dynamic_cast<StepScoringContext const*>(scoringContext.get()));
            if (getInput(stepScoringContext->segment, stepScoringContext->currentStep) and scoreCache_.find(stepScoringContext) == scoreCache_.end() and uniqueUncachedContexts.emplace(stepScoringContext).second) {
                contextBatch.push_back(stepScoringContext);
                ++numComputed;
                if (contextBatch.size() == maxBatchSize_) {
                    forwardBatch(contextBatch);
                    contextBatch.clear();
                }
            }
        }
        forwardBatch(contextBatch);
    }

    for (size_t contextIndex = 0ul; contextIndex < scoringContexts.size(); ++contextIndex) {
        StepScoringContextRef stepScoringContext(dynamic_cast<StepScoringContext const*>(scoringContexts[contextIndex].get()));
        if (not getInput(stepScoringContext->segment, stepScoringContext->currentStep)) {
            // If input is not available, this context can't be forwarded
            continue;
        }
        if (scoreCache_.find(stepScoringContext) == scoreCache_.end()) {
            forwardContext(stepScoringContext);
            ++numComputed;
        }

        scoreAccessors[contextIndex] = Core::ref(new VectorScoreAccessor(scoreCache_.at(stepScoringContext), stepScoringContext->currentStep));
    }

    countUncomputedAccessors(scoringContexts.size() - numComputed);

    return scoreAccessors;
}

std::optional<ScoreAccessorRef> NoContextOnnxLabelScorer::computeScoreAccessor(ScoringContextRef scoringContext) {
    return computeScoreAccessors({scoringContext})[0];
}

size_t NoContextOnnxLabelScorer::getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const {
    auto minTimeIndex = Core::Type<Speech::TimeframeIndex>::max;
    for (auto const& context : activeContexts.internalData()) {
        StepScoringContextRef stepHistory(dynamic_cast<StepScoringContext const*>(context.get()));
        minTimeIndex = std::min(minTimeIndex, stepHistory->currentStep);
    }

    return minTimeIndex;
}

void NoContextOnnxLabelScorer::logScoringBreakdown() const {
    statisticsChannel_ << Core::XmlFull("onnx-session-time", onnxSessionTime_.elapsedMilliseconds());
}

void NoContextOnnxLabelScorer::forwardContext(StepScoringContextRef const& scoringContext) {
    /*
     * Create session inputs
     */
    auto                 inputDataView = getInput(scoringContext->segment, scoringContext->currentStep);
    f32 const*           inputData     = inputDataView->data();
    std::vector<int64_t> inputShape    = {1ul, static_cast<int64_t>(inputDataView->size())};

    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;
    sessionInputs.emplace_back(inputFeatureName_, Onnx::Value::create(inputData, inputShape));

    /*
     * Run session
     */
    std::vector<Onnx::Value> sessionOutputs;
    onnxSessionTime_.start();
    onnxModel_->session.run(std::move(sessionInputs), {scoresName_}, sessionOutputs);
    onnxSessionTime_.stop();

    /*
     * Put resulting scores into cache map
     */
    auto scoreVec = std::make_shared<std::vector<Score>>();
    sessionOutputs.front().get(0, *scoreVec);
    scoreCache_.emplace(scoringContext, scoreVec);
}

void NoContextOnnxLabelScorer::forwardBatch(std::vector<StepScoringContextRef> const& scoringContextBatch) {
    if (scoringContextBatch.empty()) {
        return;
    }

    /*
     * Create session inputs by stacking the input features of all contexts
     */
    size_t featureSize = getInput(scoringContextBatch.front()->segment, scoringContextBatch.front()->currentStep)->size();
    auto   inputValue  = Onnx::Value::createEmpty<f32>({static_cast<int64_t>(scoringContextBatch.size()), static_cast<int64_t>(featureSize)});
    for (size_t b = 0ul; b < scoringContextBatch.size(); ++b) {
        auto inputDataView = getInput(scoringContextBatch[b]->segment, scoringContextBatch[b]->currentStep);
        verify(inputDataView and inputDataView->size() == featureSize);
        std::copy(inputDataView->data(), inputDataView->data() + featureSize, inputValue.data<f32>(b));
    }

    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;
    sessionInputs.emplace_back(inputFeatureName_, std::move(inputValue));

    /*
     * Run session
     */
    std::vector<Onnx::Value> sessionOutputs;
    onnxSessionTime_.start();
    onnxModel_->session.run(std::move(sessionInputs), {scoresName_}, sessionOutputs);
    onnxSessionTime_.stop();

    /*
     * Put resulting scores into cache map
     */
    for (size_t b = 0ul; b < scoringContextBatch.size(); ++b) {
        auto scoreVec = std::make_shared<std::vector<Score>>();
        sessionOutputs.front().get(b, *scoreVec);
        scoreCache_.emplace(scoringContextBatch[b], scoreVec);
    }
}

}  // namespace Nn
