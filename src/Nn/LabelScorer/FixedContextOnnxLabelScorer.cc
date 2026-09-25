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

#include "FixedContextOnnxLabelScorer.hh"
#include "ScoreAccessor.hh"

namespace Nn {

const Core::ParameterInt FixedContextOnnxLabelScorer::paramStartLabelIndex(
        "start-label-index",
        "Initial history in the first step is filled with this label index.",
        0);

const Core::ParameterInt FixedContextOnnxLabelScorer::paramHistoryLength(
        "history-length",
        "Number of previous labels that are passed as history.",
        1);

const Core::ParameterBool FixedContextOnnxLabelScorer::paramBlankUpdatesHistory(
        "blank-updates-history",
        "Whether previously emitted blank labels should be included in the history.",
        false);

const Core::ParameterBool FixedContextOnnxLabelScorer::paramSilenceUpdatesHistory(
        "silence-updates-history",
        "Whether previously emitted silence labels should be included in the history.",
        false);

const Core::ParameterBool FixedContextOnnxLabelScorer::paramLoopUpdatesHistory(
        "loop-updates-history",
        "Whether in the case of loop transitions every repeated emission should be separately included in the history.",
        false);

const Core::ParameterInt FixedContextOnnxLabelScorer::paramMaxBatchSize(
        "max-batch-size",
        "Max number of histories that can be fed into the ONNX model at once.",
        Core::Type<int>::max);

const Core::ParameterBool FixedContextOnnxLabelScorer::paramVerticalLabelTransition(
        "vertical-label-transition",
        "Whether (non-blank) label transitions should be vertical, i.e. not increase the time step.",
        false);

static const std::vector<Onnx::IOSpecification> ioSpec = {
        Onnx::IOSpecification{
                "input-feature",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -2}, {1, -2}}},  // [B, F] or [1, F]
        Onnx::IOSpecification{
                "encoder-states",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{1, -1, -2}, {-1, -1, -2}}},  // [1, T, E] or [B, T, E]
        Onnx::IOSpecification{
                "encoder-states-size",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{1}, {-1}}},  // [1] or [B]
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

FixedContextOnnxLabelScorer::FixedContextOnnxLabelScorer(Core::Configuration const& config, ModelCache& modelCache)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::TRANSDUCER),
          startLabelIndex_(paramStartLabelIndex(config)),
          historyLength_(paramHistoryLength(config)),
          blankUpdatesHistory_(paramBlankUpdatesHistory(config)),
          silenceUpdatesHistory_(paramSilenceUpdatesHistory(config)),
          loopUpdatesHistory_(paramLoopUpdatesHistory(config)),
          verticalLabelTransition_(paramVerticalLabelTransition(config)),
          maxBatchSize_(paramMaxBatchSize(config)),
          encoderStatesValue_(),
          encoderStatesSizeValue_(),
          scoreCache_() {
    Core::Configuration modelConfig(config, "onnx-model");
    auto                key = modelConfig.getSelection();
    onnxModel_              = modelCache.getOrCreate<Onnx::Model>(key, modelConfig, ioSpec);
    inputFeatureName_       = onnxModel_->mapping.getOnnxName("input-feature");
    encoderStatesName_      = onnxModel_->mapping.getOnnxName("encoder-states");
    encoderStatesSizeName_  = onnxModel_->mapping.getOnnxName("encoder-states-size");
    historyName_            = onnxModel_->mapping.getOnnxName("history");
    scoresName_             = onnxModel_->mapping.getOnnxName("scores");
}

void FixedContextOnnxLabelScorer::reset() {
    Precursor::reset();
    scoreCache_.clear();
    encoderStatesValue_     = Onnx::Value();
    encoderStatesSizeValue_ = Onnx::Value();
}

void FixedContextOnnxLabelScorer::addInput(DataView const& input) {
    Precursor::addInput(input);

    if (not encoderStatesValue_.empty()) {  // Any previously computed encoder-states values are outdated now, so reset them
        encoderStatesValue_     = Onnx::Value();
        encoderStatesSizeValue_ = Onnx::Value();
    }
}

ScoringContextRef FixedContextOnnxLabelScorer::getInitialScoringContext() {
    auto hist = Core::ref(new SeqStepScoringContext());
    hist->labelSeq.resize(historyLength_, startLabelIndex_);
    return hist;
}

size_t FixedContextOnnxLabelScorer::getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const {
    if (encoderStatesName_ != "" or encoderStatesSizeName_ != "") {
        // Full encoder-states sequence is needed for every scoring call, so the input buffer must not be trimmed
        return 0u;
    }

    auto minTimeIndex = Core::Type<Speech::TimeframeIndex>::max;
    for (auto const& context : activeContexts.internalData()) {
        SeqStepScoringContextRef stepHistory(dynamic_cast<SeqStepScoringContext const*>(context.get()));
        minTimeIndex = std::min(minTimeIndex, stepHistory->currentStep);
    }

    return minTimeIndex;
}

void FixedContextOnnxLabelScorer::cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) {
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

ScoringContextRef FixedContextOnnxLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) {
    bool   pushToken     = false;
    size_t timeIncrement = 0ul;
    switch (transitionType) {
        case TransitionType::BLANK_LOOP:
            pushToken     = blankUpdatesHistory_ and loopUpdatesHistory_;
            timeIncrement = 1ul;
            break;
        case TransitionType::SILENCE_LOOP:
            pushToken     = silenceUpdatesHistory_ and loopUpdatesHistory_;
            timeIncrement = 1ul;
            break;
        case TransitionType::LABEL_TO_BLANK:
        case TransitionType::INITIAL_BLANK:
            pushToken     = blankUpdatesHistory_;
            timeIncrement = 1ul;
            break;
        case TransitionType::LABEL_TO_SILENCE:
        case TransitionType::INITIAL_SILENCE:
            pushToken     = silenceUpdatesHistory_;
            timeIncrement = 1ul;
            break;
        case TransitionType::LABEL_LOOP:
            pushToken     = loopUpdatesHistory_;
            timeIncrement = not verticalLabelTransition_;
            break;
        case TransitionType::BLANK_TO_LABEL:
        case TransitionType::SILENCE_TO_LABEL:
        case TransitionType::LABEL_TO_LABEL:
        case TransitionType::INITIAL_LABEL:
        case TransitionType::SENTENCE_END:
            pushToken     = true;
            timeIncrement = not verticalLabelTransition_;
            break;
        default:
            error() << "Unknown transition type " << transitionType;
    }

    // If context is not going to be modified, return the original one to avoid copying
    if (not pushToken and timeIncrement == 0ul) {
        return scoringContext;
    }

    SeqStepScoringContextRef seqStepScoringContext(dynamic_cast<SeqStepScoringContext const*>(scoringContext.get()));

    std::vector<LabelIndex> newLabelSeq;
    newLabelSeq.reserve(seqStepScoringContext->labelSeq.size());
    if (pushToken) {
        newLabelSeq.insert(newLabelSeq.end(), seqStepScoringContext->labelSeq.begin() + 1, seqStepScoringContext->labelSeq.end());
        newLabelSeq.push_back(nextToken);
    }
    else {
        newLabelSeq = seqStepScoringContext->labelSeq;
    }

    return Core::ref(new SeqStepScoringContext(std::move(newLabelSeq), seqStepScoringContext->currentStep + timeIncrement));
}

std::vector<std::optional<ScoreAccessorRef>> FixedContextOnnxLabelScorer::getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) {
    if (scoringContexts.empty()) {
        return {};
    }

    if ((encoderStatesName_ != "" or encoderStatesSizeName_ != "") and (expectMoreFeatures_ or bufferSize() == 0)) {
        // Only allow scoring once all encoder states have been passed
        return std::vector<std::optional<ScoreAccessorRef>>(scoringContexts.size(), std::nullopt);
    }

    // Cast scoring contexts to concrete types
    std::vector<SeqStepScoringContextRef> seqStepScoringContexts;
    seqStepScoringContexts.reserve(scoringContexts.size());
    for (auto const& scoringContext : scoringContexts) {
        seqStepScoringContexts.push_back(Core::ref(dynamic_cast<SeqStepScoringContext const*>(scoringContext.get())));
    }

    /*
     * Collect all requests. If a per-frame input-feature is used, requests are grouped by their
     * timestep since the fed-in feature differs by step. The batched call otherwise uses the same
     * full encoder-states input for every step, so no such grouping is needed in that case.
     */
    std::unordered_map<size_t, std::vector<size_t>> contextsByTimestep;

    for (size_t contextIndex = 0ul; contextIndex < scoringContexts.size(); ++contextIndex) {
        auto step = seqStepScoringContexts[contextIndex]->currentStep;

        // If no input exists for this step (e.g. past the end of the segment), this context can't be forwarded
        if (not getInput(step)) {
            continue;
        }

        size_t groupKey = inputFeatureName_ != "" ? step : 0ul;

        contextsByTimestep[groupKey].push_back(contextIndex);
    }

    std::vector<std::optional<ScoreAccessorRef>> scoreAccessors(scoringContexts.size(), std::nullopt);

    /*
     * Iterate over distinct groups
     */
    for (auto const& [groupKey, contextIndices] : contextsByTimestep) {
        /*
         * Identify unique histories that still need session runs
         */
        std::unordered_set<SeqStepScoringContextRef, ScoringContextHash, ScoringContextEq> uniqueUncachedContexts;

        for (auto contextIndex : contextIndices) {
            if (scoreCache_.find(seqStepScoringContexts[contextIndex]) == scoreCache_.end()) {
                // Group by unique context
                uniqueUncachedContexts.emplace(seqStepScoringContexts[contextIndex]);
            }
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

        // Create score accessors from cache
        for (auto const& contextIndex : contextIndices) {
            auto const& scoreVec         = scoreCache_.at(seqStepScoringContexts[contextIndex]);
            scoreAccessors[contextIndex] = Core::ref(new VectorScoreAccessor(scoreVec, seqStepScoringContexts[contextIndex]->currentStep));
        }
    }

    return scoreAccessors;
}

std::optional<ScoreAccessorRef> FixedContextOnnxLabelScorer::getScoreAccessor(ScoringContextRef scoringContext) {
    return getScoreAccessors({scoringContext})[0];
}

void FixedContextOnnxLabelScorer::forwardBatch(std::vector<SeqStepScoringContextRef> const& scoringContextBatch) {
    if (scoringContextBatch.empty()) {
        return;
    }

    /*
     * Create session inputs
     */
    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;

    if (inputFeatureName_ != "") {
        // All requests in this batch share the same input feature which is set up here
        auto                 inputFeatureDataView = getInput(scoringContextBatch.front()->currentStep);
        f32 const*           inputFeatureData     = inputFeatureDataView->data();
        std::vector<int64_t> inputFeatureShape    = {1ul, static_cast<int64_t>(inputFeatureDataView->size())};
        sessionInputs.emplace_back(inputFeatureName_, Onnx::Value::create(inputFeatureData, inputFeatureShape));
    }

    if (encoderStatesName_ != "") {
        setupEncoderStatesValue();
        sessionInputs.emplace_back(encoderStatesName_, encoderStatesValue_);
    }
    if (encoderStatesSizeName_ != "") {
        setupEncoderStatesSizeValue();
        sessionInputs.emplace_back(encoderStatesSizeName_, encoderStatesSizeValue_);
    }

    // Create batched context input
    Math::FastMatrix<s32> historyMat(historyLength_, scoringContextBatch.size());
    for (size_t b = 0ul; b < scoringContextBatch.size(); ++b) {
        auto context = scoringContextBatch[b];
        std::copy(context->labelSeq.begin(), context->labelSeq.end(), &(historyMat.at(0, b)));  // Pointer to first element in column b
    }

    sessionInputs.emplace_back(historyName_, Onnx::Value::create(historyMat, true));

    /*
     * Run session
     */
    std::vector<Onnx::Value> sessionOutputs;
    onnxModel_->session.run(std::move(sessionInputs), {scoresName_}, sessionOutputs);

    /*
     * Put resulting scores into cache map
     */
    for (size_t b = 0ul; b < scoringContextBatch.size(); ++b) {
        auto scoreVec = std::make_shared<std::vector<Score>>();
        sessionOutputs.front().get(b, *scoreVec);
        scoreCache_.emplace(scoringContextBatch[b], scoreVec);
    }
}

void FixedContextOnnxLabelScorer::setupEncoderStatesValue() {
    if (not encoderStatesValue_.empty()) {
        return;
    }

    u32  T                    = bufferSize();
    auto inputFeatureDataView = getInput(0);

    encoderStatesValue_ = Onnx::Value::createEmpty<f32>({1l, static_cast<int64_t>(T), static_cast<int64_t>(inputFeatureDataView->size())});

    for (size_t t = 0ul; t < T; ++t) {
        inputFeatureDataView = getInput(t);
        std::copy(inputFeatureDataView->data(), inputFeatureDataView->data() + inputFeatureDataView->size(), encoderStatesValue_.data<f32>(0, t));
    }
}

void FixedContextOnnxLabelScorer::setupEncoderStatesSizeValue() {
    if (not encoderStatesSizeValue_.empty()) {
        return;
    }

    u32 T = bufferSize();

    encoderStatesSizeValue_ = Onnx::Value::create(std::vector<s32>{static_cast<s32>(T)});
}

}  // namespace Nn
