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

#include "StatefulOnnxLabelScorer.hh"

#include <algorithm>
#include <cstddef>
#include <utility>

#include <Core/Assertions.hh>
#include <Core/ReferenceCounting.hh>
#include <Core/XmlStream.hh>
#include <Flow/Timestamp.hh>
#include <Math/FastMatrix.hh>
#include <Mm/Module.hh>
#include <Onnx/OnnxStateVariable.hh>
#include <Onnx/Value.hh>
#include <Speech/Types.hh>

#include "OnnxEncoderStates.hh"
#include "ScoreAccessor.hh"
#include "ScoringContext.hh"

namespace Nn {

OnnxHiddenStateScoringContext::OnnxHiddenStateScoringContext()
        : labelSeq(), hiddenState(), requiresFinalize(false) {}

OnnxHiddenStateScoringContext::OnnxHiddenStateScoringContext(std::vector<LabelIndex> const& labelSeq, Onnx::OnnxNamedValuesRef state, bool requiresFinalize)
        : labelSeq(labelSeq), hiddenState(state), requiresFinalize(requiresFinalize) {}

bool OnnxHiddenStateScoringContext::isEqual(ScoringContextRef const& other) const {
    auto* otherPtr = dynamic_cast<OnnxHiddenStateScoringContext const*>(other.get());
    if (otherPtr == nullptr) {
        return false;
    }

    return labelSeqEqual(labelSeq, otherPtr->labelSeq);
}

size_t OnnxHiddenStateScoringContext::hash() const {
    return labelSeqHash(labelSeq);
}

typedef Core::Ref<OnnxHiddenStateScoringContext const> OnnxHiddenStateScoringContextRef;

/*
 * =============================
 * == StatefulOnnxLabelScorer ==
 * =============================
 */

const Core::ParameterBool StatefulOnnxLabelScorer::paramBlankUpdatesHistory(
        "blank-updates-history",
        "Whether previously emitted blank labels should be used to update the history.",
        false);

const Core::ParameterBool StatefulOnnxLabelScorer::paramSilenceUpdatesHistory(
        "silence-updates-history",
        "Whether previously emitted silence labels should be used to update the history.",
        false);

const Core::ParameterBool StatefulOnnxLabelScorer::paramLoopUpdatesHistory(
        "loop-updates-history",
        "Whether in the case of loop transitions every repeated emission should be used to update the history.",
        false);

const Core::ParameterInt StatefulOnnxLabelScorer::paramMaxBatchSize(
        "max-batch-size",
        "Max number of hidden-states that can be fed into the scorer ONNX model at once.",
        Core::Type<int>::max);

const Core::ParameterInt StatefulOnnxLabelScorer::paramMaxCachedScores(
        "max-cached-score-vectors",
        "Maximum size of cache that maps scoring contexts to scores. This prevents memory overflow in case of very long audio segments."
        " In multi-segment usage this applies to each segment separately.",
        10000);

const Core::ParameterBool StatefulOnnxLabelScorer::paramStackEncoderStates(
        "stack-encoder-states",
        "Whether contexts of different segments are forwarded together through the state updater by stacking the encoder states of"
        " their segments to shape [B, T_max, E] (zero-padded). Requires a state updater with dynamic batch dimension of the encoder"
        " states that masks the padding based on the encoder states size. Otherwise contexts are forwarded per segment.",
        false);

// The hidden states and the scores output are not part of the IO specs; they are handled by `Onnx::OnnxInitUpdateScoreBundle`.
// The scorer takes nothing but hidden states, so it has no IO spec of its own.
static const std::vector<Onnx::IOSpecification> stateInitializerModelIoSpec = {
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
                {{1}, {-1}}}};  // [1] or [B]

static const std::vector<Onnx::IOSpecification> stateUpdaterModelIoSpec = {
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
                "encoder-states-index",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{-1}}},  // [B]
        Onnx::IOSpecification{
                "token",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{1}, {-1}}}};  // [1] or [B]

StatefulOnnxLabelScorer::StatefulOnnxLabelScorer(Core::Configuration const& config, ModelCache& modelCache)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::LM),
          blankUpdatesHistory_(paramBlankUpdatesHistory(config)),
          silenceUpdatesHistory_(paramSilenceUpdatesHistory(config)),
          loopUpdatesHistory_(paramLoopUpdatesHistory(config)),
          maxBatchSize_(paramMaxBatchSize(config)),
          maxCachedScores_(paramMaxCachedScores(config)),
          stackEncoderStates_(paramStackEncoderStates(config)),
          initUpdateScoreBundle_(config, modelCache, stateInitializerModelIoSpec, stateUpdaterModelIoSpec, {}),
          segmentData_() {
    initializerEncoderStatesName_     = initUpdateScoreBundle_.stateInitializerOnnxName("encoder-states");
    initializerEncoderStatesSizeName_ = initUpdateScoreBundle_.stateInitializerOnnxName("encoder-states-size");
    updaterEncoderStatesName_         = initUpdateScoreBundle_.stateUpdaterOnnxName("encoder-states");
    updaterEncoderStatesSizeName_     = initUpdateScoreBundle_.stateUpdaterOnnxName("encoder-states-size");
    updaterEncoderStatesIndexName_    = initUpdateScoreBundle_.stateUpdaterOnnxName("encoder-states-index");
    updaterTokenName_                 = initUpdateScoreBundle_.stateUpdaterOnnxName("token");

    if (stackEncoderStates_ and updaterEncoderStatesName_ != "") {
        if (updaterEncoderStatesSizeName_ == "") {
            error() << "stack-encoder-states requires the encoder-states-size input of the state updater to mask the padding";
        }
        auto problem = checkStackableEncoderStatesInput(initUpdateScoreBundle_.stateUpdaterSession(), updaterEncoderStatesName_);
        if (not problem.empty()) {
            error() << "stack-encoder-states is not possible with this state updater: " << problem;
        }
    }
}

StatefulOnnxLabelScorer::SegmentData::SegmentData(size_t maxCachedScores)
        : encoderStatesValue(),
          encoderStatesSizeValue(),
          initialHiddenState(),
          scoreCache(maxCachedScores),
          stateCache(maxCachedScores) {}

StatefulOnnxLabelScorer::SegmentData& StatefulOnnxLabelScorer::segmentData(SegmentId segment) {
    return segmentData_.try_emplace(segment, maxCachedScores_).first->second;
}

bool StatefulOnnxLabelScorer::usesEncoderStates() const {
    return initializerEncoderStatesName_ != "" or initializerEncoderStatesSizeName_ != "" or updaterEncoderStatesName_ != "" or updaterEncoderStatesSizeName_ != "";
}

ScoringContextRef StatefulOnnxLabelScorer::getInitialSegmentScoringContext(SegmentId segment) {
    auto context     = Core::ref(new OnnxHiddenStateScoringContext());
    context->segment = segment;
    return context;
}

ScoringContextRef StatefulOnnxLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) {
    bool updateState = false;
    switch (transitionType) {
        case TransitionType::BLANK_LOOP:
            updateState = blankUpdatesHistory_ and loopUpdatesHistory_;
            break;
        case TransitionType::SILENCE_LOOP:
            updateState = silenceUpdatesHistory_ and loopUpdatesHistory_;
            break;
        case TransitionType::LABEL_TO_BLANK:
        case TransitionType::INITIAL_BLANK:
            updateState = blankUpdatesHistory_;
            break;
        case TransitionType::LABEL_TO_SILENCE:
        case TransitionType::INITIAL_SILENCE:
            updateState = silenceUpdatesHistory_;
            break;
        case TransitionType::LABEL_LOOP:
            updateState = loopUpdatesHistory_;
            break;
        case TransitionType::BLANK_TO_LABEL:
        case TransitionType::SILENCE_TO_LABEL:
        case TransitionType::LABEL_TO_LABEL:
        case TransitionType::INITIAL_LABEL:
        case TransitionType::SENTENCE_END:
            updateState = true;
            break;
        default:
            error() << "Unknown transition type " << transitionType;
    }

    // If scoring context is not going to be modified, return the original one
    if (not updateState) {
        return scoringContext;
    }

    OnnxHiddenStateScoringContextRef onnxHiddenStateScoringContext(dynamic_cast<OnnxHiddenStateScoringContext const*>(scoringContext.get()));
    std::vector<LabelIndex>          newLabelSeq(onnxHiddenStateScoringContext->labelSeq);
    newLabelSeq.push_back(nextToken);

    // Re-use previous hidden-state but mark that finalization (i.e. hidden-state update) is required
    auto newScoringContext     = Core::ref(new OnnxHiddenStateScoringContext(std::move(newLabelSeq), onnxHiddenStateScoringContext->hiddenState, true));
    newScoringContext->segment = onnxHiddenStateScoringContext->segment;

    // Only look up existing segment data here since this may be called in parallel for different segments
    auto dataIt = segmentData_.find(newScoringContext->segment);
    if (dataIt != segmentData_.end()) {
        auto hiddenState = dataIt->second.stateCache.get(newScoringContext);
        if (hiddenState) {
            newScoringContext->hiddenState      = *hiddenState;
            newScoringContext->requiresFinalize = false;
        }
    }

    return newScoringContext;
}

void StatefulOnnxLabelScorer::addSegmentInput(SegmentId segment, DataView const& input) {
    Precursor::addSegmentInput(segment, input);

    auto& data              = segmentData(segment);
    data.initialHiddenState = Onnx::OnnxNamedValuesRef();

    if (not data.encoderStatesValue.empty()) {  // Any previously computed hidden state values are outdated now so reset them
        data.encoderStatesValue.clear();
        data.encoderStatesSizeValue = Onnx::Value();
    }
}

void StatefulOnnxLabelScorer::addSegment(SegmentId segment) {
    Precursor::addSegment(segment);
    // Create the segment data here, so that it is never inserted while contexts of other segments are processed
    segmentData_.erase(segment);
    segmentData(segment);
}

void StatefulOnnxLabelScorer::removeSegment(SegmentId segment) {
    Precursor::removeSegment(segment);
    segmentData_.erase(segment);
}

void StatefulOnnxLabelScorer::resetInternal() {
    Precursor::resetInternal();
    segmentData_.clear();
}

void StatefulOnnxLabelScorer::resetStatisticsInternal() {
    Precursor::resetStatisticsInternal();
    initUpdateScoreBundle_.resetTimers();
    stateMarshallingTime_.reset();
    contextPreparationTime_.reset();
}

std::optional<ScoreAccessorRef> StatefulOnnxLabelScorer::computeScoreAccessor(ScoringContextRef scoringContext) {
    return computeScoreAccessors({scoringContext})[0];
}

std::vector<std::optional<ScoreAccessorRef>> StatefulOnnxLabelScorer::computeScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) {
    if (scoringContexts.empty()) {
        return {};
    }

    std::vector<std::optional<ScoreAccessorRef>> scoreAccessors(scoringContexts.size(), std::nullopt);

    // Only allow scoring once all encoder states of a segment have been passed
    auto segmentReady = [this](SegmentId segment) {
        return not usesEncoderStates() or (not expectMoreFeatures(segment) and bufferSize(segment) != 0);
    };

    /*
     * Identify unique scoring contexts that still need session runs
     */
    contextPreparationTime_.start();
    std::unordered_set<OnnxHiddenStateScoringContextRef, ScoringContextHash, ScoringContextEq> uniqueUncachedScoringContexts;

    for (auto const& scoringContext : scoringContexts) {
        // We need to finalize all scoring contexts before using them for scoring again.

        OnnxHiddenStateScoringContextRef onnxHiddenStateScoringContext(dynamic_cast<OnnxHiddenStateScoringContext const*>(scoringContext.get()));
        if (not segmentReady(onnxHiddenStateScoringContext->segment)) {
            continue;
        }
        if (not segmentData(onnxHiddenStateScoringContext->segment).scoreCache.contains(onnxHiddenStateScoringContext)) {
            // Group by unique scoring context
            uniqueUncachedScoringContexts.emplace(onnxHiddenStateScoringContext);
        }
    }

    /*
     * Group contexts that can be forwarded together: all contexts if the state updater doesn't need
     * encoder states or they are stacked, otherwise the contexts of each segment
     */
    bool                                                       groupBySegment = updaterEncoderStatesName_ != "" and not stackEncoderStates_;
    std::vector<std::vector<OnnxHiddenStateScoringContextRef>> contextGroups;
    std::unordered_map<SegmentId, size_t>                      segmentToGroupIndex;
    for (auto const& scoringContext : uniqueUncachedScoringContexts) {
        SegmentId groupKey  = groupBySegment ? scoringContext->segment : defaultSegmentId;
        auto [it, inserted] = segmentToGroupIndex.emplace(groupKey, contextGroups.size());
        if (inserted) {
            contextGroups.emplace_back();
        }
        contextGroups[it->second].push_back(scoringContext);
    }
    contextPreparationTime_.stop();

    countUncomputedAccessors(scoringContexts.size() - uniqueUncachedScoringContexts.size());

    /*
     * Fill state and score caches for all uncached scoring contexts
     */
    for (auto const& contextGroup : contextGroups) {
        std::vector<OnnxHiddenStateScoringContextRef> scoringContextBatch;
        scoringContextBatch.reserve(std::min(contextGroup.size(), maxBatchSize_));
        for (auto const& scoringContext : contextGroup) {
            scoringContextBatch.push_back(scoringContext);
            if (scoringContextBatch.size() == maxBatchSize_) {  // Batch is full -> forward now
                cacheStates(scoringContextBatch);
                cacheScores(scoringContextBatch);
                scoringContextBatch.clear();
            }
        }

        // Forward remaining scoring contexts
        cacheStates(scoringContextBatch);
        cacheScores(scoringContextBatch);
    }

    /*
     * Assign states from cache to scoring contexts and scores from cache to result vector
     */
    contextPreparationTime_.start();
    for (size_t contextIndex = 0ul; contextIndex < scoringContexts.size(); ++contextIndex) {
        OnnxHiddenStateScoringContextRef onnxHiddenStateScoringContext(dynamic_cast<OnnxHiddenStateScoringContext const*>(scoringContexts[contextIndex].get()));
        if (not segmentReady(onnxHiddenStateScoringContext->segment)) {
            continue;
        }
        auto& data = segmentData(onnxHiddenStateScoringContext->segment);

        if (onnxHiddenStateScoringContext->requiresFinalize) {
            auto hiddenState = data.stateCache.get(onnxHiddenStateScoringContext);
            verify(hiddenState);
            onnxHiddenStateScoringContext->hiddenState      = *hiddenState;
            onnxHiddenStateScoringContext->requiresFinalize = false;
        }

        verify(data.scoreCache.contains(onnxHiddenStateScoringContext));
        auto const& scoreVec         = data.scoreCache.get(onnxHiddenStateScoringContext)->get();
        auto const  timeframe        = onnxHiddenStateScoringContext->labelSeq.size();
        scoreAccessors[contextIndex] = Core::ref(new VectorScoreAccessor(scoreVec, timeframe));
    }
    contextPreparationTime_.stop();

    return scoreAccessors;
}

size_t StatefulOnnxLabelScorer::getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const {
    return 0u;
}

void StatefulOnnxLabelScorer::cacheScores(std::vector<OnnxHiddenStateScoringContextRef> const& scoringContextBatch) {
    if (scoringContextBatch.empty()) {
        return;
    }

    // Fetched before the timed block below because they run a session of their own
    for (auto const& scoringContext : scoringContextBatch) {
        if (scoringContext->labelSeq.empty()) {
            computeInitialHiddenState(scoringContext->segment);
        }
    }

    /*
     * Collect the hidden state of each scoring context in the batch
     */
    std::vector<Onnx::OnnxNamedValuesRef> hiddenStates;
    {
        Core::StopWatch::Scope marshallingTimer(stateMarshallingTime_);
        hiddenStates.reserve(scoringContextBatch.size());
        for (auto const& scoringContext : scoringContextBatch) {
            auto& data = segmentData(scoringContext->segment);
            hiddenStates.push_back(scoringContext->labelSeq.empty()
                                           ? data.initialHiddenState
                                           : (*data.stateCache.get(scoringContext)).get());
        }
    }

    /*
     * Run scorer and put resulting scores into cache map
     */
    auto scoreVecs = initUpdateScoreBundle_.scores(hiddenStates);
    verify(scoreVecs.size() == scoringContextBatch.size());

    Core::StopWatch::Scope marshallingTimer(stateMarshallingTime_);
    for (size_t b = 0ul; b < scoringContextBatch.size(); ++b) {
        segmentData(scoringContextBatch[b]->segment).scoreCache.put(scoringContextBatch[b], scoreVecs[b]);
    }
}

std::vector<Onnx::OnnxNamedValuesRef> StatefulOnnxLabelScorer::updatedHiddenStates(std::vector<Onnx::OnnxNamedValuesRef> const& hiddenStatesBatch, std::vector<s32> nextTokensBatch, std::vector<SegmentId> const& segmentsBatch) {
    Onnx::OnnxInitUpdateScoreBundle::SessionInputs sessionInputs;
    {
        Core::StopWatch::Scope marshallingTimer(stateMarshallingTime_);

        if (updaterEncoderStatesName_ != "" or updaterEncoderStatesSizeName_ != "") {
            addEncoderStatesInputs(
                    {.states = updaterEncoderStatesName_, .size = updaterEncoderStatesSizeName_, .index = updaterEncoderStatesIndexName_},
                    initUpdateScoreBundle_.stateUpdaterSession(),
                    segmentsBatch,
                    [this](SegmentId segment) -> Onnx::PersistentInput& {
                        setupEncoderStatesValue(segment);
                        return segmentData(segment).encoderStatesValue;
                    },
                    [this](SegmentId segment) -> Onnx::Value const& {
                        setupEncoderStatesSizeValue(segment);
                        return segmentData(segment).encoderStatesSizeValue;
                    },
                    [this](SegmentId segment) { return bufferSize(segment); },
                    sessionInputs);
        }
        if (updaterTokenName_ != "") {
            sessionInputs.emplace_back(updaterTokenName_, Onnx::Value::create(nextTokensBatch));
        }
    }

    return initUpdateScoreBundle_.updatedStates(hiddenStatesBatch, std::move(sessionInputs));
}

void StatefulOnnxLabelScorer::cacheStates(std::vector<OnnxHiddenStateScoringContextRef> const& scoringContextBatch) {
    std::vector<OnnxHiddenStateScoringContextRef> nonFinalizedContexts;
    std::vector<Onnx::OnnxNamedValuesRef>         hiddenStates;
    std::vector<s32>                              nextTokens;
    std::vector<SegmentId>                        segments;
    for (auto const& scoringContext : scoringContextBatch) {
        if (not scoringContext->requiresFinalize) {
            continue;
        }
        nonFinalizedContexts.push_back(scoringContext);
        if (scoringContext->hiddenState) {
            hiddenStates.push_back(scoringContext->hiddenState);
        }
        else {
            hiddenStates.push_back(computeInitialHiddenState(scoringContext->segment));
        }
        verify(not scoringContext->labelSeq.empty());
        nextTokens.push_back(scoringContext->labelSeq.back());
        segments.push_back(scoringContext->segment);
    }

    // If no scoring contexts need finalization, nothing has to be done
    if (nonFinalizedContexts.empty()) {
        return;
    }

    auto newHiddenStates = updatedHiddenStates(hiddenStates, nextTokens, segments);
    verify(newHiddenStates.size() == nonFinalizedContexts.size());

    for (size_t i = 0ul; i < nonFinalizedContexts.size(); ++i) {
        segmentData(nonFinalizedContexts[i]->segment).stateCache.put(nonFinalizedContexts[i], newHiddenStates[i]);
    }
}

Onnx::OnnxNamedValuesRef StatefulOnnxLabelScorer::computeInitialHiddenState(SegmentId segment) {
    verify(not expectMoreFeatures(segment) or (initializerEncoderStatesName_ == "" and initializerEncoderStatesSizeName_ == ""));

    auto& data = segmentData(segment);
    if (not data.initialHiddenState) {  // initialHiddenState is still sentinel value -> compute it
        Onnx::OnnxInitUpdateScoreBundle::SessionInputs sessionInputs;
        {
            Core::StopWatch::Scope marshallingTimer(stateMarshallingTime_);

            if (initializerEncoderStatesName_ != "") {
                setupEncoderStatesValue(segment);
                sessionInputs.emplace_back(initializerEncoderStatesName_, data.encoderStatesValue.view(initUpdateScoreBundle_.stateInitializerSession()));
            }
            if (initializerEncoderStatesSizeName_ != "") {
                setupEncoderStatesSizeValue(segment);
                sessionInputs.emplace_back(initializerEncoderStatesSizeName_, data.encoderStatesSizeValue.view());
            }
        }

        data.initialHiddenState = initUpdateScoreBundle_.initialState(std::move(sessionInputs));
    }

    return data.initialHiddenState;
}

void StatefulOnnxLabelScorer::setupEncoderStatesValue(SegmentId segment) {
    auto& data = segmentData(segment);
    if (not data.encoderStatesValue.empty()) {
        return;
    }

    u32  T                    = bufferSize(segment);
    auto inputFeatureDataView = getInput(segment, 0);

    auto encoderStates = Onnx::Value::createEmpty<f32>({1l, static_cast<int64_t>(T), static_cast<int64_t>(inputFeatureDataView->size())});

    for (size_t t = 0ul; t < T; ++t) {
        inputFeatureDataView = getInput(segment, t);
        std::copy(inputFeatureDataView->data(), inputFeatureDataView->data() + inputFeatureDataView->size(), encoderStates.data<f32>(0, t));
    }
    data.encoderStatesValue.set(std::move(encoderStates));
}

void StatefulOnnxLabelScorer::setupEncoderStatesSizeValue(SegmentId segment) {
    auto& data = segmentData(segment);
    if (not data.encoderStatesSizeValue.empty()) {
        return;
    }

    u32 T = bufferSize(segment);

    data.encoderStatesSizeValue = Onnx::Value::create(std::vector<s32>{static_cast<s32>(T)});
}

void StatefulOnnxLabelScorer::logScoringBreakdown() const {
    auto stateUpdateSessionTime = initUpdateScoreBundle_.initializerSessionTime().elapsedMilliseconds() + initUpdateScoreBundle_.updaterSessionTime().elapsedMilliseconds();
    auto stateMarshallingTime   = stateMarshallingTime_.elapsedMilliseconds() + initUpdateScoreBundle_.marshallingTime().elapsedMilliseconds();

    statisticsChannel_ << Core::XmlFull("state-update-session-time", stateUpdateSessionTime);
    statisticsChannel_ << Core::XmlFull("scorer-session-time", initUpdateScoreBundle_.scorerSessionTime().elapsedMilliseconds());
    statisticsChannel_ << Core::XmlFull("state-marshalling-time", stateMarshallingTime);
    statisticsChannel_ << Core::XmlFull("context-preparation-time", contextPreparationTime_.elapsedMilliseconds());
}

}  // namespace Nn
