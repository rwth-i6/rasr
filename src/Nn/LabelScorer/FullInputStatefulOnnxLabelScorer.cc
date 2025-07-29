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

#include "FullInputStatefulOnnxLabelScorer.hh"

#include <algorithm>
#include <cstddef>
#include <utility>

#include <Core/Assertions.hh>
#include <Core/ReferenceCounting.hh>
#include <Flow/Timestamp.hh>
#include <Math/FastMatrix.hh>
#include <Mm/Module.hh>
#include <Speech/Types.hh>

#include "LabelScorer.hh"
#include "ScoringContext.hh"

namespace Nn {

/*
 * =============================
 * == StatefulOnnxLabelScorer
 * =============================
 */

const Core::ParameterBool FullInputStatefulOnnxLabelScorer::paramBlankUpdatesHistory(
        "blank-updates-history",
        "Whether previously emitted blank labels should be used to update the history.",
        false);

const Core::ParameterBool FullInputStatefulOnnxLabelScorer::paramLoopUpdatesHistory(
        "loop-updates-history",
        "Whether in the case of loop transitions every repeated emission should be used to update the history.",
        false);

const Core::ParameterInt FullInputStatefulOnnxLabelScorer::paramMaxBatchSize(
        "max-batch-size",
        "Max number of hidden-states that can be fed into the scorer ONNX model at once.",
        Core::Type<int>::max);

const Core::ParameterInt FullInputStatefulOnnxLabelScorer::paramMaxCachedScores(
        "max-cached-scores",
        "Maximum size of cache that maps histories to scores. This prevents memory overflow in case of very long audio segments.",
        1000);

const std::vector<Onnx::IOSpecification> scorerModelIoSpec = {
        Onnx::IOSpecification{
                "scores",
                Onnx::IODirection::OUTPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -2}}}};

const std::vector<Onnx::IOSpecification> stateInitializerModelIoSpec = {
        Onnx::IOSpecification{
                "encoder-states",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{1, -1, -2}, {-1, -1, -2}}},
        Onnx::IOSpecification{
                "encoder-states-size",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{1}, {-1}}}};

const std::vector<Onnx::IOSpecification> stateUpdaterModelIoSpec = {
        Onnx::IOSpecification{
                "encoder-states",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{1, -1, -2}, {-1, -1, -2}}},
        Onnx::IOSpecification{
                "encoder-states-size",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{1}, {-1}}},
        Onnx::IOSpecification{
                "token",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{1}, {-1}}}};

FullInputStatefulOnnxLabelScorer::FullInputStatefulOnnxLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          initialHiddenState_(),
          initializerOutputToStateNameMap_(),
          updaterInputToStateNameMap_(),
          updaterOutputToStateNameMap_(),
          scorerInputToStateNameMap_(),
          blankUpdatesHistory_(paramBlankUpdatesHistory(config)),
          loopUpdatesHistory_(paramLoopUpdatesHistory(config)),
          maxBatchSize_(paramMaxBatchSize(config)),
          scorerOnnxModel_(select("scorer-model"), scorerModelIoSpec),
          stateInitializerOnnxModel_(select("state-initializer-model"), stateInitializerModelIoSpec),
          stateUpdaterOnnxModel_(select("state-updater-model"), stateUpdaterModelIoSpec),
          scorerScoresName_(scorerOnnxModel_.mapping.getOnnxName("scores")),
          initializerEncoderStatesName_(stateInitializerOnnxModel_.mapping.getOnnxName("encoder-states")),
          initializerEncoderStatesSizeName_(stateInitializerOnnxModel_.mapping.getOnnxName("encoder-states-size")),
          updaterEncoderStatesName_(stateUpdaterOnnxModel_.mapping.getOnnxName("encoder-states")),
          updaterEncoderStatesSizeName_(stateUpdaterOnnxModel_.mapping.getOnnxName("encoder-states-size")),
          updaterTokenName_(stateUpdaterOnnxModel_.mapping.getOnnxName("token")),
          encoderStatesValue_(),
          encoderStatesSizeValue_(),
          scoreCache_(paramMaxCachedScores(config)) {
    auto initializerMetadataKeys = stateInitializerOnnxModel_.session.getCustomMetadataKeys();
    auto updaterMetadataKeys     = stateUpdaterOnnxModel_.session.getCustomMetadataKeys();
    auto scorerMetadataKeys      = scorerOnnxModel_.session.getCustomMetadataKeys();

    // Map state initializer outputs to states
    std::unordered_set<std::string> initializerStateNames;
    for (auto const& key : initializerMetadataKeys) {
        if (stateInitializerOnnxModel_.session.hasOutput(key)) {
            auto stateName = stateInitializerOnnxModel_.session.getCustomMetadata(key);
            initializerOutputToStateNameMap_.emplace(key, stateName);
            initializerStateNames.insert(stateName);
        }
    }

    // Map state updater inputs and outputs to states
    std::unordered_set<std::string> updaterStateNames;
    for (auto const& key : updaterMetadataKeys) {
        if (stateUpdaterOnnxModel_.session.hasInput(key)) {
            auto stateName = stateUpdaterOnnxModel_.session.getCustomMetadata(key);
            if (initializerStateNames.find(stateName) == initializerStateNames.end()) {
                error() << "State updater input " << key << " associated with state " << stateName << " is not present in state initializer";
            }
            updaterInputToStateNameMap_.emplace(key, stateName);
        }
        if (stateUpdaterOnnxModel_.session.hasOutput(key)) {
            auto stateName = stateUpdaterOnnxModel_.session.getCustomMetadata(key);
            if (initializerStateNames.find(stateName) == initializerStateNames.end()) {
                error() << "State updater output " << key << " associated with state " << stateName << " is not present in state initializer";
            }
            updaterOutputToStateNameMap_.emplace(key, stateName);
            updaterStateNames.insert(stateName);
        }
    }

    // In the loop we checked that the updater outputs are a subset of the initializer outputs.
    // If they have the same size, they are equal. Otherwise, some initializer outputs
    // are not updater outputs.
    if (initializerStateNames.size() != updaterStateNames.size()) {
        warning() << "State initializer has states that are not updated by the state updater";
    }

    // Map scorer inputs to states
    for (auto const& key : scorerMetadataKeys) {
        if (scorerOnnxModel_.session.hasInput(key)) {
            auto stateName = scorerOnnxModel_.session.getCustomMetadata(key);
            if (initializerStateNames.find(stateName) == initializerStateNames.end()) {
                error() << "Scorer input " << key << " associated with state " << stateName << " is not present in state initializer";
            }
            scorerInputToStateNameMap_.emplace(key, stateName);
        }
    }
}

void FullInputStatefulOnnxLabelScorer::reset() {
    Precursor::reset();
    scoreCache_.clear();
}

Core::Ref<const ScoringContext> FullInputStatefulOnnxLabelScorer::getInitialScoringContext() {
    return Core::ref(new OnnxHiddenStateScoringContext());  // Sentinel empty Ref as initial hidden state
}

Core::Ref<const ScoringContext> FullInputStatefulOnnxLabelScorer::extendedScoringContext(LabelScorer::Request const& request) {
    OnnxHiddenStateScoringContextRef history(dynamic_cast<const OnnxHiddenStateScoringContext*>(request.context.get()));

    bool updateState = false;
    switch (request.transitionType) {
        case LabelScorer::TransitionType::BLANK_LOOP:
            updateState = blankUpdatesHistory_ and loopUpdatesHistory_;
            break;
        case LabelScorer::TransitionType::LABEL_TO_BLANK:
        case LabelScorer::TransitionType::INITIAL_BLANK:
            updateState = blankUpdatesHistory_;
            break;
        case LabelScorer::TransitionType::LABEL_LOOP:
            updateState = loopUpdatesHistory_;
            break;
        case LabelScorer::TransitionType::BLANK_TO_LABEL:
        case LabelScorer::TransitionType::LABEL_TO_LABEL:
        case LabelScorer::TransitionType::INITIAL_LABEL:
            updateState = true;
            break;
        case LabelScorer::TransitionType::SENTENCE_END:
            updateState = false;
            break;
        default:
            error() << "Unknown transition type " << request.transitionType;
    }

    // If history is not going to be modified, return the original one
    if (not updateState) {
        return request.context;
    }

    std::vector<LabelIndex> newLabelSeq(history->labelSeq);
    newLabelSeq.push_back(request.nextToken);

    OnnxHiddenStateRef newHiddenState;
    if (not history->hiddenState) {  // Sentinel start-state
        newHiddenState = updatedHiddenState(computeInitialHiddenState(), request.nextToken);
    }
    else {
        newHiddenState = updatedHiddenState(history->hiddenState, request.nextToken);
    }

    return Core::ref(new OnnxHiddenStateScoringContext(std::move(newLabelSeq), newHiddenState));
}

void FullInputStatefulOnnxLabelScorer::addInput(DataView const& input) {
    Precursor::addInput(input);

    initialHiddenState_ = OnnxHiddenStateRef();

    if (not encoderStatesValue_.empty()) {  // Any previously computed hidden state values are outdated now so reset them
        encoderStatesValue_     = Onnx::Value();
        encoderStatesSizeValue_ = Onnx::Value();
    }
}

std::optional<LabelScorer::ScoresWithTimes> FullInputStatefulOnnxLabelScorer::computeScoresWithTimes(std::vector<LabelScorer::Request> const& requests) {
    if (expectMoreFeatures_ or bufferSize() == 0ul) {  // Only allow scoring once all encoder states have been passed
        return {};
    }

    ScoresWithTimes result;
    result.scores.reserve(requests.size());

    /*
     * Identify unique histories that still need session runs
     */
    std::unordered_set<OnnxHiddenStateScoringContextRef, ScoringContextHash, ScoringContextEq> uniqueUncachedHistories;

    for (auto& request : requests) {
        OnnxHiddenStateScoringContextRef historyPtr(dynamic_cast<const OnnxHiddenStateScoringContext*>(request.context.get()));
        if (not scoreCache_.contains(historyPtr)) {
            // Group by unique history
            uniqueUncachedHistories.emplace(historyPtr);
        }
    }

    std::vector<OnnxHiddenStateScoringContextRef> historyBatch;
    historyBatch.reserve(std::min(uniqueUncachedHistories.size(), maxBatchSize_));
    for (auto history : uniqueUncachedHistories) {
        historyBatch.push_back(history);
        if (historyBatch.size() == maxBatchSize_) {  // Batch is full -> forward now
            forwardBatch(historyBatch);
            historyBatch.clear();
        }
    }

    forwardBatch(historyBatch);  // Forward remaining histories

    /*
     * Assign from cache map to result vector
     */
    for (const auto& request : requests) {
        OnnxHiddenStateScoringContextRef history(dynamic_cast<const OnnxHiddenStateScoringContext*>(request.context.get()));

        verify(scoreCache_.contains(history));
        auto const& scores = scoreCache_.get(history)->get();

        result.scores.push_back(scores.at(request.nextToken));

        result.timeframes.push_back(history->labelSeq.size());
    }

    return result;
}

std::optional<LabelScorer::ScoreWithTime> FullInputStatefulOnnxLabelScorer::computeScoreWithTime(LabelScorer::Request const& request) {
    auto result = computeScoresWithTimes({request});
    if (not result) {
        return {};
    }
    return ScoreWithTime{result->scores.front(), result->timeframes.front()};
}

size_t FullInputStatefulOnnxLabelScorer::getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const {
    return 0u;
}

void FullInputStatefulOnnxLabelScorer::setupEncoderStatesValue() {
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

void FullInputStatefulOnnxLabelScorer::setupEncoderStatesSizeValue() {
    if (not encoderStatesSizeValue_.empty()) {
        return;
    }

    u32 T = bufferSize();

    encoderStatesSizeValue_ = Onnx::Value::create(std::vector<s32>{static_cast<s32>(T)});
}

OnnxHiddenStateRef FullInputStatefulOnnxLabelScorer::computeInitialHiddenState() {
    verify(not expectMoreFeatures_);

    if (not initialHiddenState_) {  // initialHiddenState_ is still sentinel value -> compute it
        /*
         * Create session inputs
         */
        std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;

        if (initializerEncoderStatesName_ != "") {
            setupEncoderStatesValue();
            sessionInputs.emplace_back(initializerEncoderStatesName_, std::move(encoderStatesValue_));
        }
        if (initializerEncoderStatesSizeName_ != "") {
            setupEncoderStatesSizeValue();
            sessionInputs.emplace_back(initializerEncoderStatesSizeName_, std::move(encoderStatesSizeValue_));
        }

        std::vector<std::string> sessionOutputNames;
        std::vector<std::string> stateNames;
        for (auto const& [outputName, stateName] : initializerOutputToStateNameMap_) {
            sessionOutputNames.push_back(outputName);
            stateNames.push_back(stateName);
        }

        /*
         * Run session
         */
        std::vector<Onnx::Value> sessionOutputs;
        stateInitializerOnnxModel_.session.run(std::move(sessionInputs), sessionOutputNames, sessionOutputs);

        /*
         * Return resulting hidden state
         */
        initialHiddenState_ = Core::ref(new OnnxHiddenState(std::move(stateNames), std::move(sessionOutputs)));
    }

    return initialHiddenState_;
}

OnnxHiddenStateRef FullInputStatefulOnnxLabelScorer::updatedHiddenState(OnnxHiddenStateRef const& hiddenState, LabelIndex nextToken) {
    /*
     * Create session inputs
     */
    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;

    if (updaterEncoderStatesName_ != "") {
        setupEncoderStatesValue();
        sessionInputs.emplace_back(updaterEncoderStatesName_, std::move(encoderStatesValue_));
    }
    if (updaterEncoderStatesSizeName_ != "") {
        setupEncoderStatesSizeValue();
        sessionInputs.emplace_back(updaterEncoderStatesSizeName_, std::move(encoderStatesSizeValue_));
    }
    if (updaterTokenName_ != "") {
        sessionInputs.emplace_back(updaterTokenName_, Onnx::Value::create(std::vector<s32>{static_cast<s32>(nextToken)}));
    }

    for (auto const& [inputName, stateName] : updaterInputToStateNameMap_) {
        sessionInputs.emplace_back(inputName, hiddenState->stateValueMap.at(stateName));
    }

    /*
     * Run session
     */
    std::vector<std::string> sessionOutputNames;
    std::vector<std::string> stateNames;
    for (auto const& [outputName, stateName] : updaterOutputToStateNameMap_) {
        sessionOutputNames.push_back(outputName);
        stateNames.push_back(stateName);
    }

    std::vector<Onnx::Value> sessionOutputs;
    stateUpdaterOnnxModel_.session.run(std::move(sessionInputs), sessionOutputNames, sessionOutputs);

    /*
     * Return resulting hidden state
     */
    auto newHiddenState = Core::ref(new OnnxHiddenState(std::move(stateNames), std::move(sessionOutputs)));

    return newHiddenState;
}

void FullInputStatefulOnnxLabelScorer::forwardBatch(std::vector<OnnxHiddenStateScoringContextRef> const& historyBatch) {
    if (historyBatch.empty()) {
        return;
    }

    /*
     * Create session inputs
     */
    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;

    for (auto const& [inputName, stateName] : scorerInputToStateNameMap_) {
        // Collect a vector of individual state values of shape [1, *] and afterwards concatenate
        // them to a batched state tensor of shape [B, *]
        std::vector<Onnx::Value const*> stateValues;
        stateValues.reserve(historyBatch.size());

        for (size_t b = 0ul; b < historyBatch.size(); ++b) {
            auto history     = historyBatch[b];
            auto hiddenState = history->hiddenState;
            if (not hiddenState) {  // Sentinel hidden-state
                hiddenState = computeInitialHiddenState();
            }
            stateValues.push_back(&hiddenState->stateValueMap.at(stateName));
        }
        sessionInputs.emplace_back(inputName, Onnx::Value::concat(stateValues, 0));
    }

    /*
     * Run session
     */
    std::vector<Onnx::Value> sessionOutputs;
    scorerOnnxModel_.session.run(std::move(sessionInputs), {scorerScoresName_}, sessionOutputs);

    /*
     * Put resulting scores into cache map
     */
    for (size_t b = 0ul; b < historyBatch.size(); ++b) {
        std::vector<f32> scoreVec;
        sessionOutputs.front().get(b, scoreVec);
        scoreCache_.put(historyBatch[b], std::move(scoreVec));
    }
}

}  // namespace Nn
