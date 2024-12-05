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

#include "StatefulOnnxLabelScorer.hh"
#include <Core/Assertions.hh>
#include <Core/ReferenceCounting.hh>
#include <Flow/Timestamp.hh>
#include <Math/FastMatrix.hh>
#include <Mm/Module.hh>
#include <Speech/Types.hh>
#include <algorithm>
#include <cstddef>
#include <utility>
#include "LabelScorer.hh"
#include "ScoringContext.hh"

namespace Nn {

/*
 * =============================
 * == StatefulOnnxLabelScorer
 * =============================
 */

const Core::ParameterBool StatefulOnnxLabelScorer::paramBlankUpdatesHistory(
        "blank-updates-history",
        "Whether previously emitted blank labels should be used to update the history.",
        false);

const Core::ParameterBool StatefulOnnxLabelScorer::paramLoopUpdatesHistory(
        "loop-updates-history",
        "Whether in the case of loop transitions every repeated emission should be used to update the history.",
        false);

const Core::ParameterInt StatefulOnnxLabelScorer::paramMaxBatchSize(
        "max-batch-size",
        "Max number of hidden-states that can be fed into the ONNX model at once.",
        Core::Type<int>::max);

const Core::ParameterInt StatefulOnnxLabelScorer::paramMaxCachedScores(
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
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{1}, {-1}}}};

StatefulOnnxLabelScorer::StatefulOnnxLabelScorer(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config),
          blankUpdatesHistory_(paramBlankUpdatesHistory(config)),
          loopUpdatesHistory_(paramLoopUpdatesHistory(config)),
          maxBatchSize_(paramMaxBatchSize(config)),
          scorerOnnxModel_(select("scorer-model"), scorerModelIoSpec),
          stateInitializerOnnxModel_(select("state-initializer-model"), stateInitializerModelIoSpec),
          stateUpdaterOnnxModel_(select("state-updater-model"), stateUpdaterModelIoSpec),
          scoresName_(scorerOnnxModel_.mapping.getOnnxName("scores")),
          initializerEncoderStatesName_(stateInitializerOnnxModel_.mapping.getOnnxName("encoder-states")),
          initializerEncoderStatesSizeName_(stateInitializerOnnxModel_.mapping.getOnnxName("encoder-states-size")),
          updaterEncoderStatesName_(stateUpdaterOnnxModel_.mapping.getOnnxName("encoder-states")),
          updaterEncoderStatesSizeName_(stateUpdaterOnnxModel_.mapping.getOnnxName("encoder-states-size")),
          updaterTokenName_(stateUpdaterOnnxModel_.mapping.getOnnxName("token")),
          encoderStatesValue_(),
          encoderStatesSizeValue_(),
          scoreCache_(paramMaxCachedScores(config)) {
    std::stringstream ss;
    auto              stateNames = stateInitializerOnnxModel_.session.getAllOutputNames();

    ss << "Found state names: ";
    for (const auto& stateName : stateNames) {
        ss << "\"" << stateName << "\", ";
    }
    ss << "\n";

    // Map matching input/output state names of onnx sessions
    for (const auto& inputName : stateUpdaterOnnxModel_.session.getAllInputNames()) {
        if (inputName == updaterEncoderStatesName_ or inputName == updaterEncoderStatesSizeName_ or inputName == updaterTokenName_) {
            continue;
        }

        bool matchFound = false;
        for (const auto& stateName : stateNames) {
            if (inputName.find(stateName) != std::string::npos or stateName.find(inputName) != std::string::npos) {
                updaterInputToStateNameMap_[inputName] = stateName;
                ss << "  Input \"" << inputName << "\" of state updater model matches state \"" << stateName << "\"\n";
                matchFound = true;
                continue;
            }
        }
        if (not matchFound) {
            log() << ss.str();
            error() << "Input \"" << inputName << "\" of state updater model couldn't be matched to any of the state names";
        }
    }

    for (const auto& outputName : stateUpdaterOnnxModel_.session.getAllOutputNames()) {
        bool matchFound = false;
        for (const auto& stateName : stateNames) {
            if (outputName.find(stateName) != std::string::npos or stateName.find(outputName) != std::string::npos) {
                updaterOutputToStateNameMap_[outputName] = stateName;
                ss << "  Output \"" << outputName << "\" of state updater model matches state \"" << stateName << "\"\n";
                matchFound = true;
                continue;
            }
        }
        if (not matchFound) {
            log() << ss.str();
            error() << "Output \"" << outputName << "\" of state updater model couldn't be matched to any of the state names";
        }
    }

    for (const auto& inputName : scorerOnnxModel_.session.getAllInputNames()) {
        bool matchFound = false;
        for (const auto& stateName : stateNames) {
            if (inputName.find(stateName) != std::string::npos or stateName.find(inputName) != std::string::npos) {
                scorerInputToStateNameMap_[inputName] = stateName;
                ss << "  Input \"" << inputName << "\" of scorer model matches state \"" << stateName << "\"\n";
                matchFound = true;
                continue;
            }
        }
        if (not matchFound) {
            log() << ss.str();
            error() << "Input \"" << inputName << "\" of scorer model couldn't be matched to any of the state names";
        }
    }

    log() << ss.str();
}

void StatefulOnnxLabelScorer::reset() {
    Precursor::reset();
    scoreCache_.clear();
}

Core::Ref<const ScoringContext> StatefulOnnxLabelScorer::getInitialScoringContext() {
    return Core::ref(new HiddenStateScoringContext());  // Sentinel empty Ref as initial hidden state
}

Core::Ref<const ScoringContext> StatefulOnnxLabelScorer::extendedScoringContext(LabelScorer::Request request) {
    HiddenStateScoringContextRef history(dynamic_cast<const HiddenStateScoringContext*>(request.context.get()));

    bool updateState = false;
    switch (request.transitionType) {
        case LabelScorer::TransitionType::BLANK_LOOP:
            updateState = blankUpdatesHistory_ and loopUpdatesHistory_;
            break;
        case LabelScorer::TransitionType::LABEL_TO_BLANK:
            updateState = blankUpdatesHistory_;
            break;
        case LabelScorer::TransitionType::LABEL_LOOP:
            updateState = loopUpdatesHistory_;
            break;
        case LabelScorer::TransitionType::BLANK_TO_LABEL:
        case LabelScorer::TransitionType::LABEL_TO_LABEL:
            updateState = true;
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

    HiddenStateRef newHiddenState;
    if (not history->hiddenState) {  // Sentinel start-state
        newHiddenState = updatedHiddenState(computeInitialHiddenState(), request.nextToken);
    }
    else {
        newHiddenState = updatedHiddenState(history->hiddenState, request.nextToken);
    }

    return Core::ref(new HiddenStateScoringContext(std::move(newLabelSeq), newHiddenState));
}

void StatefulOnnxLabelScorer::addInput(f32 const* input, size_t F) {
    Precursor::addInput(input, F);
    initialHiddenState_ = HiddenStateRef();

    if (not encoderStatesValue_.empty()) {  // Computed ONNX value is outdated now so reset it
        encoderStatesValue_     = Onnx::Value();
        encoderStatesSizeValue_ = Onnx::Value();
    }
}

std::optional<LabelScorer::ScoresWithTimes> StatefulOnnxLabelScorer::getScoresWithTimes(const std::vector<LabelScorer::Request>& requests) {
    if (featuresMissing_ or inputBuffer_.empty()) {  // Only allow scoring once all encoder states have been passed
        return {};
    }

    ScoresWithTimes result;
    result.scores.reserve(requests.size());

    /*
     * Identify unique histories that still need session runs
     */
    std::unordered_set<HiddenStateScoringContextRef, HiddenStateScoringContextHash, HiddenStateScoringContextEq> uniqueUncachedHistories;

    for (auto& request : requests) {
        HiddenStateScoringContextRef historyPtr(dynamic_cast<const HiddenStateScoringContext*>(request.context.get()));
        if (not scoreCache_.contains(historyPtr)) {
            // Group by unique history
            uniqueUncachedHistories.emplace(historyPtr);
        }
    }

    std::vector<HiddenStateScoringContextRef> historyBatch;
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
        HiddenStateScoringContextRef history(dynamic_cast<const HiddenStateScoringContext*>(request.context.get()));
        result.scores.push_back(scoreCache_.get(history).at(request.nextToken));

        result.timesteps.push_back(history->labelSeq.size());
    }

    return result;
}

std::optional<LabelScorer::ScoreWithTime> StatefulOnnxLabelScorer::getScoreWithTime(LabelScorer::Request request) {
    auto result = getScoresWithTimes({request});
    if (not result.has_value()) {
        return {};
    }
    return ScoreWithTime{result->scores.front(), result->timesteps.front()};
}

HiddenStateRef StatefulOnnxLabelScorer::computeInitialHiddenState() {
    verify(not featuresMissing_);

    if (not initialHiddenState_) {  // initialHiddenState_ is still sentinel value -> compute it
        /*
         * Create session inputs
         */
        std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;

        if (encoderStatesValue_.empty()) {  // Create encoder onnx value only if it needs to be recomputed
            u32 T = inputBuffer_.size();
            u32 F = featureSize_;

            if (inputsAreContiguous_) {
                std::vector<int64_t> encoderStatesShape = {1l, T, F};

                encoderStatesValue_ = Onnx::Value::create(inputBuffer_.front().get(), encoderStatesShape);
            }
            else {
                std::vector<Math::FastMatrix<f32>> encoderStatesMats = {{F, T}};

                for (size_t t = 0ul; t < T; ++t) {
                    std::copy(inputBuffer_[t].get(), inputBuffer_[t].get() + F, &(encoderStatesMats.front().at(0, t)));  // Pointer to first element in column t
                }
                encoderStatesValue_ = Onnx::Value::create(encoderStatesMats, true);
            }

            Math::FastVector<s32> encoderStatesSize(1);
            encoderStatesSize.at(0) = T;
            encoderStatesSizeValue_ = Onnx::Value::create(encoderStatesSize);
        }

        if (initializerEncoderStatesName_ != "") {
            sessionInputs.emplace_back(initializerEncoderStatesName_, encoderStatesValue_);
        }
        if (initializerEncoderStatesSizeName_ != "") {
            sessionInputs.emplace_back(initializerEncoderStatesSizeName_, encoderStatesSizeValue_);
        }

        std::vector<std::string> sessionOutputNames = stateInitializerOnnxModel_.session.getAllOutputNames();

        /*
         * Run session
         */
        std::vector<Onnx::Value> sessionOutputs;
        stateInitializerOnnxModel_.session.run(std::move(sessionInputs), sessionOutputNames, sessionOutputs);

        /*
         * Return resulting hidden state
         */

        initialHiddenState_ = Core::ref(new HiddenState(std::move(sessionOutputNames), std::move(sessionOutputs)));
    }

    return initialHiddenState_;
}

HiddenStateRef StatefulOnnxLabelScorer::updatedHiddenState(HiddenStateRef hiddenState, LabelIndex nextToken) {
    /*
     * Create session inputs
     */
    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;

    sessionInputs.emplace_back(updaterEncoderStatesName_, encoderStatesValue_);
    if (updaterEncoderStatesSizeName_ != "") {
        sessionInputs.emplace_back(updaterEncoderStatesSizeName_, encoderStatesSizeValue_);
    }

    sessionInputs.emplace_back(updaterTokenName_, Onnx::Value::create(std::vector<s32>{static_cast<s32>(nextToken)}));

    for (const auto& inputName : stateUpdaterOnnxModel_.session.getAllInputNames()) {
        if (inputName == updaterEncoderStatesName_ or inputName == updaterEncoderStatesSizeName_ or inputName == updaterTokenName_) {
            continue;
        }
        sessionInputs.emplace_back(inputName, hiddenState->stateValueMap[updaterInputToStateNameMap_[inputName]]);
    }

    /*
     * Run session
     */
    std::vector<std::string> sessionOutputNames = stateUpdaterOnnxModel_.session.getAllOutputNames();
    std::vector<Onnx::Value> sessionOutputs;
    stateUpdaterOnnxModel_.session.run(std::move(sessionInputs), sessionOutputNames, sessionOutputs);

    /*
     * Return resulting hidden state
     */

    // Replace session output names with corresponding state names
    std::transform(sessionOutputNames.begin(), sessionOutputNames.end(), sessionOutputNames.begin(), [this](std::string outputName) { return updaterOutputToStateNameMap_[outputName]; });
    auto newHiddenState = Core::ref(new HiddenState(std::move(sessionOutputNames), std::move(sessionOutputs)));

    return newHiddenState;
}

void StatefulOnnxLabelScorer::forwardBatch(const std::vector<HiddenStateScoringContextRef> historyBatch) {
    if (historyBatch.empty()) {
        return;
    }

    /*
     * Create session inputs
     */
    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;

    for (auto& inputName : scorerOnnxModel_.session.getAllInputNames()) {
        // Collect a vector of individual state values of shape [1, *] and afterwards concatenate
        // them to a batched state tensor of shape [B, *]
        std::vector<const Onnx::Value*> stateValues;
        stateValues.reserve(historyBatch.size());
        for (size_t b = 0ul; b < historyBatch.size(); ++b) {
            auto history     = historyBatch[b];
            auto hiddenState = history->hiddenState;
            if (not hiddenState) {  // Sentinel hidden-state
                hiddenState = computeInitialHiddenState();
            }
            stateValues.push_back(&hiddenState->stateValueMap[scorerInputToStateNameMap_[inputName]]);
        }
        sessionInputs.emplace_back(inputName, Onnx::Value::concat(stateValues, 0));
    }

    /*
     * Run session
     */

    std::vector<Onnx::Value> sessionOutputs;
    scorerOnnxModel_.session.run(std::move(sessionInputs), {scoresName_}, sessionOutputs);

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
