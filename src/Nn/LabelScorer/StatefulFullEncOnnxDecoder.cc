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

#include "StatefulFullEncOnnxDecoder.hh"
#include <Core/Assertions.hh>
#include <Core/ReferenceCounting.hh>
#include <Flow/Timestamp.hh>
#include <Math/FastMatrix.hh>
#include <Mm/Module.hh>
#include <Speech/Types.hh>
#include <cstddef>
#include <string>
#include <utility>
#include "LabelHistory.hh"
#include "LabelScorer.hh"

namespace Nn {

/*
 * =============================
 * == StatefulFullEncOnnxDecoder
 * =============================
 */

const Core::ParameterBool StatefulFullEncOnnxDecoder::paramBlankUpdatesHistory(
        "blank-updates-history",
        "Whether previously emitted blank labels should be used to update the history.",
        false);

const Core::ParameterBool StatefulFullEncOnnxDecoder::paramLoopUpdatesHistory(
        "loop-updates-history",
        "Whether in the case of loop transitions every repeated emission should be used to update the history.",
        false);

const Core::ParameterInt StatefulFullEncOnnxDecoder::paramMaxBatchSize(
        "max-batch-size",
        "Max number of hidden-states that can be fed into the ONNX model at once.",
        Core::Type<int>::max);

const Core::ParameterInt StatefulFullEncOnnxDecoder::paramMaxCachedScores(
        "max-cached-scores",
        "Maximum size of cache that maps histories to scores. This prevents memory overflow in case of very long audio segments.",
        1000);

StatefulFullEncOnnxDecoder::StatefulFullEncOnnxDecoder(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config),
          blankUpdatesHistory_(paramBlankUpdatesHistory(config)),
          loopUpdatesHistory_(paramLoopUpdatesHistory(config)),
          maxBatchSize_(paramMaxBatchSize(config)),
          hiddenStateVecSize_(0ul),
          hiddenStateMatSize_(0ul),
          decoderSession_(select("decoder-session")),
          decoderValidator_(select("decoder-validator")),
          decoderMapping_(select("decoder-io-map"), decoderIoSpec_),
          stateInitializerSession_(select("state-initializer-session")),
          stateInitializerValidator_(select("state-initializer-validator")),
          stateInitializerMapping_(select("state-initializer-io-map"), stateInitializerIoSpec_),
          stateUpdaterSession_(select("state-updater-session")),
          stateUpdaterValidator_(select("state-updater-validator")),
          stateUpdaterMapping_(select("state-updater-io-map"), stateUpdaterIoSpec_),
          scoresName_(decoderMapping_.getOnnxName("scores")),
          initEncoderStatesName_(stateInitializerMapping_.getOnnxName("encoder-states")),
          initEncoderSizeName_(stateInitializerMapping_.getOnnxName("encoder-states-size")),
          updaterEncoderStatesName_(stateUpdaterMapping_.getOnnxName("encoder-states")),
          updaterEncoderSizeName_(stateUpdaterMapping_.getOnnxName("encoder-states-size")),
          updaterTokenName_(stateUpdaterMapping_.getOnnxName("token")),
          encoderStatesValue_(),
          encoderStatesSizeValue_(),
          scoreCache_(paramMaxCachedScores(config)) {
    decoderValidator_.validate(decoderIoSpec_, decoderMapping_, decoderSession_);
    stateUpdaterValidator_.validate(stateUpdaterIoSpec_, stateUpdaterMapping_, stateUpdaterSession_);
}

const std::vector<Onnx::IOSpecification> StatefulFullEncOnnxDecoder::decoderIoSpec_ = {
        Onnx::IOSpecification{
                "scores",
                Onnx::IODirection::OUTPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -2}}}};

const std::vector<Onnx::IOSpecification> StatefulFullEncOnnxDecoder::stateInitializerIoSpec_ = {
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

const std::vector<Onnx::IOSpecification> StatefulFullEncOnnxDecoder::stateUpdaterIoSpec_ = {
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

void StatefulFullEncOnnxDecoder::reset() {
    Precursor::reset();
    scoreCache_.clear();
}

Core::Ref<const LabelHistory> StatefulFullEncOnnxDecoder::getStartHistory() {
    return Core::ref(new HiddenStateLabelHistory());  // Sentinel empty Ref as initial hidden state
}

Core::Ref<const LabelHistory> StatefulFullEncOnnxDecoder::extendedHistory(LabelScorer::Request request) {
    HiddenStateLabelHistoryRef history(dynamic_cast<const HiddenStateLabelHistory*>(request.history.get()));

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
        return request.history;
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

    return Core::ref(new HiddenStateLabelHistory(std::move(newLabelSeq), newHiddenState));
}

void StatefulFullEncOnnxDecoder::addEncoderOutput(FeatureVectorRef encoderOutput) {
    Precursor::addEncoderOutput(encoderOutput);
    initialHiddenState_ = HiddenStateRef();

    if (not encoderStatesValue_.empty()) {  // Computed ONNX value is outdated now so reset it
        encoderStatesValue_     = Onnx::Value();
        encoderStatesSizeValue_ = Onnx::Value();
    }
}

std::optional<std::pair<std::vector<Score>, Core::CollapsedVector<Speech::TimeframeIndex>>> StatefulFullEncOnnxDecoder::getScoresWithTime(const std::vector<LabelScorer::Request>& requests) {
    if (requests.empty()) {
        return {};
    }

    if (not segmentEnd_) {  // Only allow scoring once all encoder states have been passed
        return {};
    }

    if (encoderOutputBuffer_.empty()) {
        return {};
    }

    Core::CollapsedVector<Speech::TimeframeIndex> timeframeResults;

    /*
     * Identify unique histories that still need session runs
     */
    std::unordered_set<HiddenStateLabelHistoryRef, HiddenStateLabelHistoryHash, HiddenStateLabelHistoryEq> uniqueUncachedHistories;

    for (auto& request : requests) {
        HiddenStateLabelHistoryRef historyPtr(dynamic_cast<const HiddenStateLabelHistory*>(request.history.get()));
        if (not scoreCache_.contains(historyPtr)) {
            // Group by unique history
            uniqueUncachedHistories.emplace(historyPtr);
        }
    }

    std::vector<HiddenStateLabelHistoryRef> historyBatch;
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
    std::vector<Score> scoreResults;
    scoreResults.reserve(requests.size());
    for (const auto& request : requests) {
        HiddenStateLabelHistoryRef history(dynamic_cast<const HiddenStateLabelHistory*>(request.history.get()));
        scoreResults.push_back(scoreCache_.get(history).at(request.nextToken));

        timeframeResults.push_back(history->labelSeq.size());
    }

    return std::make_pair(scoreResults, timeframeResults);
}

std::optional<std::pair<Score, Speech::TimeframeIndex>> StatefulFullEncOnnxDecoder::getScoreWithTime(LabelScorer::Request request) {
    auto result = getScoresWithTime({request});
    if (not result.has_value()) {
        return {};
    }
    return std::make_pair(result->first.front(), result->second.front());
}

HiddenStateRef StatefulFullEncOnnxDecoder::computeInitialHiddenState() {
    verify(segmentEnd_);

    if (not initialHiddenState_) {  // initialHiddenState_ is still sentinel value -> compute it
        /*
         * Create session inputs
         */
        std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;

        if (encoderStatesValue_.empty()) {  // Create encoder onnx value only if it needs to be recomputed
            u32 T = encoderOutputBuffer_.size();
            u32 F = encoderOutputBuffer_.front()->size();

            std::vector<Math::FastMatrix<f32>> encoderStatesMats = {{F, T}};

            for (size_t t = 0ul; t < T; ++t) {
                std::copy(encoderOutputBuffer_[t]->begin(), encoderOutputBuffer_[t]->end(), &(encoderStatesMats.front().at(0, t)));  // Pointer to first element in column t
            }
            encoderStatesValue_ = Onnx::Value::create(encoderStatesMats, true);

            Math::FastVector<s32> encoderStatesSize(1);
            encoderStatesSize.at(0) = encoderStatesMats.front().nColumns();
            encoderStatesSizeValue_ = Onnx::Value::create(encoderStatesSize);
        }

        sessionInputs.emplace_back(initEncoderStatesName_, encoderStatesValue_);
        if (initEncoderSizeName_ != "") {
            sessionInputs.emplace_back(initEncoderSizeName_, encoderStatesSizeValue_);
        }

        std::vector<std::string> sessionOutputNames = stateInitializerSession_.getAllOutputNames();

        /*
         * Run session
         */
        std::vector<Onnx::Value> sessionOutputs;
        stateInitializerSession_.run(std::move(sessionInputs), sessionOutputNames, sessionOutputs);

        /*
         * Return resulting hidden state
         */

        initialHiddenState_ = Core::ref(new HiddenState(std::move(sessionOutputNames), std::move(sessionOutputs)));
    }

    return initialHiddenState_;
}

HiddenStateRef StatefulFullEncOnnxDecoder::updatedHiddenState(HiddenStateRef hiddenState, LabelIndex nextToken) {
    /*
     * Create session inputs
     */
    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;

    sessionInputs.emplace_back(updaterEncoderStatesName_, encoderStatesValue_);
    if (updaterEncoderSizeName_ != "") {
        sessionInputs.emplace_back(updaterEncoderSizeName_, encoderStatesSizeValue_);
    }

    sessionInputs.emplace_back(updaterTokenName_, Onnx::Value::create(std::vector<s32>{static_cast<s32>(nextToken)}));

    for (auto& name : stateUpdaterSession_.getAllInputNames()) {
        if (name == updaterEncoderStatesName_ or name == updaterEncoderSizeName_ or name == updaterTokenName_) {
            continue;
        }

        // Name duplication between input and output leads to suffix ".1",
        // e.g. input "lstm_state.1" corresponds to output "lstm_state"
        // So if the input name ends in ".1", remove the suffix for lookup of the value
        std::string stateName;
        if (name.compare(name.size() - 2, 2, ".1") == 0) {
            stateName = name.substr(0, name.size() - 2);
        }
        else {
            stateName = name;
        }
        auto findResult = hiddenState->stateValueMap.find(stateName);
        if (findResult == hiddenState->stateValueMap.end()) {
            error() << "State updater expects input " << name << " which corresponds to state name " << stateName << " but that is missing from the saved hidden state";
        }
        auto& state = findResult->second;
        sessionInputs.emplace_back(name, state);
    }

    /*
     * Run session
     */
    std::vector<std::string> sessionOutputNames = stateUpdaterSession_.getAllOutputNames();
    std::vector<Onnx::Value> sessionOutputs;
    stateUpdaterSession_.run(std::move(sessionInputs), sessionOutputNames, sessionOutputs);

    /*
     * Return resulting hidden state
     */

    auto newHiddenState = Core::ref(new HiddenState(std::move(sessionOutputNames), std::move(sessionOutputs)));

    return newHiddenState;
}

void StatefulFullEncOnnxDecoder::forwardBatch(const std::vector<HiddenStateLabelHistoryRef> historyBatch) {
    if (historyBatch.empty()) {
        return;
    }

    /*
     * Create session inputs
     */
    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;

    for (auto& name : decoderSession_.getAllInputNames()) {
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
            stateValues.push_back(&hiddenState->stateValueMap.at(name));
        }
        sessionInputs.emplace_back(name, Onnx::Value::concat(stateValues, 0));
    }

    /*
     * Run session
     */

    std::vector<Onnx::Value> sessionOutputs;
    decoderSession_.run(std::move(sessionInputs), {scoresName_}, sessionOutputs);

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
