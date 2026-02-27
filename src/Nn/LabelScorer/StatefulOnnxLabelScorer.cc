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
#include <Flow/Timestamp.hh>
#include <Math/FastMatrix.hh>
#include <Mm/Module.hh>
#include <Speech/Types.hh>

#include "ScoreAccessor.hh"
#include "ScoringContext.hh"

namespace Nn {

/*
 * =============================
 * == StatefulOnnxLabelScorer ==
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
        "Max number of hidden-states that can be fed into the scorer ONNX model at once.",
        Core::Type<int>::max);

const Core::ParameterInt StatefulOnnxLabelScorer::paramMaxCachedScores(
        "max-cached-score-vectors",
        "Maximum size of cache that maps scoring contexts to scores. This prevents memory overflow in case of very long audio segments.",
        1000);

// Scorer only takes hidden states as input which are not part of the IO spec
const std::vector<Onnx::IOSpecification> scorerModelIoSpec = {
        Onnx::IOSpecification{
                "scores",
                Onnx::IODirection::OUTPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -2}}}};  // [B, V]

const std::vector<Onnx::IOSpecification> stateInitializerModelIoSpec = {
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

const std::vector<Onnx::IOSpecification> stateUpdaterModelIoSpec = {
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
                "token",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{1}, {-1}}}};  // [1] or [B]

StatefulOnnxLabelScorer::StatefulOnnxLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::LM),
          blankUpdatesHistory_(paramBlankUpdatesHistory(config)),
          loopUpdatesHistory_(paramLoopUpdatesHistory(config)),
          maxBatchSize_(paramMaxBatchSize(config)),
          scorerOnnxModel_(select("scorer-model"), scorerModelIoSpec),
          stateInitializerOnnxModel_(select("state-initializer-model"), stateInitializerModelIoSpec),
          stateUpdaterOnnxModel_(select("state-updater-model"), stateUpdaterModelIoSpec),
          initialHiddenState_(),
          initializerOutputToStateNameMap_(),
          updaterInputToStateNameMap_(),
          updaterOutputToStateNameMap_(),
          scorerInputToStateNameMap_(),
          scorerScoresName_(scorerOnnxModel_.mapping.getOnnxName("scores")),
          initializerEncoderStatesName_(stateInitializerOnnxModel_.mapping.getOnnxName("encoder-states")),
          initializerEncoderStatesSizeName_(stateInitializerOnnxModel_.mapping.getOnnxName("encoder-states-size")),
          updaterEncoderStatesName_(stateUpdaterOnnxModel_.mapping.getOnnxName("encoder-states")),
          updaterEncoderStatesSizeName_(stateUpdaterOnnxModel_.mapping.getOnnxName("encoder-states-size")),
          updaterTokenName_(stateUpdaterOnnxModel_.mapping.getOnnxName("token")),
          encoderStatesValue_(),
          encoderStatesSizeValue_(),
          scoreCache_(paramMaxCachedScores(config)),
          stateCache_(scoreCache_.maxSize()) {
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
    if (initializerStateNames.empty()) {
        error() << "State initializer does not define any hidden states.";
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
    if (updaterOutputToStateNameMap_.empty()) {
        error() << "State updater does not produce any updated hidden states";
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
    if (scorerInputToStateNameMap_.empty()) {
        error() << "Scorer does not take any input";
    }
}

void StatefulOnnxLabelScorer::reset() {
    Precursor::reset();
    stateCache_.clear();
    scoreCache_.clear();
}

ScoringContextRef StatefulOnnxLabelScorer::getInitialScoringContext() {
    return Core::ref(new OnnxHiddenStateScoringContext());
}

void StatefulOnnxLabelScorer::addInput(DataView const& input) {
    Precursor::addInput(input);

    initialHiddenState_ = OnnxHiddenStateRef();

    if (not encoderStatesValue_.empty()) {  // Any previously computed hidden state values are outdated now so reset them
        encoderStatesValue_     = Onnx::Value();
        encoderStatesSizeValue_ = Onnx::Value();
    }
}

size_t StatefulOnnxLabelScorer::getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const {
    return 0u;
}

ScoringContextRef StatefulOnnxLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) {
    bool updateState = false;
    switch (transitionType) {
        case TransitionType::BLANK_LOOP:
            updateState = blankUpdatesHistory_ and loopUpdatesHistory_;
            break;
        case TransitionType::LABEL_TO_BLANK:
        case TransitionType::INITIAL_BLANK:
            updateState = blankUpdatesHistory_;
            break;
        case TransitionType::LABEL_LOOP:
            updateState = loopUpdatesHistory_;
            break;
        case TransitionType::BLANK_TO_LABEL:
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
    auto newScoringContext = Core::ref(new OnnxHiddenStateScoringContext(std::move(newLabelSeq), onnxHiddenStateScoringContext->hiddenState, true));

    auto hiddenState = stateCache_.get(newScoringContext);
    if (hiddenState) {
        newScoringContext->hiddenState      = *hiddenState;
        newScoringContext->requiresFinalize = false;
    }

    return newScoringContext;
}

std::vector<std::optional<ScoreAccessorRef>> StatefulOnnxLabelScorer::getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) {
    if (scoringContexts.empty()) {
        return {};
    }

    std::vector<std::optional<ScoreAccessorRef>> scoreAccessors(scoringContexts.size(), std::nullopt);

    if ((initializerEncoderStatesName_ != "" or initializerEncoderStatesSizeName_ != "" or updaterEncoderStatesName_ != "" or updaterEncoderStatesSizeName_ != "") and (expectMoreFeatures_ or bufferSize() == 0)) {
        // Only allow scoring once all encoder states have been passed
        return scoreAccessors;
    }

    /*
     * Identify unique scoring contexts that still need session runs
     */
    std::unordered_set<OnnxHiddenStateScoringContextRef, ScoringContextHash, ScoringContextEq> uniqueUncachedScoringContexts;

    for (auto const& scoringContext : scoringContexts) {
        // We need to finalize all scoring contexts before using them for scoring again.

        OnnxHiddenStateScoringContextRef onnxHiddenStateScoringContext(dynamic_cast<OnnxHiddenStateScoringContext const*>(scoringContext.get()));
        if (not scoreCache_.contains(onnxHiddenStateScoringContext)) {
            // Group by unique scoring context
            uniqueUncachedScoringContexts.emplace(scoringContext);
        }
    }

    /*
     * Fill state and score caches for all uncached scoring contexts
     */
    std::vector<OnnxHiddenStateScoringContextRef> scoringContextBatch;
    scoringContextBatch.reserve(std::min(uniqueUncachedScoringContexts.size(), maxBatchSize_));
    for (auto scoringContext : uniqueUncachedScoringContexts) {
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

    /*
     * Assign states from cache to scoring contexts and scores from cache to result vector
     */
    for (size_t contextIndex = 0ul; contextIndex < scoringContexts.size(); ++contextIndex) {
        OnnxHiddenStateScoringContextRef onnxHiddenStateScoringContext(dynamic_cast<OnnxHiddenStateScoringContext const*>(scoringContexts[contextIndex].get()));

        if (onnxHiddenStateScoringContext->requiresFinalize) {
            auto hiddenState = stateCache_.get(onnxHiddenStateScoringContext);
            verify(hiddenState);
            onnxHiddenStateScoringContext->hiddenState      = *hiddenState;
            onnxHiddenStateScoringContext->requiresFinalize = false;
        }

        verify(scoreCache_.contains(onnxHiddenStateScoringContext));
        auto const& scoreVec         = scoreCache_.get(onnxHiddenStateScoringContext)->get();
        auto const  timeframe        = onnxHiddenStateScoringContext->labelSeq.size();
        scoreAccessors[contextIndex] = Core::ref(new VectorScoreAccessor(scoreVec, timeframe));
    }

    return scoreAccessors;
}

std::optional<ScoreAccessorRef> StatefulOnnxLabelScorer::getScoreAccessor(ScoringContextRef scoringContext) {
    return getScoreAccessors({scoringContext})[0];
}

void StatefulOnnxLabelScorer::setupEncoderStatesValue() {
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

void StatefulOnnxLabelScorer::setupEncoderStatesSizeValue() {
    if (not encoderStatesSizeValue_.empty()) {
        return;
    }

    u32 T = bufferSize();

    encoderStatesSizeValue_ = Onnx::Value::create(std::vector<s32>{static_cast<s32>(T)});
}

OnnxHiddenStateRef StatefulOnnxLabelScorer::computeInitialHiddenState() {
    verify(not expectMoreFeatures_);

    if (not initialHiddenState_) {  // initialHiddenState_ is still sentinel value -> compute it
        /*
         * Create session inputs
         */
        std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;

        if (initializerEncoderStatesName_ != "") {
            setupEncoderStatesValue();
            sessionInputs.emplace_back(initializerEncoderStatesName_, encoderStatesValue_);
        }
        if (initializerEncoderStatesSizeName_ != "") {
            setupEncoderStatesSizeValue();
            sessionInputs.emplace_back(initializerEncoderStatesSizeName_, encoderStatesSizeValue_);
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

std::vector<OnnxHiddenStateRef> StatefulOnnxLabelScorer::updatedHiddenStates(std::vector<OnnxHiddenStateRef> const& hiddenStatesBatch, std::vector<s32> nextTokensBatch) {
    /*
     * Create session inputs
     */
    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;

    if (updaterEncoderStatesName_ != "") {
        setupEncoderStatesValue();
        sessionInputs.emplace_back(updaterEncoderStatesName_, encoderStatesValue_);
    }
    if (updaterEncoderStatesSizeName_ != "") {
        setupEncoderStatesSizeValue();
        sessionInputs.emplace_back(updaterEncoderStatesSizeName_, encoderStatesSizeValue_);
    }
    if (updaterTokenName_ != "") {
        sessionInputs.emplace_back(updaterTokenName_, Onnx::Value::create(nextTokensBatch));
    }

    for (auto const& [inputName, stateName] : updaterInputToStateNameMap_) {
        std::vector<Onnx::Value const*> stateValues;
        stateValues.reserve(hiddenStatesBatch.size());
        for (size_t b = 0ul; b < hiddenStatesBatch.size(); ++b) {
            stateValues.push_back(&hiddenStatesBatch[b]->stateValueMap.at(stateName));
        }
        sessionInputs.emplace_back(inputName, Onnx::Value::concat(stateValues, 0));
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
     * Return resulting hidden states
     */
    std::vector<OnnxHiddenStateRef> newHiddenStates;
    for (size_t b = 0ul; b < hiddenStatesBatch.size(); ++b) {
        OnnxHiddenStateRef       newHiddenState = Core::ref(new OnnxHiddenState());
        std::vector<Onnx::Value> stateValues;
        stateValues.reserve(sessionOutputs.size());
        for (size_t i = 0; i < sessionOutputs.size(); ++i) {
            stateValues.push_back(sessionOutputs[i].slice(b, b + 1, 0));
        }
        newHiddenStates.push_back(Core::ref(new OnnxHiddenState({stateNames.begin(), stateNames.end()}, std::move(stateValues))));
    }

    return newHiddenStates;
}

void StatefulOnnxLabelScorer::cacheStates(std::vector<OnnxHiddenStateScoringContextRef> const& scoringContextBatch) {
    std::vector<OnnxHiddenStateScoringContextRef> nonFinalizedContexts;
    std::vector<OnnxHiddenStateRef>               hiddenStates;
    std::vector<s32>                              nextTokens;
    for (auto const& scoringContext : scoringContextBatch) {
        if (not scoringContext->requiresFinalize) {
            continue;
        }
        nonFinalizedContexts.push_back(scoringContext);
        if (scoringContext->hiddenState) {
            hiddenStates.push_back(scoringContext->hiddenState);
        }
        else {
            hiddenStates.push_back(computeInitialHiddenState());
        }
        verify(not scoringContext->labelSeq.empty());
        nextTokens.push_back(scoringContext->labelSeq.back());
    }

    // If no scoring contexts need finalization, nothing has to be done
    if (nonFinalizedContexts.empty()) {
        return;
    }

    auto newHiddenStates = updatedHiddenStates(hiddenStates, nextTokens);
    verify(newHiddenStates.size() == nonFinalizedContexts.size());

    for (size_t i = 0ul; i < nonFinalizedContexts.size(); ++i) {
        stateCache_.put(nonFinalizedContexts[i], newHiddenStates[i]);
    }
}

void StatefulOnnxLabelScorer::cacheScores(std::vector<OnnxHiddenStateScoringContextRef> const& scoringContextBatch) {
    if (scoringContextBatch.empty()) {
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
        stateValues.reserve(scoringContextBatch.size());

        for (size_t b = 0ul; b < scoringContextBatch.size(); ++b) {
            auto const&        scoringContext = scoringContextBatch[b];
            OnnxHiddenStateRef hiddenState;
            if (scoringContext->labelSeq.empty()) {
                hiddenState = computeInitialHiddenState();
            }
            else {
                hiddenState = (*stateCache_.get(scoringContext)).get();
            }
            verify(hiddenState);
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
    for (size_t b = 0ul; b < scoringContextBatch.size(); ++b) {
        auto scoreVec = std::make_shared<std::vector<Score>>();
        sessionOutputs.front().get(b, *scoreVec);
        scoreCache_.put(scoringContextBatch[b], scoreVec);
    }
}

}  // namespace Nn
