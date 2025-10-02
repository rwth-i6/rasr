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

#include "StatefulTransducerOnnxLabelScorer.hh"

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
 * == StatefulTransducerOnnxLabelScorer ==
 * =============================
 */

const Core::ParameterBool StatefulTransducerOnnxLabelScorer::paramBlankUpdatesHistory(
        "blank-updates-history",
        "Whether previously emitted blank labels should be used to update the history.",
        false);

const Core::ParameterBool StatefulTransducerOnnxLabelScorer::paramLoopUpdatesHistory(
        "loop-updates-history",
        "Whether in the case of loop transitions every repeated emission should be used to update the history.",
        false);

const Core::ParameterBool StatefulTransducerOnnxLabelScorer::paramVerticalLabelTransition(
        "vertical-label-transition",
        "Whether (non-blank) label transitions should be vertical, i.e. not increase the time step.",
        false);

const Core::ParameterInt StatefulTransducerOnnxLabelScorer::paramMaxBatchSize(
        "max-batch-size",
        "Max number of hidden-states that can be fed into the scorer ONNX model at once.",
        Core::Type<int>::max);

const Core::ParameterInt StatefulTransducerOnnxLabelScorer::paramMaxCachedScores(
        "max-cached-score-vectors",
        "Maximum size of cache that maps histories to scores. This prevents memory overflow in case of very long audio segments.",
        1000);

// Scorer only takes hidden states as input which are not part of the IO spec
const std::vector<Onnx::IOSpecification> scorerModelIoSpec = {
        Onnx::IOSpecification{
                "input-feature",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -2}, {1, -2}}},  // [1, E]
        Onnx::IOSpecification{
                "scores",
                Onnx::IODirection::OUTPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -2}}}};  // [B, V]

const std::vector<Onnx::IOSpecification> stateUpdaterModelIoSpec = {
        Onnx::IOSpecification{
                "token",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{1}, {-1}}}};  // [1] or [B]

StatefulTransducerOnnxLabelScorer::StatefulTransducerOnnxLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          blankUpdatesHistory_(paramBlankUpdatesHistory(config)),
          loopUpdatesHistory_(paramLoopUpdatesHistory(config)),
          verticalLabelTransition_(paramVerticalLabelTransition(config)),
          maxBatchSize_(paramMaxBatchSize(config)),
          scorerOnnxModel_(select("scorer-model"), scorerModelIoSpec),
          stateInitializerOnnxModel_(select("state-initializer-model"), {}),
          stateUpdaterOnnxModel_(select("state-updater-model"), stateUpdaterModelIoSpec),
          initialScoringContext_(),
          initializerOutputToStateNameMap_(),
          updaterInputToStateNameMap_(),
          updaterOutputToStateNameMap_(),
          scorerInputToStateNameMap_(),
          scorerInputFeatureName_(scorerOnnxModel_.mapping.getOnnxName("input-feature")),
          scorerScoresName_(scorerOnnxModel_.mapping.getOnnxName("scores")),
          updaterTokenName_(stateUpdaterOnnxModel_.mapping.getOnnxName("token")),
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
        error() << "Scorer does not take any input hidden-states";
    }
}

void StatefulTransducerOnnxLabelScorer::reset() {
    Precursor::reset();
    scoreCache_.clear();
}

Core::Ref<const ScoringContext> StatefulTransducerOnnxLabelScorer::getInitialScoringContext() {
    verify(not expectMoreFeatures_);

    if (not initialScoringContext_) {
        std::vector<std::string> sessionOutputNames;
        std::vector<std::string> stateNames;
        for (auto const& [outputName, stateName] : initializerOutputToStateNameMap_) {
            sessionOutputNames.push_back(outputName);
            stateNames.push_back(stateName);
        }

        std::vector<Onnx::Value> sessionOutputs;
        stateInitializerOnnxModel_.session.run({}, sessionOutputNames, sessionOutputs);

        auto initialHiddenState = Core::ref(new OnnxHiddenState(std::move(stateNames), std::move(sessionOutputs)));
        initialScoringContext_  = Core::ref(new StepOnnxHiddenStateScoringContext(0ul, std::vector<LabelIndex>(), initialHiddenState));
    }

    return initialScoringContext_;
}

Core::Ref<const ScoringContext> StatefulTransducerOnnxLabelScorer::extendedScoringContext(LabelScorer::Request const& request) {
    StepOnnxHiddenStateScoringContextRef scoringContext(dynamic_cast<const StepOnnxHiddenStateScoringContext*>(request.context.get()));

    bool   pushToken     = false;
    size_t timeIncrement = 0ul;
    switch (request.transitionType) {
        case LabelScorer::TransitionType::BLANK_LOOP:
            pushToken     = blankUpdatesHistory_ and loopUpdatesHistory_;
            timeIncrement = 1ul;
            break;
        case LabelScorer::TransitionType::LABEL_TO_BLANK:
        case LabelScorer::TransitionType::INITIAL_BLANK:
            pushToken     = blankUpdatesHistory_;
            timeIncrement = 1ul;
            break;
        case LabelScorer::TransitionType::LABEL_LOOP:
            pushToken     = loopUpdatesHistory_;
            timeIncrement = not verticalLabelTransition_;
            break;
        case LabelScorer::TransitionType::BLANK_TO_LABEL:
        case LabelScorer::TransitionType::LABEL_TO_LABEL:
        case LabelScorer::TransitionType::INITIAL_LABEL:
            pushToken     = true;
            timeIncrement = not verticalLabelTransition_;
            break;
        default:
            error() << "Unknown transition type " << request.transitionType;
    }

    // If scoringContext is not going to be modified, return the original one
    if (not pushToken and timeIncrement == 0ul) {
        return request.context;
    }

    std::vector<LabelIndex> newLabelSeq(scoringContext->labelSeq);
    auto                    newHiddenState   = scoringContext->hiddenState;
    bool                    requiresFinalize = false;

    if (pushToken) {
        newLabelSeq.push_back(request.nextToken);
        newHiddenState   = updatedHiddenState(scoringContext->hiddenState, request.nextToken);
        requiresFinalize = true;
    }

    auto newScoringContext              = Core::ref(new StepOnnxHiddenStateScoringContext(scoringContext->currentStep + timeIncrement, std::move(newLabelSeq), newHiddenState));
    newScoringContext->requiresFinalize = requiresFinalize;

    return newScoringContext;
}

std::optional<LabelScorer::ScoresWithTimes> StatefulTransducerOnnxLabelScorer::computeScoresWithTimes(std::vector<LabelScorer::Request> const& requests) {
    if (requests.empty()) {
        return ScoresWithTimes();
    }

    ScoresWithTimes result;
    result.scores.reserve(requests.size());

    /*
     * Collect all requests that are based on the same timestep (-> same input feature) and
     * group them together
     */
    std::unordered_map<size_t, std::vector<size_t>> requestsWithTimestep;  // Maps timestep to list of all indices of requests with that timestep

    for (size_t b = 0ul; b < requests.size(); ++b) {
        StepOnnxHiddenStateScoringContextRef context(dynamic_cast<const StepOnnxHiddenStateScoringContext*>(requests[b].context.get()));
        auto                                 step = context->currentStep;

        if (not getInput(step)) {
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
    for (auto const& [timestep, requestIndices] : requestsWithTimestep) {
        /*
         * Identify unique histories that still need session runs
         */
        std::unordered_set<StepOnnxHiddenStateScoringContextRef, ScoringContextHash, ScoringContextEq> uniqueUncachedHistories;

        for (auto requestIndex : requestIndices) {
            StepOnnxHiddenStateScoringContextRef scoringContextRef(dynamic_cast<const StepOnnxHiddenStateScoringContext*>(requests[requestIndex].context.get()));
            if (not scoreCache_.contains(scoringContextRef)) {
                // Group by unique scoringContext
                uniqueUncachedHistories.emplace(scoringContextRef);
            }
        }

        if (uniqueUncachedHistories.empty()) {
            continue;
        }

        std::vector<StepOnnxHiddenStateScoringContextRef> scoringContextBatch;
        scoringContextBatch.reserve(std::min(uniqueUncachedHistories.size(), maxBatchSize_));
        for (auto scoringContext : uniqueUncachedHistories) {
            scoringContextBatch.push_back(scoringContext);
            if (scoringContextBatch.size() == maxBatchSize_) {  // Batch is full -> forward now
                forwardBatch(scoringContextBatch);
                scoringContextBatch.clear();
            }
        }

        forwardBatch(scoringContextBatch);  // Forward remaining histories
    }

    /*
     * Assign from cache map to result vector
     */
    for (const auto& request : requests) {
        StepOnnxHiddenStateScoringContextRef scoringContext(dynamic_cast<const StepOnnxHiddenStateScoringContext*>(request.context.get()));

        verify(scoreCache_.contains(scoringContext));
        auto const& scores = scoreCache_.get(scoringContext)->get();

        result.scores.push_back(scores.at(request.nextToken));
        result.timeframes.push_back(scoringContext->labelSeq.size());
    }

    return result;
}

std::optional<LabelScorer::ScoreWithTime> StatefulTransducerOnnxLabelScorer::computeScoreWithTime(LabelScorer::Request const& request) {
    auto result = computeScoresWithTimes({request});
    if (not result) {
        return {};
    }
    return ScoreWithTime{result->scores.front(), result->timeframes.front()};
}

size_t StatefulTransducerOnnxLabelScorer::getMinActiveInputIndex(Core::CollapsedVector<ScoringContextRef> const& activeContexts) const {
    return 0u;
}

OnnxHiddenStateRef StatefulTransducerOnnxLabelScorer::updatedHiddenState(OnnxHiddenStateRef const& hiddenState, LabelIndex nextToken) {
    /*
     * Create session inputs
     */
    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;
    sessionInputs.emplace_back(updaterTokenName_, Onnx::Value::create(std::vector<s32>{static_cast<s32>(nextToken)}));

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

void StatefulTransducerOnnxLabelScorer::finalizeScoringContext(StepOnnxHiddenStateScoringContextRef const& scoringContext) {
    // If this scoring context does not need finalization, don't change it
    if (not scoringContext->requiresFinalize) {
        return;
    }

    verify(not scoringContext->labelSeq.empty());

    scoringContext->hiddenState      = updatedHiddenState(scoringContext->hiddenState, scoringContext->labelSeq.back());
    scoringContext->requiresFinalize = false;
}

void StatefulTransducerOnnxLabelScorer::forwardBatch(std::vector<StepOnnxHiddenStateScoringContextRef> const& scoringContextBatch) {
    if (scoringContextBatch.empty()) {
        return;
    }

    /*
     * Create session inputs
     */
    auto                 inputFeatureDataView = getInput(scoringContextBatch.front()->currentStep);
    f32 const*           inputFeatureData     = inputFeatureDataView->data();
    std::vector<int64_t> inputFeatureShape    = {1ul, static_cast<int64_t>(inputFeatureDataView->size())};

    std::vector<std::pair<std::string, Onnx::Value>> sessionInputs;
    sessionInputs.emplace_back(scorerInputFeatureName_, Onnx::Value::create(inputFeatureData, inputFeatureShape));

    for (auto const& [inputName, stateName] : scorerInputToStateNameMap_) {
        // Collect a vector of individual state values of shape [1, *] and afterwards concatenate
        // them to a batched state tensor of shape [B, *]
        std::vector<Onnx::Value const*> stateValues;
        stateValues.reserve(scoringContextBatch.size());

        for (size_t b = 0ul; b < scoringContextBatch.size(); ++b) {
            auto scoringContext = scoringContextBatch[b];
            auto hiddenState    = scoringContext->hiddenState;
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
        std::vector<f32> scoreVec;
        sessionOutputs.front().get(b, scoreVec);
        scoreCache_.put(scoringContextBatch[b], std::move(scoreVec));
    }
}

}  // namespace Nn
