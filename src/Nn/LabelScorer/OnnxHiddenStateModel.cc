/** Copyright 2026 RWTH Aachen University. All rights reserved.
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

#include "OnnxHiddenStateModel.hh"

#include <unordered_set>

#include <Core/Assertions.hh>
#include <Core/ReferenceCounting.hh>

namespace Nn {

// Every scorer model produces one score vector per batch entry. The rest of its IO is not part of
// the hidden-state convention and is declared by the owner.
static const Onnx::IOSpecification scorerScoresIoSpec{
        "scores",
        Onnx::IODirection::OUTPUT,
        false,
        {Onnx::ValueType::TENSOR},
        {Onnx::ValueDataType::FLOAT},
        {{-1, -2}}};  // [B, V]

OnnxHiddenStateModel::OnnxHiddenStateModel(Core::Configuration const&                config,
                                           ModelCache&                               modelCache,
                                           std::vector<Onnx::IOSpecification> const& stateInitializerIoSpec,
                                           std::vector<Onnx::IOSpecification> const& stateUpdaterIoSpec,
                                           std::vector<Onnx::IOSpecification> const& scorerIoSpec)
        : Core::Component(config),
          initializerOutputToStateNameMap_(),
          updaterInputToStateNameMap_(),
          updaterOutputToStateNameMap_(),
          scorerInputToStateNameMap_() {
    Core::Configuration initializerModelConfig(config, "state-initializer-model");
    Core::Configuration updaterModelConfig(config, "state-updater-model");
    Core::Configuration scorerModelConfig(config, "scorer-model");

    auto initializerKey = initializerModelConfig.getSelection();
    auto updaterKey     = updaterModelConfig.getSelection();
    auto scorerKey      = scorerModelConfig.getSelection();

    auto fullScorerIoSpec = scorerIoSpec;
    fullScorerIoSpec.push_back(scorerScoresIoSpec);

    stateInitializerOnnxModel_ = modelCache.getOrCreate<Onnx::Model>(initializerKey, initializerModelConfig, stateInitializerIoSpec);
    stateUpdaterOnnxModel_     = modelCache.getOrCreate<Onnx::Model>(updaterKey, updaterModelConfig, stateUpdaterIoSpec);
    scorerOnnxModel_           = modelCache.getOrCreate<Onnx::Model>(scorerKey, scorerModelConfig, fullScorerIoSpec);

    scorerScoresName_ = scorerOnnxModel_->mapping.getOnnxName("scores");

    setupStateNameMaps();
}

void OnnxHiddenStateModel::setupStateNameMaps() {
    auto initializerMetadataKeys = stateInitializerOnnxModel_->session.getCustomMetadataKeys();
    auto updaterMetadataKeys     = stateUpdaterOnnxModel_->session.getCustomMetadataKeys();
    auto scorerMetadataKeys      = scorerOnnxModel_->session.getCustomMetadataKeys();

    // Map state initializer outputs to states
    std::unordered_set<std::string> initializerStateNames;
    for (auto const& key : initializerMetadataKeys) {
        if (stateInitializerOnnxModel_->session.hasOutput(key)) {
            auto stateName = stateInitializerOnnxModel_->session.getCustomMetadata(key);
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
        if (stateUpdaterOnnxModel_->session.hasInput(key)) {
            auto stateName = stateUpdaterOnnxModel_->session.getCustomMetadata(key);
            if (initializerStateNames.find(stateName) == initializerStateNames.end()) {
                error() << "State updater input " << key << " associated with state " << stateName << " is not present in state initializer";
            }
            updaterInputToStateNameMap_.emplace(key, stateName);
        }
        if (stateUpdaterOnnxModel_->session.hasOutput(key)) {
            auto stateName = stateUpdaterOnnxModel_->session.getCustomMetadata(key);
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
        if (scorerOnnxModel_->session.hasInput(key)) {
            auto stateName = scorerOnnxModel_->session.getCustomMetadata(key);
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

std::string OnnxHiddenStateModel::stateInitializerOnnxName(std::string const& key) const {
    return stateInitializerOnnxModel_->mapping.getOnnxName(key);
}

std::string OnnxHiddenStateModel::stateUpdaterOnnxName(std::string const& key) const {
    return stateUpdaterOnnxModel_->mapping.getOnnxName(key);
}

std::string OnnxHiddenStateModel::scorerOnnxName(std::string const& key) const {
    return scorerOnnxModel_->mapping.getOnnxName(key);
}

void OnnxHiddenStateModel::addStateInputs(StateNameMap const&                    inputToStateNameMap,
                                          std::vector<OnnxHiddenStateRef> const& hiddenStatesBatch,
                                          SessionInputs&                         sessionInputs) const {
    for (auto const& [inputName, stateName] : inputToStateNameMap) {
        // Collect a vector of individual state values of shape [1, *] and afterwards concatenate
        // them to a batched state tensor of shape [B, *]
        std::vector<Onnx::Value const*> stateValues;
        stateValues.reserve(hiddenStatesBatch.size());

        for (auto const& hiddenState : hiddenStatesBatch) {
            verify(hiddenState);
            stateValues.push_back(&hiddenState->stateValueMap.at(stateName));
        }
        sessionInputs.emplace_back(inputName, Onnx::Value::concat(stateValues, 0));
    }
}

void OnnxHiddenStateModel::collectStateOutputs(StateNameMap const&       outputToStateNameMap,
                                               std::vector<std::string>& sessionOutputNames,
                                               std::vector<std::string>& stateNames) {
    sessionOutputNames.reserve(outputToStateNameMap.size());
    stateNames.reserve(outputToStateNameMap.size());
    for (auto const& [outputName, stateName] : outputToStateNameMap) {
        sessionOutputNames.push_back(outputName);
        stateNames.push_back(stateName);
    }
}

OnnxHiddenStateRef OnnxHiddenStateModel::initialHiddenState(SessionInputs extraInputs) {
    std::vector<std::string> sessionOutputNames;
    std::vector<std::string> stateNames;
    collectStateOutputs(initializerOutputToStateNameMap_, sessionOutputNames, stateNames);

    std::vector<Onnx::Value> sessionOutputs;
    stateInitializerOnnxModel_->session.run(std::move(extraInputs), sessionOutputNames, sessionOutputs);

    return Core::ref(new OnnxHiddenState(std::move(stateNames), std::move(sessionOutputs)));
}

std::vector<OnnxHiddenStateRef> OnnxHiddenStateModel::updatedHiddenStates(std::vector<OnnxHiddenStateRef> const& hiddenStatesBatch,
                                                                          SessionInputs                          extraInputs) {
    if (hiddenStatesBatch.empty()) {
        return {};
    }

    /*
     * Create session inputs
     */
    addStateInputs(updaterInputToStateNameMap_, hiddenStatesBatch, extraInputs);

    /*
     * Run session
     */
    std::vector<std::string> sessionOutputNames;
    std::vector<std::string> stateNames;
    collectStateOutputs(updaterOutputToStateNameMap_, sessionOutputNames, stateNames);

    std::vector<Onnx::Value> sessionOutputs;
    stateUpdaterOnnxModel_->session.run(std::move(extraInputs), sessionOutputNames, sessionOutputs);

    /*
     * Return resulting hidden states
     */
    std::vector<OnnxHiddenStateRef> newHiddenStates;
    newHiddenStates.reserve(hiddenStatesBatch.size());
    for (size_t b = 0ul; b < hiddenStatesBatch.size(); ++b) {
        std::vector<Onnx::Value> stateValues;
        stateValues.reserve(sessionOutputs.size());
        for (size_t i = 0ul; i < sessionOutputs.size(); ++i) {
            stateValues.push_back(sessionOutputs[i].slice(b, b + 1, 0));
        }
        newHiddenStates.push_back(Core::ref(new OnnxHiddenState({stateNames.begin(), stateNames.end()}, std::move(stateValues))));
    }

    return newHiddenStates;
}

std::vector<std::shared_ptr<std::vector<Score>>> OnnxHiddenStateModel::scores(std::vector<OnnxHiddenStateRef> const& hiddenStatesBatch,
                                                                              SessionInputs                          extraInputs) {
    if (hiddenStatesBatch.empty()) {
        return {};
    }

    /*
     * Create session inputs
     */
    addStateInputs(scorerInputToStateNameMap_, hiddenStatesBatch, extraInputs);

    /*
     * Run session
     */
    std::vector<Onnx::Value> sessionOutputs;
    scorerOnnxModel_->session.run(std::move(extraInputs), {scorerScoresName_}, sessionOutputs);

    /*
     * Return resulting score vectors
     */
    std::vector<std::shared_ptr<std::vector<Score>>> scoreVecs;
    scoreVecs.reserve(hiddenStatesBatch.size());
    for (size_t b = 0ul; b < hiddenStatesBatch.size(); ++b) {
        auto scoreVec = std::make_shared<std::vector<Score>>();
        sessionOutputs.front().get(b, *scoreVec);
        scoreVecs.push_back(scoreVec);
    }

    return scoreVecs;
}

}  // namespace Nn
