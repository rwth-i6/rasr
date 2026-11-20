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

#ifndef ONNX_INIT_UPDATE_SCORE_BUNDLE_HH
#define ONNX_INIT_UPDATE_SCORE_BUNDLE_HH

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Core/Component.hh>
#include <Core/Configuration.hh>
#include <Core/ReferenceCounting.hh>
#include <Core/StopWatch.hh>
#include <Nn/LabelScorer/ModelCache.hh>
#include <Nn/LabelScorer/Types.hh>

#include "IOSpecification.hh"
#include "Model.hh"
#include "Value.hh"

namespace Onnx {

/*
 * Hidden state represented by a dictionary of named ONNX values
 */
struct OnnxNamedValues : public Core::ReferenceCounted {
    std::unordered_map<std::string, Value> valueMap;

    OnnxNamedValues();
    OnnxNamedValues(std::vector<std::string>&& names, std::vector<Value>&& values);
};

typedef Core::Ref<OnnxNamedValues const> OnnxNamedValuesRef;

/*
 * Bundle of the three ONNX models that make up a stateful scoring pipeline together with the
 * mapping between their inputs/outputs and the hidden states that they consume and produce:
 *  - A State Initializer which produces the hidden states for the first step
 *  - A State Updater which produces updated hidden states based on the previous hidden states
 *  - A Scorer which computes scores based on the hidden states
 *
 * The hidden states can be any number of ONNX tensors of any shape and type.
 * Each ONNX model must have metadata that specifies the mapping of its input and output names to
 * the corresponding state names. These state names need to be consistent over all three models.
 *
 * For example:
 *   - The State Initializer has output called "lstm_c" and {"lstm_c": "LSTM_C"} in its metadata
 *   - The State Updater has input "lstm_c_in", output "lstm_c_out" and {"lstm_c_in": "LSTM_C", "lstm_c_out": "LSTM_C"} in its metadata
 *   - The Scorer has input "lstm_c" and {"lstm_c": "LSTM_C"} in its metadata
 * Here, "LSTM_C" is the state name and the same across all three models while the specific
 * input/output names are arbitrary.
 *
 * The State Initializer must have all states as output.
 * The State Updater must have a subset of states as input and all states as output.
 * The Scorer must have a subset of states as input.
 *
 * Any further model IO (input features, tokens, encoder states, ...) is not part of the hidden-state
 * convention and is declared by the owner via the IO specifications passed to the constructor. The
 * owner passes the corresponding values to `initialState`, `updatedStates` and `scores`
 * as extra session inputs. The `scores` output of the Scorer is added by this class itself and must
 * not be part of the Scorer IO specification.
 *
 * Batching is done by concatenating the per-context state values along the batch axis before a
 * session run and slicing the outputs apart again afterwards. This means that each context holds
 * its complete hidden state; states are never shared or sliced between contexts. Models whose state
 * requires more elaborate handling, e.g. a tree-like KV cache of a transformer model, are not
 * covered by this class and are instead handled by a RASR-side state manager
 * (see `StateManagedOnnxLabelScorer`).
 */
class OnnxInitUpdateScoreBundle : public Core::Component {
public:
    // Session inputs that are not hidden states and thus prepared by the owner
    using SessionInputs = std::vector<std::pair<std::string, Value>>;

    // The models are read from the sub-configurations "state-initializer-model", "state-updater-model"
    // and "scorer-model" of the given config
    OnnxInitUpdateScoreBundle(Core::Configuration const&          config,
                              Nn::ModelCache&                     modelCache,
                              std::vector<IOSpecification> const& stateInitializerIoSpec,
                              std::vector<IOSpecification> const& stateUpdaterIoSpec,
                              std::vector<IOSpecification> const& scorerIoSpec);

    // ONNX name that the model maps the given IO specification key to, or "" if it doesn't have that IO
    std::string stateInitializerOnnxName(std::string const& key) const;
    std::string stateUpdaterOnnxName(std::string const& key) const;
    std::string scorerOnnxName(std::string const& key) const;

    // Run the state initializer to obtain the hidden state for the first step
    OnnxNamedValuesRef initialState(SessionInputs extraInputs = {});

    // Run the state updater on a batch of hidden states to obtain one updated hidden state per batch entry
    std::vector<OnnxNamedValuesRef> updatedStates(std::vector<OnnxNamedValuesRef> const& hiddenStatesBatch,
                                                  SessionInputs                          extraInputs = {});

    // Run the scorer on a batch of hidden states to obtain one score vector per batch entry
    std::vector<std::shared_ptr<std::vector<Nn::Score>>> scores(std::vector<OnnxNamedValuesRef> const& hiddenStatesBatch,
                                                                SessionInputs                          extraInputs = {});

    // Time spent in the session runs of the individual models
    Core::StopWatch const& initializerSessionTime() const;
    Core::StopWatch const& updaterSessionTime() const;
    Core::StopWatch const& scorerSessionTime() const;

    // Time spent batching hidden states into session inputs and splitting session outputs back up
    Core::StopWatch const& marshallingTime() const;

    void resetTimers();

private:
    using StateNameMap = std::unordered_map<std::string, std::string>;

    // Fill the state name maps from the model metadata and check them for consistency
    void setupStateNameMaps();

    // Add one batched state input per entry of the given map by concatenating the corresponding
    // state values of all hidden states in the batch along the batch axis
    void addStateInputs(StateNameMap const&                    inputToStateNameMap,
                        std::vector<OnnxNamedValuesRef> const& hiddenStatesBatch,
                        SessionInputs&                         sessionInputs) const;

    // Split the given map into the output names to request from a session run and the state names
    // that the corresponding outputs belong to
    static void collectStateOutputs(StateNameMap const&       outputToStateNameMap,
                                    std::vector<std::string>& sessionOutputNames,
                                    std::vector<std::string>& stateNames);

    std::shared_ptr<Model> stateInitializerOnnxModel_;
    std::shared_ptr<Model> stateUpdaterOnnxModel_;
    std::shared_ptr<Model> scorerOnnxModel_;

    // Map input/output names of the onnx models to hidden state names
    StateNameMap initializerOutputToStateNameMap_;
    StateNameMap updaterInputToStateNameMap_;
    StateNameMap updaterOutputToStateNameMap_;
    StateNameMap scorerInputToStateNameMap_;

    std::string scorerScoresName_;

    Core::StopWatch initializerSessionTime_;
    Core::StopWatch updaterSessionTime_;
    Core::StopWatch scorerSessionTime_;
    Core::StopWatch marshallingTime_;
};

}  // namespace Onnx

#endif  // ONNX_INIT_UPDATE_SCORE_BUNDLE_HH
