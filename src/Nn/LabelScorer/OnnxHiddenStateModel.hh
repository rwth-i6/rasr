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

#ifndef ONNX_HIDDEN_STATE_MODEL_HH
#define ONNX_HIDDEN_STATE_MODEL_HH

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Core/Component.hh>
#include <Core/Configuration.hh>
#include <Onnx/IOSpecification.hh>
#include <Onnx/Model.hh>
#include <Onnx/Value.hh>

#include "ModelCache.hh"
#include "ScoringContext.hh"
#include "Types.hh"

namespace Nn {

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
 * owner passes the corresponding values to `initialHiddenState`, `updatedHiddenStates` and `scores`
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
class OnnxHiddenStateModel : public Core::Component {
public:
    // Session inputs that are not hidden states and thus prepared by the owner
    using SessionInputs = std::vector<std::pair<std::string, Onnx::Value>>;

    // The models are read from the sub-configurations "state-initializer-model", "state-updater-model"
    // and "scorer-model" of the given config
    OnnxHiddenStateModel(Core::Configuration const&                config,
                         ModelCache&                               modelCache,
                         std::vector<Onnx::IOSpecification> const& stateInitializerIoSpec,
                         std::vector<Onnx::IOSpecification> const& stateUpdaterIoSpec,
                         std::vector<Onnx::IOSpecification> const& scorerIoSpec);

    // ONNX name that the model maps the given IO specification key to, or "" if it doesn't have that IO
    std::string stateInitializerOnnxName(std::string const& key) const;
    std::string stateUpdaterOnnxName(std::string const& key) const;
    std::string scorerOnnxName(std::string const& key) const;

    // Run the state initializer to obtain the hidden state for the first step
    OnnxHiddenStateRef initialHiddenState(SessionInputs extraInputs = {});

    // Run the state updater on a batch of hidden states to obtain one updated hidden state per batch entry
    std::vector<OnnxHiddenStateRef> updatedHiddenStates(std::vector<OnnxHiddenStateRef> const& hiddenStatesBatch,
                                                        SessionInputs                          extraInputs = {});

    // Run the scorer on a batch of hidden states to obtain one score vector per batch entry
    std::vector<std::shared_ptr<std::vector<Score>>> scores(std::vector<OnnxHiddenStateRef> const& hiddenStatesBatch,
                                                            SessionInputs                          extraInputs = {});

private:
    using StateNameMap = std::unordered_map<std::string, std::string>;

    // Fill the state name maps from the model metadata and check them for consistency
    void setupStateNameMaps();

    // Add one batched state input per entry of the given map by concatenating the corresponding
    // state values of all hidden states in the batch along the batch axis
    void addStateInputs(StateNameMap const&                    inputToStateNameMap,
                        std::vector<OnnxHiddenStateRef> const& hiddenStatesBatch,
                        SessionInputs&                         sessionInputs) const;

    // Split the given map into the output names to request from a session run and the state names
    // that the corresponding outputs belong to
    static void collectStateOutputs(StateNameMap const&       outputToStateNameMap,
                                    std::vector<std::string>& sessionOutputNames,
                                    std::vector<std::string>& stateNames);

    std::shared_ptr<Onnx::Model> stateInitializerOnnxModel_;
    std::shared_ptr<Onnx::Model> stateUpdaterOnnxModel_;
    std::shared_ptr<Onnx::Model> scorerOnnxModel_;

    // Map input/output names of the onnx models to hidden state names
    StateNameMap initializerOutputToStateNameMap_;
    StateNameMap updaterInputToStateNameMap_;
    StateNameMap updaterOutputToStateNameMap_;
    StateNameMap scorerInputToStateNameMap_;

    std::string scorerScoresName_;
};

}  // namespace Nn

#endif  // ONNX_HIDDEN_STATE_MODEL_HH
