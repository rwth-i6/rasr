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

#ifndef ONNX_ENCODER_STATES_HH
#define ONNX_ENCODER_STATES_HH

#include <functional>
#include <string>
#include <vector>

#include <Onnx/PersistentInput.hh>
#include <Onnx/Session.hh>
#include <Onnx/Value.hh>

#include "Types.hh"

namespace Nn {

/*
 * Helpers for ONNX label scorers whose model receives all encoder states of a segment
 * (e.g. AED decoders with cross-attention) and that can score contexts of several segments in one session run.
 */

// Stack encoder states of shape [1, T_b, F] (one per batch entry, several entries may point to the same value)
// to shape [B, max_b T_b, F]. Shorter encoder states are zero-padded at the end.
Onnx::Value stackEncoderStates(std::vector<Onnx::Value const*> const& encoderStates);

// Check whether the given input of the session can receive stacked encoder states, i.e. its batch dimension is not
// fixed to 1. Returns an empty string if it can, otherwise a description of the problem.
std::string checkStackableEncoderStatesInput(Onnx::Session const& session, std::string const& inputName);

/*
 * ONNX input names of the encoder states (empty if the model doesn't have the input)
 */
struct EncoderStatesInputNames {
    std::string states;  // Encoder states [1, T, E] or stacked [N, T_max, E]
    std::string size;    // Encoder states sizes [1] or [N]
    std::string index;   // Optional: row of the encoder states / sizes for each batch entry [B]
};

/*
 * Add the encoder states inputs for a batch of contexts to the session inputs. `segmentsBatch` contains the segment of
 * each batch entry, `segmentStates(segment)` / `segmentSize(segment)` return the encoder states value [1, T, E] /
 * size value [1] of a segment and `segmentLength(segment)` its number of encoder states T.
 *  - If all batch entries belong to the same segment, views of its values for `session` are passed (and the index is all zeros).
 *  - Otherwise the encoder states are stacked: with an index input once per segment ([S, T_max, E], sizes [S] and
 *    the index [B]), without index input once per batch entry ([B, T_max, E], sizes [B]).
 */
void addEncoderStatesInputs(EncoderStatesInputNames const&                          names,
                            Onnx::Session const&                                    session,
                            std::vector<SegmentId> const&                           segmentsBatch,
                            std::function<Onnx::PersistentInput&(SegmentId)> const& segmentStates,
                            std::function<Onnx::Value const&(SegmentId)> const&     segmentSize,
                            std::function<size_t(SegmentId)> const&                 segmentLength,
                            std::vector<std::pair<std::string, Onnx::Value>>&       inputs);

}  // namespace Nn

#endif  // ONNX_ENCODER_STATES_HH
