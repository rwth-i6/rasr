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

#pragma once

#include <string>
#include <vector>

#pragma push_macro("ensure")
#undef ensure
#include <torch/torch.h>
#pragma pop_macro("ensure")

namespace Torch {

// Description of one state entry from the parsed I/O spec
struct StateVariableSpec {
    std::string          name;                     // semantic state name
    std::string          kind;                     // state type, e.g. conv/mhsa
    int                  inputIndex  = -1;         // relative index in state input block
    int                  outputIndex = -1;         // relative index in state output block
    int                  layer       = -1;         // owning layer index
    std::vector<int64_t> shape;                    // expected tensor shape
    torch::ScalarType    dtype = torch::kFloat32;  // expected tensor dtype
};

// Parsed model I/O information loaded from the JSON manifest
struct ParsedIoSpec {
    bool valid = false;  // whether parsing succeeded

    int featuresInputIndex       = -1;  // absolute features input index
    int featuresLengthInputIndex = -1;  // absolute length input index
    int outputsIndex             = -1;  // absolute output index
    int outputLengthIndex        = -1;  // absolute out_len output index
    int statesInputIndex         = -1;  // absolute state input block index
    int statesOutputIndex        = -1;  // absolute state output block index

    std::vector<int64_t> featuresShape;  // shape of features input
    std::vector<int64_t> outputsShape;   // shape of log_probs output

    std::vector<StateVariableSpec> states;  // parsed state entries
};

ParsedIoSpec parseIoSpecJsonFile(const std::string& path);

}  // namespace Torch