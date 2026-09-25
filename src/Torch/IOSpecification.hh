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

#ifndef _TORCH_IOSPECIFICATION_HH
#define _TORCH_IOSPECIFICATION_HH

#include <Core/Component.hh>
#include "Session.hh"

#include <string>
#include <unordered_set>
#include <vector>

#pragma push_macro("ensure")
#undef ensure
#include <torch/torch.h>
#pragma pop_macro("ensure")

namespace Torch {

/*
 * Defines a positional I/O contract for exported Torch models.
 * Each IOSpecification entry describes one expected input or output.
 * IOValidator uses this specification to build a probe input vector and
 * validate the observed runtime I/O against the expected interface.
 */

enum class IODirection : int {
    INPUT  = 0,
    OUTPUT = 1,
};

// Describes one positional model input/output for validation
struct IOSpecification {
    std::string                         name;                // semantic name for logging/debugging
    size_t                              position;            // positional index in runtime I/O
    IODirection                         ioDirection;         // input or output
    bool                                optional;            // whether this entry may be absent
    std::unordered_set<c10::ScalarType> allowedScalarTypes;  // accepted tensor dtypes
    std::vector<int64_t>                allowedRanks;        // accepted tensor ranks
};

// Validates the runtime I/O contract of a Torch model
class IOValidator : public Core::Component {
public:
    using Precursor = Core::Component;

    static const Core::ParameterBool paramStrict;
    static const Core::ParameterInt  paramProbeFeatureDimension;
    static const Core::ParameterInt  paramProbeTimeDimension;

    IOValidator(Core::Configuration const& config);
    virtual ~IOValidator() = default;

    // Runs a small probe and checks it against the given specification
    bool validate(Session& session, std::vector<IOSpecification> const& ioSpec) const;

private:
    bool   strict_;
    size_t probeFeatureDimension_;
    size_t probeTimeDimension_;

    // Emits either an error or a warning, depending on strict mode
    void finding(std::string const& s) const;
};

}  // namespace Torch

#endif  // _TORCH_IOSPECIFICATION_HH