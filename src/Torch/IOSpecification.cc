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

#include "IOSpecification.hh"

#include <sstream>

namespace {

std::string scalarTypeToString(c10::ScalarType type) {
    return c10::toString(type);
}

std::string ranksToString(std::vector<int64_t> const& ranks) {
    std::stringstream ss;
    for (size_t i = 0; i < ranks.size(); ++i) {
        if (i > 0)
            ss << ", ";
        ss << ranks[i];
    }
    return ss.str();
}

const Torch::IOSpecification* findRequiredSpec(
        std::vector<Torch::IOSpecification> const& ioSpec,
        std::string const&                         name,
        Torch::IODirection                         direction) {
    for (auto const& spec : ioSpec) {
        if (spec.name == name && spec.ioDirection == direction && !spec.optional) {
            return &spec;
        }
    }
    return nullptr;
}

size_t maxRequiredInputPosition(std::vector<Torch::IOSpecification> const& ioSpec) {
    size_t maxPos = 0;
    bool   found  = false;
    for (auto const& spec : ioSpec) {
        if (spec.ioDirection == Torch::IODirection::INPUT && !spec.optional) {
            maxPos = found ? std::max(maxPos, spec.position) : spec.position;
            found  = true;
        }
    }
    return found ? maxPos : 0;
}

}  // namespace

namespace Torch {

const Core::ParameterBool IOValidator::paramStrict(
        "strict",
        "whether to emit an error or warning upon validation failure",
        true);

const Core::ParameterInt IOValidator::paramProbeFeatureDimension(
        "probe-feature-dimension",
        "feature dimension used for model I/O probing",
        80);

const Core::ParameterInt IOValidator::paramProbeTimeDimension(
        "probe-time-dimension",
        "time dimension used for model I/O probing",
        20);

IOValidator::IOValidator(Core::Configuration const& config)
        : Precursor(config),
          strict_(paramStrict(config)),
          probeFeatureDimension_(paramProbeFeatureDimension(config)),
          probeTimeDimension_(paramProbeTimeDimension(config)) {
}

bool IOValidator::validate(Session& session, std::vector<IOSpecification> const& ioSpec) const {
    auto const* featureSpec = findRequiredSpec(ioSpec, "features", IODirection::INPUT);
    auto const* lengthSpec  = findRequiredSpec(ioSpec, "lengths", IODirection::INPUT);

    if (!featureSpec) {
        finding("Required input specification 'features' is missing");
        return false;
    }
    if (!lengthSpec) {
        finding("Required input specification 'lengths' is missing");
        return false;
    }

    std::vector<torch::Tensor> inputs(maxRequiredInputPosition(ioSpec) + 1);

    torch::Tensor features = torch::zeros(
            {1, static_cast<long>(probeTimeDimension_), static_cast<long>(probeFeatureDimension_)},
            torch::kFloat32);
    torch::Tensor lengths = torch::tensor(
            {static_cast<long>(probeTimeDimension_)},
            torch::kInt64);

    inputs[featureSpec->position] = features;
    inputs[lengthSpec->position]  = lengths;

    std::vector<torch::Tensor> outputs;
    try {
        session.run(inputs, outputs);
    }
    catch (std::exception const& e) {
        finding(std::string("Session run failed during IO validation: ") + e.what());
        return false;
    }

    bool success = true;

    for (auto const& s : ioSpec) {
        torch::Tensor value;

        if (s.ioDirection == IODirection::INPUT) {
            if (s.position >= inputs.size()) {
                if (!s.optional) {
                    std::stringstream err;
                    err << "Required input '" << s.name << "' at position " << s.position << " is missing";
                    finding(err.str());
                    success = false;
                }
                continue;
            }
            value = inputs[s.position];
        }
        else {
            if (s.position >= outputs.size()) {
                if (!s.optional) {
                    std::stringstream err;
                    err << "Required output '" << s.name << "' at position " << s.position << " is missing";
                    finding(err.str());
                    success = false;
                }
                continue;
            }
            value = outputs[s.position];
        }

        if (!value.defined()) {
            if (!s.optional) {
                std::stringstream err;
                err << (s.ioDirection == IODirection::INPUT ? "Input" : "Output")
                    << " '" << s.name << "' at position " << s.position << " is undefined";
                finding(err.str());
                success = false;
            }
            continue;
        }

        if (s.allowedScalarTypes.find(value.scalar_type()) == s.allowedScalarTypes.end()) {
            std::stringstream err;
            err << (s.ioDirection == IODirection::INPUT ? "Input" : "Output")
                << " '" << s.name << "' at position " << s.position
                << " has scalar type " << scalarTypeToString(value.scalar_type())
                << ", allowed types are: ";
            bool first = true;
            for (auto t : s.allowedScalarTypes) {
                if (!first)
                    err << ", ";
                err << scalarTypeToString(t);
                first = false;
            }
            finding(err.str());
            success = false;
        }

        const int64_t rank        = value.dim();
        bool          rankMatched = false;
        for (auto allowedRank : s.allowedRanks) {
            if (rank == allowedRank) {
                rankMatched = true;
                break;
            }
        }

        if (!rankMatched) {
            std::stringstream err;
            err << (s.ioDirection == IODirection::INPUT ? "Input" : "Output")
                << " '" << s.name << "' at position " << s.position
                << " has rank " << rank
                << ", allowed ranks are: " << ranksToString(s.allowedRanks);
            finding(err.str());
            success = false;
        }
    }

    return success;
}

void IOValidator::finding(std::string const& s) const {
    if (strict_) {
        error("%s", s.c_str());
    }
    else {
        warning("%s", s.c_str());
    }
}

}  // namespace Torch