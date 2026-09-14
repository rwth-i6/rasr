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

#include "IOSpecParser.hh"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <third_party/rapidjson/document.h>
#include <third_party/rapidjson/error/en.h>
#include <third_party/rapidjson/filereadstream.h>

namespace Torch {
namespace {

using rapidjson::Document;
using rapidjson::FileReadStream;
using rapidjson::Value;

torch::ScalarType scalarTypeFromString(const std::string& s) {
    static const std::unordered_map<std::string, torch::ScalarType> map = {
            {"torch.float16", torch::kFloat16},
            {"torch.float32", torch::kFloat32},
            {"torch.float64", torch::kFloat64},
            {"torch.bfloat16", torch::kBFloat16},
            {"torch.uint8", torch::kUInt8},
            {"torch.int8", torch::kInt8},
            {"torch.int16", torch::kInt16},
            {"torch.int32", torch::kInt32},
            {"torch.int64", torch::kInt64},
            {"torch.bool", torch::kBool},
    };

    auto it = map.find(s);
    if (it == map.end()) {
        throw std::runtime_error("Unsupported dtype string in io-spec-json: " + s);
    }
    return it->second;
}

const Value& requireObjectMember(const Value& obj, const char* name) {
    if (!obj.IsObject()) {
        throw std::runtime_error(std::string("Expected JSON object while looking for member '") + name + "'");
    }
    if (!obj.HasMember(name)) {
        throw std::runtime_error(std::string("Missing required JSON member '") + name + "'");
    }
    return obj[name];
}

std::string requireString(const Value& obj, const char* name) {
    const Value& v = requireObjectMember(obj, name);
    if (!v.IsString()) {
        throw std::runtime_error(std::string("Expected string in JSON member '") + name + "'");
    }
    return v.GetString();
}

int requireInt(const Value& obj, const char* name) {
    const Value& v = requireObjectMember(obj, name);
    if (!v.IsInt()) {
        throw std::runtime_error(std::string("Expected int in JSON member '") + name + "'");
    }
    return v.GetInt();
}

std::vector<int64_t> requireShape(const Value& obj, const char* name) {
    const Value& v = requireObjectMember(obj, name);
    if (!v.IsArray()) {
        throw std::runtime_error(std::string("Expected array in JSON member '") + name + "'");
    }

    std::vector<int64_t> shape;
    shape.reserve(v.Size());
    for (rapidjson::SizeType i = 0; i < v.Size(); ++i) {
        if (!v[i].IsInt64()) {
            throw std::runtime_error(std::string("Expected int64 in shape array '") + name + "'");
        }
        shape.push_back(v[i].GetInt64());
    }
    return shape;
}

void requireTensorEntry(const Value& parent, const char* name) {
    const Value& v = requireObjectMember(parent, name);
    if (!v.IsObject()) {
        throw std::runtime_error(std::string("Expected JSON object in member '") + name + "'");
    }
    (void)requireObjectMember(v, "index");
    (void)requireObjectMember(v, "shape");
    (void)requireObjectMember(v, "dtype");
}

}  // namespace

ParsedIoSpec parseIoSpecJsonFile(const std::string& path) {
    ParsedIoSpec ioSpec;

    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error("Could not open io-spec-json file: " + path);
    }

    std::unique_ptr<std::FILE, int (*)(std::FILE*)> fileGuard(fp, &std::fclose);

    char           buffer[65536];
    FileReadStream is(fp, buffer, sizeof(buffer));

    Document doc;
    doc.ParseStream(is);

    if (doc.HasParseError()) {
        throw std::runtime_error(
                "Failed to parse io-spec-json file '" + path + "' at offset " +
                std::to_string(doc.GetErrorOffset()) + ": " +
                rapidjson::GetParseError_En(doc.GetParseError()));
    }

    if (!doc.IsObject()) {
        throw std::runtime_error("Root of io-spec-json must be a JSON object");
    }

    const Value& inputs  = requireObjectMember(doc, "inputs");
    const Value& outputs = requireObjectMember(doc, "outputs");

    if (!inputs.IsObject() || !outputs.IsObject()) {
        throw std::runtime_error("'inputs' and 'outputs' must be JSON objects");
    }

    requireTensorEntry(inputs, "features");
    requireTensorEntry(inputs, "features_len");
    requireTensorEntry(outputs, "log_probs");

    ioSpec.featuresInputIndex       = requireInt(inputs["features"], "index");
    ioSpec.featuresLengthInputIndex = requireInt(inputs["features_len"], "index");
    ioSpec.outputsIndex             = requireInt(outputs["log_probs"], "index");

    ioSpec.featuresShape = requireShape(inputs["features"], "shape");
    ioSpec.outputsShape  = requireShape(outputs["log_probs"], "shape");

    if (outputs.HasMember("out_len")) {
        if (!outputs["out_len"].IsObject()) {
            throw std::runtime_error("Expected JSON object in member 'out_len'");
        }
        ioSpec.outputLengthIndex = requireInt(outputs["out_len"], "index");
    }

    const bool hasInputStates  = inputs.HasMember("states");
    const bool hasOutputStates = outputs.HasMember("states");

    if (hasInputStates != hasOutputStates) {
        throw std::runtime_error("io-spec-json must contain both inputs.states and outputs.states or neither");
    }

    if (hasInputStates) {
        const Value& inputStates  = inputs["states"];
        const Value& outputStates = outputs["states"];

        ioSpec.statesInputIndex  = requireInt(inputStates, "arg_index");
        ioSpec.statesOutputIndex = requireInt(outputStates, "output_index");

        const Value& inputEntries  = requireObjectMember(inputStates, "entries");
        const Value& outputEntries = requireObjectMember(outputStates, "entries");

        if (!inputEntries.IsArray() || !outputEntries.IsArray()) {
            throw std::runtime_error("inputs.states.entries and outputs.states.entries must be arrays");
        }

        std::unordered_map<int, const Value*> outputByIndex;
        for (rapidjson::SizeType i = 0; i < outputEntries.Size(); ++i) {
            const Value& outEntry = outputEntries[i];
            if (!outEntry.IsObject()) {
                throw std::runtime_error("Each outputs.states.entries element must be an object");
            }

            const int stateIndex = requireInt(outEntry, "state_index");
            outputByIndex.emplace(stateIndex, &outEntry);
        }

        ioSpec.states.clear();
        ioSpec.states.reserve(inputEntries.Size());

        for (rapidjson::SizeType i = 0; i < inputEntries.Size(); ++i) {
            const Value& inEntry = inputEntries[i];
            if (!inEntry.IsObject()) {
                throw std::runtime_error("Each inputs.states.entries element must be an object");
            }

            StateVariableSpec spec;
            spec.name       = requireString(inEntry, "name");
            spec.kind       = requireString(inEntry, "state_kind");
            spec.inputIndex = requireInt(inEntry, "state_index");
            spec.layer      = requireInt(inEntry, "layer");
            spec.shape      = requireShape(inEntry, "shape");
            spec.dtype      = scalarTypeFromString(requireString(inEntry, "dtype"));

            auto outIt = outputByIndex.find(spec.inputIndex);
            if (outIt == outputByIndex.end()) {
                throw std::runtime_error("Missing matching output state entry for state_index=" + std::to_string(spec.inputIndex));
            }

            const Value& outEntry = *outIt->second;
            spec.outputIndex      = requireInt(outEntry, "state_index");

            const std::vector<int64_t> outShape = requireShape(outEntry, "shape");
            const torch::ScalarType    outDtype = scalarTypeFromString(requireString(outEntry, "dtype"));

            if (outDtype != spec.dtype) {
                throw std::runtime_error("Input/output dtype mismatch for state_index=" + std::to_string(spec.inputIndex));
            }

            if (outShape.size() != spec.shape.size()) {
                throw std::runtime_error("Input/output rank mismatch for state_index=" + std::to_string(spec.inputIndex));
            }

            for (size_t dim = 0; dim < spec.shape.size(); ++dim) {
                if (dim == 1) {
                    // State time axis is dynamic and may differ
                    continue;
                }

                if (outShape[dim] != spec.shape[dim]) {
                    throw std::runtime_error(
                            "Input/output non-time dimension mismatch for state_index=" +
                            std::to_string(spec.inputIndex) +
                            " at dim=" + std::to_string(dim));
                }
            }

            ioSpec.states.push_back(std::move(spec));
        }
    }

    ioSpec.valid = true;
    return ioSpec;
}

}  // namespace Torch