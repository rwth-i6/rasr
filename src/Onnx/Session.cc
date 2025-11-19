/*
 * Copyright 2022 AppTek LLC. All rights reserved.
 */
#include "Session.hh"

#include <chrono>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#ifdef MODULE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef MODULE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef MODULE_CUDA
#include <cuda_runtime.h>
#endif

#include "Util.hh"

namespace Onnx {

const Core::ParameterString Session::paramFile("file",
                                               "path of the model to be loaded into the session",
                                               "");

const Core::ParameterInt Session::paramIntraOpNumThreads("intra-op-num-threads",
                                                         "number of threads to use within one op",
                                                         1);

const Core::ParameterInt Session::paramInterOpNumThreads("inter-op-num-threads",
                                                         "number of threads to use between ops",
                                                         1);

const Core::Choice Session::executionProviderChoice(
        "cpu", ExecutionProviderType::cpu,
        "cuda", ExecutionProviderType::cuda,
        Core::Choice::endMark());

const Core::ParameterChoice Session::paramExecutionProviderType(
        "execution-provider-type", &Session::executionProviderChoice, "type of execution provider", ExecutionProviderType::cpu);

const Core::ParameterString Session::paramStatePrefix("state-prefix",
                                                      "Prefix for the state keys in the metadata to distinguish from other metadata",
                                                      "STATE_");

const Core::ParameterBool Session::paramRemovePrefixFromKey("remove-prefix-from-key",
                                                            "Whether to remove the prefix from the state keys for the node name lookup",
                                                            true);

Session::Session(Core::Configuration const& config)
        : Precursor(config),
          file_(paramFile(config)),
          intraOpNumThreads_(paramIntraOpNumThreads(config)),
          interOpNumThreads_(paramInterOpNumThreads(config)),
          statePrefix_(paramStatePrefix(config)),
          removePrefixFromKey_(paramRemovePrefixFromKey(config)),
          allocator_(),
          env_(ORT_LOGGING_LEVEL_WARNING),
          session_(nullptr),
          inputNameMap_(),
          outputNameMap_() {
    Ort::SessionOptions session_opts;
    session_opts.SetIntraOpNumThreads(intraOpNumThreads_);
    session_opts.SetInterOpNumThreads(interOpNumThreads_);

    auto providers = Ort::GetAvailableProviders();
    switch (paramExecutionProviderType(config)) {
        case ExecutionProviderType::cpu: {
            if (std::find(providers.begin(), providers.end(), "CPUExecutionProvider") == providers.end()) {
                error() << "Requested CPU execution provider for ONNX session but it is not available.";
            }
            break;
        }
        case ExecutionProviderType::cuda: {
            if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") == providers.end()) {
                error() << "Requested CUDA execution provider for ONNX session but it is not available.";
            }
#ifdef MODULE_CUDA
            int deviceCount = 0;
            if (cudaGetDeviceCount(&deviceCount) != cudaSuccess or deviceCount == 0) {
                error() << "Requested CUDA execution provider but no CUDA device was found.";
            }
            OrtCUDAProviderOptionsV2* cuda_opts = nullptr;
            Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_opts));
            session_opts.AppendExecutionProvider_CUDA_V2(*cuda_opts);
            Ort::GetApi().ReleaseCUDAProviderOptions(cuda_opts);
            break;
#else
            error() << "Requested CUDA execution provider but RASR was not compiled with MODULE_CUDA which is required for it.";
#endif
        }
        default:
            error() << "Execution provider for ONNX session not known.";
    }

    session_ = Ort::Session(env_, file_.c_str(), session_opts);

    size_t num_inputs  = session_.GetInputCount();
    size_t num_outputs = session_.GetOutputCount();
    log("Created ONNX session for ") << file_ << " with " << num_inputs << " inputs and " << num_outputs << " outputs";

    std::stringstream ss;
    for (size_t i = 0ul; i < num_inputs; i++) {
        auto name                              = session_.GetInputNameAllocated(i, allocator_);
        auto type_info                         = session_.GetInputTypeInfo(i);
        inputNameMap_[std::string(name.get())] = i;
        ss << "input " << i << " : " << name.get() << " " << detail::OnnxTypeToString(type_info.GetONNXType());
        if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
            auto type_and_shape_info = type_info.GetTensorTypeAndShapeInfo();
            ss << "[" << detail::OnnxTensorElementDataTypeToString(type_and_shape_info.GetElementType()) << "]";
            ss << "(" << detail::OnnxShapeToString(type_and_shape_info) << ")";
        }
        ss << '\n';
    }
    for (size_t i = 0ul; i < num_outputs; i++) {
        auto name                               = session_.GetOutputNameAllocated(i, allocator_);
        auto type_info                          = session_.GetOutputTypeInfo(i);
        outputNameMap_[std::string(name.get())] = i;
        ss << "output " << i << " : " << name.get() << " " << detail::OnnxTypeToString(type_info.GetONNXType());
        if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
            auto type_and_shape_info = type_info.GetTensorTypeAndShapeInfo();
            ss << "[" << detail::OnnxTensorElementDataTypeToString(type_and_shape_info.GetElementType()) << "]";
            ss << "(" << detail::OnnxShapeToString(type_and_shape_info) << ")";
        }
        ss << '\n';
    }
    log("%s", ss.str().c_str());

    auto metadata = session_.GetModelMetadata();
    auto keys     = metadata.GetCustomMetadataMapKeysAllocated(allocator_);

    for (size_t i = 0ul; i < keys.size(); i++) {
        auto        value = metadata.LookupCustomMetadataMapAllocated(keys[i].get(), allocator_);
        std::string key   = std::string(keys[i].get());

        customMetadataKeys_.emplace_back(key);
        customMetadata_[key] = std::string(value.get());
    }

    initializeStateVariablesMetadata();
}

std::vector<std::string> Session::getAllInputNames() const {
    std::vector<std::string> result;
    result.reserve(inputNameMap_.size());
    for (auto& kv : inputNameMap_) {
        result.push_back(kv.first);
    }
    return result;
}

std::vector<std::string> Session::getAllOutputNames() const {
    std::vector<std::string> result;
    result.reserve(outputNameMap_.size());
    for (auto& kv : outputNameMap_) {
        result.push_back(kv.first);
    }
    return result;
}

bool Session::hasInput(std::string const& name) const {
    return inputNameMap_.find(name) != inputNameMap_.end();
}

bool Session::hasOutput(std::string const& name) const {
    return outputNameMap_.find(name) != outputNameMap_.end();
}

ValueType Session::getInputValueType(std::string const& name) const {
    auto iter = inputNameMap_.find(name);
    if (iter != inputNameMap_.end()) {
        return static_cast<ValueType>(session_.GetInputTypeInfo(iter->second).GetONNXType());
    }
    return ValueType::EMPTY;
}

ValueType Session::getOutputValueType(std::string const& name) const {
    auto iter = outputNameMap_.find(name);
    if (iter != outputNameMap_.end()) {
        return static_cast<ValueType>(session_.GetOutputTypeInfo(iter->second).GetONNXType());
    }
    return ValueType::EMPTY;
}

ValueDataType Session::getInputValueDataType(std::string const& name) const {
    auto iter = inputNameMap_.find(name);
    if (iter != inputNameMap_.end()) {
        auto     type_info = session_.GetInputTypeInfo(iter->second);
        ONNXType type      = type_info.GetONNXType();
        if (type == ONNX_TYPE_TENSOR) {
            return static_cast<ValueDataType>(type_info.GetTensorTypeAndShapeInfo().GetElementType());
        }
    }
    return ValueDataType::EMPTY;
}

ValueDataType Session::getOutputValueDataType(std::string const& name) const {
    auto iter = outputNameMap_.find(name);
    if (iter != outputNameMap_.end()) {
        auto     type_info = session_.GetOutputTypeInfo(iter->second);
        ONNXType type      = type_info.GetONNXType();
        if (type == ONNX_TYPE_TENSOR) {
            return static_cast<ValueDataType>(type_info.GetTensorTypeAndShapeInfo().GetElementType());
        }
    }
    return ValueDataType::EMPTY;
}

std::vector<int64_t> Session::getInputShape(std::string const& name) const {
    std::vector<int64_t> res;
    auto                 iter = inputNameMap_.find(name);
    if (iter != inputNameMap_.end()) {
        auto type_info = session_.GetInputTypeInfo(iter->second);
        if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
            auto type_and_shape_info = type_info.GetTensorTypeAndShapeInfo();
            res                      = type_and_shape_info.GetShape();
        }
    }
    return res;
}

std::vector<int64_t> Session::getOutputShape(std::string const& name) const {
    std::vector<int64_t> res;
    auto                 iter = outputNameMap_.find(name);
    if (iter != outputNameMap_.end()) {
        auto type_info = session_.GetOutputTypeInfo(iter->second);
        if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
            auto type_and_shape_info = type_info.GetTensorTypeAndShapeInfo();
            res                      = type_and_shape_info.GetShape();
        }
    }
    return res;
}

bool Session::run(std::vector<std::pair<std::string, Value>>&& inputs,
                  std::vector<std::string> const&              output_names,
                  std::vector<Value>&                          outputs) {
    Ort::RunOptions run_options;

    std::vector<char const*> input_names;
    std::vector<Ort::Value>  input_vals;
    for (auto&& input : inputs) {
        input_names.emplace_back(input.first.c_str());
        input_vals.emplace_back(std::move(input.second.value_));
    }

    std::vector<char const*> output_cnames;
    for (auto const& n : output_names) {
        output_cnames.emplace_back(n.c_str());
    }

    std::vector<Ort::Value> out_vals;
    try {
        out_vals = session_.Run(run_options, input_names.data(), input_vals.data(), inputs.size(), output_cnames.data(), output_cnames.size());
    }
    catch (Ort::Exception& e) {
        std::cerr << e.what() << std::endl;
        return false;
    }

    outputs.resize(out_vals.size());
    for (size_t i = 0ul; i < outputs.size(); i++) {
        outputs[i] = std::move(out_vals[i]);
    }

    return true;
}

std::string Session::getCustomMetadata(std::string const& key) const {
    std::string result = "";

    auto iter = customMetadata_.find(key);
    if (iter != customMetadata_.end()) {
        result = iter->second;
    }

    return result;
}

std::vector<std::string> const& Session::getCustomMetadataKeys() const {
    return customMetadataKeys_;
}

void Session::initializeStateVariablesMetadata() {
    for (std::string const& key : customMetadataKeys_) {
        auto state_pos = key.find(statePrefix_);

        if (state_pos != 0) {
            continue;
        }

        OnnxStateVariable state_variable;

        if (removePrefixFromKey_) {
            state_variable.input_state_key = key.substr(statePrefix_.size());
        }
        else {
            state_variable.input_state_key = key;
        }

        state_variable.output_state_key = getCustomMetadata(key);
        state_variable.shape            = getInputShape(state_variable.input_state_key);

        log("State: input_state_key=%s output_state_key=%s", state_variable.input_state_key.c_str(), state_variable.output_state_key.c_str());

        stateVariables_.push_back(state_variable);
    }
}

std::vector<OnnxStateVariable> const& Session::getStateVariablesMetadata() const {
    return stateVariables_;
}

}  // namespace Onnx
