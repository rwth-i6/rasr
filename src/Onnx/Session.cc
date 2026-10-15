/*
 * Copyright 2022 AppTek LLC. All rights reserved.
 */
#include "Session.hh"

#include <chrono>
#include <unordered_set>

#ifdef MODULE_CUDA
#include <cuda_runtime.h>
#endif

#include <Core/Application.hh>

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

const Core::ParameterBool Session::paramCudaUseTf32("cuda-use-tf32",
                                                    "Whether the CUDA execution provider may use TF32 for float matrix multiplications and convolutions"
                                                    " (onnxruntime default: yes). Faster, but results depend slightly on e.g. the batch size."
                                                    " Disabling it requires an onnxruntime version that knows the CUDA provider option `use_tf32`",
                                                    true);

Session::Session(Core::Configuration const& config)
        : Precursor(config),
          file_(paramFile(config)),
          intraOpNumThreads_(paramIntraOpNumThreads(config)),
          interOpNumThreads_(paramInterOpNumThreads(config)),
          statePrefix_(paramStatePrefix(config)),
          removePrefixFromKey_(paramRemovePrefixFromKey(config)),
          cudaUseTf32_(paramCudaUseTf32(config)),
          executionProviderType_(paramExecutionProviderType(config)),
          cudaDevice_(0),
          allocator_(),
          env_(ORT_LOGGING_LEVEL_WARNING),
          session_(nullptr),
          inputNameMap_(),
          outputNameMap_() {
    Ort::SessionOptions session_opts;
    session_opts.SetIntraOpNumThreads(intraOpNumThreads_);
    session_opts.SetInterOpNumThreads(interOpNumThreads_);

    auto providers = Ort::GetAvailableProviders();
    switch (executionProviderType_) {
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
            if (cudaGetDevice(&cudaDevice_) != cudaSuccess) {
                error() << "Could not get the current CUDA device.";
            }
            OrtCUDAProviderOptionsV2* cuda_opts = nullptr;
            Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_opts));
            // `use_tf32` is only passed if TF32 is disabled, so that the default also works with onnxruntime versions without this option
            std::string const device_id = std::to_string(cudaDevice_);
            char const*       keys[]    = {"device_id", "use_tf32"};
            char const*       values[]  = {device_id.c_str(), "0"};
            Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(cuda_opts, keys, values, cudaUseTf32_ ? 1 : 2));
            // All sessions compute on one stream, which other code writing device memory for the sessions can use as well
            Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptionsWithValue(cuda_opts, "user_compute_stream", sharedCudaStream()));
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

void* Session::sharedCudaStream() {
#ifdef MODULE_CUDA
    static cudaStream_t stream = []() {
        cudaStream_t s = nullptr;
        if (cudaStreamCreate(&s) != cudaSuccess) {
            Core::Application::us()->criticalError("Could not create the shared CUDA stream for the ONNX sessions");
        }
        return s;
    }();
    return stream;
#else
    return nullptr;
#endif
}

bool Session::run(std::vector<std::pair<std::string, Value>>&& inputs,
                  std::vector<std::string> const&              output_names,
                  std::vector<Value>&                          outputs,
                  std::vector<MemoryLocation> const&           output_locations) {
    verify(output_locations.empty() or output_locations.size() == output_names.size());

    std::unordered_set<std::string> deviceInputs;
    for (auto const& input : inputs) {
        if (input.second.isOnDevice()) {
            deviceInputs.insert(input.first);
        }
    }

    // Resolve DEFAULT: state outputs follow their state input, everything else goes to the host
    std::vector<MemoryLocation> locations(output_names.size(), MemoryLocation::HOST);
    bool                        anyDeviceOutput = false;
    for (size_t i = 0ul; i < output_names.size(); ++i) {
        auto location = output_locations.empty() ? MemoryLocation::DEFAULT : output_locations[i];
        if (location == MemoryLocation::DEFAULT) {
            auto iter = stateOutputToInput_.find(output_names[i]);
            location  = (iter != stateOutputToInput_.end() and deviceInputs.count(iter->second) > 0ul) ? MemoryLocation::DEVICE : MemoryLocation::HOST;
        }
        locations[i] = location;
        anyDeviceOutput |= location == MemoryLocation::DEVICE;
    }

    if (not deviceInputs.empty() or anyDeviceOutput) {
        return runWithBinding(inputs, output_names, outputs, locations);
    }
    return runPlain(std::move(inputs), output_names, outputs);
}

bool Session::run(std::vector<std::pair<std::string, Value>>&& inputs,
                  std::vector<std::string> const&              output_names,
                  std::vector<Value>&                          outputs) {
    return run(std::move(inputs), output_names, outputs, {});
}

bool Session::runWithBinding(std::vector<std::pair<std::string, Value>>& inputs,
                             std::vector<std::string> const&             output_names,
                             std::vector<Value>&                         outputs,
                             std::vector<MemoryLocation> const&          output_locations) {
#ifdef MODULE_CUDA
    // Configuration errors: the callers can't continue without the outputs
    if (executionProviderType_ != ExecutionProviderType::cuda) {
        criticalError() << "ONNX session inputs/outputs in device memory require the CUDA execution provider";
    }

    Ort::MemoryInfo hostMemoryInfo   = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::MemoryInfo deviceMemoryInfo = Ort::MemoryInfo("Cuda", OrtDeviceAllocator, cudaDevice_, OrtMemTypeDefault);

    std::vector<Value> results;
    try {
        Ort::IoBinding binding(session_);
        for (auto const& input : inputs) {
            binding.BindInput(input.first.c_str(), input.second.value_);
        }
        for (size_t i = 0ul; i < output_names.size(); ++i) {
            binding.BindOutput(output_names[i].c_str(), output_locations[i] == MemoryLocation::DEVICE ? deviceMemoryInfo : hostMemoryInfo);
        }

        Ort::RunOptions run_options;
        session_.Run(run_options, binding);
        binding.SynchronizeOutputs();

        auto values = binding.GetOutputValues();  // in the order of binding
        results.reserve(values.size());
        for (auto& value : values) {
            results.emplace_back(Value(std::move(value)));
        }
    }
    catch (Ort::Exception& e) {
        inputs.clear();  // like in `runPlain`, where the inputs are moved into the run
        warning() << "Exception during ONNX session run: " << e.what();
        return false;
    }

    // Free the inputs as early as the run without binding does
    inputs.clear();
    outputs = std::move(results);
    return true;
#else
    criticalError() << "ONNX session inputs/outputs in device memory require RASR compiled with MODULE_CUDA";
    return false;
#endif
}

bool Session::runPlain(std::vector<std::pair<std::string, Value>>&& inputs,
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
        warning() << "Exception during ONNX session run: " << e.what();
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

        stateOutputToInput_[state_variable.output_state_key] = state_variable.input_state_key;
        stateVariables_.push_back(state_variable);
    }
}

std::vector<OnnxStateVariable> const& Session::getStateVariablesMetadata() const {
    return stateVariables_;
}

}  // namespace Onnx
