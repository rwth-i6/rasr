/*
 * Copyright 2022 AppTek LLC. All rights reserved.
 */
#include "Session.hh"

#include <chrono>

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

Session::Session(Core::Configuration const& config)
        : Precursor(config),
          file_(paramFile(config)),
          intraOpNumThreads_(paramIntraOpNumThreads(config)),
          interOpNumThreads_(paramInterOpNumThreads(config)),
          allocator_(),
          env_(ORT_LOGGING_LEVEL_WARNING),
          session_(nullptr),
          inputNameMap_(),
          outputNameMap_() {
    Ort::SessionOptions session_opts;
    session_opts.SetIntraOpNumThreads(intraOpNumThreads_);
    session_opts.SetInterOpNumThreads(interOpNumThreads_);
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
}

bool Session::hasInput(std::string const& name) const {
    return inputNameMap_.find(name) != inputNameMap_.end();
}

size_t Session::numInputs() const {
    return inputNameMap_.size();
}

bool Session::hasOutput(std::string const& name) const {
    return outputNameMap_.find(name) != outputNameMap_.end();
}

size_t Session::numOutputs() const {
    return outputNameMap_.size();
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

}  // namespace Onnx
