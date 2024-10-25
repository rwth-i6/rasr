/*
 * Copyright 2022 AppTek LLC. All rights reserved.
 */
#ifndef _ONNX_SESSION_HH
#define _ONNX_SESSION_HH

#include <memory>

#include <onnxruntime_cxx_api.h>

#include <Core/Component.hh>

#include "Value.hh"

namespace Onnx {

class Session : public Core::Component {
public:
    using Precursor = Core::Component;

    static const Core::ParameterString paramFile;
    static const Core::ParameterInt    paramIntraOpNumThreads;
    static const Core::ParameterInt    paramInterOpNumThreads;

    Session(Core::Configuration const& config);
    virtual ~Session() = default;

    std::vector<std::string> getAllInputNames() const;
    std::vector<std::string> getAllOutputNames() const;
    bool                     hasInput(std::string const& name) const;
    bool                     hasOutput(std::string const& name) const;
    ValueType                getInputValueType(std::string const& name) const;
    ValueType                getOutputValueType(std::string const& name) const;
    ValueDataType            getInputValueDataType(std::string const& name) const;
    ValueDataType            getOutputValueDataType(std::string const& name) const;
    std::vector<int64_t>     getInputShape(std::string const& name) const;
    std::vector<int64_t>     getOutputShape(std::string const& name) const;

    bool run(std::vector<std::pair<std::string, Value>>&& inputs,
             std::vector<std::string> const&              output_names,
             std::vector<Value>&                          outputs);

    std::string                     getCustomMetadata(std::string const& key) const;
    std::vector<std::string> const& getCustomMetadataKeys() const;

private:
    const std::string file_;
    const size_t      intraOpNumThreads_;
    const size_t      interOpNumThreads_;

    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::Env                         env_;
    Ort::Session                     session_;

    std::unordered_map<std::string, size_t> inputNameMap_;
    std::unordered_map<std::string, size_t> outputNameMap_;

    std::unordered_map<std::string, std::string> customMetadata_;
    std::vector<std::string>                     customMetadataKeys_;
};

}  // namespace Onnx

#endif  // _ONNX_SESSION_HH
