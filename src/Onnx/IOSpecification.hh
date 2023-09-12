/*
 * Copyright 2022 AppTek LLC. All rights reserved.
 */
#ifndef _ONNX_IOSPECIFICATION_HH
#define _ONNX_IOSPECIFICATION_HH

#include "Session.hh"
#include "Value.hh"

#include <Core/Component.hh>

namespace Onnx {

enum class IODirection : int {
    INPUT  = 0,
    OUTPUT = 1,
};

struct IOSpecification {
    std::string                       name;
    IODirection                       ioDirection;
    bool                              optional;
    std::unordered_set<ValueType>     allowedTypes;
    std::unordered_set<ValueDataType> allowedDataTypes;
    std::vector<std::vector<int64_t>> allowedShapes;  // use -2 to match any size
};

class IOMapping : public Core::Component {
public:
    using Precursor = Core::Component;

    IOMapping(Core::Configuration const& config, std::vector<IOSpecification> const& io_spec);
    virtual ~IOMapping() = default;

    bool        hasOnnxName(std::string const& param) const;
    std::string getOnnxName(std::string const& param) const;

private:
    std::unordered_map<std::string, std::string> mapping_;
};

class IOValidator : public Core::Component {
public:
    using Precursor = Core::Component;

    static const Core::ParameterBool paramStrict;

    IOValidator(Core::Configuration const& config);
    virtual ~IOValidator() = default;

    bool validate(std::vector<IOSpecification> const& io_spec, IOMapping const& mapping, Session const& session);

private:
    bool strict_;

    void finding(std::string const& s);
};

// inline implementations

inline IOMapping::IOMapping(Core::Configuration const& config, std::vector<IOSpecification> const& io_spec)
        : Precursor(config),
          mapping_() {
    for (auto const& s : io_spec) {
        require(mapping_.find(s.name) == mapping_.end());

        Core::ParameterString param(s.name.c_str(), "onnx name", "");
        bool                  default_used = true;
        std::string           onnx_name    = param(config, &default_used);
        if (not default_used) {
            mapping_[s.name] = onnx_name;
        }
    }
}

inline bool IOMapping::hasOnnxName(std::string const& param) const {
    return mapping_.find(param) != mapping_.end();
}

inline std::string IOMapping::getOnnxName(std::string const& param) const {
    auto iter = mapping_.find(param);
    if (iter != mapping_.end()) {
        return iter->second;
    }
    return "";
}

inline IOValidator::IOValidator(Core::Configuration const& config)
        : Precursor(config), strict_(paramStrict(config)) {
}

}  // namespace Onnx

#endif  // _ONNX_IOSPECIFICATION_HH
