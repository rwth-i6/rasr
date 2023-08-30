/*
 * Copyright 2022 AppTek LLC. All rights reserved.
 */
#include "IOSpecification.hh"

#include "Util.hh"

namespace {

std::string shape_to_string(std::vector<int64_t> shape) {
    std::stringstream ss;
    ss << '(';
    for (size_t i = 0ul; i < shape.size(); i++) {
        if (i > 0ul) {
            ss << ',';
        }
        if (shape[i] == -2) {
            ss << '*';
        }
        else {
            ss << shape[i];
        }
    }
    ss << ')';
    return ss.str();
}

bool match_shape(std::vector<int64_t> shape, std::vector<int64_t> ref_shape) {
    if (shape.size() != ref_shape.size()) {
        return false;
    }
    for (size_t i = 0ul; i < shape.size(); i++) {
        if (ref_shape[i] == -2) {  // ref-shape -2 implies any value is allowed
            continue;
        }
        if (ref_shape[i] != shape[i]) {
            return false;
        }
    }
    return true;
}

}  // namespace

namespace Onnx {

const Core::ParameterBool IOValidator::paramStrict("strict", "wether to emit an error or a warning upon validation failure", true);

bool IOValidator::validate(std::vector<IOSpecification> const& io_spec, IOMapping const& mapping, Session const& session) {
    bool success = true;

    for (auto const& s : io_spec) {
        if (not mapping.hasOnnxName(s.name)) {
            if (not s.optional) {
                success = false;
                finding(std::string("required input/output '") + s.name + "' is missing from mapping");
            }
            continue;
        }
        std::string onnx_name = mapping.getOnnxName(s.name);

        std::string          direction;
        bool                 missing;
        ValueType            vt;
        ValueDataType        vdt;
        std::vector<int64_t> shape;
        if (s.ioDirection == IODirection::INPUT) {
            direction = "input";
            missing   = not session.hasInput(onnx_name);
            vt        = session.getInputValueType(onnx_name);
            vdt       = session.getInputValueDataType(onnx_name);
            shape     = session.getInputShape(onnx_name);
        }
        else {
            direction = "output";
            missing   = not session.hasOutput(onnx_name);
            vt        = session.getOutputValueType(onnx_name);
            vdt       = session.getOutputValueDataType(onnx_name);
            shape     = session.getOutputShape(onnx_name);
        }

        if (missing) {
            success = false;
            finding(std::string("mapped value '") + onnx_name + "' for " + direction + " '" + s.name + "' does not exist within the session");
            continue;
        }

        if (s.allowedTypes.find(vt) == s.allowedTypes.end()) {
            std::stringstream err;
            err << "the type of '" << onnx_name << "' " << ValueTypeToString(vt) << " for " << direction << " '" << s.name
                << "' is not in the allowed list of types:";
            for (ValueType avt : s.allowedTypes) {
                err << ' ' << ValueTypeToString(avt);
            }
            success = false;
            finding(err.str());
        }

        if (s.allowedDataTypes.find(vdt) == s.allowedDataTypes.end()) {
            std::stringstream err;
            err << "the data-type of '" << onnx_name << "' " << ValueDataTypeToString(vdt) << " for " << direction << " '" << s.name
                << "' is not in the allowed list of data-types:";
            for (ValueDataType avdt : s.allowedDataTypes) {
                err << ' ' << ValueDataTypeToString(avdt);
            }
            success = false;
            finding(err.str());
        }

        bool shape_matched = false;
        for (auto const& as : s.allowedShapes) {
            shape_matched |= match_shape(shape, as);
        }
        if (not shape_matched) {
            std::stringstream err;
            err << "the shape of '" << onnx_name << " " << shape_to_string(shape) << "' for " << direction << " '" << s.name
                << "' is not in the allowed list of shapes:";
            for (auto const& as : s.allowedShapes) {
                err << ' ' << shape_to_string(as);
            }
            success = false;
            finding(err.str());
        }
    }

    return success;
}

void IOValidator::finding(std::string const& s) {
    if (strict_) {
        error("%s", s.c_str());
    }
    else {
        warning("%s", s.c_str());
    }
}

}  // namespace Onnx
