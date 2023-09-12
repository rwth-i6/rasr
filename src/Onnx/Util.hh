/*
 * Copyright 2022 AppTek LLC. All rights reserved.
 */
#ifndef _ONNX_UTIL_HH
#define _ONNX_UTIL_HH

#include <onnxruntime_cxx_api.h>

#include "Value.hh"

namespace Onnx {

namespace detail {

inline std::string OnnxTypeToString(ONNXType type) {
    switch (type) {
        case ONNX_TYPE_UNKNOWN:
            return "unknown";
        case ONNX_TYPE_TENSOR:
            return "tensor";
        case ONNX_TYPE_SEQUENCE:
            return "sequence";
        case ONNX_TYPE_MAP:
            return "map";
        case ONNX_TYPE_OPAQUE:
            return "opaque";
        case ONNX_TYPE_SPARSETENSOR:
            return "sparse-tensor";
        case ONNX_TYPE_OPTIONAL:
            return "optional";
        default:
            return "<unk>";
    };
}

inline std::string OnnxTensorElementDataTypeToString(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return "uint16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return "string";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "double";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return "uint32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return "uint64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: return "complex64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: return "complex128";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return "bfloat16";
        default: return "<unk>";
    };
}

inline std::string OnnxShapeToStringImpl(std::vector<int64_t> const& dims) {
    std::stringstream ss;
    for (size_t i = 0ul; i < dims.size(); i++) {
        if (i > 0) {
            ss << ",";
        }
        ss << dims[i];
    }
    return ss.str();
}

inline std::string OnnxShapeToString(Ort::TensorTypeAndShapeInfo const& info) {
    return OnnxShapeToStringImpl(info.GetShape());
}

#if ORT_API_VERSION >= 14
inline std::string OnnxShapeToString(Ort::ConstTensorTypeAndShapeInfo const& info) {
    return OnnxShapeToStringImpl(info.GetShape());
}
#endif

}  // namespace detail

inline std::string ValueTypeToString(ValueType vt) {
    if (vt == ValueType::EMPTY) {
        return "<empty>";
    }
    return detail::OnnxTypeToString(static_cast<ONNXType>(vt));
}

inline std::string ValueDataTypeToString(ValueDataType vdt) {
    if (vdt == ValueDataType::EMPTY) {
        return "empty";
    }
    return detail::OnnxTensorElementDataTypeToString(static_cast<ONNXTensorElementDataType>(vdt));
}

}  // namespace Onnx

#endif /* _ONNX_UTIL_HH */
