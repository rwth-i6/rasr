/*
 * Copyright 2022 AppTek LLC. All rights reserved.
 */
#include "Value.hh"

#include "Util.hh"

namespace {

template<typename T>
struct ToDataType {
    static constexpr ONNXTensorElementDataType onnx_tensor_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
};

#define DEFINE_ONNX_TENSOR_TYPE_MAPING(TYPE, ENUM)                                  \
    template<>                                                                      \
    struct ToDataType<TYPE> {                                                       \
        static constexpr ONNXTensorElementDataType onnx_tensor_element_type = ENUM; \
    };

DEFINE_ONNX_TENSOR_TYPE_MAPING(float, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
DEFINE_ONNX_TENSOR_TYPE_MAPING(uint8_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
DEFINE_ONNX_TENSOR_TYPE_MAPING(int8_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)
DEFINE_ONNX_TENSOR_TYPE_MAPING(uint16_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)
DEFINE_ONNX_TENSOR_TYPE_MAPING(int16_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16)
DEFINE_ONNX_TENSOR_TYPE_MAPING(int32_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
DEFINE_ONNX_TENSOR_TYPE_MAPING(int64_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
DEFINE_ONNX_TENSOR_TYPE_MAPING(std::string, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)
DEFINE_ONNX_TENSOR_TYPE_MAPING(bool, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)
DEFINE_ONNX_TENSOR_TYPE_MAPING(Ort::Float16_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
DEFINE_ONNX_TENSOR_TYPE_MAPING(double, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
DEFINE_ONNX_TENSOR_TYPE_MAPING(uint32_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32)
DEFINE_ONNX_TENSOR_TYPE_MAPING(uint64_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64)
DEFINE_ONNX_TENSOR_TYPE_MAPING(std::complex<float>, ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64)
DEFINE_ONNX_TENSOR_TYPE_MAPING(std::complex<double>, ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128)
DEFINE_ONNX_TENSOR_TYPE_MAPING(Ort::BFloat16_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)

}  // namespace

namespace Onnx {

template<typename T>
Value Value::zeros(std::initializer_list<int64_t> dim) {
    Ort::AllocatorWithDefaultOptions allocator;

    int64_t total_size = std::accumulate(dim.begin(), dim.end(), 1l, [](int64_t a, int64_t b) { return a * b; });

    Value res;
    res.value_ = Ort::Value::CreateTensor<T>(allocator, &(*dim.begin()), dim.size());
    T* data    = res.value_.GetTensorMutableData<T>();
    for (int64_t i = 0ul; i < total_size; i++) {
        data[i] = T(0);
    }

    return res;
}

template Value Value::zeros<f32>(std::initializer_list<int64_t> dim);
template Value Value::zeros<f64>(std::initializer_list<int64_t> dim);
template Value Value::zeros<s64>(std::initializer_list<int64_t> dim);
template Value Value::zeros<u64>(std::initializer_list<int64_t> dim);
template Value Value::zeros<s32>(std::initializer_list<int64_t> dim);
template Value Value::zeros<u32>(std::initializer_list<int64_t> dim);
template Value Value::zeros<s16>(std::initializer_list<int64_t> dim);
template Value Value::zeros<u16>(std::initializer_list<int64_t> dim);
template Value Value::zeros<s8>(std::initializer_list<int64_t> dim);
template Value Value::zeros<u8>(std::initializer_list<int64_t> dim);

template<typename T>
Value Value::zeros(std::vector<int64_t> const& dim) {
    Ort::AllocatorWithDefaultOptions allocator;

    int64_t total_size = std::accumulate(dim.begin(), dim.end(), 1l, [](int64_t a, int64_t b) { return a * b; });

    Value res;
    res.value_ = Ort::Value::CreateTensor<T>(allocator, &(*dim.begin()), dim.size());
    T* data    = res.value_.GetTensorMutableData<T>();
    for (int64_t i = 0ul; i < total_size; i++) {
        data[i] = T(0);
    }

    return res;
}

template Value Value::zeros<f32>(std::vector<int64_t> const& dim);
template Value Value::zeros<f64>(std::vector<int64_t> const& dim);
template Value Value::zeros<s64>(std::vector<int64_t> const& dim);
template Value Value::zeros<u64>(std::vector<int64_t> const& dim);
template Value Value::zeros<s32>(std::vector<int64_t> const& dim);
template Value Value::zeros<u32>(std::vector<int64_t> const& dim);
template Value Value::zeros<s16>(std::vector<int64_t> const& dim);
template Value Value::zeros<u16>(std::vector<int64_t> const& dim);
template Value Value::zeros<s8>(std::vector<int64_t> const& dim);
template Value Value::zeros<u8>(std::vector<int64_t> const& dim);

template<typename T>
Value Value::createEmpty(std::initializer_list<int64_t> dim) {
    Ort::AllocatorWithDefaultOptions allocator;

    Value res;
    res.value_ = Ort::Value::CreateTensor<T>(allocator, &(*dim.begin()), dim.size());

    return res;
}

template Value Value::createEmpty<f32>(std::initializer_list<int64_t> dim);
template Value Value::createEmpty<f64>(std::initializer_list<int64_t> dim);
template Value Value::createEmpty<s64>(std::initializer_list<int64_t> dim);
template Value Value::createEmpty<u64>(std::initializer_list<int64_t> dim);
template Value Value::createEmpty<s32>(std::initializer_list<int64_t> dim);
template Value Value::createEmpty<u32>(std::initializer_list<int64_t> dim);
template Value Value::createEmpty<s16>(std::initializer_list<int64_t> dim);
template Value Value::createEmpty<u16>(std::initializer_list<int64_t> dim);
template Value Value::createEmpty<s8>(std::initializer_list<int64_t> dim);
template Value Value::createEmpty<u8>(std::initializer_list<int64_t> dim);

template<typename T>
Value Value::createEmpty(std::vector<int64_t> const& dim) {
    Ort::AllocatorWithDefaultOptions allocator;

    Value res;
    res.value_ = Ort::Value::CreateTensor<T>(allocator, &(*dim.begin()), dim.size());

    return res;
}

template Value Value::createEmpty<f32>(std::vector<int64_t> const& dim);
template Value Value::createEmpty<f64>(std::vector<int64_t> const& dim);
template Value Value::createEmpty<s64>(std::vector<int64_t> const& dim);
template Value Value::createEmpty<u64>(std::vector<int64_t> const& dim);
template Value Value::createEmpty<s32>(std::vector<int64_t> const& dim);
template Value Value::createEmpty<u32>(std::vector<int64_t> const& dim);
template Value Value::createEmpty<s16>(std::vector<int64_t> const& dim);
template Value Value::createEmpty<u16>(std::vector<int64_t> const& dim);
template Value Value::createEmpty<s8>(std::vector<int64_t> const& dim);
template Value Value::createEmpty<u8>(std::vector<int64_t> const& dim);

/* ------------------------- Getters ------------------------- */

std::string Value::dimInfo() const {
    std::stringstream ss;
    ss << "Shape<";
    for (int i = 0; i < numDims(); i++) {
        ss << dimSize(i);
        if (i + 1 < numDims()) {
            ss << " ";
        }
    }
    ss << ">";
    return ss.str();
}

ValueType Value::type() const {
    if (value_) {
        ONNXType type = value_.GetTypeInfo().GetONNXType();
        return static_cast<ValueType>(type);
    }
    return ValueType::EMPTY;
}

std::string Value::typeName() const {
    if (value_) {
        ONNXType type = value_.GetTypeInfo().GetONNXType();
        return detail::OnnxTypeToString(type);
    }
    return "<empty>";
}

ValueDataType Value::dataType() const {
    if (value_) {
        ONNXTensorElementDataType element_type = value_.GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementType();
        return static_cast<ValueDataType>(element_type);
    }
    return ValueDataType::EMPTY;
}

std::string Value::dataTypeName() const {
    if (value_) {
        ONNXTensorElementDataType element_type = value_.GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementType();
        return detail::OnnxTensorElementDataTypeToString(element_type);
    }
    return "empty";
}

template<typename T>
void Value::get(Math::FastMatrix<T>& mat, bool transpose) const {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_eq(numDims(), 2);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);

    T const* data = value_.GetTensorData<T>();
    u32      rows = static_cast<u32>(dimSize(transpose ? 1 : 0));
    u32      cols = static_cast<u32>(dimSize(transpose ? 0 : 1));
    mat.resize(rows, cols);

    if (transpose) {
        for (u32 c = 0u; c < mat.nColumns(); c++) {
            for (u32 r = 0u; r < mat.nRows(); r++) {
                mat.at(r, c) = data[c * rows + r];
            }
        }
    }
    else {
        for (u32 c = 0u; c < mat.nColumns(); c++) {
            for (u32 r = 0u; r < mat.nRows(); r++) {
                mat.at(r, c) = data[r * cols + c];
            }
        }
    }
}

template void Value::get<f32>(Math::FastMatrix<f32>&, bool) const;
template void Value::get<f64>(Math::FastMatrix<f64>&, bool) const;
template void Value::get<s64>(Math::FastMatrix<s64>&, bool) const;
template void Value::get<u64>(Math::FastMatrix<u64>&, bool) const;
template void Value::get<s32>(Math::FastMatrix<s32>&, bool) const;
template void Value::get<u32>(Math::FastMatrix<u32>&, bool) const;
template void Value::get<s16>(Math::FastMatrix<s16>&, bool) const;
template void Value::get<u16>(Math::FastMatrix<u16>&, bool) const;
template void Value::get<s8>(Math::FastMatrix<s8>&, bool) const;
template void Value::get<u8>(Math::FastMatrix<u8>&, bool) const;

template<typename T>
void Value::get(std::vector<Math::FastMatrix<T>>& batches, bool transpose) const {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_eq(numDims(), 3);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);

    T const* data = value_.GetTensorData<T>();
    batches.resize(dimSize(0));
    u32 rows = static_cast<u32>(dimSize(transpose ? 2 : 1));
    u32 cols = static_cast<u32>(dimSize(transpose ? 1 : 2));

    for (size_t b = 0ul; b < batches.size(); b++) {
        auto& m = batches[b];
        m.resize(rows, cols);
        if (transpose) {
            for (u32 c = 0u; c < m.nColumns(); c++) {
                for (u32 r = 0u; r < m.nRows(); r++) {
                    m.at(r, c) = data[c * rows + r];
                }
            }
        }
        else {
            for (u32 c = 0u; c < m.nColumns(); c++) {
                for (u32 r = 0u; r < m.nRows(); r++) {
                    m.at(r, c) = data[r * cols + c];
                }
            }
        }
    }
}

template void Value::get<f32>(std::vector<Math::FastMatrix<f32>>&, bool) const;
template void Value::get<f64>(std::vector<Math::FastMatrix<f64>>&, bool) const;
template void Value::get<s64>(std::vector<Math::FastMatrix<s64>>&, bool) const;
template void Value::get<u64>(std::vector<Math::FastMatrix<u64>>&, bool) const;
template void Value::get<s32>(std::vector<Math::FastMatrix<s32>>&, bool) const;
template void Value::get<u32>(std::vector<Math::FastMatrix<u32>>&, bool) const;
template void Value::get<s16>(std::vector<Math::FastMatrix<s16>>&, bool) const;
template void Value::get<u16>(std::vector<Math::FastMatrix<u16>>&, bool) const;
template void Value::get<s8>(std::vector<Math::FastMatrix<s8>>&, bool) const;
template void Value::get<u8>(std::vector<Math::FastMatrix<u8>>&, bool) const;

template<typename T>
void Value::get(Math::FastVector<T>& vec) const {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_eq(numDims(), 1);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);

    T const* data = value_.GetTensorData<T>();
    vec.resize(static_cast<size_t>(dimSize(0)));
    std::copy(data, data + vec.size(), vec.begin());
}

template void Value::get<f32>(Math::FastVector<f32>&) const;
template void Value::get<f64>(Math::FastVector<f64>&) const;
template void Value::get<s64>(Math::FastVector<s64>&) const;
template void Value::get<u64>(Math::FastVector<u64>&) const;
template void Value::get<s32>(Math::FastVector<s32>&) const;
template void Value::get<u32>(Math::FastVector<u32>&) const;
template void Value::get<s16>(Math::FastVector<s16>&) const;
template void Value::get<u16>(Math::FastVector<u16>&) const;
template void Value::get<s8>(Math::FastVector<s8>&) const;
template void Value::get<u8>(Math::FastVector<u8>&) const;

template<typename T>
void Value::get(std::vector<T>& vec) const {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_eq(numDims(), 1);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);

    T const* data = value_.GetTensorData<T>();
    vec.resize(static_cast<size_t>(dimSize(0)));
    std::copy(data, data + vec.size(), vec.begin());
}

template void Value::get<f32>(std::vector<f32>&) const;
template void Value::get<f64>(std::vector<f64>&) const;
template void Value::get<s64>(std::vector<s64>&) const;
template void Value::get<u64>(std::vector<u64>&) const;
template void Value::get<s32>(std::vector<s32>&) const;
template void Value::get<u32>(std::vector<u32>&) const;
template void Value::get<s16>(std::vector<s16>&) const;
template void Value::get<u16>(std::vector<u16>&) const;
template void Value::get<s8>(std::vector<s8>&) const;
template void Value::get<u8>(std::vector<u8>&) const;

template<typename T>
void Value::get(T& val) const {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_eq(numDims(), 0);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);

    T const* data = value_.GetTensorData<T>();
    val           = data[0];
}

template void Value::get<f32>(f32&) const;
template void Value::get<f64>(f64&) const;
template void Value::get<s64>(s64&) const;
template void Value::get<u64>(u64&) const;
template void Value::get<s32>(s32&) const;
template void Value::get<u32>(u32&) const;
template void Value::get<s16>(s16&) const;
template void Value::get<u16>(u16&) const;
template void Value::get<s8>(s8&) const;
template void Value::get<u8>(u8&) const;
template void Value::get<bool>(bool&) const;

// getters for a subset of the data (1-dim subset)

template<typename T>
void Value::get(size_t dim0_idx, Math::FastMatrix<T>& mat, bool transpose) const {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_eq(numDims(), 3);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);
    require_gt(dimSize(0), static_cast<s64>(dim0_idx));

    T const* data     = value_.GetTensorData<T>();
    u32      rows     = static_cast<u32>(dimSize(transpose ? 2 : 1));
    u32      cols     = static_cast<u32>(dimSize(transpose ? 1 : 2));
    u32      mat_size = rows * cols;

    mat.resize(rows, cols);
    if (transpose) {
        for (u32 c = 0u; c < mat.nColumns(); c++) {
            for (u32 r = 0u; r < mat.nRows(); r++) {
                mat.at(r, c) = data[dim0_idx * mat_size + c * rows + r];
            }
        }
    }
    else {
        for (u32 c = 0u; c < mat.nColumns(); c++) {
            for (u32 r = 0u; r < mat.nRows(); r++) {
                mat.at(r, c) = data[dim0_idx * mat_size + r * cols + c];
            }
        }
    }
}

template void Value::get<f32>(size_t, Math::FastMatrix<f32>&, bool) const;
template void Value::get<f64>(size_t, Math::FastMatrix<f64>&, bool) const;
template void Value::get<s64>(size_t, Math::FastMatrix<s64>&, bool) const;
template void Value::get<u64>(size_t, Math::FastMatrix<u64>&, bool) const;
template void Value::get<s32>(size_t, Math::FastMatrix<s32>&, bool) const;
template void Value::get<u32>(size_t, Math::FastMatrix<u32>&, bool) const;
template void Value::get<s16>(size_t, Math::FastMatrix<s16>&, bool) const;
template void Value::get<u16>(size_t, Math::FastMatrix<u16>&, bool) const;
template void Value::get<s8>(size_t, Math::FastMatrix<s8>&, bool) const;
template void Value::get<u8>(size_t, Math::FastMatrix<u8>&, bool) const;

template<typename T>
void Value::get(size_t dim0_idx, Math::FastVector<T>& vec) const {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_eq(numDims(), 2);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);
    require_gt(dimSize(0), static_cast<s64>(dim0_idx));

    T const* data = value_.GetTensorData<T>() + dim0_idx * dimSize(1);
    vec.resize(static_cast<size_t>(dimSize(1)));
    std::copy(data, data + vec.size(), vec.begin());
}

template void Value::get<f32>(size_t, Math::FastVector<f32>&) const;
template void Value::get<f64>(size_t, Math::FastVector<f64>&) const;
template void Value::get<s64>(size_t, Math::FastVector<s64>&) const;
template void Value::get<u64>(size_t, Math::FastVector<u64>&) const;
template void Value::get<s32>(size_t, Math::FastVector<s32>&) const;
template void Value::get<u32>(size_t, Math::FastVector<u32>&) const;
template void Value::get<s16>(size_t, Math::FastVector<s16>&) const;
template void Value::get<u16>(size_t, Math::FastVector<u16>&) const;
template void Value::get<s8>(size_t, Math::FastVector<s8>&) const;
template void Value::get<u8>(size_t, Math::FastVector<u8>&) const;

template<typename T>
void Value::get(size_t dim0_idx, std::vector<T>& vec) const {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_eq(numDims(), 2);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);
    require_gt(dimSize(0), static_cast<s64>(dim0_idx));

    T const* data = value_.GetTensorData<T>() + dim0_idx * dimSize(1);
    vec.resize(static_cast<size_t>(dimSize(1)));
    std::copy(data, data + vec.size(), vec.begin());
}

template void Value::get<f32>(size_t, std::vector<f32>&) const;
template void Value::get<f64>(size_t, std::vector<f64>&) const;
template void Value::get<s64>(size_t, std::vector<s64>&) const;
template void Value::get<u64>(size_t, std::vector<u64>&) const;
template void Value::get<s32>(size_t, std::vector<s32>&) const;
template void Value::get<u32>(size_t, std::vector<u32>&) const;
template void Value::get<s16>(size_t, std::vector<s16>&) const;
template void Value::get<u16>(size_t, std::vector<u16>&) const;
template void Value::get<s8>(size_t, std::vector<s8>&) const;
template void Value::get<u8>(size_t, std::vector<u8>&) const;

template<typename T>
void Value::get(size_t dim0_idx, T& val) const {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_eq(numDims(), 1);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);
    require_gt(dimSize(0), static_cast<s64>(dim0_idx));

    T const* data = value_.GetTensorData<T>();
    val           = data[dim0_idx];
}

template void Value::get<f32>(size_t, f32&) const;
template void Value::get<f64>(size_t, f64&) const;
template void Value::get<s64>(size_t, s64&) const;
template void Value::get<u64>(size_t, u64&) const;
template void Value::get<s32>(size_t, s32&) const;
template void Value::get<u32>(size_t, u32&) const;
template void Value::get<s16>(size_t, s16&) const;
template void Value::get<u16>(size_t, u16&) const;
template void Value::get<s8>(size_t, s8&) const;
template void Value::get<u8>(size_t, u8&) const;

// getters for a subset of the data (2-dim subset)

template<typename T>
void Value::get(size_t dim0_idx, size_t dim1_idx, Math::FastVector<T>& vec) const {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_eq(numDims(), 3);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);
    require_gt(dimSize(0), static_cast<s64>(dim0_idx));
    require_gt(dimSize(1), static_cast<s64>(dim1_idx));

    T const* data = value_.GetTensorData<T>() + dim0_idx * dimSize(1) * dimSize(2) + dim1_idx * dimSize(2);
    vec.resize(static_cast<size_t>(dimSize(2)));
    std::copy(data, data + vec.size(), vec.begin());
}

template void Value::get<f32>(size_t, size_t, Math::FastVector<f32>&) const;
template void Value::get<f64>(size_t, size_t, Math::FastVector<f64>&) const;
template void Value::get<s64>(size_t, size_t, Math::FastVector<s64>&) const;
template void Value::get<u64>(size_t, size_t, Math::FastVector<u64>&) const;
template void Value::get<s32>(size_t, size_t, Math::FastVector<s32>&) const;
template void Value::get<u32>(size_t, size_t, Math::FastVector<u32>&) const;
template void Value::get<s16>(size_t, size_t, Math::FastVector<s16>&) const;
template void Value::get<u16>(size_t, size_t, Math::FastVector<u16>&) const;
template void Value::get<s8>(size_t, size_t, Math::FastVector<s8>&) const;
template void Value::get<u8>(size_t, size_t, Math::FastVector<u8>&) const;

template<typename T>
void Value::get(size_t dim0_idx, size_t dim1_idx, std::vector<T>& vec) const {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_eq(numDims(), 3);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);
    require_gt(dimSize(0), static_cast<s64>(dim0_idx));
    require_gt(dimSize(1), static_cast<s64>(dim1_idx));

    T const* data = value_.GetTensorData<T>() + dim0_idx * dimSize(1) * dimSize(2) + dim1_idx * dimSize(2);
    vec.resize(static_cast<size_t>(dimSize(2)));
    std::copy(data, data + vec.size(), vec.begin());
}

template void Value::get<f32>(size_t, size_t, std::vector<f32>&) const;
template void Value::get<f64>(size_t, size_t, std::vector<f64>&) const;
template void Value::get<s64>(size_t, size_t, std::vector<s64>&) const;
template void Value::get<u64>(size_t, size_t, std::vector<u64>&) const;
template void Value::get<s32>(size_t, size_t, std::vector<s32>&) const;
template void Value::get<u32>(size_t, size_t, std::vector<u32>&) const;
template void Value::get<s16>(size_t, size_t, std::vector<s16>&) const;
template void Value::get<u16>(size_t, size_t, std::vector<u16>&) const;
template void Value::get<s8>(size_t, size_t, std::vector<s8>&) const;
template void Value::get<u8>(size_t, size_t, std::vector<u8>&) const;

template<typename T>
void Value::get(size_t dim0_idx, size_t dim1_idx, T& val) const {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_eq(numDims(), 2);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);
    require_gt(dimSize(0), static_cast<s64>(dim0_idx));
    require_gt(dimSize(1), static_cast<s64>(dim1_idx));

    T const* data = value_.GetTensorData<T>() + dim0_idx * dimSize(1) + dim1_idx;
    val           = *data;
}

template void Value::get<f32>(size_t, size_t, f32&) const;
template void Value::get<f64>(size_t, size_t, f64&) const;
template void Value::get<s64>(size_t, size_t, s64&) const;
template void Value::get<u64>(size_t, size_t, u64&) const;
template void Value::get<s32>(size_t, size_t, s32&) const;
template void Value::get<u32>(size_t, size_t, u32&) const;
template void Value::get<s16>(size_t, size_t, s16&) const;
template void Value::get<u16>(size_t, size_t, u16&) const;
template void Value::get<s8>(size_t, size_t, s8&) const;
template void Value::get<u8>(size_t, size_t, u8&) const;

/* ------------------------- raw data access ------------------------- */

template<typename T>
T* Value::data() {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_ge(numDims(), 0);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);

    return value_.GetTensorMutableData<T>();
}

template f32* Value::data<f32>();
template f64* Value::data<f64>();
template s64* Value::data<s64>();
template u64* Value::data<u64>();
template s32* Value::data<s32>();
template u32* Value::data<u32>();
template s16* Value::data<s16>();
template u16* Value::data<u16>();
template s8*  Value::data<s8>();
template u8*  Value::data<u8>();

template<typename T>
T const* Value::data() const {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_ge(numDims(), 0);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);

    return value_.GetTensorData<T>();
}

template f32 const* Value::data<f32>() const;
template f64 const* Value::data<f64>() const;
template s64 const* Value::data<s64>() const;
template u64 const* Value::data<u64>() const;
template s32 const* Value::data<s32>() const;
template u32 const* Value::data<u32>() const;
template s16 const* Value::data<s16>() const;
template u16 const* Value::data<u16>() const;
template s8 const*  Value::data<s8>() const;
template u8 const*  Value::data<u8>() const;

template<typename T>
T* Value::data(size_t dim0_idx) {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_ge(numDims(), 1);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);
    require_gt(dimSize(0), static_cast<s64>(dim0_idx));

    size_t factor = 1;
    for (int i = 1; i < numDims(); i++) {
        factor *= dimSize(i);
    }

    return value_.GetTensorMutableData<T>() + dim0_idx * factor;
}

template f32* Value::data<f32>(size_t);
template f64* Value::data<f64>(size_t);
template s64* Value::data<s64>(size_t);
template u64* Value::data<u64>(size_t);
template s32* Value::data<s32>(size_t);
template u32* Value::data<u32>(size_t);
template s16* Value::data<s16>(size_t);
template u16* Value::data<u16>(size_t);
template s8*  Value::data<s8>(size_t);
template u8*  Value::data<u8>(size_t);

template<typename T>
T const* Value::data(size_t dim0_idx) const {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_ge(numDims(), 1);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);
    require_gt(dimSize(0), static_cast<s64>(dim0_idx));

    size_t factor = 1;
    for (int i = 1; i < numDims(); i++) {
        factor *= dimSize(i);
    }

    return value_.GetTensorData<T>() + dim0_idx * factor;
}

template f32 const* Value::data<f32>(size_t) const;
template f64 const* Value::data<f64>(size_t) const;
template s64 const* Value::data<s64>(size_t) const;
template u64 const* Value::data<u64>(size_t) const;
template s16 const* Value::data<s16>(size_t) const;
template u16 const* Value::data<u16>(size_t) const;
template s8 const*  Value::data<s8>(size_t) const;
template u8 const*  Value::data<u8>(size_t) const;

template<typename T>
T* Value::data(size_t dim0_idx, size_t dim1_idx) {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_ge(numDims(), 2);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);
    require_gt(dimSize(0), static_cast<s64>(dim0_idx));
    require_gt(dimSize(1), static_cast<s64>(dim1_idx));

    size_t factor = 1;
    for (int i = 2; i < numDims(); i++) {
        factor *= dimSize(i);
    }

    return value_.GetTensorMutableData<T>() + dim0_idx * dimSize(1) * factor + dim1_idx * factor;
}

template f32* Value::data<f32>(size_t, size_t);
template f64* Value::data<f64>(size_t, size_t);
template s64* Value::data<s64>(size_t, size_t);
template u64* Value::data<u64>(size_t, size_t);
template s32* Value::data<s32>(size_t, size_t);
template u32* Value::data<u32>(size_t, size_t);
template s16* Value::data<s16>(size_t, size_t);
template u16* Value::data<u16>(size_t, size_t);
template s8*  Value::data<s8>(size_t, size_t);
template u8*  Value::data<u8>(size_t, size_t);

template<typename T>
T const* Value::data(size_t dim0_idx, size_t dim1_idx) const {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_ge(numDims(), 2);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);
    require_gt(dimSize(0), static_cast<s64>(dim0_idx));
    require_gt(dimSize(1), static_cast<s64>(dim1_idx));

    size_t factor = 1;
    for (int i = 2; i < numDims(); i++) {
        factor *= dimSize(i);
    }

    return value_.GetTensorData<T>() + dim0_idx * dimSize(1) * factor + dim1_idx * factor;
}

template f32 const* Value::data<f32>(size_t, size_t) const;
template f64 const* Value::data<f64>(size_t, size_t) const;
template s64 const* Value::data<s64>(size_t, size_t) const;
template u64 const* Value::data<u64>(size_t, size_t) const;
template s32 const* Value::data<s32>(size_t, size_t) const;
template u32 const* Value::data<u32>(size_t, size_t) const;
template s16 const* Value::data<s16>(size_t, size_t) const;
template u16 const* Value::data<u16>(size_t, size_t) const;
template s8 const*  Value::data<s8>(size_t, size_t) const;
template u8 const*  Value::data<u8>(size_t, size_t) const;

template<typename T>
T* Value::data(size_t dim0_idx, size_t dim1_idx, size_t dim2_idx) {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_ge(numDims(), 3);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);
    require_gt(dimSize(0), static_cast<s64>(dim0_idx));
    require_gt(dimSize(1), static_cast<s64>(dim1_idx));
    require_gt(dimSize(2), static_cast<s64>(dim2_idx));

    size_t factor = 1;
    for (int i = 3; i < numDims(); i++) {
        factor *= dimSize(i);
    }

    return value_.GetTensorMutableData<T>() + dim0_idx * dimSize(1) * dimSize(2) * factor + dim1_idx * dimSize(2) * factor + dim2_idx * factor;
}

template f32* Value::data<f32>(size_t, size_t, size_t);
template f64* Value::data<f64>(size_t, size_t, size_t);
template s64* Value::data<s64>(size_t, size_t, size_t);
template u64* Value::data<u64>(size_t, size_t, size_t);
template s32* Value::data<s32>(size_t, size_t, size_t);
template u32* Value::data<u32>(size_t, size_t, size_t);
template s16* Value::data<s16>(size_t, size_t, size_t);
template u16* Value::data<u16>(size_t, size_t, size_t);
template s8*  Value::data<s8>(size_t, size_t, size_t);
template u8*  Value::data<u8>(size_t, size_t, size_t);

template<typename T>
T const* Value::data(size_t dim0_idx, size_t dim1_idx, size_t dim2_idx) const {
    ONNXTensorElementDataType expected_dtype = ToDataType<T>::onnx_tensor_element_type;
    require(not empty());
    require(value_.IsTensor());
    require_ge(numDims(), 3);
    require_eq(value_.GetTensorTypeAndShapeInfo().GetElementType(), expected_dtype);
    require_gt(dimSize(0), static_cast<s64>(dim0_idx));
    require_gt(dimSize(1), static_cast<s64>(dim1_idx));
    require_gt(dimSize(2), static_cast<s64>(dim2_idx));

    size_t factor = 1;
    for (int i = 3; i < numDims(); i++) {
        factor *= dimSize(i);
    }

    return value_.GetTensorData<T>() + dim0_idx * dimSize(1) * dimSize(2) * factor + dim1_idx * dimSize(2) * factor + dim2_idx * factor;
}

template f32 const* Value::data<f32>(size_t, size_t, size_t) const;
template f64 const* Value::data<f64>(size_t, size_t, size_t) const;
template s64 const* Value::data<s64>(size_t, size_t, size_t) const;
template u64 const* Value::data<u64>(size_t, size_t, size_t) const;
template s32 const* Value::data<s32>(size_t, size_t, size_t) const;
template u32 const* Value::data<u32>(size_t, size_t, size_t) const;
template s16 const* Value::data<s16>(size_t, size_t, size_t) const;
template u16 const* Value::data<u16>(size_t, size_t, size_t) const;
template s8 const*  Value::data<s8>(size_t, size_t, size_t) const;
template u8 const*  Value::data<u8>(size_t, size_t, size_t) const;

/* ------------------------- Setters ------------------------- */

template<typename T>
void Value::set(Math::FastMatrix<T> const& mat, bool transpose) {
    int64_t rows = transpose ? mat.nColumns() : mat.nRows();
    int64_t cols = transpose ? mat.nRows() : mat.nColumns();

    std::vector<int64_t>             shape({rows, cols});
    Ort::AllocatorWithDefaultOptions allocator;
    value_  = Ort::Value::CreateTensor<T>(allocator, shape.data(), shape.size());
    T* data = value_.GetTensorMutableData<T>();

    if (transpose) {
        // if we transpose we can iterate over both matrices linearly
        for (u32 c = 0u; c < mat.nColumns(); c++) {
            for (u32 r = 0u; r < mat.nRows(); r++) {
                data[c * mat.nRows() + r] = mat.at(r, c);
            }
        }
    }
    else {
        // as Onnx uses row-major and sprint col major we will have nonlinear
        // memory access in at least one case, we opt to do linear writes
        for (u32 r = 0u; r < mat.nRows(); r++) {
            for (u32 c = 0u; c < mat.nColumns(); c++) {
                data[r * mat.nColumns() + c] = mat.at(r, c);
            }
        }
    }
}

template void Value::set<f64>(Math::FastMatrix<f64> const&, bool);
template void Value::set<f32>(Math::FastMatrix<f32> const&, bool);
template void Value::set<s64>(Math::FastMatrix<s64> const&, bool);
template void Value::set<u64>(Math::FastMatrix<u64> const&, bool);
template void Value::set<s32>(Math::FastMatrix<s32> const&, bool);
template void Value::set<u32>(Math::FastMatrix<u32> const&, bool);
template void Value::set<s16>(Math::FastMatrix<s16> const&, bool);
template void Value::set<u16>(Math::FastMatrix<u16> const&, bool);
template void Value::set<s8>(Math::FastMatrix<s8> const&, bool);
template void Value::set<u8>(Math::FastMatrix<u8> const&, bool);

template<typename T>
void Value::set(std::vector<Math::FastMatrix<T>> const& batches, bool transpose) {
    require_gt(batches.size(), 0ul);
    u32 rows = 0u;
    u32 cols = 0u;

    for (auto const& b : batches) {
        rows = std::max(rows, transpose ? b.nColumns() : b.nRows());
        cols = std::max(cols, transpose ? b.nRows() : b.nColumns());
    }

    std::vector<int64_t>             shape({static_cast<int64_t>(batches.size()), rows, cols});
    Ort::AllocatorWithDefaultOptions allocator;
    value_  = Ort::Value::CreateTensor<T>(allocator, shape.data(), shape.size());
    T* data = value_.GetTensorMutableData<T>();

    for (size_t b = 0ul; b < batches.size(); b++) {
        auto const& m          = batches[b];
        T*          batch_data = data + b * rows * cols;
        // in these loops we always use indices (r and c) which relate to the original matrix order,
        // but rows and cols refers to the size of the tensor, thus length checks are asymetrical in the transposed case
        if (transpose) {
            for (u32 c = 0u; c < m.nColumns(); c++) {
                for (u32 r = 0u; r < m.nRows(); r++) {
                    batch_data[c * cols + r] = m.at(r, c);
                }
                for (u32 r = m.nRows(); r < cols; r++) {
                    batch_data[c * cols + r] = T(0);
                }
            }
            for (u32 c = m.nColumns(); c < rows; c++) {
                for (u32 r = 0u; r < cols; r++) {
                    batch_data[c * cols + r] = T(0);
                }
            }
        }
        else {
            for (u32 r = 0u; r < m.nRows(); r++) {
                for (u32 c = 0u; c < m.nColumns(); c++) {
                    batch_data[r * cols + c] = m.at(r, c);
                }
                for (u32 c = m.nColumns(); c < cols; c++) {
                    batch_data[r * cols + c] = T(0);
                }
            }
            for (u32 r = m.nRows(); r < rows; r++) {
                for (u32 c = 0u; c < cols; c++) {
                    batch_data[r * cols + c] = T(0);
                }
            }
        }
    }
}

template void Value::set<f32>(std::vector<Math::FastMatrix<f32>> const&, bool);
template void Value::set<f64>(std::vector<Math::FastMatrix<f64>> const&, bool);
template void Value::set<s64>(std::vector<Math::FastMatrix<s64>> const&, bool);
template void Value::set<u64>(std::vector<Math::FastMatrix<u64>> const&, bool);
template void Value::set<s32>(std::vector<Math::FastMatrix<s32>> const&, bool);
template void Value::set<u32>(std::vector<Math::FastMatrix<u32>> const&, bool);
template void Value::set<s16>(std::vector<Math::FastMatrix<s16>> const&, bool);
template void Value::set<u16>(std::vector<Math::FastMatrix<u16>> const&, bool);
template void Value::set<s8>(std::vector<Math::FastMatrix<s8>> const&, bool);
template void Value::set<u8>(std::vector<Math::FastMatrix<u8>> const&, bool);

template<typename T>
void Value::set(Math::FastVector<T> const& vec) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t>             shape({static_cast<int64_t>(vec.size())});
    value_ = Ort::Value::CreateTensor<T>(allocator, shape.data(), shape.size());

    T* data = value_.GetTensorMutableData<T>();
    std::copy(vec.begin(), vec.end(), data);
}

template void Value::set<f32>(Math::FastVector<f32> const&);
template void Value::set<f64>(Math::FastVector<f64> const&);
template void Value::set<s64>(Math::FastVector<s64> const&);
template void Value::set<u64>(Math::FastVector<u64> const&);
template void Value::set<s32>(Math::FastVector<s32> const&);
template void Value::set<u32>(Math::FastVector<u32> const&);
template void Value::set<s16>(Math::FastVector<s16> const&);
template void Value::set<u16>(Math::FastVector<u16> const&);
template void Value::set<s8>(Math::FastVector<s8> const&);
template void Value::set<u8>(Math::FastVector<u8> const&);

template<typename T>
void Value::set(std::vector<T> const& vec) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t>             shape({static_cast<int64_t>(vec.size())});
    value_ = Ort::Value::CreateTensor<T>(allocator, shape.data(), shape.size());

    T* data = value_.GetTensorMutableData<T>();
    std::copy(vec.begin(), vec.end(), data);
}

template void Value::set<f32>(std::vector<f32> const&);
template void Value::set<f64>(std::vector<f64> const&);
template void Value::set<s64>(std::vector<s64> const&);
template void Value::set<u64>(std::vector<u64> const&);
template void Value::set<s32>(std::vector<s32> const&);
template void Value::set<u32>(std::vector<u32> const&);
template void Value::set<s16>(std::vector<s16> const&);
template void Value::set<u16>(std::vector<u16> const&);
template void Value::set<s8>(std::vector<s8> const&);
template void Value::set<u8>(std::vector<u8> const&);

template<typename T>
void Value::set(T const& val) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t>             shape;
    value_ = Ort::Value::CreateTensor<T>(allocator, shape.data(), shape.size());

    T* data = value_.GetTensorMutableData<T>();
    data[0] = val;
}

template void Value::set<f32>(f32 const&);
template void Value::set<f64>(f64 const&);
template void Value::set<s64>(s64 const&);
template void Value::set<u64>(u64 const&);
template void Value::set<s32>(s32 const&);
template void Value::set<u32>(u32 const&);
template void Value::set<s16>(s16 const&);
template void Value::set<u16>(u16 const&);
template void Value::set<s8>(s8 const&);
template void Value::set<u8>(u8 const&);
template void Value::set<bool>(bool const&);

template<typename T>
void Value::save(std::string const& path) const {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    for (int i = 0; i < numDims(); i++) {
        if (i > 0) {
            out << ' ';
        }
        out << dimSize(i);
    }
    out << '\n';

    T const* data = value_.GetTensorData<T>();
    int      rows = 1;
    int      cols = dimSize(numDims() - 1);
    for (int i = 0; i < numDims() - 1; i++) {
        rows *= dimSize(i);
    }

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            out << data[r * cols + c];
            if (c + 1 < cols) {
                out << ' ';
            }
        }
        out << '\n';
    }
}

template void Value::save<f32>(std::string const&) const;
template void Value::save<f64>(std::string const&) const;
template void Value::save<s64>(std::string const&) const;
template void Value::save<u64>(std::string const&) const;
template void Value::save<s32>(std::string const&) const;
template void Value::save<u32>(std::string const&) const;
template void Value::save<s16>(std::string const&) const;
template void Value::save<u16>(std::string const&) const;
template void Value::save<s8>(std::string const&) const;
template void Value::save<u8>(std::string const&) const;
template void Value::save<bool>(std::string const&) const;

}  // namespace Onnx
