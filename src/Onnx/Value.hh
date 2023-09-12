/*
 * Copyright 2022 AppTek LLC. All rights reserved.
 */
#ifndef _ONNX_TENSOR_HH
#define _ONNX_TENSOR_HH

#include <onnxruntime_cxx_api.h>

#include <Math/FastMatrix.hh>
#include <Math/FastVector.hh>

namespace Onnx {

class Session;

enum class ValueType : int {
    EMPTY = -1,

    UNKNOWN      = ONNX_TYPE_UNKNOWN,
    TENSOR       = ONNX_TYPE_TENSOR,
    SEQUENCE     = ONNX_TYPE_SEQUENCE,
    MAP          = ONNX_TYPE_MAP,
    OPAQUE       = ONNX_TYPE_OPAQUE,
    SPARSETENSOR = ONNX_TYPE_SPARSETENSOR,
    OPTIONAL     = ONNX_TYPE_OPTIONAL,
};

enum class ValueDataType : int {
    EMPTY = -1,

    FLOAT      = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    INT8       = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
    UINT16     = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
    INT16      = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
    INT32      = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
    INT64      = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    STRING     = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
    BOOL       = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
    FLOAT16    = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
    DOUBLE     = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
    UINT32     = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
    UINT64     = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
    COMPLEX64  = ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
    COMPLEX128 = ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
    BFLOAT16   = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
};

class Value {
public:
    friend Session;

    template<typename... Args>
    static Value create(Args... value);

    template<typename T>
    static Value zeros(std::initializer_list<int64_t> dim);

    template<typename T>
    static Value zeros(std::vector<int64_t> const& dim);

    Value();
    Value(Value&& other);
    ~Value() = default;

    bool empty() const;

    Value& operator=(Value&& other);

    /* -------------------- Getters -------------------- */

    int           numDims() const;
    int64_t       dimSize(int d) const;
    std::string   dimInfo() const;  // usefull for debugging, format: Shape<dim0, dim1, ...>
    ValueType     type() const;
    std::string   typeName() const;
    ValueDataType dataType() const;
    std::string   dataTypeName() const;

    template<typename T>
    void get(Math::FastMatrix<T>& mat, bool transpose = false) const;

    template<typename T>
    void get(std::vector<Math::FastMatrix<T>>& batches, bool transpose = false) const;

    template<typename T>
    void get(Math::FastVector<T>& vec) const;

    template<typename T>
    void get(std::vector<T>& vec) const;

    template<typename T>
    void get(T& val) const;

    // getters for a subset of the data (1-dim subset)

    template<typename T>
    void get(size_t dim0_idx, Math::FastMatrix<T>& mat, bool transpose = false) const;

    template<typename T>
    void get(size_t dim0_idx, Math::FastVector<T>& vec) const;

    template<typename T>
    void get(size_t dim0_idx, std::vector<T>& vec) const;

    template<typename T>
    void get(size_t dim0_idx, T& val) const;

    // getters for a subset of the data (2-dim subset)

    template<typename T>
    void get(size_t dim0_idx, size_t dim1_idx, Math::FastVector<T>& vec) const;

    template<typename T>
    void get(size_t dim0_idx, size_t dim1_idx, std::vector<T>& vec) const;

    template<typename T>
    void get(size_t dim0_idx, size_t dim1_idx, T& val) const;

    // raw data access

    template<typename T>
    T* data();

    template<typename T>
    T const* data() const;

    template<typename T>
    T* data(size_t dim0_idx);

    template<typename T>
    T const* data(size_t dim0_idx) const;

    template<typename T>
    T* data(size_t dim0_idx, size_t dim1_idx);

    template<typename T>
    T const* data(size_t dim0_idx, size_t dim1_idx) const;

    template<typename T>
    T* data(size_t dim0_idx, size_t dim1_idx, size_t dim2_idx);

    template<typename T>
    T const* data(size_t dim0_idx, size_t dim1_idx, size_t dim2_idx) const;

    /* -------------------- Setters -------------------- */

    template<typename T>
    void set(Math::FastMatrix<T> const& mat, bool transpose = false);

    template<typename T>
    void set(std::vector<Math::FastMatrix<T>> const& batches, bool transpose = false);

    template<typename T>
    void set(Math::FastVector<T> const& vec);

    template<typename T>
    void set(std::vector<T> const& vec);

    template<typename T>
    void set(T const& val);

    template<typename T>
    void save(std::string const& path) const;

protected:
    Value(Ort::Value&& value);

    Ort::Value value_;

    Ort::Value const* rawValue() const;
};

// Implementations for some of Values functions (which likely can be inlined)

template<typename... Args>
inline Value Value::create(Args... value) {
    Value res;
    res.set(std::forward<Args>(value)...);
    return res;
}

inline Value::Value()
        : value_(nullptr) {
}

inline Value::Value(Value&& value)
        : value_(std::move(value.value_)) {
}

inline bool Value::empty() const {
    return value_ == nullptr or not value_.HasValue();
}

inline Value& Value::operator=(Value&& other) {
    value_ = std::move(other.value_);
    return *this;
}

inline int Value::numDims() const {
    if (empty() or not value_.IsTensor()) {
        return -1;
    }
    Ort::TensorTypeAndShapeInfo info = value_.GetTensorTypeAndShapeInfo();

    return static_cast<int>(info.GetDimensionsCount());
}

inline int64_t Value::dimSize(int d) const {
    if (not value_.IsTensor()) {
        return -1;
    }

    Ort::TensorTypeAndShapeInfo info = value_.GetTensorTypeAndShapeInfo();
    std::vector<int64_t>        dims = info.GetShape();
    if (d >= 0 and d <= static_cast<int64_t>(info.GetDimensionsCount())) {
        return dims[d];
    }

    return -1;
}

inline Value::Value(Ort::Value&& value)
        : value_(nullptr) {
    value_ = std::move(value);
}

}  // namespace Onnx

#endif  // _ONNX_TENSOR_HH
