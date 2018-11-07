/** Copyright 2018 RWTH Aachen University. All rights reserved.
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
#include "Tensor.hh"

namespace {

namespace tf = tensorflow;

template<typename T>
struct ToDataType {
    typedef T                     cpp_type;
    static constexpr tf::DataType tf_type = tf::DataTypeToEnum<T>::value;
};

template struct ToDataType<f32>;
template struct ToDataType<f64>;
template struct ToDataType<s32>;
template struct ToDataType<u32>;

// tf::DataTypeToEnum does not have entries for s64 and u64 (as they are (unsigned) long)
// instead it has entries for long long and unsigned long long. For our supported data-model
// (i.e. not Windows) these are equivalent. Thus we implement template specializations for
// these two cases separately

template<>
struct ToDataType<s64> {
    typedef tf::int64             cpp_type;
    static constexpr tf::DataType tf_type = tf::DT_INT64;
};

template<>
struct ToDataType<u64> {
    typedef tf::uint64            cpp_type;
    static constexpr tf::DataType tf_type = tf::DT_UINT64;
};

}  // namespace

namespace Tensorflow {

/* ------------------------- Getters ------------------------- */

std::string Tensor::dimInfo() const {
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

std::string Tensor::dataTypeName() const {
    if (tensor_) {
        return tf::DataType_Name(tensor_->dtype());
    }
    return "<empty>";
}

template<typename T>
void Tensor::get(Math::FastMatrix<T>& mat, bool transpose) const {
    tf::DataType expected_dtype = ToDataType<T>::tf_type;
    require(not empty());
    require_eq(tensor_->dims(), 2);
    require_eq(tensor_->dtype(), expected_dtype);

    auto tensor_map = tensor_->flat_outer_dims<typename ToDataType<T>::cpp_type>();
    u32  rows       = static_cast<u32>(tensor_->dim_size(transpose ? 1 : 0));
    u32  cols       = static_cast<u32>(tensor_->dim_size(transpose ? 0 : 1));
    mat.resize(rows, cols);

    if (transpose) {
        for (u32 c = 0u; c < mat.nColumns(); c++) {
            for (u32 r = 0u; r < mat.nRows(); r++) {
                mat.at(r, c) = tensor_map(c, r);
            }
        }
    }
    else {
        for (u32 c = 0u; c < mat.nColumns(); c++) {
            for (u32 r = 0u; r < mat.nRows(); r++) {
                mat.at(r, c) = tensor_map(r, c);
            }
        }
    }
}

template void Tensor::get<f32>(Math::FastMatrix<f32>&, bool) const;
template void Tensor::get<f64>(Math::FastMatrix<f64>&, bool) const;
template void Tensor::get<s64>(Math::FastMatrix<s64>&, bool) const;
template void Tensor::get<u64>(Math::FastMatrix<u64>&, bool) const;
template void Tensor::get<s32>(Math::FastMatrix<s32>&, bool) const;
template void Tensor::get<u32>(Math::FastMatrix<u32>&, bool) const;

template<typename T>
void Tensor::get(std::vector<Math::FastMatrix<T>>& batches, bool transpose) const {
    tf::DataType expected_dtype = ToDataType<T>::tf_type;
    require(not empty());
    require_eq(tensor_->dims(), 3);
    require_eq(tensor_->dtype(), expected_dtype);

    auto tensor_map = tensor_->flat_outer_dims<typename ToDataType<T>::cpp_type, 3>();
    batches.resize(tensor_->dim_size(0));
    u32 rows = static_cast<u32>(tensor_->dim_size(transpose ? 2 : 1));
    u32 cols = static_cast<u32>(tensor_->dim_size(transpose ? 1 : 2));

    for (size_t b = 0ul; b < batches.size(); b++) {
        auto& m = batches[b];
        m.resize(rows, cols);
        if (transpose) {
            for (u32 c = 0u; c < m.nColumns(); c++) {
                for (u32 r = 0u; r < m.nRows(); r++) {
                    m.at(r, c) = tensor_map(b, c, r);
                }
            }
        }
        else {
            for (u32 c = 0u; c < m.nColumns(); c++) {
                for (u32 r = 0u; r < m.nRows(); r++) {
                    m.at(r, c) = tensor_map(b, r, c);
                }
            }
        }
    }
}

template void Tensor::get<f32>(std::vector<Math::FastMatrix<f32>>&, bool) const;
template void Tensor::get<f64>(std::vector<Math::FastMatrix<f64>>&, bool) const;
template void Tensor::get<s64>(std::vector<Math::FastMatrix<s64>>&, bool) const;
template void Tensor::get<u64>(std::vector<Math::FastMatrix<u64>>&, bool) const;
template void Tensor::get<s32>(std::vector<Math::FastMatrix<s32>>&, bool) const;
template void Tensor::get<u32>(std::vector<Math::FastMatrix<u32>>&, bool) const;

template<typename T>
void Tensor::get(Math::FastVector<T>& vec) const {
    tf::DataType expected_dtype = ToDataType<T>::tf_type;
    require(not empty());
    require_eq(tensor_->dims(), 1);
    require_eq(tensor_->dtype(), expected_dtype);
    require_ge(tensor_->dim_size(0), 0);

    vec.resize(static_cast<size_t>(tensor_->dim_size(0)));
    auto tensor_map = tensor_->flat<typename ToDataType<T>::cpp_type>();
    std::copy(&tensor_map(0), &tensor_map(vec.size()), vec.begin());
}

template void Tensor::get<f32>(Math::FastVector<f32>&) const;
template void Tensor::get<f64>(Math::FastVector<f64>&) const;
template void Tensor::get<s64>(Math::FastVector<s64>&) const;
template void Tensor::get<u64>(Math::FastVector<u64>&) const;
template void Tensor::get<s32>(Math::FastVector<s32>&) const;
template void Tensor::get<u32>(Math::FastVector<u32>&) const;

template<typename T>
void Tensor::get(std::vector<T>& vec) const {
    tf::DataType expected_dtype = ToDataType<T>::tf_type;
    require(not empty());
    require_eq(tensor_->dims(), 1);
    require_eq(tensor_->dtype(), expected_dtype);
    require_ge(tensor_->dim_size(0), 0);

    vec.resize(static_cast<size_t>(tensor_->dim_size(0)));
    auto tensor_map = tensor_->flat<typename ToDataType<T>::cpp_type>();
    std::copy(&tensor_map(0), &tensor_map(vec.size()), vec.begin());
}

template void Tensor::get<f32>(std::vector<f32>&) const;
template void Tensor::get<f64>(std::vector<f64>&) const;
template void Tensor::get<s32>(std::vector<s32>&) const;
template void Tensor::get<u32>(std::vector<u32>&) const;
template void Tensor::get<s64>(std::vector<s64>&) const;
template void Tensor::get<u64>(std::vector<u64>&) const;

template<typename T>
void Tensor::get(T& val) const {
    tf::DataType expected_dtype = ToDataType<T>::tf_type;
    require(not empty());
    require_eq(tensor_->dims(), 0);
    require_eq(tensor_->dtype(), expected_dtype);

    val = tensor_->scalar<typename ToDataType<T>::cpp_type>()();
}

template void Tensor::get<f32>(f32&) const;
template void Tensor::get<f64>(f64&) const;
template void Tensor::get<s32>(s32&) const;
template void Tensor::get<u32>(u32&) const;
template void Tensor::get<s64>(s64&) const;
template void Tensor::get<u64>(u64&) const;
template void Tensor::get<std::string>(std::string&) const;

// getters for a subset of the data (1-dim subset)

template<typename T>
void Tensor::get(size_t dim0_idx, Math::FastVector<T>& vec) const {
    tf::DataType expected_dtype = ToDataType<T>::tf_type;
    require(not empty());
    require_eq(tensor_->dims(), 2);
    require_eq(tensor_->dtype(), expected_dtype);
    require_gt(tensor_->dim_size(0), static_cast<s64>(dim0_idx));

    vec.resize(static_cast<size_t>(tensor_->dim_size(1)));
    auto tensor_map = tensor_->flat_outer_dims<typename ToDataType<T>::cpp_type>();
    for (size_t i = 0ul; i < vec.size(); i++) {
        vec[i] = tensor_map(static_cast<s64>(dim0_idx), static_cast<s64>(i));
    }
}

template void Tensor::get<f32>(size_t, Math::FastVector<f32>&) const;
template void Tensor::get<f64>(size_t, Math::FastVector<f64>&) const;
template void Tensor::get<s32>(size_t, Math::FastVector<s32>&) const;
template void Tensor::get<u32>(size_t, Math::FastVector<u32>&) const;
template void Tensor::get<s64>(size_t, Math::FastVector<s64>&) const;
template void Tensor::get<u64>(size_t, Math::FastVector<u64>&) const;

template<typename T>
void Tensor::get(size_t dim0_idx, std::vector<T>& vec) const {
    tf::DataType expected_dtype = ToDataType<T>::tf_type;
    require(not empty());
    require_eq(tensor_->dims(), 2);
    require_eq(tensor_->dtype(), expected_dtype);
    require_gt(tensor_->dim_size(0), static_cast<s64>(dim0_idx));

    vec.resize(static_cast<size_t>(tensor_->dim_size(1)));
    auto tensor_map = tensor_->flat_outer_dims<typename ToDataType<T>::cpp_type>();
    for (size_t i = 0ul; i < vec.size(); i++) {
        vec[i] = tensor_map(static_cast<s64>(dim0_idx), static_cast<s64>(i));
    }
}

template void Tensor::get<f32>(size_t, std::vector<f32>&) const;
template void Tensor::get<f64>(size_t, std::vector<f64>&) const;
template void Tensor::get<s32>(size_t, std::vector<s32>&) const;
template void Tensor::get<u32>(size_t, std::vector<u32>&) const;
template void Tensor::get<s64>(size_t, std::vector<s64>&) const;
template void Tensor::get<u64>(size_t, std::vector<u64>&) const;

template<typename T>
void Tensor::get(size_t dim0_idx, T& val) const {
    tf::DataType expected_dtype = ToDataType<T>::tf_type;
    require(not empty());
    require_eq(tensor_->dims(), 1);
    require_eq(tensor_->dtype(), expected_dtype);
    require_gt(tensor_->dim_size(0), static_cast<s64>(dim0_idx));

    auto tensor_map = tensor_->flat<typename ToDataType<T>::cpp_type>();
    val             = tensor_map(static_cast<s64>(dim0_idx));
}

template void Tensor::get<f32>(size_t, f32&) const;
template void Tensor::get<f64>(size_t, f64&) const;
template void Tensor::get<s32>(size_t, s32&) const;
template void Tensor::get<u32>(size_t, u32&) const;
template void Tensor::get<s64>(size_t, s64&) const;
template void Tensor::get<u64>(size_t, u64&) const;
template void Tensor::get<std::string>(size_t, std::string&) const;

// getters for a subset of the data (2-dim subset)

template<typename T>
void Tensor::get(size_t dim0_idx, size_t dim1_idx, Math::FastVector<T>& vec) const {
    tf::DataType expected_dtype = ToDataType<T>::tf_type;
    require(not empty());
    require_eq(tensor_->dims(), 3);
    require_eq(tensor_->dtype(), expected_dtype);
    require_gt(tensor_->dim_size(0), static_cast<s64>(dim0_idx));

    vec.resize(static_cast<size_t>(tensor_->dim_size(2)));
    auto tensor_map = tensor_->flat_outer_dims<typename ToDataType<T>::cpp_type, 3>();
    for (size_t i = 0ul; i < vec.size(); i++) {
        vec[i] = tensor_map(static_cast<s64>(dim0_idx), static_cast<s64>(dim1_idx), static_cast<s64>(i));
    }
}

template void Tensor::get<f32>(size_t, size_t, Math::FastVector<f32>&) const;
template void Tensor::get<f64>(size_t, size_t, Math::FastVector<f64>&) const;
template void Tensor::get<s32>(size_t, size_t, Math::FastVector<s32>&) const;
template void Tensor::get<u32>(size_t, size_t, Math::FastVector<u32>&) const;
template void Tensor::get<s64>(size_t, size_t, Math::FastVector<s64>&) const;
template void Tensor::get<u64>(size_t, size_t, Math::FastVector<u64>&) const;

template<typename T>
void Tensor::get(size_t dim0_idx, size_t dim1_idx, std::vector<T>& vec) const {
    tf::DataType expected_dtype = ToDataType<T>::tf_type;
    require(not empty());
    require_eq(tensor_->dims(), 3);
    require_eq(tensor_->dtype(), expected_dtype);
    require_gt(tensor_->dim_size(0), static_cast<s64>(dim0_idx));

    vec.resize(static_cast<size_t>(tensor_->dim_size(2)));
    auto tensor_map = tensor_->flat_outer_dims<typename ToDataType<T>::cpp_type, 3>();
    for (size_t i = 0ul; i < vec.size(); i++) {
        vec[i] = tensor_map(static_cast<s64>(dim0_idx), static_cast<s64>(dim1_idx), static_cast<s64>(i));
    }
}

template void Tensor::get<f32>(size_t, size_t, std::vector<f32>&) const;
template void Tensor::get<f64>(size_t, size_t, std::vector<f64>&) const;
template void Tensor::get<s32>(size_t, size_t, std::vector<s32>&) const;
template void Tensor::get<u32>(size_t, size_t, std::vector<u32>&) const;
template void Tensor::get<s64>(size_t, size_t, std::vector<s64>&) const;
template void Tensor::get<u64>(size_t, size_t, std::vector<u64>&) const;

template<typename T>
void Tensor::get(size_t dim0_idx, size_t dim1_idx, T& val) const {
    tf::DataType expected_dtype = ToDataType<T>::tf_type;
    require(not empty());
    require_eq(tensor_->dims(), 2);
    require_eq(tensor_->dtype(), expected_dtype);
    require_gt(tensor_->dim_size(0), static_cast<s64>(dim0_idx));
    require_gt(tensor_->dim_size(1), static_cast<s64>(dim1_idx));

    auto tensor_map = tensor_->flat_outer_dims<typename ToDataType<T>::cpp_type>();
    val             = tensor_map(static_cast<s64>(dim0_idx), static_cast<s64>(dim1_idx));
}

template void Tensor::get<f32>(size_t, size_t, f32&) const;
template void Tensor::get<f64>(size_t, size_t, f64&) const;
template void Tensor::get<s32>(size_t, size_t, s32&) const;
template void Tensor::get<u32>(size_t, size_t, u32&) const;
template void Tensor::get<s64>(size_t, size_t, s64&) const;
template void Tensor::get<u64>(size_t, size_t, u64&) const;
template void Tensor::get<std::string>(size_t, size_t, std::string&) const;

/* ------------------------- Setters ------------------------- */

template<typename T>
void Tensor::set(Math::FastMatrix<T> const& mat, bool transpose) {
    tf::int64 rows = transpose ? mat.nColumns() : mat.nRows();
    tf::int64 cols = transpose ? mat.nRows() : mat.nColumns();
    tensor_.reset(new tf::Tensor(ToDataType<T>::tf_type, {rows, cols}));
    auto tensor_map = tensor_->flat_outer_dims<typename ToDataType<T>::cpp_type>();
    if (transpose) {
        // if we transpose we can iterate over both matrices linearly
        for (u32 c = 0u; c < mat.nColumns(); c++) {
            for (u32 r = 0u; r < mat.nRows(); r++) {
                tensor_map(c, r) = mat.at(r, c);
            }
        }
    }
    else {
        // as tf uses row-major and sprint col major we will have nonlinear
        // memory access in at least one case, we opt to do linear writes
        for (u32 r = 0u; r < mat.nRows(); r++) {
            for (u32 c = 0u; c < mat.nColumns(); c++) {
                tensor_map(r, c) = mat.at(r, c);
            }
        }
    }
}

template void Tensor::set<f64>(Math::FastMatrix<f64> const&, bool);
template void Tensor::set<f32>(Math::FastMatrix<f32> const&, bool);
template void Tensor::set<s64>(Math::FastMatrix<s64> const&, bool);
template void Tensor::set<u64>(Math::FastMatrix<u64> const&, bool);
template void Tensor::set<s32>(Math::FastMatrix<s32> const&, bool);
template void Tensor::set<u32>(Math::FastMatrix<u32> const&, bool);

template<typename T>
void Tensor::set(std::vector<Math::FastMatrix<T>> const& batches, bool transpose) {
    require_gt(batches.size(), 0ul);
    u32 rows = 0u;
    u32 cols = 0u;

    for (auto const& b : batches) {
        rows = std::max(rows, transpose ? b.nColumns() : b.nRows());
        cols = std::max(cols, transpose ? b.nRows() : b.nColumns());
    }

    tensor_.reset(new tf::Tensor(ToDataType<T>::tf_type, {static_cast<tf::int64>(batches.size()), rows, cols}));
    auto tensor_map = tensor_->flat_outer_dims<typename ToDataType<T>::cpp_type, 3>();
    for (size_t b = 0ul; b < batches.size(); b++) {
        auto const& m = batches[b];
        // in these loops we always use indices (r and c) which relate to the original matrix order,
        // but rows and cols refers to the size of the tensorflow tensor, thus length checks are asymetrical in the transposed case
        if (transpose) {
            for (u32 c = 0u; c < m.nColumns(); c++) {
                for (u32 r = 0u; r < m.nRows(); r++) {
                    tensor_map(b, c, r) = m.at(r, c);
                }
                for (u32 r = m.nRows(); r < cols; r++) {
                    tensor_map(b, c, r) = T(0);
                }
            }
            for (u32 c = m.nColumns(); c < rows; c++) {
                for (u32 r = 0u; r < cols; r++) {
                    tensor_map(b, c, r) = T(0);
                }
            }
        }
        else {
            for (u32 r = 0u; r < m.nRows(); r++) {
                for (u32 c = 0u; c < m.nColumns(); c++) {
                    tensor_map(b, r, c) = m.at(r, c);
                }
                for (u32 c = m.nColumns(); c < cols; c++) {
                    tensor_map(b, r, c) = T(0);
                }
            }
            for (u32 r = m.nRows(); r < rows; r++) {
                for (u32 c = 0u; c < cols; c++) {
                    tensor_map(b, r, c) = T(0);
                }
            }
        }
    }
}

template void Tensor::set<f32>(std::vector<Math::FastMatrix<f32>> const&, bool);
template void Tensor::set<f64>(std::vector<Math::FastMatrix<f64>> const&, bool);
template void Tensor::set<s64>(std::vector<Math::FastMatrix<s64>> const&, bool);
template void Tensor::set<u64>(std::vector<Math::FastMatrix<u64>> const&, bool);
template void Tensor::set<s32>(std::vector<Math::FastMatrix<s32>> const&, bool);
template void Tensor::set<u32>(std::vector<Math::FastMatrix<u32>> const&, bool);

template<typename T>
void Tensor::set(Math::FastVector<T> const& vec) {
    tensor_.reset(new tf::Tensor(ToDataType<T>::tf_type, {vec.size()}));
    auto tensor_map = tensor_->flat<typename ToDataType<T>::cpp_type>();
    std::copy(vec.begin(), vec.end(), &tensor_map(0));
}

template void Tensor::set<f32>(Math::FastVector<f32> const&);
template void Tensor::set<f64>(Math::FastVector<f64> const&);
template void Tensor::set<s64>(Math::FastVector<s64> const&);
template void Tensor::set<u64>(Math::FastVector<u64> const&);
template void Tensor::set<s32>(Math::FastVector<s32> const&);
template void Tensor::set<u32>(Math::FastVector<u32> const&);

template<typename T>
void Tensor::set(std::vector<T> const& vec) {
    tensor_.reset(new tf::Tensor(ToDataType<T>::tf_type, {static_cast<tf::int64>(vec.size())}));
    auto tensor_map = tensor_->flat<typename ToDataType<T>::cpp_type>();
    std::copy(vec.begin(), vec.end(), &tensor_map(0));
}

template void Tensor::set<f32>(std::vector<f32> const&);
template void Tensor::set<f64>(std::vector<f64> const&);
template void Tensor::set<s32>(std::vector<s32> const&);
template void Tensor::set<u32>(std::vector<u32> const&);
template void Tensor::set<s64>(std::vector<s64> const&);
template void Tensor::set<u64>(std::vector<u64> const&);

template<typename T>
void Tensor::set(T const& val) {
    tensor_.reset(new tf::Tensor(ToDataType<T>::tf_type, {}));
    tensor_->scalar<typename ToDataType<T>::cpp_type>()() = val;
}

template void Tensor::set<f32>(f32 const&);
template void Tensor::set<f64>(f64 const&);
template void Tensor::set<s32>(s32 const&);
template void Tensor::set<u32>(u32 const&);
template void Tensor::set<s64>(s64 const&);
template void Tensor::set<u64>(u64 const&);
template void Tensor::set<std::string>(std::string const&);

}  // namespace Tensorflow
