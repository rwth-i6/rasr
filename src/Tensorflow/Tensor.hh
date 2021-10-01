/** Copyright 2020 RWTH Aachen University. All rights reserved.
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
#ifndef _TENSORFLOW_TENSOR_HH
#define _TENSORFLOW_TENSOR_HH

#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/version.h>

#include <Math/FastMatrix.hh>
#include <Math/FastVector.hh>

namespace Tensorflow {

namespace tf = tensorflow;
using int64  = tf::int64;

#if TF_MAJOR_VERSION < 2
typedef std::string tstring;
#else
typedef tf::tstring tstring;
#endif

class Session;

class Tensor {
public:
    friend Session;

    template<typename... Args>
    static Tensor create(Args... value);

    template<typename T>
    static Tensor zeros(std::initializer_list<int64> dim);

    template<typename T>
    static Tensor zeros(std::vector<int64> const& dim);

    static Tensor concat(Tensor const& a, Tensor const& b, int axis);

    Tensor();
    Tensor(Tensor const& other);
    Tensor(Tensor&& other) = default;
    ~Tensor()              = default;

    bool empty() const;

    Tensor& operator=(Tensor& other);
    Tensor& operator=(Tensor&& other);

    /* -------------------- Getters -------------------- */

    int         numDims() const;
    tf::int64   dimSize(int d) const;
    std::string dimInfo() const;  // usefull for debugging, format: Shape<dim0, dim1, ...>
    std::string dataTypeName() const;

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

    Tensor slice(std::vector<int> const& start, std::vector<int> const& end);

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
    std::unique_ptr<tf::Tensor> tensor_;

    Tensor(tf::Tensor&& tensor);
};

// Implementations for some of Tensors functions (which likely can be inlined)

template<typename... Args>
inline Tensor Tensor::create(Args... value) {
    Tensor res;
    res.set(std::forward<Args>(value)...);
    return res;
}

inline Tensor::Tensor()
        : tensor_(nullptr) {
}

inline Tensor::Tensor(Tensor const& other) {
    if (not other.empty()) {
        tensor_.reset(new tf::Tensor(*other.tensor_));
    }
}

inline bool Tensor::empty() const {
    return tensor_.get() == nullptr;
}

inline Tensor::Tensor(tf::Tensor&& tensor)
        : tensor_(new tf::Tensor(tensor)) {
}

inline Tensor& Tensor::operator=(Tensor& other) {
    *tensor_ = *other.tensor_;
    return *this;
}

inline Tensor& Tensor::operator=(Tensor&& other) {
    std::swap(tensor_, other.tensor_);
    return *this;
}

inline int Tensor::numDims() const {
    if (tensor_) {
        return tensor_->dims();
    }
    return -1;
}

inline tf::int64 Tensor::dimSize(int d) const {
    if (tensor_) {
        return tensor_->dim_size(d);
    }
    return -1;
}

}  // namespace Tensorflow

#endif  // _TENSORFLOW_TENSOR_HH
