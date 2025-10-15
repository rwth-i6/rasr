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
#ifndef CUDAVECTOR_HH_
#define CUDAVECTOR_HH_

#include <Math/CudaDataStructure.hh>
#include <Math/CudaMatrixKernelsWrapper.hh>
#include <Math/FastVector.hh>
#include <algorithm>

namespace Math {

template<typename T>
class CudaMatrix;

/*
 *
 * check maximum vector dimensions (needs to work with Cuda)
 *
 */

/*
 * CudaVector
 *
 * Vector class that makes use of GPU parallelization when compile with MODULE_CUDA and GPU is available.
 * Derives from FastVector.
 * Design analogous to CudaMatrix
 */

template<typename T>
class CudaVector : protected FastVector<T>, public CudaDataStructure {
    typedef FastVector<T> Precursor;
    friend class CudaVector<float>;
    friend class CudaVector<double>;
    friend class CudaMatrix<T>;
    friend class CudaMatrix<float>;
    friend class CudaMatrix<double>;

protected:
    using CudaDataStructure::cublasHandle;
    using CudaDataStructure::gpuMode_;
    using Precursor::elem_;
    using Precursor::nRows_;

protected:
    mutable bool                 isComputing_;
    T*                           d_elem_;
    std::map<int, cudaStream_t*> streamPool_;
    CudaVector<T>*               d_vec_ones_;

public:
    // constructor with memory allocation
    CudaVector(u32 nRows = 0);

    // copy constructor
    CudaVector(const CudaVector<T>& vector);

    virtual ~CudaVector();

private:
    bool allocateGpuMemory();

public:
    void resize(u32 newSize, T value = 0, bool allocOnly = false);
    void clear();
    u32  nRows() const {
        return Precursor::nRows();
    }
    u32 nColumns() const {
        return 1;
    }
    u32 size() const {
        return nRows();
    }
    bool empty() const {
        return Precursor::empty();
    }
    T&       operator()(u32 index);
    T&       operator[](u32 index);
    const T& operator()(u32 index) const;
    const T& operator[](u32 index) const;
    T&       at(u32 index);
    const T& at(u32 index) const;

public:  // iterators
    typedef T        value_type;
    typedef T*       iterator;
    typedef const T* const_iterator;
    iterator         begin();
    const_iterator   begin() const;
    iterator         end();
    const_iterator   end() const;
    T*               d_elem() {
        return d_elem_;
    }
    T* elem() {
        return Precursor::elem_;
    }

public:
    // memory copy
    template<typename S>
    void copy(const CudaVector<S>& vector);

    template<typename S>
    void copy(const Vector<S>& vector);

    // this[offset+i] := x[i] for 0 <= i < num_elements
    template<typename S>
    void setAtOffset(const Math::CudaVector<S>& x, u32 offset = 0, u32 num_elements = 0);

    // convert to Math::Vector
    template<typename S>
    void convert(Vector<S>& vector) const;

    // resize to size of x & allocate
    // side effect: after resize content is meaningless
    void copyStructure(const CudaVector<T>& x);

    bool isFinite() const;

public:
    // addition of a vector (scaling of the vector possible)
    template<typename S>
    void add(const CudaVector<S>& vector, S scale = 1);

    template<typename S>
    void addAtOffset(const CudaVector<S>& vector, S scale = 1, u32 offset = 0, u32 num_elements = 0);

    // just a wrapper
    template<typename S>
    void axpy(int n, S alpha, const S* x, int incx, T* y, int incy);

    // add a constant to each element of the vector
    void addConstantElementwise(T c);

    // scaling of the vector
    void scale(T value);

    T sumOfSquares() const {
        return dot(*this);
    }

    // vector dot product (result = this^T * v)
    T dot(const CudaVector<T>& vector) const;

    // set i-th component of vector to inner product of i-th column of A and i-th column of B
    void columnwiseInnerProduct(const CudaMatrix<T>& A, const CudaMatrix<T>& B);

    // multiply corresponding elements (this = this .* v)
    void elementwiseMultiplication(const CudaVector<T>& v);

    // divide corresponding elements (this = this ./ v)
    void elementwiseDivision(const CudaVector<T>& v);

    // division by a constant
    void divide(T value);

    // set all elements to zero
    void setToZero();
    void setToZeroAsync(int stream = 0);

    // set all elements to value
    void fill(T value);

    // set all values < threshold to threshold
    void ensureMinimalValue(const T threshold);

public:
    // l1-norm of vector
    T asum() const;
    // just an alias
    T l1norm() const;

    T sum();
    T mean() const {
        return this->sum() / this->nRows();
    }

    // *this = (*this) + scale * matrixColumnSum
    void addSummedRows(const CudaMatrix<T>& matrix, const T scale = 1.0);

    // slightly faster version of addSummedRows that uses intermediate storage
    void addSummedRows(const CudaMatrix<T>& matrix, CudaMatrix<T>& tmp, const T scale = 1.0);

    // *this = (*this) + scale * matrixRowSum
    void addSummedColumns(const CudaMatrix<T>& matrix, const T scale = 1.0);
    template<typename S>
    void addSummedColumns(const CudaMatrix<S>& matrix, const S scale = 1.0);

    // like addSummedColumns, but squares each matrix entry before summation
    void addSquaredSummedColumns(const CudaMatrix<T>& matrix, const T scale = 1.0);

    // this = maximum of each column in X
    void getMaxOfColumns(const CudaMatrix<T>& X);

    // slightly faster version of getMaxOfColumns that uses intermediate storage
    void getMaxOfColumns(const CudaMatrix<T>& X, CudaMatrix<T>& tmp);

    void l1clipping(const T value);

    void clip(const T maxAbsValue);

    // euclidean norm => ?nrm2 s, d, sc, dz Vector 2-norm (Euclidean norm) a normal
    T normEuclidean() const;

    // apply exp to each element of matrix
    void exp();

    // need assignment operator, because we have a copy constructor
    // pass by value ! (needed for temporary object creation)
    CudaVector<T>& operator=(CudaVector<T> X);

    void swap(CudaVector<T>& x);

    // for each i: this[i] = sign(X[i]) with sign(0) := 0
    void sign(const CudaVector<T>& X);

    // this[i] = pow(this[i], p)
    void pow(T p);

    // this = entropy of each column in X
    void columnEntropy(const CudaMatrix<T>& X);

public:  // GPU handling
    void initComputation(bool sync = true) const;
    void finishComputation(bool sync = true) const;
    bool isComputing() const {
        return isComputing_;
    }

public:
    void print(std::ostream& os) const;
    void printToFile(const std::string& filename) const;
    void write(Core::XmlWriter& os) const;
    void show() const;
    void syncAndShow() const;
};

// constructors

template<typename T>
CudaVector<T>::CudaVector(u32 nRows)
        : Precursor(nRows),
          CudaDataStructure(),
          isComputing_(false),
          d_elem_(0),
          d_vec_ones_(0) {
    allocateGpuMemory();
}

template<typename T>
CudaVector<T>::CudaVector(const CudaVector<T>& vector)
        : Precursor(vector),
          CudaDataStructure(vector),
          isComputing_(false),
          d_elem_(0),
          d_vec_ones_(0) {
    require(!isComputing_);
    allocateGpuMemory();
}

template<typename T>
bool CudaVector<T>::allocateGpuMemory() {
    int result = 0;
    if (gpuMode_) {
        if (d_elem_) {
            result = Cuda::free(d_elem_);
            require_eq(result, 0);
        }
        result = Cuda::alloc(d_elem_, nRows_);
        require_eq(result, 0);
        if ((d_elem_ == 0) && (nRows_ > 0)) {
            Core::Application::us()->criticalError("GPU: Failed to allocate memory.");
        }
    }
    return true;
}

template<typename T>
CudaVector<T>::~CudaVector() {
    if (gpuMode_) {
        Cuda::free(d_elem_);
    }
}

template<typename T>
void CudaVector<T>::resize(u32 newSize, T value, bool allocOnly) {
    u32  oldSize      = nRows_;
    bool allocOnlyCpu = allocOnly || (gpuMode_ && isComputing_);
    bool allocOnlyGpu = allocOnly || (gpuMode_ && !isComputing_);
    Precursor::resize(newSize, value, allocOnlyCpu);
    if (gpuMode_) {
        T* old_d_elem = d_elem_;
        nRows_        = newSize;
        if (newSize == 0 && d_elem_) {
            cudaError_t result = Cuda::free(d_elem_);
            require_eq(result, cudaSuccess);
            nRows_  = 0;
            d_elem_ = 0;
            return;
        }
        bool reallocate = newSize != oldSize;
        if (reallocate) {
            cudaError_t result;
            // allocate memory
            result = Cuda::alloc(d_elem_, newSize);
            require_eq(result, cudaSuccess);
            if ((d_elem_ == 0) && (newSize > 0))
                Core::Application::us()->criticalError("GPU: Failed to allocate memory.");
            if (allocOnlyGpu)
                return;

            // copy old values
            u32 nElements = std::min(oldSize, newSize);
            result        = Cuda::memcpy(d_elem_, old_d_elem, nElements);
            require_eq(result, cudaSuccess);
            // free old memory
            if (old_d_elem) {
                result = Cuda::free(old_d_elem);
                require_eq(result, cudaSuccess);
            }
            // fill new part
            if (newSize > oldSize)
                Cuda::fill(d_elem_ + oldSize, value, newSize - oldSize, 1);
        }
    }
}

template<typename T>
void CudaVector<T>::clear() {
    if (gpuMode_ && d_elem_) {
        Cuda::free(d_elem_);
        d_elem_ = 0;
    }
    Precursor::clear();
}

template<typename T>
T& CudaVector<T>::operator()(u32 index) {
    require(!isComputing_);
    return elem_[index];
}

template<typename T>
T& CudaVector<T>::operator[](u32 index) {
    require(!isComputing_);
    return (*this)(index);
}

template<typename T>
const T& CudaVector<T>::operator()(u32 index) const {
    require(!isComputing_);
    return elem_[index];
}

template<typename T>
const T& CudaVector<T>::operator[](u32 index) const {
    require(!isComputing_);
    return (*this)(index);
}

template<typename T>
T& CudaVector<T>::at(u32 index) {
    require(!isComputing_);
    return Precursor::at(index);
}

template<typename T>
const T& CudaVector<T>::at(u32 index) const {
    require(!isComputing_);
    return Precursor::at(index);
}

template<typename T>
T* CudaVector<T>::begin() {
    require(!isComputing_);
    return elem_;
}

template<typename T>
const T* CudaVector<T>::begin() const {
    require(!isComputing_);
    return elem_;
}

template<typename T>
T* CudaVector<T>::end() {
    require(!isComputing_);
    return &elem_[nRows_];
}

template<typename T>
const T* CudaVector<T>::end() const {
    require(!isComputing_);
    return &elem_[nRows_];
}

// TODO CUDA implementation works only for identical types !!
template<typename T>
template<typename S>
void CudaVector<T>::copy(const Math::CudaVector<S>& x) {
    require(isComputing_ == x.isComputing_);
    if (gpuMode_ && isComputing_) {
        require(typeid(S) == typeid(T));  // TODO...
        require_eq(x.nRows(), nRows_);
        require(d_elem_);
        cudaError_t result = Cuda::memcpy(d_elem_, x.d_elem_, nRows_);
        if (result != cudaSuccess) {
            const char* msg = Cuda::getErrorString(result);
            Core::Application::us()->criticalError("GPU: call to Cuda::memcpy() failed: ")
                    << msg << " (" << result << ")";
        }
    }
    else
        Precursor::copy(x);
}

template<typename T>
template<typename S>
void CudaVector<T>::copy(const Vector<S>& vector) {
    require(!isComputing_);
    Precursor::copy(vector);
}

template<typename T>
template<typename S>
void CudaVector<T>::setAtOffset(const Math::CudaVector<S>& x, u32 offset, u32 num_elements) {
    require(isComputing_ == x.isComputing_);
    if (gpuMode_ && isComputing_) {
        require(typeid(S) == typeid(T));  // TODO...
        require_le(num_elements + offset, nRows_);
        require(d_elem_);
        cudaError_t result = Cuda::memcpy(d_elem_ + offset, x.d_elem_, num_elements);
        if (result != cudaSuccess) {
            const char* msg = Cuda::getErrorString(result);
            Core::Application::us()->criticalError("GPU: call to Cuda::setAtOffset() failed: ")
                    << msg << " (" << result << ")";
        }
    }
    else
        Precursor::setAtOffset(x, offset, num_elements);
}

template<typename T>
template<typename S>
void CudaVector<T>::convert(Vector<S>& x) const {
    require(!isComputing_);
    Precursor::convert(x);
}

template<typename T>
void CudaVector<T>::copyStructure(const Math::CudaVector<T>& x) {
    if (x.nRows_ != nRows_)
        resize(x.nRows_);
}

template<typename T>
bool CudaVector<T>::isFinite() const {
    require(!isComputing_);
    return Precursor::isFinite();
}

// ----------------------------------------------------------------------------
//		Math operations
// ----------------------------------------------------------------------------

template<typename T>
template<typename S>
void CudaVector<T>::add(const CudaVector<S>& vector, S scale) {
    require(isComputing_);
    require(vector.isComputing_);
    if (gpuMode_) {
        require_eq(nRows_, vector.nRows());
        int result = Cuda::axpy(cublasHandle, nRows_, scale, vector.d_elem_, 1, d_elem_, 1);
        require_eq(result, 0);
    }
    else {
        Precursor::add(vector, scale);
    }
}

template<typename T>
template<typename S>
void CudaVector<T>::addAtOffset(const CudaVector<S>& vector, S scale, u32 offset, u32 num_elements) {
    require(isComputing_);
    require(vector.isComputing_);
    if (gpuMode_) {
        require_le(offset + num_elements, nRows_);
        require_lt(num_elements, vector.nRows());
        int result = Cuda::axpy(cublasHandle, num_elements, scale, vector.d_elem_, 1, d_elem_ + offset, 1);
        require_eq(result, 0);
    }
    else {
        Math::axpy<S, T>(num_elements, scale, vector.begin(), 1, elem_ + offset, 1);
    }
}

template<typename T>
template<typename S>
void CudaVector<T>::axpy(int n, S alpha, const S* x, int incx, T* y, int incy) {
    if (gpuMode_) {
        int result = Cuda::axpy(cublasHandle, n, alpha, x, incx, y, incy);
        require_eq(result, 0);
    }
    else {
        Math::axpy<S, T>(n, alpha, x, incx, y, incy);
    }
}

template<typename T>
void CudaVector<T>::addConstantElementwise(T c) {
    require(isComputing_);
    if (gpuMode_)
        Cuda::addConstantElementwise(c, d_elem_, nRows_, 1);
    else
        Precursor::addConstantElementwise(c);
}

template<typename T>
void CudaVector<T>::scale(T scale) {
    require(isComputing_);
    if (gpuMode_) {
        int result = Cuda::scal(cublasHandle, nRows_, scale, d_elem_, 1);
        require_eq(result, 0);
    }
    else {
        Precursor::scale(scale);
    }
}

template<typename T>
T CudaVector<T>::dot(const CudaVector<T>& vector) const {
    require(isComputing_);
    require(vector.isComputing_);
    if (gpuMode_) {
        T   dotProduct = 0;
        int result     = Cuda::dot(cublasHandle, nRows_, vector.d_elem_, 1, d_elem_, 1, dotProduct);
        require_eq(result, 0);
        return dotProduct;
    }
    else {
        return Precursor::dot(vector);
    }
}

template<typename T>
void CudaVector<T>::columnwiseInnerProduct(const Math::CudaMatrix<T>& A, const Math::CudaMatrix<T>& B) {
    require(isComputing_);
    require(A.isComputing());
    require(B.isComputing());
    if (gpuMode_) {
        require_eq(A.nRows(), B.nRows());
        require_eq(A.nColumns(), B.nColumns());
        require_eq(nRows_, A.nColumns());
        u32 matrixRows = A.nRows();
        // TODO: for now only parallelized within the columns, implement a better parallelization
        for (u32 column = 0; column < A.nColumns(); column++) {
            T   dotProduct = 0;
            int result     = Cuda::dot(cublasHandle, matrixRows, A.d_elem_ + column * matrixRows, 1,
                                       B.d_elem_ + column * matrixRows, 1, dotProduct);
            require_eq(result, 0);
            Cuda::copyToGpu(d_elem_ + column, &dotProduct, 1);
        }
    }
    else {
        Precursor::columnwiseInnerProduct(A, B);
    }
}

template<typename T>
void CudaVector<T>::elementwiseMultiplication(const CudaVector<T>& v) {
    require(isComputing_);
    require(v.isComputing_);
    if (gpuMode_) {
        require_eq(nRows_, v.nRows_);
        Cuda::elementwiseMultiplication(d_elem_, v.d_elem_, v.nRows_, 1);
    }
    else {
        Precursor::elementwiseMultiplication(v);
    }
}

template<typename T>
void CudaVector<T>::elementwiseDivision(const CudaVector<T>& v) {
    require(isComputing_);
    require(v.isComputing_);
    if (gpuMode_) {
        require_eq(nRows_, v.nRows_);
        Cuda::elementwiseDivision(d_elem_, v.d_elem_, v.nRows_, 1);
    }
    else {
        Precursor::elementwiseDivision(v);
    }
}

template<typename T>
void CudaVector<T>::divide(T value) {
    require(isComputing_);
    if (gpuMode_) {
        scale((T)1 / value);
    }
    else {
        Precursor::divide(value);
    }
}

template<typename T>
void CudaVector<T>::setToZero() {
    if (gpuMode_ && isComputing_) {
        int result = Cuda::memSet(d_elem_, 0, nRows_);
        require_eq(result, 0);
    }
    else {
        Precursor::setToZero();
    }
}
template<typename T>
void CudaVector<T>::setToZeroAsync(int stream) {
    if (gpuMode_ && isComputing_) {
        if (streamPool_.find(stream) == streamPool_.end()) {
            cudaStream_t* str   = Cuda::streamCreate();
            streamPool_[stream] = str;
        }
        int result = Cuda::memSet(d_elem_, 0, nRows_, *(streamPool_[stream]));
        require_eq(result, 0);
    }
    else
        Precursor::setToZero();
}

template<typename T>
void CudaVector<T>::fill(T value) {
    require(isComputing_);
    if (gpuMode_) {
        Cuda::fill(d_elem_, value, nRows_, 1);
    }
    else {
        Precursor::fill(value);
    }
}

template<typename T>
void CudaVector<T>::ensureMinimalValue(const T threshold) {
    require(isComputing_);
    if (gpuMode_) {
        Cuda::ensureMinimalValue(d_elem_, threshold, nRows_, 1);
    }
    else {
        Precursor::ensureMinimalValue(threshold);
    }
}

template<typename T>
T CudaVector<T>::asum() const {
    require(isComputing_);
    int result = 0;
    if (gpuMode_) {
        T sum  = 0;
        result = Cuda::asum(cublasHandle, nRows_, d_elem_, 1, &sum);
        require_eq(result, 0);
        return sum;
    }
    else {
        return Precursor::asum();
    }
}

template<typename T>
T CudaVector<T>::l1norm() const {
    return asum();
}

template<typename T>
T CudaVector<T>::sum() {
    require(isComputing_);
    if (gpuMode_) {
        T sum = 0;
        if (d_vec_ones_ == 0) {
            d_vec_ones_ = new CudaVector<T>;
            d_vec_ones_->initComputation();
        }
        if (d_vec_ones_->size() != nRows_) {
            d_vec_ones_->resize(nRows_);
            d_vec_ones_->fill(1);
        }
        int result = Cuda::dot(cublasHandle, nRows_, d_elem_, 1, d_vec_ones_->d_elem_, 1, sum);
        require_eq(result, 0);
        return sum;
    }
    else {
        return Precursor::sum();
    }
}

template<typename T>
template<typename S>
void CudaVector<T>::addSummedColumns(const CudaMatrix<S>& matrix, const S scale) {
    require(isComputing_);
    require(matrix.isComputing());
    require_eq(matrix.nRows(), nRows_);
    if (gpuMode_) {
        Cuda::addSummedColumns(d_elem_, matrix.d_elem_, matrix.nRows_, matrix.nColumns_, scale);
    }
    else {
        Precursor::addSummedColumns(matrix, scale);
    }
}

template<typename T>
void CudaVector<T>::addSummedColumns(const CudaMatrix<T>& matrix, const T scale) {
    require(isComputing_);
    require(matrix.isComputing());
    require_eq(matrix.nRows(), nRows_);
    if (gpuMode_) {
        if (d_vec_ones_ == 0) {
            d_vec_ones_ = new CudaVector<T>;
            d_vec_ones_->initComputation();
        }
        if (d_vec_ones_->size() != matrix.nColumns_) {
            d_vec_ones_->resize(matrix.nColumns_);
            d_vec_ones_->fill(1);
        }

        int result = Cuda::gemv(cublasHandle, /*transposed=*/false, matrix.nRows_, matrix.nColumns_, scale, matrix.d_elem_, matrix.nRows_,
                                d_vec_ones_->d_elem_, 1, T(1.0),
                                /*output=*/d_elem_, 1);
        require_eq(result, 0);
    }
    else {
        Precursor::addSummedColumns(matrix, scale);
    }
}

template<typename T>
void CudaVector<T>::addSquaredSummedColumns(const CudaMatrix<T>& matrix, const T scale) {
    require(isComputing_);
    require(matrix.isComputing());
    require_eq(matrix.nRows(), nRows_);
    if (gpuMode_) {
        Cuda::addSquaredSummedColumns(d_elem_, matrix.d_elem_, matrix.nRows_, matrix.nColumns_, scale);
    }
    else {
        Precursor::addSquaredSummedColumns(matrix, scale);
    }
}

template<typename T>
void CudaVector<T>::addSummedRows(const CudaMatrix<T>& matrix, const T scale) {
    require(isComputing_);
    require(matrix.isComputing());
    require_eq(matrix.nColumns(), nRows_);
    if (gpuMode_) {
        Cuda::addSummedRows(d_elem_, matrix.d_elem_, matrix.nRows_, matrix.nColumns_, scale);
    }
    else {
        Precursor::addSummedRows(matrix, scale);
    }
}

template<typename T>
void CudaVector<T>::addSummedRows(const CudaMatrix<T>& matrix, CudaMatrix<T>& tmp, const T scale) {
    require(isComputing_);
    require(matrix.isComputing());
    require(tmp.isComputing());
    require_eq(matrix.nColumns(), nRows_);
    require_eq(tmp.nColumns(), matrix.nColumns());
    if (gpuMode_) {
        Cuda::addSummedRows(d_elem_, matrix.d_elem_, matrix.nRows_, matrix.nColumns_, tmp.d_elem_, tmp.nRows_, scale);
    }
    else {
        Precursor::addSummedRows(matrix, scale);
    }
}

template<typename T>
void CudaVector<T>::getMaxOfColumns(const CudaMatrix<T>& matrix) {
    require(isComputing_);
    require(matrix.isComputing());
    require_eq(matrix.nColumns(), nRows_);
    if (gpuMode_) {
        Cuda::getMaxOfColumns(d_elem_, matrix.d_elem_, matrix.nRows_, matrix.nColumns_);
    }
    else {
        Precursor::getMaxOfColumns(matrix);
    }
}

template<typename T>
void CudaVector<T>::getMaxOfColumns(const CudaMatrix<T>& X, CudaMatrix<T>& tmp) {
    require(isComputing_);
    require(X.isComputing());
    require(tmp.isComputing());
    require_eq(X.nColumns(), nRows_);
    require_eq(tmp.nColumns(), X.nColumns());
    if (gpuMode_)
        Cuda::getMaxOfColumns(d_elem_, X.d_elem_, X.nRows_, X.nColumns_, tmp.d_elem_, tmp.nRows_);
    else
        Precursor::getMaxOfColumns(X);
}

template<typename T>
void CudaVector<T>::l1clipping(const T value) {
    require(isComputing_);
    if (gpuMode_) {
        Cuda::l1clipping(d_elem_, nRows_, 1, value);
    }
    else {
        Precursor::l1clipping(value);
    }
}

template<typename T>
void CudaVector<T>::clip(const T maxAbsValue) {
    require(isComputing_);
    if (gpuMode_) {
        Cuda::clip(d_elem_, nRows_, maxAbsValue);
    }
    else {
        Precursor::l1clipping(maxAbsValue);
    }
}

template<typename T>
T CudaVector<T>::normEuclidean() const {
    require(isComputing_);
    if (gpuMode_) {
        T result = 0;
        Cuda::nrm2(cublasHandle, nRows_, d_elem_, 1, &result);
        return result;
    }
    else {
        return Precursor::normEuclidean();
    }
}

template<typename T>
void CudaVector<T>::exp() {
    require(isComputing_);
    if (gpuMode_)
        Cuda::exp(d_elem_, nRows_, 1);
    else
        Precursor::exp();
}

template<typename T>
CudaVector<T>& CudaVector<T>::operator=(CudaVector<T> rhs) {
    swap(rhs);
    return *this;
}

template<typename T>
void CudaVector<T>::swap(CudaVector<T>& x) {
    require_eq(x.gpuMode_, gpuMode_);
    require_eq(x.isComputing_, isComputing_);
    Precursor::swap(x);
    std::swap(d_elem_, x.d_elem_);
}

template<typename T>
void CudaVector<T>::sign(const CudaVector<T>& X) {
    require(isComputing_);
    require(X.isComputing_);
    require_eq(X.nRows(), nRows_);
    if (gpuMode_) {
        Cuda::sign(d_elem_, X.d_elem_, nRows_, 1);
    }
    else {
        Core::Application::us()->error("CudaVector::sign not implemented on CPU yet.");
    }
}

template<typename T>
void CudaVector<T>::pow(T p) {
    require(isComputing_);
    if (gpuMode_) {
        Cuda::pow(d_elem_, nRows_, 1, p);
    }
    else {
        Precursor::pow(p);
    }
}

template<typename T>
void CudaVector<T>::columnEntropy(const CudaMatrix<T>& X) {
    require(isComputing_);
    if (gpuMode_) {
        Core::Application::us()->error("CudaVector::columnEntropy not implemented on GPU yet.");
    }
    else {
        Precursor::columnEntropy(X);
    }
}

// ----------------------------------------------------------------------------
//		GPU handling
// ----------------------------------------------------------------------------

template<typename T>
void CudaVector<T>::initComputation(bool sync) const {
    if (gpuMode_ && !isComputing_) {
        if (sync) {
            Cuda::copyToGpu(d_elem_, elem_, nRows_);
            Math::Cuda::deviceSync(Math::CudaDataStructure::hasGpu());
        }
    }
    isComputing_ = true;
}

template<typename T>
void CudaVector<T>::finishComputation(bool sync) const {
    if (gpuMode_ && isComputing_) {
        if (d_elem_ && sync)
            Cuda::copyFromGpu(elem_, d_elem_, nRows_);
    }
    isComputing_ = false;
}

template<typename T>
void CudaVector<T>::print(std::ostream& os) const {
    bool wasComputing = isComputing_;
    if (wasComputing)
        finishComputation(true);
    Precursor::print(os);
    if (wasComputing)
        initComputation(false);
}

template<typename T>
void CudaVector<T>::printToFile(const std::string& filename) const {
    bool wasComputing = isComputing_;
    if (wasComputing)
        finishComputation(true);
    Precursor::printToFile(filename);
    if (wasComputing)
        initComputation(false);
}

template<typename T>
void CudaVector<T>::write(Core::XmlWriter& os) const {
    require(!isComputing_);
    Precursor::write(os);
}

template<typename T>
Core::XmlWriter& operator<<(Core::XmlWriter& os, const CudaVector<T>& v) {
    v.write(os);
    return os;
}

template<typename T>
void CudaVector<T>::show() const {
    require(!isComputing_);
    Precursor::show();
}

template<typename T>
void CudaVector<T>::syncAndShow() const {
    if (isComputing_ && gpuMode_) {
        Cuda::copyFromGpu(elem_, d_elem_, nRows_);
    }
    Precursor::show();
}

}  // namespace Math

#endif /* CUDAVECTOR_HH_ */
