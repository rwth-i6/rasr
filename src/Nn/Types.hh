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
#ifndef TYPES_HH_
#define TYPES_HH_

#include <Core/Types.hh>
#include <Flow/Vector.hh>
#include <Search/Types.hh>
#include <Speech/Types.hh>
#include <stdexcept>

namespace Math {
template<typename T>
class CudaMatrix;
template<typename T>
class CudaVector;
}  // namespace Math

namespace Nn {

template<typename T>
class Types {
public:
    typedef Math::CudaMatrix<T> NnMatrix;
    typedef Math::CudaVector<T> NnVector;
};

typedef Flow::Vector<f32>            FeatureVector;
typedef Flow::DataPtr<FeatureVector> FeatureVectorRef;

typedef Search::Score Score;

/*
 * Vector that is automatically collapsed down to one element if all entries have the same value
 * From the outside it can be used like a normal vector, the collapse handling is done internally
 */
template<typename T>
struct CollapsedVector : private std::vector<T> {
    using Precursor = std::vector<T>;

    CollapsedVector()
            : Precursor(), logicalSize_(0ul) {}

    void push_back(const T& value) {
        if (logicalSize_ == Precursor::size()) {  // Vector is currently not collapsed -> just push back as usual
            Precursor::push_back(value);
        }
        else if (value != front()) {  // Vector is currently collapsed and new values is different -> expand out and then push back as usual
            Precursor::resize(logicalSize_, front());
            Precursor::push_back(value);
        }  // else vector is collapsed and new value is the same -> don't push back, just increase logical size
        ++logicalSize_;
    }

    const T& operator[](size_t idx) const {
        if (isCollapsed()) {
            return front();
        }
        return Precursor::operator[](idx);
    }

    const T& at(size_t idx) const {
        if (idx >= logicalSize_) {
            throw std::out_of_range("Trying to access illegal index of CollapsedVector");
        }
        return operator[](idx);
    }

    size_t size() const noexcept {
        return logicalSize_;
    }

    void clear() noexcept {
        Precursor::clear();
        logicalSize_ = 0ul;
    }

    void reserve(size_t size) {
        Precursor::reserve(size);
    }

    const T& front() const {
        return Precursor::front();
    }

private:
    bool isCollapsed() const {
        return Precursor::size() <= 1ul;
    }

    size_t logicalSize_;
};

}  // namespace Nn

#endif /* TYPES_HH_ */
