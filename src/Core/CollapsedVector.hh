/** Copyright 2024 RWTH Aachen University. All rights reserved.
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

#ifndef COLLAPSED_VECTOR_HH
#define COLLAPSED_VECTOR_HH

#include <stdexcept>
#include <vector>

namespace Core {

/*
 * Vector that is automatically collapsed down to one element if all entries have the same value.
 * From the outside it behaves like a normal vector. Whether it is collapsed or not only matters internally.
 *
 * Example:
 *
 * CollapsedVector<int> v;  // Internal data of v is [] with logical size 0
 * v.push_back(5);  // Normal push. Internal data of v is [5] with logical size 1
 * v.push_back(5);  // v stays collapsed. Internal data of v is [5] with logical size 2
 * std::cout << v.size() << std::endl;  // Prints "2"
 * std::cout << v[1] << std::endl;  // Prints "5"
 * v.push_back(6);  // v is expanded because the new value is different. Internal data of v is [5, 5, 6] with logical size 3
 * std::cout << v[2] << std::endl;  // Prints "6"
 * v.clear();  // Internal data of v is [] with logical size 0.
 */
template<typename T>
class CollapsedVector {
public:
    CollapsedVector();
    CollapsedVector(size_t size, const T& value);

    void     push_back(const T& value);
    const T& operator[](size_t idx) const;
    const T& at(size_t idx) const;
    size_t   size() const noexcept;
    void     clear() noexcept;
    void     reserve(size_t size);
    const T& front() const;

private:
    std::vector<T> data_;
    size_t         logicalSize_;
};

/*
 * Implementations
 */

template<typename T>
CollapsedVector<T>::CollapsedVector()
        : data_(), logicalSize_(0ul) {}

template<typename T>
CollapsedVector<T>::CollapsedVector(size_t size, const T& value)
        : data_(1ul, value), logicalSize_(size) {}

template<typename T>
void CollapsedVector<T>::push_back(const T& value) {
    if (data_.empty() or data_.size() > 1ul) {
        // Vector is not collapsed so the value can be pushed as usual
        data_.push_back(value);
    }
    else if (value != data_.front()) {
        // Data contains only one element and thus may be collapsed.
        // If the new value is different to the present one, uncollapse it
        // and push the new value
        data_.reserve(logicalSize_ + 1);
        data_.resize(logicalSize_, data_.front());  // Note: this is a no-op if the logical size is 1
        data_.push_back(value);
    }
    // Otherwise the value is the same as the present one so the vector stays collapsed
    // and the new value does not have to be pushed

    ++logicalSize_;  // Logical size always increases
}

template<typename T>
const T& CollapsedVector<T>::operator[](size_t idx) const {
    if (data_.size() == 1ul) {  // Vector may be collapsed
        return front();
    }
    return data_.operator[](idx);
}

template<typename T>
const T& CollapsedVector<T>::at(size_t idx) const {
    if (idx >= logicalSize_) {
        throw std::out_of_range("Trying to access illegal index of CollapsedVector");
    }
    return operator[](idx);
}

template<typename T>
size_t CollapsedVector<T>::size() const noexcept {
    return logicalSize_;
}

template<typename T>
void CollapsedVector<T>::clear() noexcept {
    data_.clear();
    logicalSize_ = 0ul;
}

template<typename T>
void CollapsedVector<T>::reserve(size_t size) {
    data_.reserve(size);
}

template<typename T>
const T& CollapsedVector<T>::front() const {
    return data_.front();
}

}  // namespace Core

#endif  // COLLAPSED_VECTOR_HH
