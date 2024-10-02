
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

#ifndef COLLAPSED_VECTOR_HH
#define COLLAPSED_VECTOR_HH

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace Core {

/*
 * Vector that is automatically collapsed down to one element if all entries have the same value
 * From the outside it can be used like a normal vector, the collapse handling is done internally
 */
template<typename T>
class CollapsedVector : private std::vector<T> {
public:
    using Precursor = std::vector<T>;

    CollapsedVector();

    bool     isCollapsed() const;
    void     push_back(const T& value);
    const T& operator[](size_t idx) const;
    const T& at(size_t idx) const;
    size_t   size() const noexcept;
    void     clear() noexcept;
    void     reserve(size_t size);
    const T& front() const;

private:
    size_t logicalSize_;
};

/*
 * Implementations
 */

template<typename T>
CollapsedVector<T>::CollapsedVector()
        : Precursor(), logicalSize_(0ul) {}

template<typename T>
bool CollapsedVector<T>::isCollapsed() const {
    return Precursor::size() <= 1ul;
}

template<typename T>
void CollapsedVector<T>::push_back(const T& value) {
    if (logicalSize_ == Precursor::size()) {  // Vector is currently not collapsed -> just push back as usual
        Precursor::push_back(value);
    }
    else if (value != front()) {  // Vector is currently collapsed and new values is different -> expand out and then push back as usual
        Precursor::resize(logicalSize_, front());
        Precursor::push_back(value);
    }  // else vector is collapsed and new value is the same -> don't push back, just increase logical size
    ++logicalSize_;
}

template<typename T>
const T& CollapsedVector<T>::operator[](size_t idx) const {
    if (isCollapsed()) {
        return front();
    }
    return Precursor::operator[](idx);
}

template<typename T>
const T& CollapsedVector<T>::at(size_t idx) const {
    if (idx >= logicalSize_) {
        throw std::out_of_range("Trying to access illegal index of CollapsedVector<T>");
    }
    return operator[](idx);
}

template<typename T>
size_t CollapsedVector<T>::size() const noexcept {
    return logicalSize_;
}

template<typename T>
void CollapsedVector<T>::clear() noexcept {
    Precursor::clear();
    logicalSize_ = 0ul;
}

template<typename T>
void CollapsedVector<T>::reserve(size_t size) {
    Precursor::reserve(size);
}

template<typename T>
const T& CollapsedVector<T>::front() const {
    return Precursor::front();
}

}  // namespace Core

#endif  // COLLAPSED_VECTOR_HH
