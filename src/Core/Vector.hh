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
#ifndef _CORE_VECTOR_HH
#define _CORE_VECTOR_HH

#include <vector>
#include "Assertions.hh"

namespace Core {

template<class T>
class Vector : public std::vector<T> {
public:
    typedef std::vector<T>                     Precursor;
    typedef typename Precursor::iterator       iterator;
    typedef typename Precursor::const_iterator const_iterator;

public:
    Vector()
            : Precursor() {}
    Vector(size_t size)
            : Precursor(size) {}
    Vector(size_t size, const T& def)
            : Precursor(size, def) {}
    Vector(const Precursor& vector)
            : Precursor(vector) {}

    /** grow to size maxId+1. */
    void grow(size_t maxId, const T& def = T()) {
        if (maxId >= this->size()) {
            if (maxId >= this->capacity())
                this->reserve(std::max(2 * this->capacity(), maxId + 1));
            this->insert(this->end(), maxId - this->size() + 1, def);
        }
    }

    /** Free over-allocated storage. */
    void yield() {
        std::vector<T> tmp(*this);
        this->swap(tmp);
        ensure(this->capacity() == this->size());
    }

    void minimize() {
        yield();
    }

    /** safe set*/
    void set(size_t id, const T& val, const T& def = T()) {
        grow(id, def);
        (*this)[id] = val;
    }

    /** safe get*/
    T get(size_t id, const T& def = T()) const {
        /* Earlier, we returned `const T&` here. This is of course dangerous if we might want to return
         * the default value (which might be a temporary object).
         * Returning `T` instead is also faster in most cases (because of the indirection +
         * because maybe even sizeof(void*) >= sizeof(T)).
         * sizeof(T) <= sizeof(long) * 4 might be a somewhat sane check for this to be at least as fast.
         * Otherwise, let's force the user of this `Vector` to code somewhat more explicitely.
         * Note that with compiler optimizations, the compiler will most likely do the same anyway,
         * no matter if we have `const T&` or `T` as the return here -- only in the unsafe cases, it will
         * use safe code instead.
         */
        static_assert(sizeof(T) <= sizeof(long) * 4, "size too big");
        if (id < this->size())
            return (*this)[id];
        return def;
    }

    size_t getMemoryUsed() const {
        return sizeof(T) * this->capacity() + sizeof(*this);
    }
};

}  // namespace Core

#endif  // _CORE_VECTOR_HH
