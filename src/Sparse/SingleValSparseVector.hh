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
#ifndef _SPARSE_SINGLE_VALUE_SPARSE_VECTOR_
#define _SPARSE_SINGLE_VALUE_SPARSE_VECTOR_

#include <Core/Debug.hh>
#include <Math/Vector.hh>
#include <map>
#include <vector>
#include "Core/BinaryStream.hh"
#include "Core/Types.hh"
#include "Core/XmlStream.hh"

namespace Sparse {

/**
 *  Sparse vector.
 *  Implements basic functions of a sparse vector.
 * For performance purposes, this vector does <b>not</b> provide random access.
 * Elements of the vector can be accessed by
 * (1) sparse vector iterators (recommended), i.e. by iterating from begin to end.
 *     Given an iterator, the value can be accessed by dereferencing or by using the pseudo-random
 *     access operator[](SparseVectorIterator).
 *     complexity: constant in the number of active entries
 * (2) using the pseudo random access operator operator[](index_type index)
 *     complexity: linear
 * Instead, only
 * The SparseVectorIterator supports reset, increment and decrement, but no other access. The iterator can
 * be converted to an integer, which is needed by methods providing matrix or vector multiplications.
 *
 * The class is not thread-safe.
 *
 * Internally the vector is a vector of index-value pairs corresponding to the active elements
 */

template<typename Content>
class SingleValueSparseVector {
public:
    typedef u32 index_type;

private:
    typedef std::vector<std::pair<index_type, Content>> InternalVector;
    InternalVector                                      v_;
    // dimension of the vector (including zero components)
    size_t size_;
    // value of the non-active elements (default 0)
    Content default_;

public:
    template<class T, class V>
    class IteratorBase {
        typedef T InternalIterator;
        // typedef typename InternalIterator::value_type::second_type ValueType;
        // doesn't work because const pair<A,B>::second_type is not const
        typedef IteratorBase<T, V> Self;

    public:
        typedef typename std::forward_iterator_tag iterator_category;
        typedef V                                  value_type;
        typedef std::size_t                        difference_type;
        typedef value_type*                        pointer;
        typedef const value_type*                  const_pointer;
        typedef value_type&                        reference;
        typedef const value_type&                  const_reference;

    private:
        // disallow basic and copy construtor
        IteratorBase();

    public:
        IteratorBase(InternalIterator i)
                : index_(i) {}
        template<class OT, class OV>
        IteratorBase(const IteratorBase<OT, OV>& other)
                : index_(other.base()) {}

        ~IteratorBase() {}

        Self& operator++() {
            ++index_;
            return *this;
        }

        Self& operator--() {
            --index_;
            return *this;
        }

        index_type pos() const {
            return index_->first;
        }

        template<class OT, class OV>
        bool operator==(const IteratorBase<OT, OV>& other) const {
            return index_ == other.base();
        }

        template<class OT, class OV>
        bool operator!=(const IteratorBase<OT, OV>& other) const {
            return !operator==(other);
        }

        value_type& operator*() {
            return index_->second;
        }

        Self& operator=(const Self& other) {
            index_ = other.index_;
            return (*this);
        }

        const InternalIterator& base() const {
            return index_;
        }

    private:
        InternalIterator index_;
    };

    typedef IteratorBase<typename InternalVector::iterator,
                         typename InternalVector::iterator::value_type::second_type>
            iterator;
    typedef IteratorBase<typename InternalVector::const_iterator,
                         const typename InternalVector::const_iterator::value_type::second_type>
            const_iterator;

    template<class T, class V>
    friend class IteratorBase;

private:
    mutable typename InternalVector::iterator pos_;

public:
    SingleValueSparseVector(size_t size = 0, Content defaultValue = 0);

    virtual ~SingleValueSparseVector() {}

#if 0
    inline void operator=(const SingleValueSparseVector<Content>& sv) {
        v_ = sv.v_;
        size_ = sv.size_;
        default_ = sv.default_;
        pos_ = v_->begin();
    }
#endif

    inline SingleValueSparseVector(const SingleValueSparseVector<Content>& sv)
            : v_(sv.v_),
              size_(sv.size_),
              default_(sv.default_),
              pos_(v_.begin()) {}

    inline bool operator==(const SingleValueSparseVector& sv) {
        return (v_ == sv.v_ and size_ == sv.size_);
    }

    inline bool operator!=(const SingleValueSparseVector& sv) {
        return !operator==(sv);
    }

    /** Returns a begin iterator. */
    inline iterator begin() {
        return iterator(v_.begin());
    }

    /** Returns an "after end" iterator. */
    inline iterator end() {
        return iterator(v_.end());
    }

    /** Returns a begin const_iterator. */
    inline const_iterator begin() const {
        return const_iterator(v_.begin());
    }

    /** Returns an "after end" const_iterator. */
    inline const_iterator end() const {
        return const_iterator(v_.end());
    }

    /**
     * complexity: constant
     * @param i const_iterator pointing to the content.
     * @return reference to the const_iterator's position.
     */
    inline const Content& operator[](const const_iterator& i) const {
        return *i;
    }

    /**
     * complexity: constant
     * @param i iterator pointing to the content.
     @return content at the given iterator's position.
     */
    inline Content& operator[](const iterator& i) {
        return *i;
    }

    /**
     * complexity: linear
     * @param index component index
     * @return content at the given index
     */
    inline const Content& get(const index_type index) const {
        if (v_.empty() || v_.back().first < index)
            return default_;
        if (pos_->first > index) {
            while (pos_ != v_.begin() && pos_->first > index)
                --pos_;
        }
        while (pos_ != v_.end() && pos_->first < index)
            ++pos_;
        if (pos_->first == index) {
            if (pos_ == v_.end()) {
                std::cerr << "Something wrong at SingleValueSparseVector::get(index) with index=" << index << " content of vector is: " << std::endl;
                for (const_iterator countI = this->begin(); countI != this->end(); ++countI) {
                    std::cerr << "pos=" << countI.pos() << " count=" << *countI << "  ";
                }
                std::cerr << std::endl;
            }
            return pos_->second;
        }
        else {
            if (pos_ != v_.begin())
                --pos_;
            return default_;
        }
    }

    /**
     * complexity: linear
     * @return reference to the component at the given index, if the component at the given index is not
     *         active, a null entry is inserted
     * @param index component index
     */
    inline Content& operator[](const index_type index) {
        if (v_.empty() || v_.back().first < index) {
            push_back(index, default_);
            pos_ = v_.end();
            --pos_;
        }
        else {
            if (pos_->first > index)
                pos_ = v_.begin();
            while (pos_ != v_.end() && pos_->first < index)
                ++pos_;
            if (pos_->first != index) {
                pos_ = v_.insert(pos_, std::pair<index_type, Content>(index, default_));
            }
        }
        verify(pos_->first == index);
        return pos_->second;
    }

    /**
     * sets all active elements to null
     * the size of the vector is unchanged
     */
    void clear() {
        v_.clear();
    }

    /**
     * sets the value of non-active components
     */
    void setDefault(Content defaultValue) {
        default_ = defaultValue;
    }

    /**
     * get the value of non-active components
     */
    Content getDefault() {
        return default_;
    }

    /**
     * sets the dimension of the vector
     * @param new_size new dimension
     */
    void resize(size_t new_size) {
        size_ = new_size;
    }

    /**
     * sets the dimension of the vector and changes the value of ALL non-active components
     * @param new_size new dimension
     * @param null value of non-active components
     */
    void resize(size_t new_size, Content defaultValue) {
        size_    = new_size;
        default_ = defaultValue;
    }

    /**
     * @return reference to the zeroth component
     */
    Content& front() {
        return (*this)[0];
    }
    /**
     * push_back of a new active entry
     * requires: index > index of last active entry
     * if index < size: size is unchanged
     * if index >= size: size = index + 1
     * @param index  index of new entry
     * @param value value of new entry
     */
    void push_back(index_type index, Content value);

    /** @return vector's size */
    inline size_t size() const {
        return size_;
    }

    inline bool empty() const {
        return v_.empty();
    }

    inline size_t nActiveElements() const {
        return v_.size();
    }

    /* Concatenates a sparse vector to the current sparse vector.
     * @param bsv vector to concatenate
     */
    void concatenate(const SingleValueSparseVector<Content>& bsv);
    /**
     * Concatenates a standard vector to the current sparse vector.
     * @param v Vector to concatenate.
     * @param startPos Start position.
     */
    void concatenate(const std::vector<Content>& v, u32 startPosition);

    /** Scales the content of a block sparse vector with all its elements. */
    SingleValueSparseVector<Content>& operator*=(const Content&);

    /** Adds a block sparse vector to the recent one. */
    SingleValueSparseVector<Content>& operator+=(const SingleValueSparseVector<Content>&);
#if 0  // not tested
    SingleValueSparseVector<Content>& operator+=(const std::map<u32, Content>&);
#endif

    /// in place weighted operator+=
    template<typename C>
    SingleValueSparseVector<Content>& add(const Content& weight, const SingleValueSparseVector<C>&);
    template<typename C>
    SingleValueSparseVector<Content>& add(const Content& weight, const std::map<u32, C>&);

    virtual Core::XmlWriter& dump(Core::XmlWriter& o) const;
    virtual bool             read(Core::BinaryInputStream& i);
    virtual bool             write(Core::BinaryOutputStream& o) const;
    template<typename C>
    friend Core::BinaryInputStream& operator>>(Core::BinaryInputStream&, SingleValueSparseVector<C>&);
    template<typename C>
    friend Core::BinaryOutputStream& operator<<(Core::BinaryOutputStream&, const SingleValueSparseVector<C>&);
    template<typename C>
    friend Core::XmlWriter& operator<<(Core::XmlWriter&, const SingleValueSparseVector<C>&);
};

#if 0
template<typename Content>
inline bool
operator==(const SingleValueSparseVector<Content>::iterator& x,
        const SingleValueSparseVector<Content>::const_iterator& y)
        { return y==x; }

template<typename Content>
inline bool
operator!=(const SingleValueSparseVector<Content>::iterator& x,
        const SingleValueSparseVector<Content>::const_iterator& y)
        { return y!=x; }
#endif

/**
 * Creates a new sparse vector of given size.
 * @param size Size.
 */
template<typename Content>
SingleValueSparseVector<Content>::SingleValueSparseVector(size_t size, Content defaultValue)
        : v_(0),
          size_(size),
          default_(defaultValue),
          pos_(v_.begin()) {}

/**
 * Adds a new element.
 * requires: index > index of last active component
 * size of vector = max(index + 1, previous dimension)
 * @param index Position where the new element should be added.
 * @param value New element to add.
 */
template<typename Content>
void SingleValueSparseVector<Content>::push_back(index_type index, Content value) {
    require(v_.size() == 0 || index > v_.back().first);
    std::pair<index_type, Content> newEntry(index, value);
    v_.push_back(newEntry);
    // pos_ might be invalid
    pos_  = v_.begin();
    size_ = std::max((size_t)index + 1, size_);
}

/** Concatenates an other block sparse vector
 * @param bsv vector to concatenate
 */
template<typename Content>
void SingleValueSparseVector<Content>::concatenate(const SingleValueSparseVector<Content>& bsv) {
    size_t recentSize         = size_;
    size_t bsvNActiveElements = bsv.v_.size();
    for (size_t i = 0; i < bsvNActiveElements; ++i) {
        std::pair<index_type, Content> newEntry(bsv.v_[i].first + recentSize, bsv.v_[i].second);
        v_.push_back(newEntry);
    }
    size_ += bsv.size_;
    // pos_ might be invalid
    pos_ = v_.begin();
}

/** Concatenates a full vector at a given position.
 * requires startPos >= previous dimension
 * @param v vector to concatenate
 * @param startPos position where v is inserted
 */
template<typename Content>
void SingleValueSparseVector<Content>::concatenate(const std::vector<Content>& v, u32 startPos) {
    require(startPos >= size_);
    size_t vSize = v.size();
    for (size_t i = 0; i < vSize; ++i) {
        std::pair<index_type, Content> newEntry(startPos + i, v[i]);
        v_.push_back(newEntry);
    }
    size_ = startPos + v.size();
    // pos_ might be invalid
    pos_ = v_.begin();
}

template<typename Content>
SingleValueSparseVector<Content>& SingleValueSparseVector<Content>::operator*=(const Content& factor) {
    for (size_t i = 0; i < v_.size(); ++i) {
        v_[i].second *= factor;
    }
    return (*this);
}

template<typename Content>
SingleValueSparseVector<Content>& SingleValueSparseVector<Content>::operator+=(const SingleValueSparseVector<Content>& bsv) {
    if (size_ != bsv.size_) {
        Core::Application::us()->error() << "Addition size error: size = " << size_ << ", bsv.size = " << bsv.size_;
        defect();
    }

    SingleValueSparseVector<Content> temp;
    size_t                           thisIndex = 0, bsvIndex = 0;
    while (thisIndex < v_.size() and bsvIndex < bsv.v_.size()) {
        if (v_[thisIndex].first < bsv.v_[bsvIndex].first) {
            temp.push_back(v_[thisIndex].first, v_[thisIndex].second);
            ++thisIndex;
        }
        else if (v_[thisIndex].first > bsv.v_[bsvIndex].first) {
            temp.push_back(bsv.v_[bsvIndex].first, bsv.v_[bsvIndex].second);
            ++bsvIndex;
        }
        else {  // thisIter.pos() == bsvIter.pos()
            temp.push_back(v_[thisIndex].first, v_[thisIndex].second + bsv.v_[bsvIndex].second);
            ++thisIndex;
            ++bsvIndex;
        }
    }
    if (thisIndex == v_.size()) {
        while (bsvIndex < bsv.v_.size()) {
            temp.push_back(bsv.v_[bsvIndex].first, bsv.v_[bsvIndex].second);
            ++bsvIndex;
        }
    }
    else {
        while (thisIndex < v_.size()) {
            temp.push_back(v_[thisIndex].first, v_[thisIndex].second);
            ++thisIndex;
        }
    }

    v_   = temp.v_;
    pos_ = v_.begin();
    return *this;
}

#if 0  // not tested
template<typename Content>
SingleValueSparseVector<Content>& SingleValueSparseVector<Content>::operator+=(const std::map<u32, Content>& bsv){
        SingleValueSparseVector<Content> temp;
        size_t thisIndex = 0;
        typename std::map<u32, Content>::const_iterator bsvIter = bsv->begin();
        while (thisIndex < v_.size() and bsvIter != bsv.end()) {
                if (v_[thisIndex].first < bsvIter.first) {
                        temp.push_back(v_[thisIndex].first, v_[thisIndex].second);
                        ++thisIndex;
                }
                else if (v_[thisIndex].first > bsvIter.first) {
                        temp.push_back(bsvIter.first, bsvIter.second);
                        ++bsvIter;
                }
                else { //thisIter.pos() == bsvIter.pos()
                        temp.push_back(v_[thisIndex].first, v_[thisIndex].second + bsvIter.second);
                        ++thisIndex;
                        ++bsvIter;
                }
        }
        if (thisIndex == v_.size()) {
                while (bsvIter != bsv.end()) {
                        temp.push_back(bsvIter.first, bsvIter.second);
                        ++bsvIter;
                }
        }
        else {
                while (thisIndex < v_.size()) {
                        temp.push_back(v_[thisIndex].first, v_[thisIndex].second);
                        ++thisIndex;
                }
        }


        v_ = temp.v_;
        pos_ = v_.begin();
        return *this;
}
#endif

template<typename Content>
template<typename C>
SingleValueSparseVector<Content>& SingleValueSparseVector<Content>::add(const Content& weight, const SingleValueSparseVector<C>& bsv) {
    if (size_ != bsv.size()) {
        Core::Application::us()->error() << "Addition size error: size = " << size_ << ", bsv.size = " << bsv.size();
        defect();
    }

    size_t                                              thisIndex = 0;
    typename SingleValueSparseVector<C>::const_iterator bsvIter   = bsv.begin();
    while (thisIndex < v_.size() and bsvIter != bsv.end()) {
        const Content weightupdate = weight * (*bsvIter);
        if (v_[thisIndex].first < bsvIter.pos()) {
            ++thisIndex;
        }
        else if (v_[thisIndex].first > bsvIter.pos()) {
            v_.insert(v_.begin() + thisIndex, std::pair<index_type, Content>(bsvIter.pos(), weightupdate));
            ++bsvIter;
        }
        else {  // thisIter.pos() == bsvIter.pos()
            v_[thisIndex].second += weightupdate;
            ++thisIndex;
            ++bsvIter;
        }
    }
    if (thisIndex == v_.size()) {
        while (bsvIter != bsv.end()) {
            this->push_back(bsvIter.pos(), weight * (*bsvIter));
            ++bsvIter;
        }
    }

    pos_ = v_.begin();
    return *this;
}

template<typename Content>
template<typename C>
SingleValueSparseVector<Content>& SingleValueSparseVector<Content>::add(const Content& weight, const std::map<index_type, C>& bsv) {
    size_t                                                 thisIndex = 0;
    typename std::map<index_type, Content>::const_iterator bsvIter   = bsv.begin();
    while (thisIndex < v_.size() and bsvIter != bsv.end()) {
        const Content weightupdate = weight * (bsvIter->second);
        if (v_[thisIndex].first < bsvIter->first) {
            ++thisIndex;
        }
        else if (v_[thisIndex].first > bsvIter->first) {
            v_.insert(v_.begin() + thisIndex, std::pair<index_type, Content>(bsvIter->first, weightupdate));  // TODO: correct?
            ++bsvIter;
        }
        else {  // thisIter.pos() == bsvIter.pos()
            v_[thisIndex].second += weightupdate;
            ++thisIndex;
            ++bsvIter;
        }
    }
    if (thisIndex == v_.size()) {
        while (bsvIter != bsv.end()) {
            this->push_back(bsvIter->first, weight * (bsvIter->second));
            ++bsvIter;
        }
    }

    pos_ = v_.begin();
    return *this;
}

/**
 * Dumps the sparse vector to an XML stream.
 * @param o XML output stream to write to.
 */
template<typename Content>
Core::XmlWriter& SingleValueSparseVector<Content>::dump(Core::XmlWriter& o) const {
    if (!v_.empty()) {
        for (size_t i = 0; i < v_.size(); ++i) {
            o << Core::XmlEmpty("element") + Core::XmlAttribute("position", v_[i].first) + Core::XmlAttribute("value", v_[i].second);
        }
    }
    return o;
}

/**
 * Reads a sparse vector from a binary stream.
 * @param i Stream to read from.
 */
template<typename Content>
bool SingleValueSparseVector<Content>::read(Core::BinaryInputStream& i) {
    u32 size = 0, nActiveElements = 0;
    i >> size;
    i >> nActiveElements;
    size_ = size;
    v_.resize(nActiveElements);
    for (size_t it = 0; it < nActiveElements; ++it) {
        i >> v_[it].first;
        i >> v_[it].second;
    }
    pos_ = v_.begin();
    return i;
}

/**
 * Writes a sparse vector to a binary stream.
 * @param o Stream to write to.
 */
template<typename Content>
bool SingleValueSparseVector<Content>::write(Core::BinaryOutputStream& o) const {
    o << (u32)size_;
    o << (u32)v_.size();
    for (size_t it = 0; it < v_.size(); ++it) {
        o << v_[it].first;
        o << v_[it].second;
    }
    return o;
}

template<typename Content>
Core::BinaryInputStream& operator>>(Core::BinaryInputStream& i, SingleValueSparseVector<Content>& v) {
    v.read(i);
    return i;
}

template<typename Content>
Core::BinaryOutputStream& operator<<(Core::BinaryOutputStream& o, const SingleValueSparseVector<Content>& v) {
    v.write(o);
    return o;
}

template<typename Content>
Core::XmlWriter& operator<<(Core::XmlWriter& o, const SingleValueSparseVector<Content>& v) {
    v.dump(o);
    return o;
}

template<typename T, typename P, typename Content>
Math::Vector<T, P> operator+(const Math::Vector<T, P>& m, const SingleValueSparseVector<Content>& bsv) {
    typename SingleValueSparseVector<Content>::const_iterator iter = bsv.begin();
    typename SingleValueSparseVector<Content>::const_iterator end  = bsv.end();
    Math::Vector<T, P>                                        r(m);
    for (; iter != end; ++iter)
        r[iter.pos()] += *iter;
    return r;
}

}  // namespace Sparse

namespace Core {
template<typename T>
class NameHelper<Sparse::SingleValueSparseVector<T>> : public std::string {
public:
    NameHelper()
            : std::string("single-value-sparse-vector-" + NameHelper<T>()) {}
};

}  // namespace Core

#endif
