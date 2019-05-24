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
#ifndef SEARCH_LINEARMINIHASH_HH
#define SEARCH_LINEARMINIHASH_HH

#include <Core/Types.hh>
#include <vector>

namespace Search {
template<class Key>
struct StandardValueHash {
    inline u32 operator()(Key a) const {
        // a = (a+0x7ed55d16) + (a<<12);
        // a = ( a^0xc761c23c ) ^ ( a>>19 );
        // a = (a+0x165667b1) + (a<<5);
        // a = (a+0xd3a2646c) ^ (a<<9);
        // a = ( a+0xfd7046c5 ) + ( a<<3 );
        // a = (a^0xb55a4f09) ^ (a>>16);
        return a;
    }
};

/**
 * A specialized hash-map, which is optimized for the following case:
 * - The size of the hash-table is known beforehand
 * - It never happens that an inserted items needs to be deleted
 *
 * Under these circumstances, the hash-map is very compact and efficient.
 *
 * The hash-map _can_ resize the table on-demand, but that is quite inefficient, so a good estimate size should be used.
 * */

template<class Key, Key invalidKey, class Value, class Hash = StandardValueHash<Key>>
class LinearMiniHash {
    inline u32 constrain(u32 val) const {
        return val & mask_;
    }

public:
    LinearMiniHash(Value defaultValue)
            : size_(0), mask_(0), defaultValue_(defaultValue) {
    }

    void read(Core::MappedArchiveReader reader) {
        reader >> size_ >> mask_ >> defaultValue_ >> sparseValues_;
    }

    void write(Core::MappedArchiveWriter writer) const {
        writer << size_ << mask_ << defaultValue_ << sparseValues_;
    }

    u32 hashSize() const {
        return sparseValues_.size();
    }

    void clear(u32 minHashSize = 0) {
        size_ = 0;

        if (minHashSize == 0) {
            sparseValues_.clear();
            mask_ = 0;
            return;
        }

        // Compute the next-higher power-of-2 hash size
        u32 hashSize = 1;

        while (hashSize < minHashSize)
            hashSize <<= 1;

        verify(hashSize >= minHashSize);

        mask_ = hashSize - 1;

        u32 fillSize = std::min(hashSize, (u32)sparseValues_.size());
        sparseValues_.resize(hashSize, std::make_pair(invalidKey, defaultValue_));

        for (u32 a = 0; a < fillSize; ++a)
            sparseValues_[a] = std::make_pair(invalidKey, defaultValue_);
    }

    void swap(LinearMiniHash<Key, invalidKey, Value>& rhs) {
        sparseValues_.swap(rhs.sparseValues_);
        swapValues(size_, rhs.size_);

        swapValues(defaultValue_, rhs.defaultValue_);
        swapValues(mask_, rhs.mask_);
    }

    // Resize at resizeAtFraction out of 256
    u32 checkResize(int resizeAtFraction) {
        if (size_ >= ((sparseValues_.size() * resizeAtFraction) >> 8)) {
            // Should not happen, a good prediction is expected
            LinearMiniHash<Key, invalidKey, Value, Hash> newHash(defaultValue_);
            newHash.clear(hashSize() * 2);

            for (typename std::vector<std::pair<Key, Value>>::iterator it = sparseValues_.begin(); it != sparseValues_.end(); ++it)
                if ((*it).first != invalidKey)
                    newHash.insert((*it).first, (*it).second);

            verify(newHash.size() == size_);

            mask_ = newHash.mask_;

            sparseValues_.swap(newHash.sparseValues_);
            return sparseValues_.size();
        }
        return 0;
    }

    // Returns the required skips
    u32 insert(Key id, Value score) {
        verify_(id != invalidKey);

        ++size_;

        u32 pos = constrain(hash(id));
        if (sparseValues_[pos].first == invalidKey) {
            // The bin is empty, insert just here
            sparseValues_[pos].first  = id;
            sparseValues_[pos].second = score;
            return 0;
        }
        else {
            verify_(sparseValues_[pos].first != id);

            u32 cnt = 0;

            // Iterate linearly, searching for a free bin
            for (u32 p = constrain(pos + 1); p != pos; p = constrain(p + 1)) {
                ++cnt;
                if (sparseValues_[p].first == invalidKey) {
                    // The bin is empty, insert just here
                    sparseValues_[p].first  = id;
                    sparseValues_[p].second = score;
                    return cnt;
                }
                else {
                    verify_(sparseValues_[p].first != id);  // Every item must be inserted only once
                }
            }

            verify(0);  // Should never happen
        }
    }

    // Returns the default value if the item wasn't found
    inline const Value& operator[](Key id) const {
        u32                          pos = constrain(hash(id));
        const std::pair<Key, Value>& item(sparseValues_[pos]);

        if (item.first == id) {
            return item.second;
        }
        else if (item.first == invalidKey) {
            return defaultValue_;  // The id wasn't found
        }
        else {
            // The first iteration of the loop is unrolled, for performance reasons
            u32 p = constrain(pos + 1);

            const std::pair<Key, Value>& item2(sparseValues_[p]);

            if (item2.first == id)
                return item2.second;
            else if (item2.first == invalidKey)
                return defaultValue_;  // The id wasn't found

            // Iterate linearly, searching for the bin
            for (p = constrain(p + 1); p != pos; p = constrain(p + 1)) {
                const std::pair<Key, Value>& item3(sparseValues_[p]);
                if (item3.first == id)
                    return item3.second;
                else if (item3.first == invalidKey)
                    return defaultValue_;  // The id wasn't found
            }

            // Should never happen, as the hash is never 100% full
            verify(0);
        }
    }

    inline bool get(Key id, Value& target) const {
        u32                          pos = constrain(hash(id));
        const std::pair<Key, Value>& item(sparseValues_[pos]);

        ifSparseCollisionStats(sparseSkipHash[id].second += 1;);

        if (item.first == invalidKey)
            return false;

        if (item.first == id) {
            target = item.second;
            return true;
        }
        else {
#ifdef EXTENSIVE_SPARSE_COLLISION_STATS
            sparseCollisionHash[std::make_pair<u32, u32>(id, item.first)] += 1;
#endif
        }

        ifSparseCollisionStats(sparseSkipHash[id].first += 1;);

        // The first iteration of the loop is unrolled, for performance reasons
        u32 p = constrain(pos + 1);

        const std::pair<Key, Value>& item2(sparseValues_[p]);

        if (item2.first == id) {
            target = item2.second;
            return true;
        }
        else if (item2.first == invalidKey) {
            return false;  // The id wasn't found
        }

#ifdef EXTENSIVE_SPARSE_COLLISION_STATS
        sparseCollisionHash[std::make_pair<u32, u32>(id, item2.first)] += 1;
#endif

        ifSparseCollisionStats(sparseSkipHash[id].first += 1;);

        // Iterate linearly, searching for the bin
        for (p = constrain(p + 1); p != pos; p = constrain(p + 1)) {
            const std::pair<Key, Value>& item3(sparseValues_[p]);
            if (item3.first == id) {
                target = item3.second;
                return true;
            }
            else if (item3.first == invalidKey)
                return false;  // The id wasn't found

#ifdef EXTENSIVE_SPARSE_COLLISION_STATS
            sparseCollisionHash[std::make_pair<u32, u32>(id, item3.first)] += 1;
#endif

            ifSparseCollisionStats(sparseSkipHash[id].first += 1;);
        }

        // Should never happen, as the hash is never 100% full
        verify(0);
    }

    bool contains(Key id) const {
        u32 pos = constrain(hash(id));
        if (sparseValues_[pos].first == id) {
            return true;
        }
        else if (sparseValues_[pos].first == invalidKey) {
            return false;
        }
        else {
            // Iterate linearly, searching for the bin
            for (u32 p = constrain(pos + 1); p != pos; p = constrain(p + 1)) {
                if (sparseValues_[p].first == id)
                    return true;
                else if (sparseValues_[p].first == invalidKey)
                    return false;  // The id wasn't found
            }

            // The id wasn't found
            return false;
        }
    }

    u32 size() const {
        return size_;
    }

private:
    template<class T>
    inline void swapValues(T& l, T& r) {
        T t = l;
        l   = r;
        r   = t;
    }

    inline u32 hash(Key a) const {
        return Hash()(a);
    }

    std::vector<std::pair<Key, Value>> sparseValues_;
    u32                                size_;
    int                                mask_;
    Value                              defaultValue_;
};
}  // namespace Search

#endif  // SEARCH_LINEARMINIHASH_HH
