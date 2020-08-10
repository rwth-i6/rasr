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
#ifndef SEARCH_APPROX_LINEARMINIHASH_HH
#define SEARCH_APPROX_LINEARMINIHASH_HH

#include <Core/Types.hh>
#include <vector>

#define F32_MAX +3.40282347e+38F

namespace Search {
template<class Key>
struct StandardApproxValueHash {
    inline u32 operator()(Key a) {
        /* a = (a+0x7ed55d16) + (a<<12);
           a = ( a^0xc761c23c ) ^ ( a>>19 );
           a = (a+0x165667b1) + (a<<5);
           a = (a+0xd3a2646c) ^ (a<<9);
           a = ( a+0xfd7046c5 ) + ( a<<3 );
           a = (a^0xb55a4f09) ^ (a>>16);*/
        return a;
    }
};

template<class Value>
struct MinimumCombine {
    inline Value operator()(Value old, Value _new) const {
        return old < _new ? old : _new;
    }
};

/**
 * For efficiency-reasons, the default-value is hardcoded to F32_MAX
 * */

template<class Key, Key invalidKey, class Value, bool preventFalsePositives = false, bool useHashChain = false, bool powerOfTwoSize = true, class Hash = StandardApproxValueHash<Key>, class Combine = MinimumCombine<Value>>
class ApproxLinearMiniHash {
    inline u32 constrain(u32 val) const {
        if (powerOfTwoSize)
            return val & mask_;
        else
            return val % mask_;
    }

public:
    ApproxLinearMiniHash()
            : size_(0), mask_(0) {
        verify(sizeof(float) == sizeof(u32));  // Expected in all of the code
    }

    u32 hashSize() const {
        return sparseValues_.size();
    }

    void clear(u32 minHashSize = 0) {
        size_ = 0;

        if (minHashSize == 0) {
            sparseValues_.clear();
            if (preventFalsePositives)
                sparseKeys_.clear();
            mask_ = 0;
            return;
        }

        u32 hashSize;

        if (powerOfTwoSize) {
            hashSize = 1;

            while (hashSize < minHashSize)
                hashSize <<= 1;

            verify(hashSize >= minHashSize);

            mask_ = hashSize - 1;
        }
        else {
            hashSize = 89;

            while (hashSize < minHashSize)
                hashSize = (2 * hashSize) + 1;

            verify(hashSize >= minHashSize);

            mask_ = hashSize;
        }

        u32 fillSize = std::min(hashSize, (u32)sparseValues_.size());
        sparseValues_.resize(hashSize, F32_MAX);
        if (preventFalsePositives)
            sparseKeys_.resize(hashSize, invalidKey);

        for (u32 a = 0; a < fillSize; ++a) {
            sparseValues_[a] = F32_MAX;
            if (preventFalsePositives)
                sparseKeys_[a] = invalidKey;
        }
    }

    // Returns the new hash-size if a resize is required
    inline u32 checkResize(int resizeAtFraction) {
        if (size_ >= ((sparseValues_.size() * resizeAtFraction) >> 8))
            return hashSize() * 2;
        return 0;
    }

    typedef __attribute__((__may_alias__)) float AliasableFloat;

    inline float mark(float value, unsigned char marker) const {
        AliasableFloat avalue                        = value;
        reinterpret_cast<unsigned char*>(&avalue)[0] = marker;
        return avalue;
    }

    inline bool isMarked(float value, unsigned char marker) const {
        AliasableFloat avalue = value;
        return reinterpret_cast<unsigned char*>(&avalue)[0] == marker;
    }

    // Returns the number of conflicts
    u32 insert(Key id, Value value) {
        const u32           h      = hash(id);
        const unsigned char marker = h >> 16;
        if (!preventFalsePositives) {
            value = mark(value, marker);
            verify(isMarked(value, marker));
        }

        u32 pos = constrain(h);
        ++size_;

        if (!useHashChain) {
            if (sparseValues_[pos] == F32_MAX) {
                sparseValues_[pos] = value;
                if (preventFalsePositives)
                    sparseValues_[pos] = id;
                return 0;
            }
            else {
                if (preventFalsePositives) {
                    // hard-coded minimum-combine
                    if (sparseValues_[pos] > value) {
                        sparseValues_[pos] = value;
                        sparseKeys_[pos]   = id;
                    }
                }
                else {
                    sparseValues_[pos] = Combine()(sparseValues_[pos], value);
                }
                return 1;
            }
        }
        else {
            if (sparseValues_[pos] == F32_MAX) {
                // The bin is empty, insert just here
                if (preventFalsePositives)
                    sparseKeys_[pos] = id;
                sparseValues_[pos] = value;
                return 0;
            }
            else {
                u32 cnt = 0;

                // Iterate linearly, searching for a free bin
                for (u32 p = constrain(pos + 1); p != pos; p = constrain(p + 1)) {
                    ++cnt;
                    if (sparseValues_[p] == F32_MAX) {
                        // The bin is empty, insert just here
                        if (preventFalsePositives)
                            sparseKeys_[p] = id;
                        sparseValues_[p] = value;
                        return cnt;
                    }
                    else {
                        if (preventFalsePositives)
                            verify_(sparseKeys_[p] != id);  // Every item must be inserted only once
                    }
                }

                verify(0);  // Should never happen
            }
        }
    }

    inline Value getQuick(Key id) const {
        return sparseValues_[constrain(id)];
    }

    inline bool get(Key id, Value& target) const {
        const u32 pos = constrain(id);
        if (!useHashChain) {
            if (preventFalsePositives && sparseKeys_[pos] != id)
                return false;

            Value t = sparseValues_[pos];
            if (t != F32_MAX && isMarked(t, id >> 16)) {
                target = t;
                return true;
            }
            else {
                return false;
            }
        }
        else {
            const unsigned char marker = id >> 16;

            if (sparseValues_[pos] == F32_MAX)
                return false;

            if ((preventFalsePositives && sparseKeys_[pos] == id) || (!preventFalsePositives && isMarked(sparseValues_[pos], marker))) {
                target = sparseValues_[pos];
                return true;
            }

            // The first iteration of the loop is unrolled, for performance reasons
            u32   p          = constrain(pos + 1);
            Value item2Value = sparseValues_[p];

            if ((preventFalsePositives && sparseKeys_[p] == id) || (!preventFalsePositives && isMarked(item2Value, marker))) {
                target = item2Value;
                return true;
            }
            else if (item2Value == F32_MAX) {
                return false;  // The id wasn't found
            }

            // Iterate linearly, searching for the bin
            for (p = constrain(p + 1); p != pos; p = constrain(p + 1)) {
                Value item3Value = sparseValues_[p];
                if ((preventFalsePositives && sparseKeys_[p] == id) || (!preventFalsePositives && isMarked(item3Value, marker))) {
                    target = item3Value;
                    return true;
                }
                else if (item3Value == F32_MAX)
                    return false;  // The id wasn't found
            }

            // Should never happen, as the hash is never 100% full
            verify(0);
        }
    }

    u32 size() const {
        return size_;
    }

private:
    inline u32 hash(Key a) const {
        return Hash()(a);
    }

    std::vector<Value> sparseValues_;
    std::vector<Key>   sparseKeys_;
    u32                size_;
    int                mask_;
};
}  // namespace Search

#endif  // SEARCH_LINEARMINIHASH_HH
