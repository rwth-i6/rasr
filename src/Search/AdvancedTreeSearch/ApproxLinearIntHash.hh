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
#ifndef SEARCH_APPROX_LINEARINTHASH_HH
#define SEARCH_APPROX_LINEARINTHASH_HH

#include <Core/Types.hh>
#include <vector>

#define U16_MAX 65535

namespace Search {
/**
 * For efficiency-reasons, the default-value is hardcoded to U16_MAX
 * This is a tiny bit faster than the standard approximative mini-hash, but also a bit less precise
 * */

template<class Key, Key invalidKey, bool useHashChain = false>
class ApproxLinearIntHash {
    struct Value {
        Value()
                : value(U16_MAX), mark(0) {
        }
        u16 value;
        u16 mark;
    };
    inline u32 constrain(u32 val) const {
        return val & mask_;
    }

public:
    ApproxLinearIntHash()
            : size_(0), mask_(0) {
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

        u32 hashSize;

        hashSize = 1;

        while (hashSize < minHashSize)
            hashSize <<= 1;

        verify(hashSize >= minHashSize);

        mask_ = hashSize - 1;

        u32 fillSize = std::min(hashSize, (u32)sparseValues_.size());
        sparseValues_.resize(hashSize, Value());

        for (u32 a = 0; a < fillSize; ++a) {
            sparseValues_[a] = Value();
        }
    }

    // Returns the new hash-size if a resize is required
    inline u32 checkResize(int resizeAtFraction) {
        if (size_ >= ((sparseValues_.size() * resizeAtFraction) >> 8))
            return hashSize() * 2;
        return 0;
    }

    // Returns the number of conflicts
    u32 insert(const Key h, const float _value) {
        u16       value  = (u16)_value;
        const u16 marker = h >> 16;

        u32 pos = constrain(h);
        ++size_;

        if (!useHashChain) {
            if (sparseValues_[pos].value == U16_MAX) {
                sparseValues_[pos].value = value;
                sparseValues_[pos].mark  = marker;
                return 0;
            }
            else {
                // hard-coded minimum-combine
                ///@todo eventually simply always use this
                if (sparseValues_[pos].value > value) {
                    sparseValues_[pos].value = value;
                    sparseValues_[pos].mark  = marker;
                }
                return 1;
            }
        }
        else {
            if (sparseValues_[pos].value == U16_MAX) {
                // The bin is empty, insert just here
                sparseValues_[pos].value = value;
                sparseValues_[pos].mark  = value;
                return 0;
            }
            else {
                u32 cnt = 0;

                // Iterate linearly, searching for a free bin
                for (u32 p = constrain(pos + 1); p != pos; p = constrain(p + 1)) {
                    ++cnt;
                    if (sparseValues_[p].value == U16_MAX) {
                        // The bin is empty, insert just here
                        sparseValues_[p].value = value;
                        sparseValues_[p].mark  = marker;
                        return cnt;
                    }
                }

                verify(0);  // Should never happen
            }
        }
    }

    inline bool get(Key id, float& target) const {
        const u32 pos = constrain(id);
        if (!useHashChain) {
            const Value& t = sparseValues_[pos];
            if (t.value != U16_MAX && t.mark == ((u16)(id >> 16))) {
                target = t.value;
                return true;
            }
            else {
                return false;
            }
        }
        else {
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
    u32                size_;
    int                mask_;
};
}  // namespace Search

#endif  // SEARCH_LINEARMINIHASH_HH
