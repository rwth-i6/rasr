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
#ifndef LMCACHE_HH
#define LMCACHE_HH

#include <Lm/LanguageModel.hh>
// #include "LinearMiniHash.hh"

template<class Key>
struct StandardValueHash {
    inline u32 operator()(Key a) const {
        // a = (a+0x7ed55d16) + (a<<12);
        a = (a ^ 0xc761c23c) ^ (a >> 19);
        // a = (a+0x165667b1) + (a<<5);
        // a = (a+0xd3a2646c) ^ (a<<9);
        a = (a + 0xfd7046c5) + (a << 3);
        // a = (a^0xb55a4f09) ^ (a>>16);
        return a;
    }
};

namespace Search {
struct LmCacheKey {
    LmCacheKey(Lm::HistoryHandle h, Bliss::LemmaPronunciation::Id _pron)
            : history(h),
              pron(_pron),
              hash_(StandardValueHash<Bliss::LemmaPronunciation::Id>()(pron) + (((size_t)history) * 311) + ((size_t)history) / sizeof(void*)) {
    }

    size_t hash() const {
        return hash_;
    }

    bool operator==(const LmCacheKey& rhs) const {
        return pron == rhs.pron && history == rhs.history;
    }

    const Lm::HistoryHandle             history;
    const Bliss::LemmaPronunciation::Id pron;
    const size_t                        hash_;
};

struct LmCacheHash {
    size_t operator()(const LmCacheKey& key) const {
        return key.hash();
    }
};

struct LmCacheEqual {
    bool operator()(const LmCacheKey& key1, const LmCacheKey& key2) const {
        return key1 == key2;
    }
};

struct LmCacheItem {
    Lm::Score score;

    inline LmCacheItem(Lm::Score s)
            : score(s) {
    }
};

class LmCache {
    ///@todo Make sure that the handles of the cached histories are kept alive
public:
    ///Should be called regularly to clean up the cache
    ///All items that were not requested since the last call to clean()
    ///will be removed.
    ///@return the count of items remaining in the cache
    u32 clean() {
        oldCache_.swap(newCache_);
        newCache_.clear();
        return oldCache_.size();
    }

    ///@return Core::Type<Lm::Score>::max if the item needs to be calculated
    LmCacheItem& retrieve(const LmCacheKey& key) {
        Hash::iterator it = newCache_.find(key);
        if (it != newCache_.end())
            return (*it).second;

        it = oldCache_.find(key);

        LmCacheItem item = LmCacheItem(Core::Type<Lm::Score>::max);

        if (it != oldCache_.end()) {
            item = (*it).second;
            oldCache_.erase(it);
        }

        return newCache_.insert(std::make_pair(key, item)).first->second;
    }

private:
    typedef std::unordered_map<LmCacheKey, LmCacheItem, LmCacheHash, LmCacheEqual> Hash;
    Hash                                                                           oldCache_;
    Hash                                                                           newCache_;
};
}  // namespace Search

#endif
