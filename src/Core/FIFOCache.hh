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

#ifndef FIFO_CACHE_HH
#define FIFO_CACHE_HH

#include <cstddef>
#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>

namespace Core {

/*
 * Cache based on an unordered_map with a maximum size such that when the cache is full and
 * a new item is trying to be added, the oldest item in the cache gets removed.
 * Note: oldest is determined by order of insertion, not order of last access.
 *
 * Example:
 *
 * FIFOCache<int, std::string> cache(2);  // Cache has room for two items
 * cache.put(1, "one");  // Internal data: {1: "one"}
 * cache.put(2, "two");  // Internal data: {1: "one", 2: "two"}
 * cache.put(3, "three");  // Oldest element is deleted since max size was reached. Internal data: {2: "two", 3: "three"}
 */
template<typename Key, typename Value, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
class FIFOCache {
public:
    inline FIFOCache(size_t maxSize);

    // Insert or update a key-value pair (oldest inserted elements get removed first)
    inline void put(const Key& key, const Value& value);

    inline std::optional<std::reference_wrapper<Value>>       get(const Key& key);
    inline std::optional<std::reference_wrapper<const Value>> get(const Key& key) const;
    inline Value&                                             operator[](const Key& key);
    inline bool                                               contains(const Key& key) const;
    inline void                                               clear();
    inline size_t                                             size() const;
    inline size_t                                             maxSize() const;

private:
    using Map         = typename std::unordered_map<Key, Value, Hash, KeyEqual>;
    using MapIterator = typename std::unordered_map<Key, Value, Hash, KeyEqual>::iterator;

    Map    cacheMap_;
    size_t maxSize_;

    // Ring-vector with the keys to all the elements of the cacheMap in order of insertion.
    std::vector<Key> cacheElementKeys_;
    size_t           oldestElementPos_;  // Position of the oldest element inside `cacheElementIters_`
};

/*
 * Implementations
 */

template<typename Key, typename Value, typename Hash, typename KeyEqual>
inline FIFOCache<Key, Value, Hash, KeyEqual>::FIFOCache(size_t maxSize)
        : cacheMap_(), maxSize_(maxSize), cacheElementKeys_(), oldestElementPos_(0ul) {
    cacheMap_.reserve(maxSize_);
    cacheElementKeys_.reserve(maxSize_);
}

// Insert or update a key-value pair (oldest inserted elements get removed first)
template<typename Key, typename Value, typename Hash, typename KeyEqual>
inline void FIFOCache<Key, Value, Hash, KeyEqual>::put(const Key& key, const Value& value) {
    auto it = cacheMap_.find(key);
    if (it != cacheMap_.end()) {
        // Key exists; update the value but don't change the order in the list
        it->second = value;
    }
    else {  // Key doesn't exist yet
        // If the cache is not full yet, just insert the new item
        if (cacheElementKeys_.size() < maxSize_) {
            cacheMap_.emplace(key, value);
            cacheElementKeys_.push_back(key);  // Store key of the newly inserted element
        }
        else {  // Cache is full -> replace oldest item with the new item
            cacheMap_.erase(cacheElementKeys_[oldestElementPos_]);
            cacheMap_.emplace(key, value);

            // Replace old key in vector with new key
            cacheElementKeys_[oldestElementPos_] = key;

            // Next oldest element becomes new oldest
            oldestElementPos_ = (oldestElementPos_ + 1ul) % maxSize_;  // Wrap position around at the end of the vector
        }
    }
}

template<typename Key, typename Value, typename Hash, typename KeyEqual>
inline std::optional<std::reference_wrapper<Value>> FIFOCache<Key, Value, Hash, KeyEqual>::get(const Key& key) {
    auto it = cacheMap_.find(key);
    if (it != cacheMap_.end()) {
        return std::ref(it->second);
    }
    return {};
}

template<typename Key, typename Value, typename Hash, typename KeyEqual>
inline std::optional<std::reference_wrapper<const Value>> FIFOCache<Key, Value, Hash, KeyEqual>::get(const Key& key) const {
    auto it = cacheMap_.find(key);
    if (it != cacheMap_.end()) {
        return std::ref(it->second);
    }
    return {};
}

template<typename Key, typename Value, typename Hash, typename KeyEqual>
inline Value& FIFOCache<Key, Value, Hash, KeyEqual>::operator[](const Key& key) {
    auto it = cacheMap_.find(key);
    if (it != cacheMap_.end()) {
        return it->second;
    }

    put(key, {});
    return cacheMap_[key];
}

template<typename Key, typename Value, typename Hash, typename KeyEqual>
inline bool FIFOCache<Key, Value, Hash, KeyEqual>::contains(const Key& key) const {
    return cacheMap_.find(key) != cacheMap_.end();
}

template<typename Key, typename Value, typename Hash, typename KeyEqual>
inline void FIFOCache<Key, Value, Hash, KeyEqual>::clear() {
    cacheMap_.clear();
    cacheElementKeys_.clear();
    oldestElementPos_ = 0ul;
}

template<typename Key, typename Value, typename Hash, typename KeyEqual>
inline size_t FIFOCache<Key, Value, Hash, KeyEqual>::size() const {
    return cacheMap_.size();
}

template<typename Key, typename Value, typename Hash, typename KeyEqual>
inline size_t FIFOCache<Key, Value, Hash, KeyEqual>::maxSize() const {
    return maxSize_;
}

}  // namespace Core

#endif  // FIFO_CACHE_HH
