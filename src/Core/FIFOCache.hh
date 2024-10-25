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

#ifndef FIFO_CACHE_HH
#define FIFO_CACHE_HH

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace Core {

/*
 * Cache based on an unordered_map with a maximum size such that when the cache is full and
 * a new item is trying to be added, the oldest item in the cache gets removed.
 */
template<typename Key, typename Value, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
class FIFOCache {
public:
    FIFOCache(size_t maxSize)
            : cacheMap_(), maxSize_(maxSize), cacheElementIters_(), oldestElementPos_(0ul) {
        cacheMap_.reserve(maxSize_);
        cacheElementIters_.reserve(maxSize_);
    }

    Value& get(const Key& key) {
        return cacheMap_.at(key);
    }

    // Insert or update a key-value pair (oldest inserted elements get removed first)
    void put(const Key& key, const Value& value) {
        auto it = cacheMap_.find(key);
        if (it != cacheMap_.end()) {
            // Key exists, update the value but don't change the order in the list
            it->second = value;
        }
        else {  // Key doesn't exist yet

            // If the cache is not full yet, just insert the new item
            if (cacheElementIters_.size() < maxSize_) {
                auto it = cacheMap_.emplace(key, value);
                cacheElementIters_.push_back(it.first);  // Store iterator to the newly inserted element
            }
            else {  // Cache is full -> replace oldest item with the new item
                auto& oldestElementIt = cacheElementIters_[oldestElementPos_];
                cacheMap_.erase(oldestElementIt);
                auto it = cacheMap_.emplace(key, value);

                // Replace old iterator with iterator of newly inserted element
                cacheElementIters_[oldestElementPos_] = it.first;

                // Wrap around at the end of the vector
                oldestElementPos_ = (oldestElementPos_ + 1ul) % maxSize_;
            }
        }
    }

    bool contains(const Key& key) const {
        return cacheMap_.find(key) != cacheMap_.end();
    }

    void clear() {
        cacheMap_.clear();
        cacheElementIters_.clear();
        oldestElementPos_ = 0ul;
    }

    size_t size() const {
        return cacheMap_.size();
    }

    size_t maxSize() const {
        return maxSize_;
    }

private:
    using Map         = typename std::unordered_map<Key, Value, Hash, KeyEqual>;
    using MapIterator = typename std::unordered_map<Key, Value, Hash, KeyEqual>::iterator;

    Map    cacheMap_;
    size_t maxSize_;

    // List with iterators to all the elements of the cacheMap.
    // Note: since full size of the cacheMap is reserved in advance, existing iterators will not be invalidated when inserting new elements
    std::vector<MapIterator> cacheElementIters_;  // TODO: Delete by key instead of by iterator so that it's robust against iterator invalidation
    size_t                   oldestElementPos_;   // Position of the oldest element inside `cacheElementIters_`
};

}  // namespace Core

#endif  // FIFO_CACHE_HH
