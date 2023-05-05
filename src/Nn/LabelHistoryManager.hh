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

#ifndef LABEL_HISTORY_MANAGER_HH
#define LABEL_HISTORY_MANAGER_HH

#include <Core/Assertions.hh>
#include <Core/MurmurHash.hh>
#include <Search/Types.hh>

// boost way of hash merging (we use 0 for special case: mostly initial or N.A.)
// Note: not 100% collision-free, better with additional safety where it's applied
static size_t updateHashKey(size_t hash, size_t update) {
    // nothing to update
    if (update == 0) {
        return hash;
    }
    if (hash == 0) {
        return update;
    }
    return hash ^ (update + 0x9e3779b9 + (hash << 6) + (hash >> 2));
}

namespace Nn {

class LabelHistory;

typedef Search::Index           LabelIndex;
typedef std::vector<LabelIndex> LabelSequence;

inline size_t label_sequence_hash(const LabelSequence& ls) {
    return Core::MurmurHash3_x64_64(reinterpret_cast<void const*>(ls.data()),
                                    ls.size() * sizeof(LabelSequence::value_type), 0x78b174eb);
}

// Note: all history have to inherit from LabelHistoryBase
struct LabelHistoryBase {
    size_t        ref_count, cacheHash;
    LabelSequence labelSeq;  // always right-most latest

    LabelHistoryBase()
            : ref_count(0), cacheHash(0) {}
    LabelHistoryBase(const LabelHistoryBase& ref)
            : ref_count(0), cacheHash(0), labelSeq(ref.labelSeq) {}

    virtual ~LabelHistoryBase() = default;
};

typedef LabelHistoryBase*                              LabelHistoryHandle;
typedef std::unordered_map<size_t, LabelHistoryHandle> HistoryCache;
typedef std::pair<HistoryCache::iterator, bool>        CacheUpdateResult;

// LabelHistoryObject handling (caching, reference counting and clean up ...): all inline
class LabelHistoryManager {
public:
    LabelHistoryManager() {}
    ~LabelHistoryManager() {
        verify_(cache_.empty());
    }

    LabelHistory history(LabelHistoryHandle lhd) const;
    void         reset() {
        cache_.clear();
    }

    bool isEqualSequence(const LabelHistoryHandle lhd, const LabelHistoryHandle rhd) const {
        return label_sequence_hash(lhd->labelSeq) == label_sequence_hash(rhd->labelSeq);
    }

    bool isEqualSequence(const LabelHistoryHandle lhd, LabelIndex lIdx, const LabelHistoryHandle rhd) const {
        return extendedHashKey(lhd, lIdx) == label_sequence_hash(rhd->labelSeq);
    }

    const HistoryCache& historyCache() {
        return cache_;
    }
    // check existence for to-be-extended history
    CacheUpdateResult checkCache(const LabelHistoryHandle lhd, LabelIndex lIdx, u32 updateHash);
    CacheUpdateResult checkCache(const LabelHistoryHandle lhd, u32 updateHash);
    CacheUpdateResult updateCache(LabelHistoryHandle lhd, u32 updateHash);

    size_t hashKey(const LabelHistoryHandle lhd) const {
        return label_sequence_hash(lhd->labelSeq);
    }

    size_t reducedHashKey(const LabelSequence& labelSeq, s32 limit) const;
    size_t reducedHashKey(const LabelHistoryHandle lhd, s32 limit) const {
        return reducedHashKey(lhd->labelSeq, limit);
    }

    size_t extendedHashKey(const LabelHistoryHandle lhd, LabelIndex lIdx) const;
    size_t reducedExtendedHashKey(const LabelHistoryHandle lhd, s32 limit, LabelIndex lIdx) const;

protected:
    friend class LabelHistory;
    LabelHistoryHandle acquire(LabelHistoryHandle lhd) const;
    void               release(LabelHistoryHandle lhd) const;

private:
    mutable HistoryCache cache_;
};

class LabelHistory {
public:
    LabelHistory()
            : mang_(0), desc_(0) {}
    LabelHistory(const LabelHistory& ref)
            : mang_(ref.mang_), desc_(ref.desc_) {
        if (desc_)
            mang_->acquire(desc_);
    }

    ~LabelHistory() {
        if (desc_)
            mang_->release(desc_);
    }

    const LabelHistory& operator=(const LabelHistory& rhs);

    const LabelHistoryManager* manager() const {
        return mang_;
    }
    const LabelHistoryHandle handle() const {
        return desc_;
    }

    bool isValid() const {
        return mang_ != 0;
    }

    size_t hashKey() const;
    size_t reducedHashKey(s32 limit) const;
    size_t reducedExtendedHashKey(s32 limit, LabelIndex lIdx) const;

    struct Hash {
        inline size_t operator()(const LabelHistory& lh) const {
            return lh.isValid() ? lh.hashKey() : 0;
        }
    };

    LabelIndex getLastLabel() const;

    // debug
    void format() const;

private:
    friend class LabelHistoryManager;
    LabelHistory(const LabelHistoryManager* lhm, LabelHistoryHandle lhd)
            : mang_(lhm), desc_(mang_->acquire(lhd)) {}

private:
    const LabelHistoryManager* mang_;
    LabelHistoryHandle         desc_;
};

inline LabelHistoryHandle LabelHistoryManager::acquire(LabelHistoryHandle lhd) const {
    if (lhd)
        ++(lhd->ref_count);
    return lhd;
}

inline void LabelHistoryManager::release(LabelHistoryHandle lhd) const {
    if (lhd) {
        require_gt(lhd->ref_count, 0);
        --(lhd->ref_count);
        if (lhd->ref_count == 0) {
            // remove from cache
            cache_.erase(lhd->cacheHash);
            delete lhd;
        }
    }
}

inline size_t LabelHistoryManager::reducedHashKey(const LabelSequence& labelSeq, s32 limit) const {
    if (limit < 0 || (u32)limit >= labelSeq.size())
        return label_sequence_hash(labelSeq);
    LabelSequence reducedLabelSeq(labelSeq.end() - limit, labelSeq.end());
    return label_sequence_hash(reducedLabelSeq);
}

inline size_t LabelHistoryManager::extendedHashKey(const LabelHistoryHandle lhd, LabelIndex lIdx) const {
    LabelSequence extendedLabelSeq(lhd->labelSeq);
    extendedLabelSeq.push_back(lIdx);
    return label_sequence_hash(extendedLabelSeq);
}

inline size_t LabelHistoryManager::reducedExtendedHashKey(const LabelHistoryHandle lhd, s32 limit, LabelIndex lIdx) const {
    if (limit < 0 || (u32)limit > lhd->labelSeq.size())
        return extendedHashKey(lhd, lIdx);
    LabelSequence reducedLabelSeq(lhd->labelSeq.end() - (limit - 1), lhd->labelSeq.end());
    reducedLabelSeq.push_back(lIdx);
    return label_sequence_hash(reducedLabelSeq);
}

inline LabelHistory LabelHistoryManager::history(LabelHistoryHandle lhd) const {
    return LabelHistory(this, lhd);
}

// check existence for to-be-extended history
inline CacheUpdateResult LabelHistoryManager::checkCache(const LabelHistoryHandle lhd,
                                                         LabelIndex lIdx, u32 updateHash) {
    size_t                 hash = updateHashKey(extendedHashKey(lhd, lIdx), updateHash);
    HistoryCache::iterator iter = cache_.find(hash);
    return std::make_pair(iter, iter != cache_.end());
}

inline CacheUpdateResult LabelHistoryManager::checkCache(const LabelHistoryHandle lhd, u32 updateHash) {
    size_t                 hash = updateHashKey(hashKey(lhd), updateHash);
    HistoryCache::iterator iter = cache_.find(hash);
    return std::make_pair(iter, iter != cache_.end());
}

inline CacheUpdateResult LabelHistoryManager::updateCache(LabelHistoryHandle lhd, u32 updateHash) {
    size_t hash    = updateHashKey(hashKey(lhd), updateHash);
    lhd->cacheHash = hash;
    return cache_.insert(std::make_pair(hash, lhd));
}

inline const LabelHistory& LabelHistory::operator=(const LabelHistory& rhs) {
    if (rhs.desc_)
        rhs.mang_->acquire(rhs.desc_);
    if (desc_)
        mang_->release(desc_);
    mang_ = rhs.mang_;
    desc_ = rhs.desc_;
    return *this;
}

inline size_t LabelHistory::hashKey() const {
    if (desc_)
        return mang_->hashKey(desc_);
    return 0;
}

inline size_t LabelHistory::reducedHashKey(s32 limit) const {
    if (desc_ && limit != 0)
        return mang_->reducedHashKey(desc_, limit);
    return 0;
}

inline size_t LabelHistory::reducedExtendedHashKey(s32 limit, LabelIndex lIdx) const {
    if (desc_ && limit != 0)
        return mang_->reducedExtendedHashKey(desc_, limit, lIdx);
    return 0;
}

inline LabelIndex LabelHistory::getLastLabel() const {
    if (desc_ && !desc_->labelSeq.empty())
        return desc_->labelSeq.back();
    return Core::Type<LabelIndex>::max;
}

// debug
inline void LabelHistory::format() const {
    std::cout << "  LabelHistory: ";
    if (desc_)
        for (LabelIndex label : desc_->labelSeq)
            std::cout << label << " ";
    std::cout << std::endl;
}

}  // namespace Nn

#endif
