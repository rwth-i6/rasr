
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

#include <Search/Types.hh>
#include <unordered_map>
#include "Core/MurmurHash.hh"

namespace Nn {

typedef Search::Index           LabelIndex;
typedef size_t                  LabelHistoryHash;
typedef std::vector<LabelIndex> LabelSequence;

struct LabelSequencePtrHash {
    LabelHistoryHash operator()(LabelSequence const* labelSeq) const {
        return Core::MurmurHash3_x64_64(reinterpret_cast<void const*>(labelSeq->data()), labelSeq->size() * sizeof(LabelSequence::value_type), 0x78b174eb);
    }
};

struct LabelSequencePtrEq {
    bool operator()(LabelSequence const* labelSeq1, LabelSequence const* labelSeq2) {
        if (labelSeq1 == labelSeq2) {
            return true;
        }

        if (labelSeq1->size() != labelSeq2->size()) {
            return false;
        }
        for (size_t idx = 0ul; idx < labelSeq1->size(); ++idx) {
            if (labelSeq1->at(idx) != labelSeq2->at(idx)) {
                return false;
            }
        }

        return true;
    }
};

// LabelHistoryManager managers a collection of label histories using a cache map
// supports reference counting and clean up
template<typename LabelHistoryType        = LabelSequence,
         typename LabelHistoryPtrHashType = LabelSequencePtrHash,
         typename LabelHistoryPtrEqType   = LabelSequencePtrEq>
class LabelHistoryManager {
public:
    // Internal class that allows reference counting
    struct RefCountedLabelHistory {
        size_t           refCount;
        LabelHistoryType labelHist;

        RefCountedLabelHistory() = default;
        RefCountedLabelHistory(const RefCountedLabelHistory& ref)
                : refCount(0), labelHist(ref.labelHist) {}

        virtual ~RefCountedLabelHistory() = default;
    };

    typedef std::unordered_map<LabelHistoryType const*, RefCountedLabelHistory*, LabelHistoryPtrHashType, LabelHistoryPtrEqType> CacheMap;

    LabelHistoryManager() = default;
    ~LabelHistoryManager() {
        reset();
    }

    void reset() {
        for (auto& keyval : cacheMap_) {
            delete keyval.second;
        }
        cacheMap_.clear;
    }

private:
    CacheMap cacheMap_;
};
}  // namespace Nn

#endif
