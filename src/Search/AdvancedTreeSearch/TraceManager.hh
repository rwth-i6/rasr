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
#ifndef SEARCH_TRACEMANAGER_HH
#define SEARCH_TRACEMANAGER_HH

#include <Core/ReferenceCounting.hh>
#include "Trace.hh"

namespace Search {
#define FAST_TRACE_MODIFICATION

typedef u32          TraceId;
static const TraceId invalidTraceId = Core::Type<TraceId>::max;

struct TraceItem final {
public:
    TraceItem(Core::Ref<Trace> t, Lm::History rch, Lm::History lah, Lm::History sch)
            : trace(t), recombinationHistory(rch), lookaheadHistory(lah), scoreHistory(sch) {
    }
    TraceItem() {
    }
    TraceItem(TraceItem const& ti) = default;
    TraceItem(TraceItem&& ti) {
        trace                = ti.trace;
        recombinationHistory = std::move(ti.recombinationHistory);
        lookaheadHistory     = std::move(ti.lookaheadHistory);
        scoreHistory         = std::move(ti.scoreHistory);
    }
    ~TraceItem() {
    }

    TraceItem& operator=(TraceItem const& ti) = default;
    TraceItem& operator                       =(TraceItem&& ti) {
        trace                = ti.trace;
        recombinationHistory = std::move(ti.recombinationHistory);
        lookaheadHistory     = std::move(ti.lookaheadHistory);
        scoreHistory         = std::move(ti.scoreHistory);
        return *this;
    }

    Core::Ref<Trace> trace;
    Lm::History      recombinationHistory;
    Lm::History      lookaheadHistory;
    Lm::History      scoreHistory;

private:
    friend class TraceManager;
};

template<typename T>
class SparseVector {
public:
    T& operator[](size_t idx) {
        require_lt(idx, items_.size());
        require(used_[idx]);
        return items_[idx];
    }

    T const& operator[](size_t idx) const {
        require_lt(idx, items_.size());
        require(used_[idx]);
        return items_[idx];
    }

    size_t insert(T const& item) {
        if (freeList_.empty()) {
            size_t idx = items_.size();
            items_.push_back(item);
            used_.push_back(true);
            return idx;
        }
        else {
            size_t idx = freeList_.top();
            require(not used_[idx]);
            items_[idx] = item;
            used_[idx]  = true;
            freeList_.pop();
            return idx;
        }
    }

    void erase(size_t idx) {
        require_lt(idx, items_.size());
        if (used_[idx]) {
            items_[idx] = T();
            used_[idx]  = false;
            freeList_.push(idx);
        }
    }

    void clear() {
        items_.clear();
        used_.clear();
        while (not freeList_.empty()) {
            freeList_.pop();
        }
    }

    void filter(std::vector<bool> const& keep) {
        require_le(keep.size(), items_.size());
        size_t i = 0ul;
        for (; i < keep.size(); i++) {
            if (not keep[i]) {
                erase(i);
            }
        }
        for (; i < items_.size(); i++) {
            erase(i);
        }
    }

    size_t pos(T const* item) const {
        return std::distance(items_.data(), item);
    }

    size_t size() const {
        return items_.size() - freeList_.size();
    }

    size_t storageSize() const {
        return items_.size();
    }

private:
    std::vector<T>       items_;
    std::vector<bool>    used_;
    std::stack<unsigned> freeList_;
};

class TraceManager {
private:
    enum {
        ModifyMask = 0xff000000
    };
    enum {
        UnModifyMask = 0x00ffffff
    };

public:
    void clear() {
        items_.clear();
        modifications_.clear();
    }

    /// Returns a trace-id that represents only the given item
    TraceId getTrace(const TraceItem& item) {
        return items_.insert(item);
    }

    /// Returns the trace-id of a trace that already is managed by the TraceManager
    TraceId getManagedTraceId(const TraceItem* item) {
        return items_.pos(item);
    }

    /// Returns the current number of existing trace-items
    uint numTraceItems() const {
        return items_.size();
    }

    /// Returns the maximum number of trace-items
    uint maxTraceItems() const {
        return UnModifyMask;
    }

    /// Returns whether a cleanup is currently strictly necessary (see cleanup())
    bool needCleanup() const {
        return numTraceItems() > maxTraceItems() / 2;
    }

    /// Returns whether the given trace-id is additionally modified by a custom value
    inline bool isModified(const TraceId trace) const {
        return trace & ModifyMask;
    }

    struct Modification {
        Modification(u32 _first = 0, u32 _second = 0, u32 _third = 0)
                : first(_first),
                  second(_second),
                  third(_third) {}
        bool operator==(const Modification& rhs) const {
            return first == rhs.first && second == rhs.second && third == rhs.third;
        }
        u32 first, second, third;
    };

    /// Returns the custom modification-value that was attached to the trace-id. Must only be called if isModified(trace).
    inline Modification getModification(TraceId trace) const {
        verify_(isModified(trace));
        u32          mod = ((trace & ModifyMask) >> 24);
        Modification ret;
        if (mod == 255) {
            ret = modifications_[trace & UnModifyMask].second;
        }
        else {
            ret.first = mod;
        }

        ret.first -= 1;  // Remove the offset that we have applied in modify(...)

        return ret;
    }

    /// Returns the unmodified version of the given trace.
    inline TraceId getUnmodified(TraceId trace) const {
        verify_(isModified(trace));
        if ((trace & ModifyMask) == ModifyMask) {
            return modifications_[trace & UnModifyMask].first;
        }
        else {
            return trace & UnModifyMask;
        }
    }
    inline TraceId getUnmodified(TraceId trace) {
        verify_(isModified(trace));
        if ((trace & ModifyMask) == ModifyMask) {
            return modifications_[trace & UnModifyMask].first;
        }
        else {
            return trace & UnModifyMask;
        }
    }

    /// Modifies the given trace-id with a specific value. The value can later be retrieved through
    /// getModification(traceid), where traceid is the returned value.
    TraceId modify(TraceId trace, u32 value, u32 value2 = 0, u32 value3 = 0) {
        verify_(trace != invalidTraceId);
        verify_(!isModified(trace));
        TraceId ret;

        value += 1;  // Offset by 1, so we can also modify with 0

#ifdef FAST_TRACE_MODIFICATION
        if (value < 255 && value2 == 0) {
            ret = (value << 24) | trace;
        }
        else {
#endif
            ret = modifications_.insert(std::make_pair<u32, Modification>((u32)trace, Modification(value, value2, value3)));
            ret |= ModifyMask;
#ifdef FAST_TRACE_MODIFICATION
        }
#endif

        verify_(getModification(ret) == Modification(value - 1, value2, value3));
        return ret;
    }

    /// Returns the (first) trace-item associated to the given trace-id
    /// The trace-id must be valid
    inline TraceItem const& traceItem(TraceId trace) const {
        return items_[getUnmodified(trace)];
    }
    inline TraceItem & traceItem(TraceId trace) {
        return items_[getUnmodified(trace)];
    }

    // Helper structure to cleanup the Tracemanager
    struct Cleaner {
        SparseVector<TraceItem>                    &items_;
        SparseVector<std::pair<u32, Modification>> &modifications_;
        std::vector<bool> item_filter;
        std::vector<bool> mod_filter;


        Cleaner(SparseVector<TraceItem> &items, SparseVector<std::pair<u32, Modification>> &modifications)
                : items_(items),
                  modifications_(modifications),
                  item_filter(items.storageSize(), false),
                  mod_filter(modifications.storageSize(), false) {
        }
        ~Cleaner() = default;

        void visit(TraceId traceid) {
            TraceId idx = traceid & UnModifyMask;
            if ((traceid & ModifyMask) == ModifyMask) {
                mod_filter[idx]                        = true;
                item_filter[modifications_[idx].first] = true;
            }
            else {
                item_filter[idx] = true;
            }
        }

        void clean() {
            items_.filter(item_filter);
            modifications_.filter(mod_filter);
        }
    };
    Cleaner getCleaner() {
      return Cleaner(items_, modifications_);
    }

private:
    SparseVector<TraceItem>                    items_;
    SparseVector<std::pair<u32, Modification>> modifications_;
};


}  // namespace Search

#endif  // SEARCH_TRACEMANAGER_HH
