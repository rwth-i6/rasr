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
#ifndef CONDITIONEDTREESEARCHTRACE_HH
#define CONDITIONEDTREESEARCHTRACE_HH

#include <Core/ReferenceCounting.hh>
#include <Search/Search.hh>
#include <Search/StateTree.hh>
#include <Search/Traceback.hh>
#include <Search/Types.hh>
#include "PathTrace.hh"

namespace Search {

struct AlternativeHistory {
    Lm::History            hist;
    ScoreVector            offset;
    Core::Ref<class Trace> trace;
};

struct AlternativeHistoryCompare {
    bool operator()(AlternativeHistory const& a, AlternativeHistory const& b) {
        return a.offset < b.offset;
    }
};

template<typename T, typename Container, typename Compare>
class AccessiblePriorityQueue : public std::priority_queue<T, Container, Compare> {
public:
    Container const& container() const {
        return this->c;
    }
    Container& container() {
        return this->c;
    }
};

using AlternativeHistoryQueue = AccessiblePriorityQueue<AlternativeHistory, std::vector<AlternativeHistory>, AlternativeHistoryCompare>;

class Trace : public Core::ReferenceCounted,
              public TracebackItem {
public:
    Core::Ref<Trace> predecessor;
    Core::Ref<Trace> sibling;
    PathTrace        pathTrace;
    uintptr_t        pruningMark;  // used by BestTracePruning
    bool             mark;

    AlternativeHistoryQueue alternativeHistories;

    Trace(Core::Ref<Trace> const&          pre,
          Bliss::LemmaPronunciation const* p,
          TimeframeIndex                   t,
          ScoreVector                      s,
          Transit const&                   transit);

    Trace(TimeframeIndex t, ScoreVector s, const Transit& transit);

    void write(std::ostream& os, Core::Ref<const Bliss::PhonemeInventory> phi) const;

    void getLemmaSequence(std::vector<Bliss::Lemma*>& lemmaSequence) const;

    u32 wordCount() const {
        u32 count = 0;
        if (pronunciation)
            ++count;
        if (predecessor)
            count += predecessor->wordCount();

        return count;
    }
};

}  // namespace Search

#endif  // CONDITIONEDTREESEARCHTRACE_HH
