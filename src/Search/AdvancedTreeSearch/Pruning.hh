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
#ifndef PRUNING_HH
#define PRUNING_HH

#include <algorithm>
#include <iterator>
#include "SearchSpace.hh"

namespace Search {
struct SearchSpace::AcousticPruning {
    AcousticPruning(SearchSpace& ss, Score relativeThreshold = 0, Score minimum = 0)
            : relativeThreshold_(relativeThreshold),
              absoluteThreshold_(Core::Type<Score>::max),
              minimum_(minimum) {
        if (relativeThreshold_ == 0)
            relativeThreshold_ = ss.acousticPruning_;
        if (minimum_ == 0)
            minimum_ = ss.bestProspect_;
        if (minimum_ != Core::Type<Score>::max)
            absoluteThreshold_ = minimum_ + relativeThreshold_;
    }

    inline void startInstance(InstanceKey const& key) {}

    inline void prepare(const StateHypothesis& hyp) {
        if (hyp.prospect < minimum_) {
            minimum_           = hyp.prospect;
            absoluteThreshold_ = minimum_ + relativeThreshold_;
        }
    }

    inline bool prune(const StateHypothesis& hyp) const {
        return hyp.prospect > absoluteThreshold_;
    }
    inline bool prune(const TraceManager &, const StateHypothesis& hyp) const {
        return prune(hyp);
    }

    enum {
        CanPrune = 1
    };

    Score relativeThreshold_;
    Score absoluteThreshold_;
    Score minimum_;
};

struct SearchSpace::PerInstanceAcousticPruning {
    PerInstanceAcousticPruning(SearchSpace& ss, Score relativeThreshold = 0, Score instanceRelativeThresholdScale = 0, Score minimum = 0)
            : ss_(ss),
              relativeThreshold_(relativeThreshold),
              instanceRelativeThresholdScale_(instanceRelativeThresholdScale),
              absoluteThreshold_(Core::Type<Score>::max),
              minimum_(minimum),
              instanceMinimum_(Core::Type<Score>::max),
              instanceThreshold_(Core::Type<Score>::max) {
        if (relativeThreshold_ == 0)
            relativeThreshold_ = ss.acousticPruning_;
        if (instanceRelativeThresholdScale_ == 0)
            instanceRelativeThresholdScale_ = ss.perInstanceAcousticPruningScale_;
        if (minimum_ == 0)
            minimum_ = ss.bestProspect_;
        if (minimum_ != Core::Type<Score>::max)
            absoluteThreshold_ = minimum_ + relativeThreshold_;
    }

    ~PerInstanceAcousticPruning() {
        ss_.bestInstanceProspect_[prevInstance_] = instanceMinimum_;
    }

    inline void startInstance(InstanceKey const& key) {
        ss_.bestInstanceProspect_[prevInstance_] = instanceMinimum_;
        prevInstance_                            = key;
        auto iter                                = ss_.bestInstanceProspect_.find(key);
        if (iter != ss_.bestInstanceProspect_.end()) {
            instanceMinimum_   = iter->second;
            instanceThreshold_ = instanceMinimum_ + relativeThreshold_ * instanceRelativeThresholdScale_;
        }
        else {
            instanceMinimum_   = Core::Type<Score>::max;
            instanceThreshold_ = Core::Type<Score>::max;
        }
    }

    inline void prepare(const StateHypothesis& hyp) {
        if (hyp.prospect < minimum_) {
            minimum_           = hyp.prospect;
            absoluteThreshold_ = minimum_ + relativeThreshold_;
        }
        if (hyp.prospect < instanceMinimum_) {
            instanceMinimum_   = hyp.prospect;
            instanceThreshold_ = instanceMinimum_ + relativeThreshold_ * instanceRelativeThresholdScale_;
        }
    }

    inline bool prune(const StateHypothesis& hyp) const {
        return hyp.prospect > absoluteThreshold_ or hyp.prospect > instanceThreshold_;
    }
    inline bool prune(const TraceManager &, const StateHypothesis& hyp) const {
        return prune(hyp);
    }

    enum {
        CanPrune = 1
    };

    SearchSpace& ss_;
    Score        relativeThreshold_;
    Score        instanceRelativeThresholdScale_;
    Score        absoluteThreshold_;
    Score        minimum_;
    Score        instanceMinimum_;
    Score        instanceThreshold_;
    InstanceKey  prevInstance_;
};

struct SearchSpace::RecordMinimum {
    RecordMinimum(SearchSpace& ss)
            : ss_(ss),
              minimum_(Core::Type<Score>::max) {}

    virtual ~RecordMinimum() {
        if (minimum_ < ss_.bestProspect_)
            ss_.bestProspect_ = minimum_;
    }

    inline void startInstance(InstanceKey const& key) {}

    inline void prepare(const StateHypothesis& hyp) {
        if (hyp.prospect < minimum_)
            minimum_ = hyp.prospect;
    }

    inline bool prune(const StateHypothesis& hyp) const {
        return false;
    }

    enum {
        CanPrune = 0
    };

    SearchSpace& ss_;
    Score        minimum_;
};

struct SearchSpace::RecordMinimumPerInstance {
    RecordMinimumPerInstance(SearchSpace& ss)
            : ss_(ss),
              minimum_(Core::Type<Score>::max),
              instanceMinimum_(Core::Type<Score>::max),
              prevInstance_() {
        ss_.bestInstanceProspect_.clear();
    }

    virtual ~RecordMinimumPerInstance() {
        if (minimum_ < ss_.bestProspect_)
            ss_.bestProspect_ = minimum_;
        ss_.bestInstanceProspect_[prevInstance_] = instanceMinimum_;
    }

    inline void startInstance(InstanceKey const& key) {
        ss_.bestInstanceProspect_[prevInstance_] = instanceMinimum_;
        instanceMinimum_                         = Core::Type<Score>::max;
        prevInstance_                            = key;
    }

    inline void prepare(const StateHypothesis& hyp) {
        if (hyp.prospect < minimum_) {
            minimum_ = hyp.prospect;
        }
        if (hyp.prospect < instanceMinimum_) {
            instanceMinimum_ = hyp.prospect;
        }
    }

    inline bool prune(const StateHypothesis& hyp) const {
        return false;
    }

    enum {
        CanPrune = 0
    };

    SearchSpace& ss_;
    Score        minimum_;
    Score        instanceMinimum_;
    InstanceKey  prevInstance_;
};

struct SearchSpace::NoPruning {
    NoPruning(SearchSpace& ss) {
    }

    inline void startInstance(InstanceKey const& key) {}

    inline bool prune(const StateHypothesis&) const {
        return false;
    }

    inline void prepare(const StateHypothesis&) const {
    }

    enum {
        CanPrune = 0
    };
};

enum {
    MaxFadeInPruningDistance = 255
};

struct SearchSpace::BestTracePruning {
    uintptr_t                   root_ptr;
    std::unordered_set<TraceId> live_traces;
    std::unordered_set<TraceId> dead_traces;

    BestTracePruning(Core::Ref<Trace> root)
            : root_ptr(reinterpret_cast<uintptr_t>(root.get())) {
        root_ptr = root->pruningMark;
    }

    inline void startInstance(InstanceKey const& key) {}

    inline bool prune(TraceManager &trace_manager, StateHypothesis const& hyp) {
        const uintptr_t invalidPruningMark = root_ptr ^ -1;
        if (live_traces.count(hyp.trace)) {
            return false;
        }
        if (dead_traces.count(hyp.trace)) {
            return true;
        }

        TraceItem           &currentItem = trace_manager.traceItem(hyp.trace);
        Trace*              current = currentItem.trace.get();
        std::vector<Trace*> chain;
        bool                should_prune = true;
        while (current) {
            chain.push_back(current);
            if (current->pruningMark == root_ptr) {
                should_prune = false;
                break;
            }
            else if (current->pruningMark == invalidPruningMark) {
                break;
            }
            current = current->predecessor.get();
        }
        if (should_prune) {
            dead_traces.insert(hyp.trace);
            for (Trace* e : chain) {
                e->pruningMark = invalidPruningMark;
            }
        }
        else {
            live_traces.insert(hyp.trace);
            for (Trace* e : chain) {
                e->pruningMark = root_ptr;
            }
        }
        return should_prune;
    }

    inline void prepare(const StateHypothesis&) const {}

    enum {
        CanPrune = 1
    };
};

}  // namespace Search

#endif
