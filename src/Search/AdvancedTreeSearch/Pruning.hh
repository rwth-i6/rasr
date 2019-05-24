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

    inline void startTree(const Instance&) {}

    inline void prepare(const StateHypothesis& hyp) {
        if (hyp.prospect < minimum_) {
            minimum_           = hyp.prospect;
            absoluteThreshold_ = minimum_ + relativeThreshold_;
        }
    }

    inline bool prune(const StateHypothesis& hyp) const {
        return hyp.prospect > absoluteThreshold_;
    }

    enum {
        CanPrune = 1
    };

    Score relativeThreshold_;
    Score absoluteThreshold_;
    Score minimum_;
};

struct SearchSpace::RecordMinimum {
    RecordMinimum(SearchSpace& ss)
            : ss_(ss),
              minimum_(Core::Type<Score>::max) {}

    inline bool needTreeNotifications() const {
        return false;
    }
    inline void startTree(const Instance&) {}

    virtual ~RecordMinimum() {
        if (minimum_ < ss_.bestProspect_)
            ss_.bestProspect_ = minimum_;
    }

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

struct SearchSpace::NoPruning {
    NoPruning(SearchSpace& ss) {
    }

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
}  // namespace Search

#endif
