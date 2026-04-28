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
#ifndef PATHRECOMBINATION_HH
#define PATHRECOMBINATION_HH

#include <Core/MappedArchive.hh>
#include <Search/PersistentStateTree.hh>
#include <Search/TreeStructure.hh>

#include "Helpers.hh"

namespace Search {
class PathRecombination {
public:
    PathRecombination(const PersistentStateTree& network, const Core::Configuration& config);
    /**
     * Returns the expected interval when all followup hypotheses of b have been recombined with some followup hypotheses of a
     * */
    u32 recombinationInterval(StateId a, StateId b) const;

    void logStatistics();

private:
    // Determines the recombination-states
    void buildRecombinationStates();
    // Connects the recombination-states
    void buildRecombinationNetwork();
    // Computes distances between recombination-states
    void buildDistances();

    void computeDistancesForState(Search::StateId state);

    // Time until recombination on a pivot with distances d1 and d2
    inline uint t(uint d1, uint d2) const {
        if (d2 > d1)
            std::swap(d1, d2);

        return t_a(d1 - d2) +
               t_s(d1 - t_a(d1 - d2) * delta_ * convergenceFactor_);
    }

    // Time until asymetric recombination with pivot distance d
    inline uint t_a(uint d) const {
        return d / asymmetryFactor_;
    }

    // Time until symmetric recombination with pivot distance d
    inline uint t_s(uint d) const {
        return d / delta_;
    }

    u32 linearChainLength(StateId from, StateId to) const {
        if (offsetAndDistanceStateForState_[from].second == offsetAndDistanceStateForState_[to].second)
            if (offsetAndDistanceStateForState_[from].first >= offsetAndDistanceStateForState_[to].first)
                return offsetAndDistanceStateForState_[from].first - offsetAndDistanceStateForState_[to].first;
        return Core::Type<u32>::max;
    }

    // Returns 0 if no unique successor exists
    StateId getUniqueSuccessor(StateId state) const;

    const PersistentStateTree& network_;
    std::string                cache_;
    float                      convergenceFactor_, delta_;
    float                      asymmetryFactor_;
    // Maps each recombination-state to its unique index. Zero for non-recombination states.
    std::vector<u32>  recombinationStateMap_;
    std::vector<bool> visitingRecombinationState_;

    struct RecombinationState {
        RecombinationState()
                : state(0),
                  loop(false) {
        }
        // The network-state this recombination-state is assigned to
        StateId state;
        bool    loop;  // Whether this state has itself as successor

        struct Successor {
            Successor()
                    : shortestDistance(Core::Type<u32>::max),
                      longestDistance(0),
                      state(0) {
            }
            u32     shortestDistance;
            u32     longestDistance;
            StateId state;
        };

        // Successor recombination-states
        std::vector<Successor> successors;
    };

    std::vector<RecombinationState> recombinationStates_;

    struct DistanceState {
        struct DistanceItem {
            DistanceItem()
                    : shortestDistance(Core::Type<u32>::max) {
            }
            u32 shortestDistance;
        };
        // A pair of the longest distance and the
        // Recombination-state indices of all states that are direct successors
        std::vector<std::pair<u32, u32>> directSuccessorStates;
        std::vector<DistanceItem>        distances;
    };

    std::vector<DistanceState> distances_;
    // Pair of offset and distance
    std::vector<std::pair<u32, u32>> offsetAndDistanceStateForState_;

    Core::MMappedFile    mappedCacheFile_;
    const DistanceState* distancesPtr_;
    std::pair<u32, u32>* offsetAndDistanceStateForStatePtr_;

    struct StatePairHash {
        std::size_t operator()(const std::pair<StateId, StateId>& item) const {
            return item.first * 31231 + item.first / 4182 + item.second + item.second / 30;
        }
    };

    typedef std::unordered_map<std::pair<StateId, StateId>, uint, StatePairHash> IntervalCache;
    mutable IntervalCache                                                        intervalCache_;
    u32                                                                          maxCacheSize_, maxDepth_;
    mutable u32                                                                  currentVisits_;
    mutable u32                                                                  nVisits_;
    mutable u64                                                                  totalVisits_;
    bool                                                                         truncateNotPromising_, approximateLinearSequences_;
    u32                                                                          promisingApproximation_;

    typedef std::unordered_map<std::pair<StateId, StateId>, bool, StatePairHash> PromisingCache;

    PromisingCache promisingCache_;
    s32            maxExactInterval_;

    bool isNotPromising(u32 recombinationState, const DistanceState& distancesA);

    u32 R(u32 offsetA, const Search::PathRecombination::DistanceState& distancesA, u32 distB, const Search::PathRecombination::DistanceState& distancesB, u32 recombinationState, u32 depth);
};
}  // namespace Search

#endif  // PATHRECOMBINATION_HH
