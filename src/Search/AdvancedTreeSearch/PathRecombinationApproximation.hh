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
#ifndef SEARCH_PATHRECOMBINATIONAPPROXIMATION_HH
#define SEARCH_PATHRECOMBINATIONAPPROXIMATION_HH

#include "PathRecombination.hh"
#include "PersistentStateTree.hh"

namespace Search {
class PathRecombinationApproximation {
public:
    PathRecombinationApproximation(const PersistentStateTree& network, const Core::Configuration& config, const PathRecombination& pathrec);
    ~PathRecombinationApproximation();

private:
    const PersistentStateTree& network_;
    const PathRecombination&   pathrec_;
    std::vector<s32>           cliqueSizes_;

    class CliquePartition {
    public:
        CliquePartition(const Search::PersistentStateTree& network, const Search::PathRecombination& pathrec, u32 cliqueSize);

    private:
        std::vector<u32>               cliqueForState_;
        std::vector<std::set<StateId>> statesForClique_;
        std::vector<u32>               recombinationIntervalForClique;
        std::vector<u32>               recombinationIntervalForCliqueWithoutState;
        const PersistentStateTree&     network_;
        const PathRecombination&       pathrec_;
        u32                            nCliques_;
        void                           removeFromClique(u32 clique, StateId state);
        u32                            cliqueRecombinationInterval(u32 clique);
        u32                            cliqueWithoutStateRecombinationInterval(u32 clique, StateId state);
        u32                            symmetricStateCliqueRecombinationInterval(u32 clique, StateId state);
        void                           addToClique(u32 clique, StateId state, u32 symmetricLocalRecombinationInterval);
    };

    std::map<u32, CliquePartition*> partitionForCliqueSize_;

    void initialize();
};
}  // namespace Search

#endif  // SEARCH_PATHRECOMBINATIONAPPROXIMATION_HH
