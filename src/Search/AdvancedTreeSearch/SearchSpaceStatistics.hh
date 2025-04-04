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
#ifndef SEARCH_CONDITIONEDTREESEARCHSPACESTATISTICS_HH
#define SEARCH_CONDITIONEDTREESEARCHSPACESTATISTICS_HH

#include <Core/Hash.hh>
#include <Core/Statistics.hh>
#include <Search/Types.hh>
#include <map>

namespace Search {
// ===========================================================================
struct SearchSpaceStatistics {
    /// Can be used to easily do statistics, when performance does not matter
    Core::Statistics<f32>&     customStatistics(const std::string& name);
    Core::HistogramStatistics& customHistogramStatistics(const std::string& name, u32 buckets = 10);
    ~SearchSpaceStatistics();

    Core::Statistics<u32> treesBeforePruning, treesAfterPrePruning, treesAfterPruning,
            statesBeforePruning, statesAfterPrePruning, statesAfterPruning,
            wordEndsBeforePruning, wordEndsAfterPruning,
            epsilonWordEndsAdded, wordEndsAfterRecombination, wordEndsAfterSecondPruning;

    Core::Statistics<Score> acousticHistogramPruningThreshold, lmHistogramPruningThreshold;

    Core::HistogramStatistics entryStateHypotheses;
    Core::HistogramStatistics rootStateHypothesesPerTree;

    SearchSpaceStatistics();
    void clear();
    void write(Core::XmlWriter&) const;

private:
    std::map<std::string, Core::Statistics<f32>*>     customStatistics_;
    std::map<std::string, Core::HistogramStatistics*> customHistogramStatistics_;
};
}  // namespace Search

#endif
