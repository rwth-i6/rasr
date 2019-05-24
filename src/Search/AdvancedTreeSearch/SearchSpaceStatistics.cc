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
#include "SearchSpaceStatistics.hh"

using namespace Search;

void SearchSpaceStatistics::write(Core::XmlWriter& os) const {
    os << Core::XmlOpen("search-space-statistics")
       << treesBeforePruning
       << treesAfterPrePruning
       << treesAfterPruning
       << statesBeforePruning
       << statesAfterPrePruning
       << statesAfterPruning
       << wordEndsBeforePruning
       << wordEndsAfterPruning
       << epsilonWordEndsAdded
       << wordEndsAfterRecombination
       << wordEndsAfterSecondPruning
       << acousticHistogramPruningThreshold
       << lmHistogramPruningThreshold
       << entryStateHypotheses
       << rootStateHypothesesPerTree;

    for (std::map<std::string, Core::Statistics<f32>*>::const_iterator it = customStatistics_.begin(); it != customStatistics_.end(); ++it)
        os << *(*it).second;

    for (std::map<std::string, Core::HistogramStatistics*>::const_iterator it = customHistogramStatistics_.begin(); it != customHistogramStatistics_.end(); ++it)
        os << *(*it).second;

    os << Core::XmlClose("search-space-statistics");
}

SearchSpaceStatistics::SearchSpaceStatistics()
        : treesBeforePruning("trees before pruning"),
          treesAfterPrePruning("trees after pre-pruning"),
          treesAfterPruning("trees after  pruning"),
          statesBeforePruning("states before pruning"),
          statesAfterPrePruning("states after pre-pruning"),
          statesAfterPruning("states after pruning"),
          wordEndsBeforePruning("ending words before pruning"),
          wordEndsAfterPruning("ending words after pruning"),
          epsilonWordEndsAdded("epsilon word ends added"),
          wordEndsAfterRecombination("ending words after recombi"),
          wordEndsAfterSecondPruning("ending words after 2nd pruning"),
          acousticHistogramPruningThreshold("acoustic histogram pruning threshold"),
          lmHistogramPruningThreshold("lm histogram pruning threshold"),
          entryStateHypotheses("entry state hypotheses"),
          rootStateHypothesesPerTree("entry state hypotheses per network") {}

void SearchSpaceStatistics::clear() {
    treesBeforePruning.clear();
    treesAfterPrePruning.clear();
    treesAfterPruning.clear();
    statesBeforePruning.clear();
    statesAfterPrePruning.clear();
    statesAfterPruning.clear();
    wordEndsBeforePruning.clear();
    wordEndsAfterPruning.clear();
    epsilonWordEndsAdded.clear();
    wordEndsAfterRecombination.clear();
    wordEndsAfterSecondPruning.clear();
    acousticHistogramPruningThreshold.clear();
    lmHistogramPruningThreshold.clear();
    entryStateHypotheses.clear();
    rootStateHypothesesPerTree.clear();

    for (std::map<std::string, Core::Statistics<f32>*>::const_iterator it = customStatistics_.begin(); it != customStatistics_.end(); ++it)
        (*it).second->clear();

    for (std::map<std::string, Core::HistogramStatistics*>::const_iterator it = customHistogramStatistics_.begin(); it != customHistogramStatistics_.end(); ++it)
        (*it).second->clear();
}

Core::Statistics<f32>& Search::SearchSpaceStatistics::customStatistics(const std::string& name) {
    std::map<std::string, Core::Statistics<f32>*>::iterator it = customStatistics_.find(name);
    if (it == customStatistics_.end())
        it = customStatistics_.insert(std::make_pair(name, new Core::Statistics<f32>(name.c_str()))).first;

    return *(*it).second;
}

Core::HistogramStatistics& Search::SearchSpaceStatistics::customHistogramStatistics(const std::string& name, u32 buckets) {
    std::map<std::string, Core::HistogramStatistics*>::iterator it = customHistogramStatistics_.find(name);
    if (it == customHistogramStatistics_.end())
        it = customHistogramStatistics_.insert(std::make_pair(name, new Core::HistogramStatistics(name.c_str(), buckets))).first;

    return *(*it).second;
}

Search::SearchSpaceStatistics::~SearchSpaceStatistics() {
    for (std::map<std::string, Core::Statistics<f32>*>::const_iterator it = customStatistics_.begin(); it != customStatistics_.end(); ++it)
        delete (*it).second;

    for (std::map<std::string, Core::HistogramStatistics*>::const_iterator it = customHistogramStatistics_.begin(); it != customHistogramStatistics_.end(); ++it)
        delete (*it).second;

    customHistogramStatistics_.clear();
}
