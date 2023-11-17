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
 *
 *  author: Wei Zhou
 */

#ifndef SEQ2SEQ_SEARCH_SPACE_STATISTICS
#define SEQ2SEQ_SEARCH_SPACE_STATISTICS

#include <Core/Statistics.hh>
#include <map>

namespace Seq2SeqTreeSearch {

// min|max|avg search space statistics for better pruning settings (all inline)
// TODO maybe add harded coded statistics to speed up ?
class SearchSpaceStatistics {
  public:
    SearchSpaceStatistics() {}
    ~SearchSpaceStatistics() {} 

    // easily add statistics by name key (not performance optimized)
    Core::Statistics<f32>& customStatistics(const std::string& name) {
      StatisticsMap::iterator it = customStatistics_.find(name);
      if (it == customStatistics_.end())
        it = customStatistics_.insert(std::make_pair(name, Core::Statistics<f32>(name.c_str()))).first;
      return it->second;
    }

    void write(Core::XmlWriter& os) const {
      os << Core::XmlOpen("search-space-statistics");
      for(StatisticsMap::const_iterator it = customStatistics_.begin(); it != customStatistics_.end(); ++it)
        os << it->second;
      os << Core::XmlClose("search-space-statistics");
    }

    void clear() {
      for(StatisticsMap::iterator it = customStatistics_.begin(); it != customStatistics_.end(); ++it)
        it->second.clear();
    }

  private:
    typedef std::map<std::string, Core::Statistics<f32>> StatisticsMap;
    StatisticsMap customStatistics_;
};

}
#endif
