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
#ifndef FILTER_HH
#define FILTER_HH
#include <set>
#include <vector>
#include "PersistentStateTree.hh"

namespace Core
{
class Configuration;
}

namespace Search
{
class PersistentStateTree;
class SearchSpace;
struct StateHypothesis;

class PrefixFilter
{
public:
  // Must be initialized before the outputs are removed from the network
  PrefixFilter( const PersistentStateTree& tree, Bliss::LexiconRef lexicon, Core::Configuration config );
  bool prune( const StateHypothesis& hyp ) const;
  bool haveFilter() const {
    return prefixSequence_.size();
  }
private:
  const PersistentStateTree& tree_;
  Bliss::LexiconRef lexicon_;
  std::vector<Bliss::Lemma*> prefixSequence_;
  std::set<const Bliss::Lemma*> nonWordLemmas_;
  std::vector<int> reachability_;
  std::set<StateId> nonWordLemmaNodes_;
  std::vector<std::set<StateId> > prefixReachability_;

  void setPrefixWords( std::string prefixWords );
  void prepareReachability();
  bool reachable( StateId state, const Bliss::Lemma* lemma );
};
}

#endif
