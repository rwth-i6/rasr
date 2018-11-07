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
#ifndef CONDITIONEDTREESEARCHTRACE_HH
#define CONDITIONEDTREESEARCHTRACE_HH

#include <Core/ReferenceCounting.hh>
#include <Search/StateTree.hh>
#include <Search/Types.hh>
#include <Search/Search.hh>
#include "PathTrace.hh"

namespace Search
{
class Trace :
  public Core::ReferenceCounted,
  public SearchAlgorithm::TracebackItem
{
public:
  Core::Ref<Trace> predecessor, sibling;

  Trace( const Core::Ref<Trace> &pre, const Bliss::LemmaPronunciation *p,
    TimeframeIndex t, SearchAlgorithm::ScoreVector s, const Transit &transit );
  Trace( TimeframeIndex t, SearchAlgorithm::ScoreVector s, const Transit &transit );

  void write( std::ostream &os, Core::Ref<const Bliss::PhonemeInventory> phi ) const;

  void getLemmaSequence( std::vector<Bliss::Lemma*>& lemmaSequence ) const;

  u32 wordCount() const {
    u32 count = 0;
    if( pronunciation )
      ++count;
    if( predecessor )
      count += predecessor->wordCount();

    return count;
  }

  PathTrace pathTrace;
};
}

#endif // CONDITIONEDTREESEARCHTRACE_HH
