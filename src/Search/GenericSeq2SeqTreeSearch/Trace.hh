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

#ifndef SEQ2SEQ_TRACE_HH
#define SEQ2SEQ_TRACE_HH

#include <Search/Search.hh>

namespace Seq2SeqTreeSearch {

class Trace : public Core::ReferenceCounted,
              public Search::SearchAlgorithm::TracebackItem {
public:
  Core::Ref<Trace> predecessor, sibling;

  // Note: assign only for end traces (otherwise memory explosion)
  Lm::History recombinationHistory;
  Lm::History scoreHistory; // only for fallback trace
  Nn::LabelHistory labelHistory;

  u32 nLabels;
  u32 nWords;
  // only for ending traces (pruning and decision making)
  Search::Score prospect; 

  Trace(Search::Index n, Search::SearchAlgorithm::ScoreVector s) : 
      TracebackItem(nullptr, nullptr, n, s, 0), nLabels(0), nWords(0), prospect(0) {}

  Trace(const Core::Ref<Trace> &pre, const Bliss::LemmaPronunciation *p, const Bliss::Lemma* l,
        Search::Index n, Search::SearchAlgorithm::ScoreVector s,
        u32 nlabels, u32 nwords, u32 pos) :
      TracebackItem(p, l, n, s, pos), predecessor(pre), nLabels(nlabels), nWords(nwords), prospect(0)
  {}
};

} // namespace

#endif
