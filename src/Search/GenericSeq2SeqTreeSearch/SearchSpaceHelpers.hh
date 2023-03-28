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

#ifndef SEQ2SEQ_SEARCH_SPACE_HELPERS_HH
#define SEQ2SEQ_SEARCH_SPACE_HELPERS_HH

#include <Core/Types.hh>
#include "LabelTree.hh"
#include "LanguageModelLookahead.hh"
#include "Trace.hh"

using namespace Search;

// label-history-dependent label hypothesis
struct LabelHypothesis {
  NodeId treeNodeId;

  SearchAlgorithm::ScoreVector score;
  Score prospect;

  u32 traceId;
  Nn::LabelHistory labelHistory; // for label scorer scoring

  // length information
  u32 nLabels, nWords;

  // additional position infomation (index of encodings)
  u32 position;

  bool isBlank;
  bool isLoop; 
  u32 nLoop; // loop occurance (for min_duration)

  size_t hash; // sequence dependency for recombination

  LabelHypothesis(NodeId tnId, SearchAlgorithm::ScoreVector s, u32 tId,
                  const Nn::LabelHistory& lh, u32 nL, u32 nW, u32 pos) :
      treeNodeId(tnId), score(s), traceId(tId), labelHistory(lh), nLabels(nL), nWords(nW), 
      position(pos), isBlank(false), isLoop(false), nLoop(0), hash(0) {}
};

typedef std::vector<LabelHypothesis> LabelHypothesesList;
// explicit separation of tree NodeId and other hash: less colision
typedef std::unordered_map<size_t, LabelHypothesesList::iterator> LabelHashMap;
typedef std::unordered_map<NodeId, LabelHashMap> LabelHypothesesMap;


// light-weighted WordEndHypothesis used before pruning
// Note: broad-level word -> actually transcription output unit (orthography)
struct EarlyWordEndHypothesis {
  Nn::LabelHistory labelHistory;

  NodeId treeNodeId; // last labelTree node of this word
  u32 traceId;
  u32 exitId;
  bool isLoop; // needed for delayed label history extension

  SearchAlgorithm::ScoreVector score;
  Score prospect; // score for pruning

  u32 nLabels, nWords;
  u32 position;

  EarlyWordEndHypothesis(const Nn::LabelHistory& lh, u32 tnId, u32 tId, u32 eId, bool loop,
                         SearchAlgorithm::ScoreVector s, u32 nL, u32 nW, u32 pos) :
      labelHistory(lh), treeNodeId(tnId), traceId(tId), exitId(eId), isLoop(loop),
      score(s), prospect(0), nLabels(nL), nWords(nW), position(pos) {}
};

typedef std::vector<EarlyWordEndHypothesis> EarlyWordEndHypothesesList;


typedef Core::Ref<Seq2SeqTreeSearch::Trace> TraceRef;
typedef std::vector<TraceRef> TraceList;
typedef std::unordered_map<size_t, TraceRef> HistoryTraceMap;

// LM-history-dependent word hypothesis
// Note: broad-level word -> actually transcription output unit (orthography)
struct WordEndHypothesis {
  Nn::LabelHistory labelHistory; 

  TraceRef trace;

  Lm::History recombinationHistory;
  Lm::History scoreHistory;
  Lm::History lookaheadHistory;

  SearchAlgorithm::ScoreVector score;
  Score prospect; // score for pruning

  NodeId treeNodeId; // last labelTree node of this word
  u32 exitId;

  // length information
  u32 nLabels, nWords;
  u32 position;

  WordEndHypothesis(const Nn::LabelHistory& lbh, const TraceRef& t,
                    const Lm::History& rch, const Lm::History& sch, const Lm::History& lah,
                    SearchAlgorithm::ScoreVector s, Score p, NodeId tnId, u32 exit, 
                    u32 nL, u32 nW, u32 pos) :
      labelHistory(lbh), trace(t),
      recombinationHistory(rch), scoreHistory(sch), lookaheadHistory(lah),
      score(s), prospect(p), treeNodeId(tnId), exitId(exit),
      nLabels(nL), nWords(nW), position(pos) {}
};

typedef std::vector<WordEndHypothesis> WordEndHypothesesList;
typedef std::unordered_map<size_t, WordEndHypothesesList::iterator> WordEndLabelMap;
typedef std::unordered_map<size_t, WordEndLabelMap> WordEndHypothesesMap;


// key for tree instance (so far LM-history only)
struct TreeInstanceKey {
  Lm::History history;
  TreeInstanceKey(const Lm::History& h) : history(h) {}
  bool operator==(const TreeInstanceKey& rhs) const { return history == rhs.history; }

  struct Hash { 
    inline size_t operator()(const TreeInstanceKey& s) const { 
      return s.history.isValid() ? s.history.hashKey() : 0; 
    }
  };
};

typedef Seq2SeqTreeSearch::LanguageModelLookahead::ContextLookaheadReference LmLookahead;

// LM-history-dependent tree wraper
struct TreeInstance {
  u32 inactive;
  const TreeInstanceKey key;

  Lm::History scoreHistory;
  Lm::History lookaheadHistory;
  LmLookahead lookahead;

  // best non-end label local score at current step
  // per tree based (not each beam anymore)
  Score bestNonEndLocal;

  // entryTraces (no global TraceManager)
  TraceList entryTraces;
  // entryLabels
  LabelHypothesesList entryLabelHypotheses;

  // nWords of entered traces (possible length-based pruning)
  std::unordered_set<u32> entryNWords;
 
  // LabelHyp range
  struct LabelRange { 
    u32 begin, end;
    LabelRange() : begin(0), end(0) {}
    inline bool empty() const { return begin == end; }
    inline void clear() { begin = end = 0; }
    inline u32 size() const { return end - begin; }
    inline bool contains(u32 idx) { return (idx >= begin && idx < end); }
  } labels;

  // EarlyWordEndHyp range
  u32 earlyWehBegin, earlyWehEnd;
  bool earlyWehContains(u32 idx) { return (idx >= earlyWehBegin && idx < earlyWehEnd); }

  TreeInstance(const TreeInstanceKey& k, const Lm::History& sch, const Lm::History& lah) :
      inactive(0), key(k), scoreHistory(sch), lookaheadHistory(lah),
      bestNonEndLocal(Core::Type<Score>::max) {
    earlyWehBegin = 0;
    earlyWehEnd = 0;
  }

  ~TreeInstance() {
    entryTraces.clear();
    lmCache.clear();
  }

  void enter(NodeId tnId, const WordEndHypothesis& weh) {
    u32 traceId = entryTraces.size();
    entryTraces.push_back(weh.trace);
    entryLabelHypotheses.emplace_back(tnId, weh.score, traceId, weh.labelHistory, weh.nLabels, weh.nWords, weh.position);
    entryNWords.insert(weh.nWords);
  }

  // general lm scores handling 
  // cache the LM scores in the context of this tree for more efficient access
  // also work for NNLM, but mainly for backoffLM since score is not stored anywhere
  typedef std::unordered_map<const Bliss::Lemma*, Score> SimpleLMCache;
  mutable SimpleLMCache lmCache;

  // get and cache lm score without pronunciation sore
  Score getLmScore(const Core::Ref<const Lm::ScaledLanguageModel>& lm, const Bliss::Lemma* lemma);
};

typedef std::vector<TreeInstance*> InstanceList;
typedef std::unordered_map<TreeInstanceKey, TreeInstance*, TreeInstanceKey::Hash> KeyInstanceMap;


inline Score TreeInstance::getLmScore(const Core::Ref<const Lm::ScaledLanguageModel>& lm, const Bliss::Lemma* lemma) {
  if (!lemma)
    return 0;
  SimpleLMCache::const_iterator iter = lmCache.find(lemma);
  if (iter == lmCache.end()) {
    Score lmScore = 0;
    Lm::addLemmaScoreOmitExtension(lm, lemma, lm->scale(), scoreHistory, lmScore);
    iter = lmCache.insert(std::make_pair(lemma, lmScore)).first;
  }
  return iter->second;
}

#endif
