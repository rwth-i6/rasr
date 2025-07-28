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
#ifndef SEARCH_SPACE_HELPERS
#define SEARCH_SPACE_HELPERS

#include <Core/Types.hh>
#include <Search/LanguageModelLookahead.hh>
#include <Search/TreeStructure.hh>

#include "TraceManager.hh"

namespace Search {
typedef s32 StateHypothesisIndex;

/**
 * Object that contains the information that _conditions_ a network, and thereby makes the network unique.
 * In TCS this is mainly the start timeframe, while in WCS it is the word history.
 */
struct InstanceKey {
    // Only valid for time conditioned trees
    TimeframeIndex                time;
    StateId                       transitNode;
    Bliss::LemmaPronunciation::Id conditionPronunciation;

    // Only valid for word conditioned trees
    Lm::History history;

    InstanceKey()
            : time(InvalidTimeframeIndex),
              transitNode(invalidTreeNodeIndex),
              conditionPronunciation(Bliss::LemmaPronunciation::invalidId) {
    }

    InstanceKey(TimeframeIndex _time, Bliss::LemmaPronunciation::Id _conditionPronunciation, StateId _transitNode)
            : time(_time),
              transitNode(_transitNode),
              conditionPronunciation(_conditionPronunciation) {
    }

    InstanceKey(const Lm::History& _history, Bliss::LemmaPronunciation::Id conditionPron = Bliss::LemmaPronunciation::invalidId)
            : time(InvalidTimeframeIndex),
              transitNode(invalidTreeNodeIndex),
              conditionPronunciation(conditionPron),
              history(_history) {
    }

    bool isTimeKey() const {
        return time != InvalidTimeframeIndex;
    }

    bool operator==(const InstanceKey& rhs) const {
        return time == rhs.time &&
               transitNode == rhs.transitNode &&
               conditionPronunciation == rhs.conditionPronunciation &&
               history == rhs.history;
    }

    struct Hash {
        size_t operator()(const InstanceKey& s) const {
            return s.time * 17 +
                   s.conditionPronunciation * 31823 +
                   s.transitNode * 31201 +
                   (s.history.isValid() ? s.history.hashKey() : 0);
        }
    };
};

/**
 * A state hypothesis consists of the probability of a network state within one specific network copy,
 * together with its back-pointer (the trace).
 */
struct StateHypothesis {
    TraceId   trace;
    Score     score;
    Score     prospect;
    StateId   state;  // used to be node
    PathTrace pathTrace;

    StateHypothesis(Search::StateId _node, Search::TraceId _trace, Search::Score _score)
            : trace(_trace),
              score(_score),
              prospect(_score),
              state(_node) {
    }

    struct ProspectCompare {
        inline bool operator()(const StateHypothesis& lhs, const StateHypothesis& rhs) const {
            return lhs.prospect < rhs.prospect;
        }
    };
};

struct WordEndHypothesis;
struct EarlyWordEndHypothesis;

typedef std::vector<WordEndHypothesis>      WordEndHypothesisList;
typedef std::vector<EarlyWordEndHypothesis> EarlyWordEndHypothesisList;

struct Instance {
    Instance(const InstanceKey& _key, Instance* backOffParent)
            : key(_key),
              inactive(0),
              backOffInstance(0),
              backOffScore(0),
              backOffParent(backOffParent),
              totalBackOffOffset(0) {
        if (backOffParent) {
            verify(backOffParent->backOffInstance == 0);
            backOffParent->backOffInstance = this;
        }
    }

    virtual ~Instance();

    const InstanceKey                                 key;
    LanguageModelLookahead::ContextLookaheadReference lookahead;

    // List of state hypotheses that should be transferred into this network
    std::vector<StateHypothesisIndex> transfer;

    /// number of time-frames this instance has been inactive
    u32 inactive;

    virtual bool mayDeactivate() {
        return true;
    }

    struct StateRange {  // actually, StateHypothesisRange.
        StateRange()
                : begin(0), end(0) {
        }
        StateHypothesisIndex begin, end;

        inline bool empty() const {
            return begin == end;
        }
        inline void clear() {
            begin = end = 0;
        }
        inline u32 size() const {
            return end - begin;
        }
        inline bool contains(StateHypothesisIndex idx) {
            return (idx >= begin && idx < end);
        }
    } states;

    std::vector<StateHypothesis> rootStateHypotheses;

    /// Enter this tree with the given trace, entry-node and score
    void enter(TraceManager& trace_manager, Core::Ref<Trace> trace, StateId entryNode, Score score);

    /// Enter this tree with the given StateHypothesis whose trace can have longer histories than tree's
    void enterWithState(const StateHypothesis& st);

    /// Alternative history that should be used for look-ahead
    Lm::History lookaheadHistory;

    /// History used for scoring
    Lm::History scoreHistory;

    /// Returns the total number of states in this back-off chain (eg. in this tree, its back-off parents, and its back-off trees)
    u32 backOffChainStates() const;

    /// Adds the LM score to the early word end hypothesis, using the tree-wise cache
    void addLmScore(EarlyWordEndHypothesis&                         hyp,
                    Bliss::LemmaPronunciation::Id                   pron,
                    const Core::Ref<const Lm::ScaledLanguageModel>& lm,
                    const Bliss::LexiconRef&                        lexicon,
                    Score                                           wpScale) const;

    /// Adds the LM score to the word end hypothesis, using the tree-wise cache
    void addLmScore(WordEndHypothesis&                              hyp,
                    Bliss::LemmaPronunciation::Id                   pron,
                    const Core::Ref<const Lm::ScaledLanguageModel>& lm,
                    const Bliss::LexiconRef&                        lexicon,
                    Score                                           wpScale) const;

    /// Back-off tree of this tree
    Instance* backOffInstance;
    Score     backOffScore;

    /// The tree this one is a back-off tree of
    Instance* backOffParent;

    /// Total back-off offset of the scores within this tree, relative to all backoff parents combined.
    Score totalBackOffOffset;

    /// LM-Cache caching the LM scores in the context of this tree for more efficient access
    typedef std::unordered_map<Bliss::LemmaPronunciation::Id, Score> SimpleLMCache;
    mutable SimpleLMCache                                            lmCache;
};

struct EarlyWordEndHypothesis {
    TraceId     trace;
    ScoreVector score;
    u32         exit;
    PathTrace   pathTrace;

    EarlyWordEndHypothesis(TraceId _trace, const ScoreVector& _score, u32 _exit, PathTrace _pathTrace)
            : trace(_trace), score(_score), exit(_exit), pathTrace(_pathTrace) {
    }
    EarlyWordEndHypothesis()
            : trace(0), score(0, 0), exit(0) {
    }
};

struct WordEndHypothesis {
    Lm::History                      recombinationHistory;
    Lm::History                      lookaheadHistory;
    Lm::History                      scoreHistory;
    StateId                          transitState;
    const Bliss::LemmaPronunciation* pronunciation;
    ScoreVector                      score;
    Core::Ref<Trace>                 trace;
    u32                              endExit;  // Exit from which this word end hypothesis was constructed
    PathTrace                        pathTrace;

    WordEndHypothesis(const Lm::History& rch, const Lm::History& lah, const Lm::History& sch, StateId e,
                      const Bliss::LemmaPronunciation* p, ScoreVector s,
                      const Core::Ref<Trace>& t, u32 _endExit, PathTrace _pathTrace)
            : recombinationHistory(rch),
              lookaheadHistory(lah),
              scoreHistory(sch),
              transitState(e),
              pronunciation(p),
              score(s),
              trace(t),
              endExit(_endExit),
              pathTrace(_pathTrace) {
    }

    ~WordEndHypothesis() {
    }

    WordEndHypothesis(const WordEndHypothesis& rhs)
            : recombinationHistory(rhs.recombinationHistory),
              lookaheadHistory(rhs.lookaheadHistory),
              scoreHistory(rhs.scoreHistory),
              transitState(rhs.transitState),
              pronunciation(rhs.pronunciation),
              score(rhs.score),
              trace(rhs.trace),
              endExit(rhs.endExit),
              pathTrace(rhs.pathTrace) {
    }

    inline u32 hashKey() const {
        u32 hash = recombinationHistory.hashKey();
        hash     = (hash << 5 | hash >> 27) ^ transitState;
        return hash;
    }

    struct ProbabilityCompare {
        inline bool operator()(const WordEndHypothesis& lhs, const WordEndHypothesis& rhs) const {
            return lhs.score < rhs.score;
        }
    };

    struct Hash {
        inline size_t operator()(const WordEndHypothesis* weh) const {
            return weh->hashKey();
        }

        inline size_t operator()(const WordEndHypothesisList::iterator weh) const {
            return weh->hashKey();
        }
    };

    static int meshHistoryPhones;  // TODO: Find a better way than static (needed in MeshHash and MeshEquality)

    struct MeshHash {
        inline size_t operator()(const WordEndHypothesisList::iterator weh) const {
            size_t ret = MyStandardValueHash<u32>()(weh->transitState);

            if (meshHistoryPhones == 0)
                return ret;
            else if (meshHistoryPhones < 0)
                return MyStandardValueHash<u32>()(ret + weh->pronunciation->id());

            int len        = weh->pronunciation->pronunciation()->length();
            int compareLen = std::min(len, meshHistoryPhones);
            for (int a = 0; a < compareLen; ++a)
                ret = MyStandardValueHash<u32>()(ret + weh->pronunciation->pronunciation()->phonemes()[len - compareLen + a]);
            return ret;
        }
    };
    struct MeshEquality {
        inline bool operator()(const WordEndHypothesisList::iterator l,
                               const WordEndHypothesisList::iterator r) const {
            if (l->transitState != r->transitState)
                return false;
            if (meshHistoryPhones == 0)
                return true;
            if (l->pronunciation == r->pronunciation)
                return true;
            if (meshHistoryPhones < 0)
                return false;
            int len1 = l->pronunciation->pronunciation()->length();
            int len2 = r->pronunciation->pronunciation()->length();

            if ((len1 < meshHistoryPhones || len2 < meshHistoryPhones) && len1 != len2)
                return false;

            int compareLen = std::min(len1, meshHistoryPhones);

            return memcmp(l->pronunciation->pronunciation()->phonemes() + len1 - compareLen,
                          r->pronunciation->pronunciation()->phonemes() + len2 - compareLen,
                          sizeof(Bliss::Phoneme::Id) * compareLen) == 0;
        }
    };
    struct Equality {
        inline bool operator()(const WordEndHypothesis* l, const WordEndHypothesis* r) const {
            return (l->recombinationHistory == r->recombinationHistory) && (l->transitState == r->transitState);
        }

        inline bool operator()(const WordEndHypothesisList::iterator l,
                               const WordEndHypothesisList::iterator r) const {
            return (l->recombinationHistory == r->recombinationHistory) && (l->transitState == r->transitState);
        }
    };
};

typedef std::pair<Lm::History, StateId> ReducedContextRecombinationKey;

// same as for WordEndHypothesis
struct HistoryStateHash {
    inline size_t operator()(ReducedContextRecombinationKey const& k) const {
        u32 hash = k.first.hashKey();
        hash     = (hash << 5 | hash >> 27) ^ k.second;
        return hash;
    }
};

typedef std::unordered_map<ReducedContextRecombinationKey, WordEndHypothesisList::iterator, HistoryStateHash> ReducedContextRecombinationMap;
}  // namespace Search

#endif
