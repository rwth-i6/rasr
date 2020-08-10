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
// $Id$

#ifndef _SEARCH_LANGUAGEMODELLOOKAHEAD_HH
#define _SEARCH_LANGUAGEMODELLOOKAHEAD_HH

#include <iostream>
#include <list>

#include <Core/Component.hh>
#include <Core/Hash.hh>
#include <Core/Parameter.hh>
#include <Core/ReferenceCounting.hh>
#include <Lm/ScaledLanguageModel.hh>
#include <Search/StateTree.hh>

#include "LinearPrediction.hh"
#include "PersistentStateTree.hh"
#include "TreeStructure.hh"

// #define EXTENSIVE_SPARSE_COLLISION_STATS

#ifdef EXTENSIVE_SPARSE_COLLISION_STATS
template<class T>
struct PairHash {
    std::size_t operator()(const std::pair<T, T>& item) {
        return item.first * 31231 + item.first / 4182 + item.second + item.second / 30;
    }
};

// Maps from pairs <needed, encountered> to the number of times this happened
extern std::unordered_map<std::pair<u32, u32>, u32, PairHash<u32>> sparseCollisionHash;
// Maps from node-number to a pair <accumulator, count>
extern std::unordered_map<u32, std::pair<u32, u32>> sparseSkipHash;

#define ifSparseCollisionStats(X) X
#else
#define ifSparseCollisionStats(X)
#endif

#include <Core/MappedArchive.hh>
#include "ApproxLinearMiniHash.hh"
#include "LinearMiniHash.hh"

// We don't use Core::Type<f32>::max in critical places,
// so that the compiler can glue the value right into the code without help from the linker
#define F32_MAX +3.40282347e+38F

namespace AdvancedTreeSearch {
/** Language model look-ahead */

class LanguageModelLookahead : public Core::Component {
public:
    typedef u32 LookaheadId;
    typedef f32 Score;

private:
    static const LookaheadId                 invalidId;
    s32                                      historyLimit_;
    s32                                      cutoffDepth_;
    u32                                      minimumRepresentation_;
    Lm::Score                                wpScale_;
    u32                                      maxDepth_;
    Core::Ref<const Lm::ScaledLanguageModel> lm_;
    Search::HMMStateNetwork const&           tree_;

    // Predicts the number of sparse look-ahead nodes based on the number of context-scores
    mutable Search::LinearPrediction sparseNodesPrediction_;

    class ConstructionNode;
    class ConstructionTree;
    friend class LanguageModelLookahead::ConstructionTree;

    struct Node;
    typedef std::vector<const Bliss::LemmaPronunciation*> Ends;
    Ends                                                  ends_;
    Core::ConstantVector<Score>                           endOffsets_;
    typedef Core::ConstantVector<LookaheadId>             Successors;
    Successors                                            successors_;
    Successors                                            parents_;
    Core::ConstantVector<Node>                            nodes_;

    // Maps the specific ending-node and a score-offset to each Bliss::Token
    Core::ConstantVector<std::pair<LookaheadId, Score>> nodeForToken_;
    Core::ConstantVector<u32>                           firstNodeForToken_;
    int                                                 invalidFirstNodeForTokenIndex_;

    LookaheadId nEntries_;

    Core::ConstantVector<LookaheadId> nodeId_;        // StateTree::StateId -> states_ index
    Core::ConstantVector<u32>         hashForState_;  // StateTree::StateId -> unique hash index
    Core::ConstantVector<u32>         hashForNode_;

    bool shouldPruneConstructionNode(const ConstructionNode& sn) const;
    void buildCompressesLookaheadStructure(u32 nodeStart, u32 numNodes, ConstructionTree const&);
    void buildBatchRequest();
    void buildHash();
    void buildLookaheadStructure(Search::HMMStateNetwork const& tree, Search::StateId rootNode, std::vector<Search::PersistentStateTree::Exit> const& exits);

    const Lm::CompiledBatchRequest* batchRequest_;
    void                            computeScores(Lm::History const&, std::vector<Score>&) const;

public:
    class ContextLookahead;

private:
    // Returns whether the scores were actually computed
    template<bool approx>
    bool computeScoresSparse(ContextLookahead& lookahead) const;
    friend class ContextLookahead;
    u32                                                                           cacheSizeHighMark_, cacheSizeLowMark_;
    typedef std::list<ContextLookahead*>                                          List;
    mutable List                                                                  tables_, freeTables_;
    mutable u32                                                                   nTables_, nFreeTables_;
    typedef std::unordered_map<Lm::History, ContextLookahead*, Lm::History::Hash> Map;
    mutable Map                                                                   map_;

    ContextLookahead* acquireTable(Lm::History const&) const;
    ContextLookahead* getCachedTable(Lm::History const&) const;
    void              releaseTable(ContextLookahead const*) const;

    struct CacheStatistics;
    CacheStatistics*                   cacheStatistics_;
    mutable Core::XmlChannel           statisticsChannel_;
    bool                               considerBackOffInMaximization_;
    bool                               considerPronunciationScore_;
    bool                               considerExitPenalty_;
    bool                               sparseThresholdExpectationBased_;
    float                              logSemiringFactor_;
    f64                                sparseLookAheadThreshold_;
    f32                                sparseHashSizeFactor_;
    u32                                sparseHashResizeAtFillFraction_;
    Core::Ref<const Am::AcousticModel> acousticModel_;

public:
    static const Core::ParameterInt    paramHistoryLimit;
    static const Core::ParameterInt    paramTreeCutoff;
    static const Core::ParameterInt    paramMinimumRepresentation;
    static const Core::ParameterInt    paramCacheSizeLow, paramCacheSizeHigh;
    static const Core::ParameterBool   paramConsiderBackOffInMaximization;
    static const Core::ParameterBool   paramSparseThresholdExpectationBased;
    static const Core::ParameterBool   paramConsiderPronunciationScore;
    static const Core::ParameterBool   paramConsiderExitPenalty;
    static const Core::ParameterFloat  paramSparseLookAheadThreshold;
    static const Core::ParameterFloat  paramSparseHashSizeFactor;
    static const Core::ParameterFloat  paramSparseHashResizeAtFill;
    static const Core::ParameterFloat  paramLmLookaheadScale;
    static const Core::ParameterString paramObservationStats;
    static const Core::ParameterInt    paramCollisionHashSize;
    static const Core::ParameterFloat  paramMaxCollisionDeviation;
    static const Core::ParameterString paramCacheArchive;

    LanguageModelLookahead(Core::Configuration const&,
                           Lm::Score wpScale,
                           Core::Ref<const Lm::ScaledLanguageModel>,
                           Search::HMMStateNetwork const&                        tree,
                           Search::StateId                                       rootNode,
                           std::vector<Search::PersistentStateTree::Exit> const& exits,
                           Core::Ref<const Am::AcousticModel>);

    ~LanguageModelLookahead();

    void draw(std::ostream&) const;

    LookaheadId lookaheadId(Search::StateTree::StateId s) const {
        require_(0 <= s && s < Search::StateTree::StateId(nodeId_.size()));
        LookaheadId result = nodeId_[s];
        ensure_(result < nEntries_);
        return result;
    };

    uint lookaheadHash(Search::StateTree::StateId s) const {
        return hashForState_[s];
    };

    class ContextLookahead : public Core::ReferenceCounted {
    private:
        const LanguageModelLookahead* la_;
        Lm::History                   history_;
        List::iterator                pos_, freePos_;
        std::vector<Score>            scores_;  //If this is empty, the look-ahead is sparse
        bool                          isFilled_;

        Search::LinearMiniHash<LookaheadId, (LanguageModelLookahead::LookaheadId)-1, Score>                             sparseScores_;
        typedef Search::ApproxLinearMiniHash<LookaheadId, (LanguageModelLookahead::LookaheadId)-1, Score, false, false> ApproxHash;
        ApproxHash                                                                                                      approxSparseScores_;
        Score                                                                                                           backOffScore_;

        friend class Core::Ref<const ContextLookahead>;
        void free() const {
            la_->releaseTable(this);
        }

    protected:
        friend class LanguageModelLookahead;
        ContextLookahead(const LanguageModelLookahead*,
                         const Lm::History&);
        bool isActive() const {
            return freePos_ == la_->freeTables_.end();
        }

    public:
        const Lm::History& history() const {
            return history_;
        }

        // Only nonzero if this lookahead is sparse
        inline Score backOffScore() const {
            return backOffScore_;
        }

        inline bool getScoreForLookAheadHashSparse(u32 hash, Score& target) const {
            return sparseScores_.get(hash, target);
        }

        inline bool getScoreForLookAheadHashSparseApprox(u32 hash, Score& target) const {
            return approxSparseScores_.get(hash, target);
        }

        inline Score scoreForLookAheadIdNormal(LookaheadId id) const {
            return scores_[id];
        }

        inline bool isSparse() const {
            return scores_.empty();
        }

        // DEBUG_AREA
        bool checkScores() {
            int nScores = scores_.size();
            int nAbnorm = 0;
            for (std::vector<Score>::iterator i = scores_.begin(); i != scores_.end(); i++) {
                if ((*i > +1.0e+20F) || (*i < -1.0e+20F))
                    nAbnorm++;
            }

            std::cout << "checkScores: abnormal scores:" << nAbnorm << "/"
                      << nScores << std::endl;

            return (nAbnorm == 0);
        }
        //END_DEBUG
    };

public:
    typedef Core::Ref<const ContextLookahead> ContextLookaheadReference;

private:
    class LookAheadNodesForDepth {
        enum {
            MinimumReservedArraySize = 1000
        };

    public:
        LookAheadNodesForDepth()
                : size_(0) {
            nodes_.reserve(MinimumReservedArraySize);
        }

        inline void push_back(const std::pair<LookaheadId, Score>& node) {
            if (size_ == nodes_.size())
                nodes_.resize(size_ + 100);
            nodes_[size_] = node;
            ++size_;
        }

        inline void clear() {
            size_ = 0;
        }

        inline u32 size() const {
            return size_;
        }

        void shrink(u32 size) {
            size_ = size;
        }

        inline std::pair<LookaheadId, Score>& operator[](u32 p) {
            return nodes_[p];
        }

    private:
        u32                                        size_;
        std::vector<std::pair<LookaheadId, Score>> nodes_;
    };

    mutable std::vector<LookAheadNodesForDepth> waitingLookaheadNodesByDepth_;
    mutable std::vector<u32>                    nodeRecombination_;

    bool accumulateNodeObservations_;
    f32  scale_;

    Score getLmScale() const;

    std::string archiveEntry() const;

    u32 nTables() const {
        verify_(nTables_ == tables_.size());
        return nTables_;
    }
    u32 nActiveTables() const {
        verify_(nTables_ == tables_.size());
        verify_(nFreeTables_ == freeTables_.size());
        return nTables_ - nFreeTables_;
    }

    void buildDepths();
    template<class Hash>
    void assignHashes(std::string hashName, Hash& hash, u32 testHashSize);
    void propagateDepth(int node, int depth);
    bool readPersistentCache();
    void writePersistentCache();

public:
    int nodeDepth(LookaheadId node) const;

    Lm::History getReducedHistory(const Lm::History& history) const;
    /**
     * Returns the LM look-ahead table for the given history.
     * The table may be unfinished, fill(..) must be called to fill it.
     * */
    ContextLookaheadReference getLookahead(const Lm::History&, bool noHistoryLimit = false) const;
    /**
     * Returns the LM look-ahead table for the given history.
     * The table is always filled.
     * */
    ContextLookaheadReference tryToGetLookahead(const Lm::History&, bool noHistoryLimit = false) const;

    /**
     * Fills the LM look-ahead table. This must be called before tables retrieved through getLookahead can be used.
     * */
    void fill(ContextLookaheadReference lah, bool sparse = false, bool approx = false);

    /**
     * Fills the LM look-ahead table with zeroes (non-sparse).
     * */
    void fillZero(ContextLookaheadReference lah);

    LookaheadId lastNodeOnDepth(int depth) const;

    u32 numNodes() const {
        return nEntries_;
    }

    void collectStatistics() const;
    void logStatistics() const;

    /// returns true if the given look-ahead node leads to exactly one word-end
    inline bool isSingleWordNode(LookaheadId node) const;
};

struct LanguageModelLookahead::Node {
    u32 firstEnd, firstSuccessor, firstParent, depth;
};

bool LanguageModelLookahead::isSingleWordNode(LanguageModelLookahead::LookaheadId node) const {
    const LanguageModelLookahead::Node& n(nodes_[node]);
    const LanguageModelLookahead::Node& next(nodes_[node + 1]);
    return (next.firstEnd - n.firstEnd == 1) && (next.firstSuccessor == n.firstSuccessor);
}
}  // namespace AdvancedTreeSearch

#endif  //_SEARCH_LANGUAGEMODELLOOKAHEAD_HH
