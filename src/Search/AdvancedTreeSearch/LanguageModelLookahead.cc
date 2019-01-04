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
// $Id$

#include <unordered_map>
#include <unordered_set>

#include <Core/Debug.hh>
#include <Core/MappedArchive.hh>
#include <Core/Statistics.hh>
#include <Core/Utility.hh>
#include <Lm/BackingOff.hh>
#include <Search/Types.hh>
#include "LanguageModelLookahead.hh"
#include "Helpers.hh"

using namespace AdvancedTreeSearch;
using namespace Search;
using Core::tie;

#ifdef EXTENSIVE_SPARSE_COLLISION_STATS
std::unordered_map<std::pair<u32, u32>, u32, PairHash<u32> > sparseCollisionHash;
std::unordered_map<u32, std::pair<u32, u32> > sparseSkipHash;
std::unordered_map<u32, Score> unigramScores;

static struct EvaluateSparseStats {
  ~EvaluateSparseStats() {
    std::multimap<u32, u32> totalSkips;
    std::multimap<u32, u32> averageSkips;
    std::multimap<u32, std::pair<u32, u32> > totalCollisions;

    u64 totalTotalCollisions = 0;
    u64 totalAverageSkips = 0;
    u64 totalAverageCount = 0;

    for( std::unordered_map<u32, std::pair<u32, u32> >::const_iterator it = sparseSkipHash.begin(); it != sparseSkipHash.end(); ++it )
    {
      totalSkips.insert( std::make_pair<u32, u32>( ( *it ).second.first, ( *it ).first ) );
      averageSkips.insert( std::make_pair<u32, u32>( ( ( *it ).second.first / ( *it ).second.second ), ( *it ).first ) );
      totalAverageSkips += ( *it ).second.first;
      totalAverageCount += ( *it ).second.second;
    }

    for( std::unordered_map<std::pair<u32, u32>, u32, PairHash<u32> >::iterator it = sparseCollisionHash.begin(); it != sparseCollisionHash.end(); ++it )
    {
      totalCollisions.insert( std::make_pair<u32, std::pair<u32, u32> >( ( *it ).second, ( *it ).first ) );
      totalTotalCollisions += ( *it ).second;
    }

    std::cout << "totally average skips: " << ( ( (f64)totalAverageSkips ) / totalAverageCount ) << std::endl;

    const u32 nBest = 100;
    {
      u32 cnt = 0;
      for( std::multimap<u32, u32>::const_reverse_iterator it = averageSkips.rbegin(); it != averageSkips.rend(); ++it )
      {
        if( ++cnt == nBest )
          break;
        std::cout << "average skips for node " << ( *it ).second << ": " << ( *it ).first << " unigram-score: " << unigramScores[( *it ).second] << std::endl;
      }
    }
    std::cout << std::endl;
    {
      u32 cnt = 0;
      for( std::multimap<u32, u32>::const_reverse_iterator it = totalSkips.rbegin(); it != totalSkips.rend(); ++it )
      {
        if( ++cnt == nBest )
          break;
        std::cout << "total skips for node " << ( *it ).second << ": " << ( *it ).first << " average " << ( ( (float)sparseSkipHash[( *it ).second].first ) / sparseSkipHash[( *it ).second].second ) << " unigram-score: " << unigramScores[( *it ).second] << std::endl;
      }
    }
    std::cout << std::endl;
    std::cout << "total collisions: " << totalTotalCollisions << std::endl;
    {
      u32 cnt = 0;
      for( std::multimap<u32, std::pair<u32, u32> >::const_reverse_iterator it = totalCollisions.rbegin(); it != totalCollisions.rend(); ++it )
      {
        if( ++cnt == nBest )
          break;
        std::cout << "total collisions for pair (" << ( *it ).second.first << " [" << unigramScores[( *it ).second.first] << "], " << ( *it ).second.second << " [" << unigramScores[( *it ).second.second] << "]): " << ( *it ).first << std::endl;
      }
    }
    std::cout << std::endl;
  }
} evalualte;

#endif

/*
 * To enhance opportunities for future reuse the look-ahead tree
 * structure is independent from the HMM state-tree structure.
 *
 * Memory consumption of LM look-ahead is determined by two
 * parameters: The size of the look-ahead tables, and the number of
 * tables kept im memory.
 *
 * The look-ahead table size is determined by the level of detail
 * reflected in the look-ahead structure.  This can be controled the
 * parameters tree-cutoff and minimum-representation.  By default the
 * look-ahead tables are not pruned, so memory consumption is
 * (unnecessarily) large.
 *
 * The number of tables kept in memory is affected by history-limit,
 * cache-size-low and cache-size-high.
 *
 * By default tables are deleted as soon as they are not immediately
 * needed (i.e. all relevant trees are pruned).  We say, the table
 * becomes inactive.  This saves memory but wastes computation time.
 * The caching strategy works as follows: As long as there are less
 * than cache-size-high tables in memory, no table is deleted.  Above
 * that limit the table that has least recently become inactive is
 * freed whenwever a table is release.  Note that cache-size-high is
 * not a strict upper bound on the number of tables.  It will be
 * exceeded when the search process requires it.
 *
 * When a new table is requested, usually one of the inactive tables
 * is re-used.  But when there are less than cache-size-low tables in
 * memory, a new table is created instead.  cache-size-low is thus a
 * (soft) lower bound on the number of tables in memory.  (Of course
 * there may be less tables, if the search never requires that many
 * different contexts.)
 *
 * Tree cutoff: In the full look-ahead structure non-branching state
 * sequences are represented by a single node.  The MINIMUM depth of
 * these states is used as pruning criterion: If it is larger than
 * "network-cutoff", the look-ahead node is merged with its parent.
 * (This minimum criterion is believed to best reproduce the behavior
 * of the old standard system.  However, other criteria might be
 * better.)
 *
 * TODO:
 * Optimizations to be considered:
 * - Different LA table cut-off criteria (may-be)
 * - Lazy-evaluation of LA tables (may-be; partially done by "lazy lookahead")
 */

///@todo Properly manage re-usage and caching of tables (considering sparse tables), cleanup
///@todo Memory-usage statistics

// Finds the integer square root of a positive number of any type
template <typename type>
type isqrt( type remainder ) {
  if ( remainder < 0 ) // if type is unsigned this will be ignored = no runtime
    return 0;  // negative number ERROR

  type place = (type)1 << ( sizeof ( type ) * 8 - 2 );   // calculated by precompiler = same runtime as: place = 0x40000000
  while ( place > remainder )
    place /= 4;  // optimized by compiler as place >>= 2

  type root = 0;
  while ( place )
  {
    if ( remainder >= root + place )
    {
      remainder -= root + place;
      root += place * 2;
    }
    root /= 2;
    place /= 4;
  }
  return root;
}

const LanguageModelLookahead::LookaheadId LanguageModelLookahead::invalidId = Core::Type<LanguageModelLookahead::LookaheadId>::max;

struct SparseStatistics {
  SparseStatistics() {
    clear();
  }

  void clear();

  void write( Core::XmlWriter& w ) const;

  u32 totalScoreCount;
  u32 potentialLookaheadNodes, backOffLookaheadNodes;
  u32 backOffLookaheadNodeHashIterations;
  u32 totalHashSize;
  u32 expectedLookAheadNodes;
  u64 lookAheadNodesExpectationDeviation;
  u32 sparseTables;
  u32 resizedTables, uniqueResizedTables;
};

struct LanguageModelLookahead::CacheStatistics {
  enum CacheEvent {shareInCacheHit, freeCacheHit, cacheMiss};
  static const Core::Choice cacheEventChoice;
  Core::ChoiceStatistics cacheEvents;
  Core::Statistics<u32> nTables, nActiveTables;
  void clear();
  void write( Core::XmlWriter& ) const;
  CacheStatistics();

  SparseStatistics sparseStats;
};

const Core::ParameterInt LanguageModelLookahead::paramHistoryLimit(
  "history-limit",
  "length of history considered for look-ahead (effective m-grammity of the look-ahead model - 1). -1 for unlimited history.",
  -1, -1 );
const Core::ParameterInt LanguageModelLookahead::paramTreeCutoff(
  "network-cutoff",
  "maximum depth of state network covered by look-ahead (number of HMM state covered)",
  Core::Type<s32>::max, 0 );
const Core::ParameterInt LanguageModelLookahead::paramMinimumRepresentation(
  "minimum-representation",
  "minimum number of HMM states represented by one look-ahead node",
  1, 1 );
const Core::ParameterInt LanguageModelLookahead::paramCacheSizeLow(
  "cache-size-low",
  "number of look-ahead tables retained before starting to re-use inactive tables",
  3500, 0 );
const Core::ParameterInt LanguageModelLookahead::paramCacheSizeHigh(
  "cache-size-high",
  "number of look-ahead tables allowed before starting to delete inactive tables",
  4500, 0 );
const Core::ParameterBool LanguageModelLookahead::paramConsiderBackOffInMaximization(
  "consider-backoff-in-maximization",
  "Disabling this makes the look-ahead much faster, without causing problems",
  false );
const Core::ParameterBool LanguageModelLookahead::paramConsiderPronunciationScore(
  "consider-pronunciation-score",
  "",
  true );
const Core::ParameterBool LanguageModelLookahead::paramConsiderExitPenalty(
  "consider-exit-penalty",
  "consider the phoneme exit penalty in the look-ahead (only beneficial if some penalties are very high)",
  false );
const Core::ParameterFloat LanguageModelLookahead::paramSparseLookAheadThreshold(
  "sparse-threshold",
  "only create a sparse look-ahead table if the fraction of words with real scores is lower than this",
  0.5 );
const Core::ParameterFloat LanguageModelLookahead::paramSparseHashSizeFactor(
  "sparse-hash-size-factor",
  "",
  1.8 );
const Core::ParameterFloat LanguageModelLookahead::paramSparseHashResizeAtFill(
  "sparse-hash-size-resize-at-fill",
  "",
  0.75 );
const Core::ParameterBool LanguageModelLookahead::paramSparseThresholdExpectationBased(
  "sparse-threshold-expectation-based",
  "",
  true );
const Core::ParameterFloat LanguageModelLookahead::paramLmLookaheadScale(
  "lm-lookahead-scale",
  "",
  1.0 );
const Core::ParameterFloat paramUseLogSemiring(
  "log-semiring-factor",
  "1.0 if the log-semiring should be used (eg. probability-sums instead of maximum probability, like in WFST search). Inefficient.",
  0, 0, 1.0 );
const Core::ParameterInt LanguageModelLookahead::paramCollisionHashSize(
  "collision-prevention-hash-size",
  "",
  65536 );

const Core::ParameterFloat LanguageModelLookahead::paramMaxCollisionDeviation(
  "collision-prevention-max-average-deviation",
  "",
  1.3 );

const Core::ParameterFloat paramEnforceLocality(
  "enforce-locality",
  "",
  1.0, 0.0, 1.0 );

const Core::ParameterString LanguageModelLookahead::paramCacheArchive(
  "cache-archive",
  "cache archive in which the look-ahead should be cached",
  "global-cache" );

static const int predictionArraySize = 100;

// If this is enabled, then the maximization can also consider the backing-off. That is more correct,
// but inefficient, so it's not really useful for practical usage.
// #define ALLOW_CONSIDER_BACK_OFF_IN_MAXIMIZATION

LanguageModelLookahead::LanguageModelLookahead(
  const Core::Configuration &c,
  Lm::Score wpScale,
  Core::Ref<const Lm::ScaledLanguageModel> lm,
  HMMStateNetwork& tree,
  StateId rootNode,
  const std::vector<PersistentStateTree::Exit>& exits,
  Core::Ref<const Am::AcousticModel> acousticModel ) :
  Core::Component( c ),
  wpScale_( wpScale ),
  maxDepth_( 0 ),
  tree_( tree ),
  lm_( lm ),
  sparseNodesPrediction_( predictionArraySize, isqrt( lm_->lexicon()->nLemmas() ) + 1 ),
  batchRequest_( 0 ),
  nTables_( 0 ), nFreeTables_( 0 ),
  statisticsChannel_( config, "statistics" ) {
  acousticModel_ = acousticModel;

  log() << "using pronunciation scale " << wpScale_;

  verify( approximatelyEqual( scaledLogAdd( -std::log( 0.1 ), -std::log( 0.2 ), 1.0, 1.0 ), -std::log( 0.3 ) ) );
  verify( approximatelyEqual( scaledLogAdd( -std::log( 0.5 ) * 2, -std::log( 0.4 ) * 2, 2.0, 0.5 ), -std::log( 0.9 ) * 2 ) );
  verify( approximatelyEqual( scaledLogAdd( -std::log( 0.0001 ) * 20, -std::log( 0.0005 ) * 20, 20.0, 1 / 20.0 ), -std::log( 0.0006 ) * 20 ) );

  scale_ = paramLmLookaheadScale( config );

  // Somehow initialize the prediction
  sparseNodesPrediction_.add( 0, 0 );
  sparseNodesPrediction_.add( isqrt( lm_->lexicon()->nLemmas() ), lm_->lexicon()->nLemmas() * 5 );

  historyLimit_ = paramHistoryLimit( config );
  cutoffDepth_  = paramTreeCutoff( config );
  minimumRepresentation_ = paramMinimumRepresentation( config );
  cacheSizeHighMark_ = paramCacheSizeHigh( config );
  cacheSizeLowMark_  = paramCacheSizeLow( config );

  sparseThresholdExpectationBased_ = paramSparseThresholdExpectationBased( config );

  sparseLookAheadThreshold_ = paramSparseLookAheadThreshold( config );

#ifdef ALLOW_CONSIDER_BACK_OFF_IN_MAXIMIZATION
  considerBackOffInMaximization_ = paramConsiderBackOffInMaximization( config );
#else
  considerBackOffInMaximization_ = false;
#endif
  considerPronunciationScore_ = paramConsiderPronunciationScore( config );
  considerExitPenalty_ = paramConsiderExitPenalty( config );

  sparseHashSizeFactor_ = paramSparseHashSizeFactor( config );
  sparseHashResizeAtFillFraction_ = paramSparseHashResizeAtFill( config ) * 256;
  if( sparseHashResizeAtFillFraction_ < 1 )
    sparseHashResizeAtFillFraction_ = 1;

  if( sparseHashResizeAtFillFraction_ > 254 )
    sparseHashResizeAtFillFraction_ = 254;

  logSemiringFactor_ = paramUseLogSemiring( config );

  if( logSemiringFactor_ && considerPronunciationScore_ )
  {
    log( "pronunciation score can not be considered when using summation" );
    considerPronunciationScore_ = false;
  }

  cacheStatistics_ = new CacheStatistics;

  if( historyLimit_ == -1 )
    log( "using unlimited look-ahead history" );
  else
    log( "look-ahead history limit is %d (usually means %d-gram look-ahead)",
      historyLimit_, historyLimit_ + 1 );
  buildLookaheadStructure( tree, rootNode, exits );
}

LanguageModelLookahead::~LanguageModelLookahead() {
  delete cacheStatistics_;
  for ( List::iterator t = tables_.begin(); t != tables_.end(); ++t ) {
    delete *t;
  }
  delete batchRequest_;
}

// ===========================================================================
// static structure

class LanguageModelLookahead::ConstructionNode
{
public:
  LookaheadId id;
  struct { StateTree::Depth min, max; } depth;
  typedef std::vector<ConstructionNode*> Successors;
  typedef std::vector<StateTree::StateId> Represents;
  Represents represents;

private:
  Ends ends_;
  Successors successors_;

  enum Consolidation { dirty, unique, domineesValid, hashValid };
  mutable Consolidation consolidation_;
  mutable Ends dominees_;
  mutable u32 hash_;

public:
  bool isUnique() const {
    return consolidation_ >= unique;
  }

  void makeUnique() {
    std::sort( ends_.begin(), ends_.end() );
    ends_.erase( std::unique( ends_.begin(), ends_.end() ), ends_.end() );
    std::sort( successors_.begin(), successors_.end() );
    successors_.erase( std::unique( successors_.begin(), successors_.end() ), successors_.end() );
    consolidation_ = unique;
  }

private:
  void updateDominiees() const {
    require( consolidation_ >= unique );
    dominees_ = ends_;
    Ends curr;
    for ( Successors::const_iterator s = successors_.begin(); s != successors_.end(); ++s ) {
      dominees_.swap( curr );
      dominees_.clear();
      const Ends &succ( ( *s )->dominees() );
      std::set_union( curr.begin(), curr.end(),
        succ.begin(), succ.end(),
        std::back_inserter( dominees_ ) );
    }
    consolidation_ = domineesValid;
  }

  void updateHash() const {
    require( consolidation_ >= domineesValid );
    hash_ = 0;
    for ( Ends::const_iterator e = dominees_.begin(); e != dominees_.end(); ++e )
      hash_ = ( ( hash_ << 3 ) | ( hash_ >> 29 ) ) ^ u32( (long) /* cast from pointer */ *e );
    consolidation_ = hashValid;
  }

  struct DominationEquality;
  friend struct ConstructionNode::DominationEquality;
  friend class LanguageModelLookahead::ConstructionTree;

public:
  ConstructionNode() :
    id( invalidId ),
    consolidation_( dirty ) {
    depth.min = Core::Type<StateTree::Depth>::max;
    depth.max = Core::Type<StateTree::Depth>::min;
  }

  const Ends &dominees() const {
    switch ( consolidation_ ) {
      case dirty: require( consolidation_ > dirty );
      case unique: updateDominiees();
      case domineesValid:;
      case hashValid:;
    }
    verify_( consolidation_ >= domineesValid );
    return dominees_;
  }

  u32 hash() const {
    switch ( consolidation_ ) {
      case dirty: require( consolidation_ > dirty );
      case unique: updateDominiees();
      case domineesValid: updateHash();
      case hashValid:;
    }
    verify_( consolidation_ >= hashValid );
    return hash_;
  }

  const Ends &ends() const {
    return ends_;
  }
  Ends &ends() {
    consolidation_ = dirty;
    return ends_;
  }

  const Successors &successors() const {
    return successors_;
  }
  Successors &successors() {
    consolidation_ = dirty;
    return successors_;
  }

protected:

  struct DominationHash {
    u32 operator()( const ConstructionNode *n ) const {
      return n->hash();
    }
  };

private:
  struct DominationEquality {
    bool operator()( const ConstructionNode *l, const ConstructionNode *r ) const {
      if ( l->consolidation_ >= hashValid && r->consolidation_ >= hashValid )
        if ( l->hash_ != r->hash_ ) return false;
      const Ends &ld( l->dominees() );
      const Ends &rd( r->dominees() );
      if ( ld.size() != rd.size() ) return false;
      return std::equal( ld.begin(), ld.end(), rd.begin() );
    }
  };
};

class LanguageModelLookahead::ConstructionTree
{
private:
  typedef std::vector<ConstructionNode*> NodeList;
  NodeList nodeList_;
  struct LevelStatistics;
public:
  ConstructionTree();
  ~ConstructionTree();

  bool isWellOrdered() const;
  void writeStatistics( std::ostream& ) const;
  void build( HMMStateNetwork& tree, StateId rootNode, const std::vector<PersistentStateTree::Exit>& exits, Bliss::LexiconRef lexicon );
  void prune( const LanguageModelLookahead *master );
  void purge();
  LookaheadId nNodes() const {
    return nodeList_.size();
  }
  const ConstructionNode &node( LookaheadId i ) const {
    ensure( nodeList_[i]->isUnique() );
    return *nodeList_[i];
  }
};

LanguageModelLookahead::ConstructionTree::ConstructionTree() {
}

LanguageModelLookahead::ConstructionTree::~ConstructionTree() {
  for ( NodeList::const_iterator i = nodeList_.begin(); i != nodeList_.end(); ++i )
    delete *i;
}

/** Check wether each node has a lower index than its parent. */

bool LanguageModelLookahead::ConstructionTree::isWellOrdered() const {
  bool result = true;
  for ( u32 ci = 0; ci < nodeList_.size(); ++ci ) {
    const ConstructionNode &cn( *nodeList_[ci] );
    if ( cn.id == invalidId ) continue;
    verify( cn.id == ci );
    for ( ConstructionNode::Successors::const_iterator si = cn.successors().begin(); si != cn.successors().end(); ++si ) {
      result = result && ( ( *si )->id != invalidId );
      result = result && ( ( *si )->id < ci );
    }
  }
  return result;
}

struct LanguageModelLookahead::ConstructionTree::LevelStatistics {
  u32 nNodes, nSuccessors, nEnds;
};

void LanguageModelLookahead::ConstructionTree::writeStatistics( std::ostream &os ) const {
  typedef std::map<StateTree::Depth, LevelStatistics> LevelStatisticsMap;
  LevelStatisticsMap levels;
  u32 totalEnds = 0;
  u32 totalSuccessors = 0;

  for ( u32 ci = 0; ci < nodeList_.size(); ++ci ) {
    const ConstructionNode &cn( *nodeList_[ci] );
    if ( cn.id == invalidId ) continue;
    LevelStatistics &ls( levels[cn.depth.min] );
    ls.nNodes      += 1;
    ls.nSuccessors += cn.successors().size();
    ls.nEnds       += cn.ends().size();
    totalEnds += cn.ends().size();
    totalSuccessors += cn.successors().size();
  }
  for ( std::map<StateTree::Depth, LevelStatistics>::const_iterator l = levels.begin(); l != levels.end(); ++l ) {
    StateTree::Depth depth = l->first;
    const LevelStatistics &thisLevel( l->second );
    os << Core::form( "level %3d: %6d nodes, branching factor %3.2f, %4d ends\n",
      depth, thisLevel.nNodes, f32( thisLevel.nSuccessors )  / f32( thisLevel.nNodes ), thisLevel.nEnds );
  }
}

void LanguageModelLookahead::ConstructionTree::build( HMMStateNetwork& tree, StateId rootNode, const std::vector<PersistentStateTree::Exit>& exits, Bliss::LexiconRef lexicon ) {
  std::vector<LookaheadId> nodeId(tree.stateCount(), invalidId );
  typedef std::unordered_set<ConstructionNode*,
    ConstructionNode::DominationHash,
    ConstructionNode::DominationEquality> NodeSet;
  NodeSet nodeSet;

  u32 totalEncounteredWordEnds = 0;

  static u32 wordEndsHash;
  wordEndsHash = 0;

  static u32 totalSuccessors;
  totalSuccessors = 0;

  static u32 totalDeletedNodes;
  totalDeletedNodes = 0;

  static u32 totalDeletedNodeSuccessors;
  totalDeletedNodeSuccessors = 0;

  struct Builder {
    std::vector<LookaheadId>& nodeId;
    NodeSet& nodeSet;
    const std::vector<PersistentStateTree::Exit>& exits;
    Bliss::LexiconRef lexicon;
    NodeList& nodeList_;
    HMMStateNetwork& tree_;
    u32& totalEncounteredWordEnds;

    Builder( HMMStateNetwork& tree, const std::vector<PersistentStateTree::Exit>& _exits,
      Bliss::LexiconRef _lexicon, NodeList& nodeList, std::vector<LookaheadId>& _nodeId,
      NodeSet& _nodeSet, u32& _totalEncounteredWordEnds ) :
      nodeId( _nodeId ),
      nodeSet( _nodeSet ),
      exits( _exits ),
      lexicon( _lexicon ),
      nodeList_( nodeList ),
      tree_( tree ),
      totalEncounteredWordEnds( _totalEncounteredWordEnds ) {
    }

    void build( StateId node, u32 depth ) {
      bool hasWordEnd = false;

      if( nodeId[node] != invalidId ) {
        return;   //Happens while collecting coarticulated nodes
      }

      std::vector<u32> successors;

      for( HMMStateNetwork::SuccessorIterator target = tree_.successors(node); target; ++target )
      {
        if( not target.isLabel() )
        {
          build( *target, depth + 1 );
          successors.push_back( *target );
        }else{
          verify( exits[target.label()].pronunciation != Bliss::LemmaPronunciation::invalidId );
          // Ignore skip-wordends
          if(tree_.state(exits[target.label()].transitState).stateDesc.transitionModelIndex != Am::TransitionModel::entryM2 )
            hasWordEnd = true;
        }
      }

      ConstructionNode *cn = 0;

      if( successors.size() == 1 && !hasWordEnd ) {
        ///Forward the node. No branching, so no new node needs to be created
        verify( successors.front() < nodeId.size() );
        verify( nodeId[successors.front()] != invalidId );
        verify( nodeId[successors.front()] < nodeList_.size() );
        cn = nodeList_[nodeId[successors.front()]];
      }else{
        ///Branching happens in this place, need a new node that has the others as followers
        cn = new ConstructionNode;

        //Care about word ends
        for( HMMStateNetwork::SuccessorIterator target = tree_.successors(node); target; ++target ) {
          if(target.isLabel() ) {
            ++totalEncounteredWordEnds;

            u32 exitIndex = target.label();

            verify( exitIndex < exits.size() );
            if( exits[exitIndex].pronunciation != Bliss::LemmaPronunciation::invalidId &&
                    tree_.state(exits[exitIndex].transitState).stateDesc.transitionModelIndex != Am::TransitionModel::entryM2 )
            {
              const Bliss::LemmaPronunciation* pron = lexicon->lemmaPronunciation( exits[exitIndex].pronunciation );
              wordEndsHash += pron->id() * depth;
              cn->ends().push_back( pron );
            }
          }
        }

        for( u32 a = 0; a < successors.size(); ++a ) {
          verify( nodeId[successors[a]] != invalidId );
          cn->successors().push_back( nodeList_[nodeId[successors[a]]] );
          ++totalSuccessors;
        }

        verify( successors.size() || hasWordEnd );
        verify( cn->successors().size() || cn->ends().size() );

        verify( cn->successors().size() == successors.size() );

        cn->makeUnique();

        NodeSet::iterator cni = nodeSet.find( cn );
        if ( cni == nodeSet.end() ) {
          cn->id = nodeList_.size();
          nodeList_.push_back( cn );
          nodeSet.insert( cn );
        } else {
          totalDeletedNodeSuccessors += cn->successors().size();
          delete cn;
          cn = *cni;
        }
      }

      verify( depth < Core::Type<s16>::max );

      nodeId[node] = cn->id;
      cn->depth.min = std::min( cn->depth.min, (s16)depth );
      cn->depth.max = std::max( cn->depth.max, (s16)depth );
      cn->represents.push_back( node );

      verify( nodeId[node] != invalidId );
    }

    s32 collectTopologicalStates( StateId node, int depth,
      std::vector<std::vector<StateId> >& topologicalStates,
      std::vector<s32>& collected ) {
      if( topologicalStates.size() <= depth )
        topologicalStates.resize( depth + 1 );

      verify( collected[node] != -2 );

      if( collected[node] != -1 )
        return collected[node];

      collected[node] = -2;

      for( HMMStateNetwork::SuccessorIterator edges = tree_.successors(node); edges; ++edges )
      {
        if( not edges.isLabel() )
        {
          int depth2 = collectTopologicalStates( *edges, depth + 1, topologicalStates, collected );
          if( depth2 - 1 < depth )
            depth = depth2 - 1;
        }
      }

      verify( depth >= 0 );

      collected[node] = depth;
      topologicalStates[depth].push_back( node );
      return depth;
    }
  };

  std::vector<std::vector<StateId> > topologicalStates;
  std::vector<s32> collected(tree.stateCount(), -1 );

  Builder builder( tree, exits, lexicon, nodeList_, nodeId, nodeSet, totalEncounteredWordEnds );

  builder.collectTopologicalStates( rootNode, 100, topologicalStates, collected );

  for( StateId node = 1; node < tree.stateCount(); ++node ) {
    if( collected[node] == -1 ) {
      builder.collectTopologicalStates( node, 100, topologicalStates, collected );
    }
  }

  // Compress depths
  for( int a = ( (int)( topologicalStates.size() ) ) - 1; a >= 0; --a ) {
    if( topologicalStates[a].empty() )
      topologicalStates.erase( topologicalStates.begin() + a );
  }

  // Build
  for( int a = ( (int)( topologicalStates.size() ) ) - 1; a >= 0; --a ) {
    for( std::vector<StateId>::const_reverse_iterator it = topologicalStates[a].rbegin();
         it != topologicalStates[a].rend(); ++it ) {
      builder.build( *it, a );
    }
  }

  verify( nodeId[rootNode] != invalidId );

  //Collect coarticulated nodes
  for( StateId node = 1; node < tree.stateCount(); ++node ) {
    verify( nodeId[node] != invalidId );
  }

  ensure( isWellOrdered() );

  u32 totalSucc = 0;
  for ( u32 ci = 0; ci < nodeList_.size(); ++ci ) {
    const ConstructionNode &cn( *nodeList_[ci] );
    totalSucc += cn.successors().size();
  }
}

/** Decide whether a look-ahead node should be merged with its father. */

bool LanguageModelLookahead::shouldPruneConstructionNode( const ConstructionNode &sn ) const {
  bool isTooDeep = ( sn.depth.min > cutoffDepth_ );
  bool isTooSmall = ( sn.represents.size() < minimumRepresentation_ );
  return isTooDeep || isTooSmall;
}

/**
 * Merge nodes of construction tree which should be pruned with their
 * parents.  Proceed in in bottom-up direction.  A pruned nodes is
 * indicated by its @c id field being invalid.  Call purge() to remove
 * pruned nodes.
 */

void LanguageModelLookahead::ConstructionTree::prune(
  const LanguageModelLookahead *master ) {
  for ( NodeList::iterator ci = nodeList_.begin(); ci != nodeList_.end(); ++ci ) {
    ConstructionNode &cn( **ci );
    verify( cn.successors().size() || cn.ends().size() );
    ConstructionNode::Successors newSuccessors;
    for ( ConstructionNode::Successors::iterator si = cn.successors().begin(); si != cn.successors().end(); ++si ) {
      ConstructionNode &sn( **si );
      if ( sn.id == invalidId || master->shouldPruneConstructionNode( sn ) ) {
        std::copy( sn.ends().begin(), sn.ends().end(), std::back_insert_iterator<Ends>( cn.ends() ) );
        std::copy( sn.successors().begin(), sn.successors().end(),
          std::back_insert_iterator<ConstructionNode::Successors>( newSuccessors ) );
        std::copy( sn.represents.begin(), sn.represents.end(),
          std::back_insert_iterator<ConstructionNode::Represents>( cn.represents ) );
        cn.depth.min = std::min( cn.depth.min, sn.depth.min );
        cn.depth.max = std::max( cn.depth.max, sn.depth.max );
        sn.id = invalidId;
      } else {
        newSuccessors.push_back( *si );
      }
    }
    cn.successors().swap( newSuccessors );
    verify( cn.successors().size() || cn.ends().size() );
    cn.makeUnique();
  }
  ensure( isWellOrdered() );
}

/** Remove pruned nodes from construction tree. */

void LanguageModelLookahead::ConstructionTree::purge() {
  for ( NodeList::iterator ci = nodeList_.begin(); ci != nodeList_.end(); ++ci ) {
    if ( ( *ci )->id == invalidId ) {
      delete *ci; *ci = 0;
    }
  }
  // reassign ids
  nodeList_.erase( std::remove( nodeList_.begin(), nodeList_.end(), (ConstructionNode*) 0 ), nodeList_.end() );
  for ( LookaheadId id = 0; id < nodeList_.size(); ++id ) {
    nodeList_[id]->id = id;
  }
  ensure( isWellOrdered() );
}

struct KnuthHash {
  u32 operator()( u32 a ) const {
    return a * 2654435761u;
  }
};

template <class Key>
struct StandardHash {
  inline u32 operator()( Key a ) {
    a = ( a ^ 0xc761c23c ) ^ ( a >> 19 );
    a = ( a + 0xfd7046c5 ) + ( a << 3 );
    return a;
  }
};

struct RandomHash {
  RandomHash() {
    srand( 0 );
  }

  u32 operator()( u32 id ) {
    u32 hash = 0;

    bool ready = false;
    while( !ready )
    {
      hash = rand();
      if( haveKeys.count( hash ) )
        continue;
      ready = true;
    }

    haveKeys.insert( hash );
    return hash;
  }

  std::unordered_set<u32> haveKeys;
};

struct RandomHashKnuth {
  RandomHashKnuth() {
    srand( 0 );
  }

  u32 operator()( u32 id ) {
    u32 hash = 0;

    bool ready = false;
    while( !ready )
    {
      hash = KnuthHash() ( rand() );
      if( haveKeys.count( hash ) )
        continue;
      ready = true;
    }

    haveKeys.insert( hash );
    return hash;
  }

  std::unordered_set<u32> haveKeys;
};

struct DistributedRandomHash {
  DistributedRandomHash( u32 hashSize, float _maxDeviation ) : testHashSize( hashSize ), maxDeviation( _maxDeviation ), testHash( testHashSize, 0 ), hashFill( 0 ) {
    srand( 0 );
  }

  u32 operator()( u32 id ) {
    u32 hash = 0;
    u32 iter = 0;

    const u32 maxIter = testHashSize * 2;

    float averageFill = ( (float)hashFill ) / testHash.size();

    bool ready = false;
    while( !ready )
    {
      ++iter;
      hash = rand();
      if( haveKeys.count( hash ) )
        continue;
      u32 cell = hash % testHash.size();
      if( testHash[cell] < ( averageFill * maxDeviation ) + 1 || iter > maxIter )
        ready = true;
    }

    if( iter > maxIter )
      std::cerr << "maximum number of iterations reached while assigning hash to look-ahead id " << id << std::endl;

    haveKeys.insert( hash );
    testHash[hash % testHash.size()] += 1;
    ++hashFill;
    return hash;
  }

  u32 testHashSize;
  float maxDeviation;
  std::vector<u32> testHash;
  std::unordered_set<u32> haveKeys;
  u32 hashFill;
};

struct DepthDistributedRandomHash {
  DepthDistributedRandomHash( LanguageModelLookahead* lookahead, u32 hashSize, float _maxDeviation ) : lookahead_( lookahead ), testHashSize( hashSize ), maxDeviation( _maxDeviation ), testHash( testHashSize, 0 ), hashFill( 0 ) {
    srand( 0 );
  }

  u32 operator()( u32 id ) {
    u32 hash = 0;
    u32 iter = 0;

    const u32 maxIter = testHashSize * 2;

    float averageFill = ( (float)hashFill ) / testHash.size();
    float weight = 100.0 / ( 50 + lookahead_->nodeDepth( id ) );

    bool ready = false;
    while( !ready )
    {
      ++iter;
      hash = rand();
      if( haveKeys.count( hash ) )
        continue;
      u32 cell = hash % testHash.size();
      if( testHash[cell] < ( averageFill * maxDeviation ) + 1 || iter > maxIter )
        ready = true;
    }

    if( iter > maxIter )
      std::cerr << "maximum number of iterations reached while assigning hash to look-ahead id " << id << std::endl;

    haveKeys.insert( hash );
    testHash[hash % testHash.size()] += weight;
    hashFill += weight;
    return hash;
  }

  LanguageModelLookahead* lookahead_;
  u32 testHashSize;
  float maxDeviation;
  std::vector<float> testHash;
  std::unordered_set<u32> haveKeys;
  float hashFill;
};

struct WeightedDistributedRandomHash {
  WeightedDistributedRandomHash( const std::vector<u32>& _weights, u32 hashSize, float _maxDeviation ) : weights( _weights ), testHashSize( hashSize ), maxDeviation( _maxDeviation ), testHash( testHashSize, 0 ), hashFill( 0 ) {
    srand( 0 );
  }

  u32 operator()( u32 id ) {
    u32 hash = 0;
    u32 iter = 0;

    const u32 maxIter = testHashSize * 2;

    float averageFill = ( (float)hashFill ) / testHash.size();
    float weight = weights[id] + 1;

    bool ready = false;
    while( !ready )
    {
      ++iter;
      hash = rand();
      if( haveKeys.count( hash ) )
        continue;
      u32 cell = hash % testHash.size();
      if( testHash[cell] <= ( averageFill * maxDeviation ) || iter > maxIter )
        ready = true;
    }

    if( iter > maxIter )
      std::cerr << "maximum number of iterations reached while assigning hash to look-ahead id " << id << std::endl;

    haveKeys.insert( hash );
    testHash[hash % testHash.size()] += weight;
    hashFill += weight;
    return hash;
  }

  const std::vector<u32>& weights;
  u32 testHashSize;
  float maxDeviation;
  std::vector<float> testHash;
  std::unordered_set<u32> haveKeys;
  float hashFill;
};

/// @todo MixedHash: Einfach die indizes zufällig durchwürfeln

template <class Hash>
struct WeightedDistributedStandardHash {
  WeightedDistributedStandardHash( const std::vector<u32>& _weights, u32 hashSize, float _maxDeviation, float _locality ) : weights( _weights ), testHashSize( hashSize ), maxDeviation( _maxDeviation ), testHash( testHashSize, 0 ), hashFill( 0 ), locality( _locality ), previous( 0 ) {
    srand( 0 );
  }

  u32 operator()( u32 id ) {
    u32 hash = 0;
    u32 iter = 0;
    const u32 maxIter = testHash.size() + 1;

    float averageFill = ( (float)hashFill ) / testHash.size();
    float weight = weights[id] + 1;

    while( true )
    {
      hash = Hash() ( id + iter );
      ++iter;
      if( haveKeys.count( hash ) )
        continue;
      s32 cell = hash % testHash.size();
      verify( cell >= 0 && cell < testHash.size() );
      s32 previousCell = previous % testHash.size();
      verify( previousCell >= 0 && previousCell < testHash.size() );
      float currentLocality = abs( cell - previousCell ) / (float)testHash.size();
      verify( currentLocality <= 1.0 && currentLocality >= 0.0 );
      if( ( currentLocality == 0 || currentLocality  > locality + ( 1.0 / testHash.size() ) ) && iter < maxIter )
        continue;
      if( testHash[cell] <= ( averageFill * maxDeviation ) )
        break;
      if( iter >= maxIter )
      {
        std::cerr << "max-iterations reached for " << id << std::endl;
        break;
      }
    }

    haveKeys.insert( hash );
    testHash[hash % testHash.size()] += weight;
    hashFill += weight;
    previous = hash;
    return hash;
  }

  const std::vector<u32>& weights;
  u32 testHashSize;
  float maxDeviation;
  std::vector<float> testHash;
  std::unordered_set<u32> haveKeys;
  float hashFill;
  float locality;
  u32 previous;
};

/** Initialize internal compact look-ahead structure from construction tree. */

void LanguageModelLookahead::buildCompressesLookaheadStructure(
  u32 nodeStart,
  u32 numNodes,
  const ConstructionTree &ct ) {
  require( ct.isWellOrdered() );
  require( ct.nNodes() > 0 );

  nodeId_.resize( numNodes, invalidId );

  for ( LookaheadId ci = 0; ci < ct.nNodes(); ++ci ) {
    const ConstructionNode &cn( ct.node( ci ) );
    verify( ci == nodes_.size() );
    {
      Node n;
      n.firstEnd = ends_.size();
      n.firstSuccessor = successors_.size();
      nodes_.push_back( n );
      verify( nodes_.back().firstEnd == ends_.size() );
    }

    std::copy( cn.ends().begin(), cn.ends().end(), std::back_insert_iterator<Ends>( ends_ ) );
    for ( ConstructionNode::Successors::const_iterator si = cn.successors().begin(); si != cn.successors().end(); ++si )
      successors_.push_back( ( *si )->id );
    for ( std::vector<StateTree::StateId>::const_iterator ri = cn.represents.begin(); ri != cn.represents.end(); ++ri )
      nodeId_.edit( *ri ) = ci;
  }

  for ( StateTree::StateId si = nodeStart; si < numNodes; ++si )
    verify( nodeId_[si] != invalidId );

  // add sentinel
  {
    Node n;
    n.firstEnd = ends_.size();
    n.firstSuccessor = successors_.size();
    nodes_.push_back( n );
  }

  nEntries_ = nodes_.size() - 1;

  // Set the offsets for ends
  for( Ends::const_iterator e = ends_.begin(); e != ends_.end(); ++e )
  {
    Score offset = 0;

    const Bliss::SyntacticTokenSequence& tokens( ( *e )->lemma()->syntacticTokenSequence() );

    for ( u32 ti = 0; ti < tokens.length(); ++ti )
      offset += lm_->scale() * tokens[ti]->classEmissionScore();

    offset *= scale_;

    if( considerPronunciationScore_ )
      offset += wpScale_ * ( *e )->pronunciationScore();

    if( considerExitPenalty_ )
    {
      // Add the exit penalty to the offset
      u32 len = ( *e )->pronunciation()->length();
      if( len )
      {
        Bliss::Phoneme::Id phonemeId = ( *e )->pronunciation()->phonemes()[len - 1];

        s16 boundary = Am::Allophone::isFinalPhone;
        if ( len == 1 )
          boundary |= Am::Allophone::isInitialPhone;

        // create an allophone with the correct boundaries
//         Am::Allophone allophone((*(acousticModel_->phonology()))(*(*e)->pronunciation(), len-1), boundary);

        const Am::Allophone *allo = acousticModel_->allophoneAlphabet()->allophone( Am::Allophone( phonemeId, boundary ) );
        verify( allo );
        verify( acousticModel_->hmmTopology( phonemeId ) );

        Am::AllophoneState alloState = acousticModel_->allophoneStateAlphabet()->allophoneState( allo, acousticModel_->hmmTopology( phonemeId )->nPhoneStates() - 1 );
        Am::AcousticModel::StateTransitionIndex transitionModel =  acousticModel_->stateTransitionIndex( alloState, acousticModel_->hmmTopology( phonemeId )->nSubStates() - 1 );

        verify( transitionModel < acousticModel_->nStateTransitions() );
        verify( acousticModel_->stateTransition( transitionModel ) );

        Am::StateTransitionModel::Score penalty = ( *acousticModel_->stateTransition( transitionModel ) )[Am::StateTransitionModel::exit];

        offset += penalty;
      }
    }

    endOffsets_.push_back( offset );
  }

  // Build 'parentNodes' structure

  //Maps a parent-node to each node. -1 if the node has no parent
  typedef std::unordered_multimap<LookaheadId, LookaheadId, StandardValueHash<u32> > ParentMap;
  ParentMap parentNodes;

  for( LookaheadId n = 0; n < nodes_.size() - 1; ++n )
    for( u32 s = nodes_[n].firstSuccessor; s != nodes_[n + 1].firstSuccessor; ++s )
      parentNodes.insert( std::make_pair( successors_[s], (LookaheadId)n ) );

  for( LookaheadId n = 0; n < nodes_.size() - 1; ++n )
  {
    nodes_.edit( n ).firstParent = parents_.size();

    std::pair<ParentMap::const_iterator, ParentMap::const_iterator> parents = parentNodes.equal_range( n );
    for( ParentMap::const_iterator parentIt = parents.first; parentIt != parents.second; ++parentIt )
    {
      parents_.push_back( ( *parentIt ).second );
      verify( parents_.back() > n );
    }
  }
  nodes_.edit( nodes_.size() - 1 ).firstParent = parents_.size();

  // Build nodeForToken_ structure

  typedef std::unordered_multimap<Bliss::Token::Id, std::pair<LookaheadId, Score>, StandardValueHash<u32> > TokenNodeMap;
  TokenNodeMap nodeForTokenMap;

  for( int n = nodes_.size() - 2; n >= 0; --n )
  {
    for( u32 e = nodes_[n].firstEnd; e != nodes_[n + 1].firstEnd; ++e )
    {
      const Bliss::SyntacticTokenSequence& seq = ends_[e]->lemma()->syntacticTokenSequence();
      if( seq.length() > 1 )
        Core::Application::us()->log() << "Warning: A pronunciation has an unsupported token-length for look-ahead: " << seq.length();

      Bliss::Token::Id token = Bliss::Token::invalidId;

      if( seq.length() )
        token = seq[0]->id();

      std::pair<TokenNodeMap::iterator, TokenNodeMap::iterator> it = nodeForTokenMap.equal_range( token );

      // We insert each pair of token and look-ahead node only once
      bool had = false;
      for(; it.first != it.second; ++it.first )
      {
        if( ( *it.first ).second.first == (LookaheadId)n )
        {
          had = true;
          if( ( *it.first ).second.second > endOffsets_[e] )
            ( *it.first ).second.second = endOffsets_[e];
        }
      }

      if( !had )
        nodeForTokenMap.insert( std::make_pair( token, std::make_pair( (LookaheadId)n, endOffsets_[e] ) ) );
    }
  }

  //Turn the multi-map into a more efficent index-based map

  for( int token = 0; token < lm_->tokenInventory().size(); ++token )
  {
    firstNodeForToken_.push_back( nodeForToken_.size() );
    std::pair<TokenNodeMap::iterator, TokenNodeMap::iterator> range = nodeForTokenMap.equal_range( token );
    for(; range.first != range.second; ++range.first )
      nodeForToken_.push_back( ( *range.first ).second );
  }

  verify( firstNodeForToken_.size() == lm_->tokenInventory().size() );
  invalidFirstNodeForTokenIndex_ = firstNodeForToken_.size();

  {
    //Add the look-ahead nodes for the invalid id
    int token = Bliss::Token::invalidId;

    firstNodeForToken_.push_back( nodeForToken_.size() );

    std::pair<TokenNodeMap::iterator, TokenNodeMap::iterator> range = nodeForTokenMap.equal_range( token );
    for(; range.first != range.second; ++range.first )
      nodeForToken_.push_back( ( *range.first ).second );
  }

  firstNodeForToken_.push_back( nodeForToken_.size() );

  buildDepths();
}

void LanguageModelLookahead::buildHash() {
  f32 maxDeviation = paramMaxCollisionDeviation( config );
  u32 testHashSize = paramCollisionHashSize( config );
#ifdef TEST_HASHES
  {
    StandardHash<u32> h;
    assignHashes( "standard", h, testHashSize );
  }
  {
    KnuthHash h;
    assignHashes( "knuth", h, testHashSize );
  }
  {
    DistributedRandomHash h( testHashSize, maxDeviation );
    assignHashes( "distributed random", h, testHashSize );
  }
  {
    RandomHash h;
    assignHashes( "random", h, testHashSize );
  }
  {
    RandomHashKnuth h;
    assignHashes( "random + knuth", h, testHashSize );
  }
  {
    DepthDistributedRandomHash h( this, testHashSize, maxDeviation );
    assignHashes( "depth-weighted distributed random", h, testHashSize );
  }
/*  if( !nodeObservations_.empty() && !accumulateNodeObservations_ ) {
    WeightedDistributedRandomHash h( nodeObservations_, testHashSize, maxDeviation );
    assignHashes( "observation-weighted distributed random", h, testHashSize );
   }*/
#endif
  {
    verify( lm_ );
    Lm::History hi = lm_->startHistory();
    hi = lm_->reducedHistory( hi, 0 );
    ContextLookaheadReference unigramLah = getLookahead( hi );
    fill( unigramLah );
    verify( unigramLah );
    std::vector<u32> weights;
    for( u32 l = 0; l < nEntries_; ++l )
    {
      Score s = unigramLah->scoreForLookAheadIdNormal( l );
      weights.push_back( exp( -s ) );
    }

#ifdef TEST_HASHES
    {
      WeightedDistributedRandomHash h( weights, testHashSize, maxDeviation );
      assignHashes( "unigram-weighted distributed random", h, testHashSize );
    }
#endif
    {
      WeightedDistributedStandardHash<KnuthHash> h( weights, testHashSize, maxDeviation, paramEnforceLocality( config ) );
      assignHashes( "unigram-weighted distributed knuth", h, testHashSize );
    }

#ifdef EXTENSIVE_SPARSE_COLLISION_STATS
    unigramScores.clear();
    for( u32 l = 0; l < nEntries_; ++l )
    {
      Score s = unigramLah->scoreForLookAheadIdNormal( l );
      unigramScores.insert( std::make_pair<u32, Score>( hashForNode_[l], s ) );
    }
#endif
  }
}

template <class Hash>
void LanguageModelLookahead::assignHashes( std::string hashName, Hash& hash, u32 testHashSize ) {
  hashForNode_.resize( nEntries_ );

  std::vector<u32> testHash( testHashSize, 0 );

  for( LookaheadId id = 0; id < nEntries_; ++id )
  {
    hashForNode_.edit( id ) = hash( id );
    testHash[hashForNode_[id] % testHash.size()] += 1;
/*    if( id > 1 )
    {
      s32 cell = hashForNode_[id] % testHash.size();
      verify( cell >= 0 && cell < testHash.size() );
      s32 previousCell = hashForNode_[id-1] % testHash.size();
      verify( previousCell >= 0 && previousCell < testHash.size() );
      float currentLocality = abs(cell - previousCell) / (float)testHash.size();
      std::cout << "achieved locality " << currentLocality << std::endl;
    }*/
  }

  f64 averageLocality = 0.0;
  f64 quadraticLocalityDeviation = 0.0;

  hashForState_.resize( nodeId_.size() );

  for( u32 s = 1; s < hashForState_.size(); ++s )
    hashForState_.edit( s ) = hashForNode_[nodeId_[s]];

  for( u32 s = 1; s < nEntries_; ++s )
  {
    // Locality statistics
    s32 cell = hashForNode_[s] % testHash.size();
    verify( cell >= 0 && cell < testHash.size() );
    s32 previousCell = hashForNode_[s - 1] % testHash.size();
    verify( previousCell >= 0 && previousCell < testHash.size() );
    float currentLocality = abs( cell - previousCell ) / (float)testHash.size();
    averageLocality += currentLocality;
  }

  averageLocality /= ( nEntries_ - 1 );

  for( u32 s = 1; s < nEntries_; ++s )
  {
    s32 cell = hashForNode_[s] % testHash.size();
    verify( cell >= 0 && cell < testHash.size() );
    s32 previousCell = hashForNode_[s - 1] % testHash.size();
    verify( previousCell >= 0 && previousCell < testHash.size() );
    float currentLocality = abs( cell - previousCell ) / (float)testHash.size();
    quadraticLocalityDeviation += ( currentLocality - averageLocality ) * ( currentLocality - averageLocality );
  }

  quadraticLocalityDeviation /= ( nEntries_ - 1 );

  float averageFill = ( (float)nEntries_ ) / testHash.size();
  float quadraticDeviation = 0;
  for( u32 a = 0; a < testHash.size(); ++a )
  {
    float deviation = ( testHash[a] - averageFill );
    quadraticDeviation += deviation * deviation;
  }

  float standardDeviation = sqrt( quadraticDeviation / testHash.size() );
}

void LanguageModelLookahead::propagateDepth( int node, int depth ) {
  if( nodes_[node].depth == Core::Type<u32>::max )
  {
    nodes_.edit( node ).depth = depth;
  }else{
    if( depth > nodes_[node].depth )
      nodes_.edit( node ).depth = depth;

    depth = nodes_[node].depth;
  }

  for ( u32 s = nodes_[node].firstSuccessor; s < nodes_[node + 1].firstSuccessor; ++s )
    propagateDepth( successors_[s], depth + 1 );
}

void LanguageModelLookahead::buildDepths() {
  for( u32 a = 0; a < nEntries_; ++a )
    nodes_.edit( a ).depth = Core::Type<u32>::max;

  for( int a = nEntries_ - 1; a >= 0; --a )
  {
    if( nodes_[a].depth == Core::Type<u32>::max )
      propagateDepth( a, 0 );
  }

  // Re-distribute the depths from the back
  for( u32 a = 0; a < nEntries_; ++a )
  {
    for( u32 p = nodes_[a].firstParent; p < nodes_[a + 1].firstParent; ++p )
    {
      LookaheadId parentNode = parents_[p];
      int parentDepth = ( (int)nodes_[a].depth ) - 1;
      if( parentDepth > nodes_[parentNode].depth )
        propagateDepth( parentNode, parentDepth );
    }
  }

  maxDepth_ = 0;

  // Verify the consistency. The only important thing is that each successor is on a deeper level than its parents.
  for( u32 a = 0; a < nEntries_; ++a )
  {
    for( u32 p = nodes_[a].firstParent; p < nodes_[a + 1].firstParent; ++p )
      verify( nodes_[parents_[p]].depth < nodes_[a].depth );
    if( nodes_[a].depth > maxDepth_ )
      maxDepth_ = nodes_[a].depth;
  }
  verify( maxDepth_ != Core::Type<u32>::max );
}

std::string LanguageModelLookahead::archiveEntry() const
{
  return isBackwardRecognition( config ) ? "backward-lm-lookahead" : "lm-lookahead";
}

LanguageModelLookahead::Score LanguageModelLookahead::getLmScale() const
{
  Lm::BatchRequest batch;
  Lm::CompiledBatchRequest* req = lm_->compileBatchRequest( batch );
  Score ret = req->scale();
  delete req;
  return ret;
}

void LanguageModelLookahead::buildBatchRequest() {
  require( !batchRequest_ );

  Lm::BatchRequest batch;
  for ( u32 n = 0; n < nEntries_; ++n ) {
    for ( Ends::const_iterator
          e     = ends_.begin() + nodes_[n].firstEnd,
          e_end = ends_.begin() + nodes_[n + 1].firstEnd;
          e != e_end; ++e )
    {
      Lm::Request request( ( *e )->lemma()->syntacticTokenSequence(), n );

      request.offset = endOffsets_[e - ends_.begin()];

      batch.push_back( request );
    }
  }

  Lm::CompiledBatchRequest* req = lm_->compileBatchRequest( batch );
  req->setScale( req->scale() * scale_ );
  batchRequest_ = req;
}

const u32 formatVersion = 0xa8312;

void LanguageModelLookahead::writePersistentCache()
{
  Core::MappedArchiveWriter writer = Core::Application::us()->getCacheArchiveWriter( paramCacheArchive( config ), archiveEntry() );

  if( !writer.good() )
    return;

  log( "writing persistent LM look-ahead cache" );

  u32 checksum = tree_.getChecksum();
  f32 lmScale = getLmScale();

  std::vector<int> mappedEnds;
  for( Ends::const_iterator it = ends_.begin(); it != ends_.end(); ++it )
    mappedEnds.push_back( ( *it )->id() );

  writer << formatVersion << checksum << lmScale << invalidFirstNodeForTokenIndex_ << nEntries_ << maxDepth_;
  writer << firstNodeForToken_ << endOffsets_ << successors_ << parents_ << nodes_;
  writer << nodeForToken_ << nodeId_ << hashForNode_ << hashForState_ << mappedEnds;
}

bool LanguageModelLookahead::readPersistentCache()
{
  Core::MappedArchiveReader reader = Core::Application::us()->getCacheArchiveReader( paramCacheArchive( config ), archiveEntry() );

  if( !reader.good() )
  {
    return false;
  }

  u32 treeChecksum, version;
  reader >> version >> treeChecksum;

  if( treeChecksum != tree_.getChecksum() || version != formatVersion )
  {
    log( "failed loading persistent LM-lookahead cache because the version mismatched" );
    return false;
  }

  f32 lmScale;
  reader >> lmScale;

  if( lmScale != getLmScale() )
  {
    log( "failed loading persistent LM-lookahead cache because the lm-scale mismatched: real %i stored %i", getLmScale(), lmScale );
    return false;
  }

  std::vector<int> mappedEnds;

  reader >> invalidFirstNodeForTokenIndex_ >> nEntries_ >> maxDepth_;

  reader >> firstNodeForToken_ >> endOffsets_ >> successors_ >> parents_ >> nodes_;
  reader >> nodeForToken_ >> nodeId_ >> hashForNode_ >> hashForState_ >> mappedEnds;

  ends_.reserve( mappedEnds.size() );
  for( std::vector<int>::const_iterator it = mappedEnds.begin(); it != mappedEnds.end(); ++it )
    ends_.push_back( lm_->lexicon()->lemmaPronunciation( *it ) );

  verify( nodes_.isConstant() );

  return reader.good();
}

void LanguageModelLookahead::buildLookaheadStructure( HMMStateNetwork& tree, StateId rootNode, const std::vector<PersistentStateTree::Exit>& exits ) {
  log( "building look-ahead structure..." );
  verify( lm_ );
  ConstructionTree ct;

  if( !readPersistentCache() )
  {
    ct.build( tree, rootNode, exits, lm_->lexicon() );

    log( "full look-ahead network: %d nodes", ct.nNodes() );
    ct.writeStatistics( log( "full look-ahead network statistics:\n" ) );

    ct.prune( this );
    ct.purge();

    log( "reduced look-ahead network: %d nodes", ct.nNodes() );
    ct.writeStatistics( log( "reduced look-ahead network statistics:\n" ) );

    buildCompressesLookaheadStructure( 1, tree.stateCount(), ct );

    buildBatchRequest(); // Must be done before buildHash

    buildHash();

    writePersistentCache();
  }else{
    log( "look-ahead was read from mapped cache" );
    buildBatchRequest();
  }

  verify( maxDepth_ != 0 );
  waitingLookaheadNodesByDepth_.resize( maxDepth_ + 1 );

  log( "table size (%d entries): %zd bytes", nEntries_,
    sizeof( ContextLookahead ) + nEntries_ * sizeof( Score ) );

  Core::Channel dc( config, "dot" );
  if ( dc.isOpen() ) draw( dc );
}

void LanguageModelLookahead::draw( std::ostream &os ) const {
  os << "digraph \"" << fullName() << "\" {" << std::endl
     << "ranksep = 1.5" << std::endl
     << "rankdir = LR" << std::endl
     << "node [fontname=\"Helvetica\"]" << std::endl
     << "edge [fontname=\"Helvetica\"]" << std::endl;

  for ( u32 ni = 0; ni < nEntries_; ++ni ) {
    const Node &n( nodes_[ni] );
    os << Core::form( "n%d [label=\"%d\\n", ni, ni );
    for ( StateTree::StateId si = 0; si < StateTree::StateId( nodeId_.size() ); ++si )
      if ( nodeId_[si] == ni ) os << Core::form( "%d ", si );
    for ( u32 e = n.firstEnd; e < nodes_[ni + 1].firstEnd; ++e )
      os << Core::form( "\\n%s", ends_[e]->lemma()->preferredOrthographicForm().str() );
    os << Core::form( "\"]\n" );
    for ( u32 s = n.firstSuccessor; s < nodes_[ni + 1].firstSuccessor; ++s ) {
      os << Core::form( "n%d -> n%d\n", ni, successors_[s] );
    }
  }

  os << "}" << std::endl;
}

// ===========================================================================
// dynamic data and caching

void LanguageModelLookahead::computeScores( const Lm::History &history, std::vector<Score> &scores ) const {
  if( scores.size() == nEntries_ ) {
    std::fill( scores.begin(), scores.end(), Core::Type<Score>::max );
  }else{
    verify( scores.empty() );
    scores.resize( nEntries_, Core::Type<Score>::max );
  }

//  log("computing look-ahead table for history ") << lm_->formatHistory(history);

  lm_->getBatch( history, batchRequest_, scores );

  std::vector<Score>::iterator score = scores.begin();

  if( logSemiringFactor_ )
  {
    Score lmScale = batchRequest_->scale();
    Score invertedLmScale = 1 / lmScale;

    for ( Core::ConstantVector<Node>::const_iterator n = nodes_.begin(); n != nodes_.end() - 1; ++n ) {
      Score sum = *score;
      Score minScore = *score;
      for ( Successors::const_iterator
            s     = successors_.begin() +  n->firstSuccessor,
            s_end = successors_.begin() + ( n + 1 )->firstSuccessor;
            s != s_end; ++s )
      {
        verify_( *s < LookaheadId( score - scores.begin() ) );
        sum = scaledLogAdd( sum, scores[*s], lmScale, invertedLmScale );
        if ( minScore > scores[*s] )
          minScore = scores[*s];
      }

      verify( sum != Core::Type<Score>::max );

      *score++ = ( sum * logSemiringFactor_ + minScore * ( 1.0 - logSemiringFactor_ ) );
    }
  }else{
    for ( Core::ConstantVector<Node>::const_iterator n = nodes_.begin(); n != nodes_.end() - 1; ++n ) {
      Score minScore = *score;
      for ( Successors::const_iterator
            s     = successors_.begin() +  n->firstSuccessor,
            s_end = successors_.begin() + ( n + 1 )->firstSuccessor;
            s != s_end; ++s )
      {
        verify_( *s < LookaheadId( score - scores.begin() ) );
        if ( minScore > scores[*s] )
          minScore = scores[*s];
      }
      *score++ = minScore;
    }
  }
  verify( score == scores.end() );
}

template <bool approx>
bool LanguageModelLookahead::computeScoresSparse( LanguageModelLookahead::ContextLookahead& lookahead ) const {
  const Lm::BackingOffLm* lm = dynamic_cast<const Lm::BackingOffLm*>( lm_->unscaled().get() );
  verify( lm );
  const Lm::History& history( lookahead.history_ );

  Lm::BackingOffLm::BackOffScores backoff = lm->getBackOffScores( history, 0 );
  u32 contextScoreCount = ( ( (size_t)backoff.end ) - ( (size_t)backoff.start ) ) / sizeof( Lm::BackingOffLm::WordScore );

  if( not sparseThresholdExpectationBased_ && contextScoreCount > sparseLookAheadThreshold_ * lm_->lexicon()->nLemmas() )
    return false;

  u32 predictionKey = isqrt( contextScoreCount );

  u32 expectedNodeCount = sparseNodesPrediction_.predict( predictionKey );
  if( expectedNodeCount < 10 )
    expectedNodeCount = 10;

  if( sparseThresholdExpectationBased_ && expectedNodeCount > nEntries_ * sparseLookAheadThreshold_ )
    return false;

  cacheStatistics_->sparseStats.expectedLookAheadNodes += expectedNodeCount;
  ++cacheStatistics_->sparseStats.sparseTables;

  if( approx )
  {
    lookahead.approxSparseScores_.clear( expectedNodeCount * sparseHashSizeFactor_ );
    lookahead.sparseScores_.clear(); ///@todo Not required once tables are managed properly
  }else{
    lookahead.sparseScores_.clear( expectedNodeCount * sparseHashSizeFactor_ );
    lookahead.approxSparseScores_.clear();
  }

  lookahead.scores_.clear();

  bool resized = false;

  cacheStatistics_->sparseStats.totalScoreCount += contextScoreCount;

  u32 insertedSparseScoreSkips = 0;

  Score scale = batchRequest_->scale();
  int historyLength = lm->historyLenght( history );
  cacheStatistics_->sparseStats.potentialLookaheadNodes += nEntries_;

#ifdef ALLOW_CONSIDER_BACK_OFF_IN_MAXIMIZATION

  // A pair of the look-ahead, and the back-off score that needs to be applied to the look-ahead
  std::pair<ContextLookaheadReference, Score> backOffLookAheads[historyLength];

  if( considerBackOffInMaximization_ ) {
    Score backOffScore = backoff.backOffScore * scale;

    for( int a = historyLength - 1; a >= 0; --a )
    {
      Lm::History h = lm->reducedHistory( history, a );
      verify( not ( h == history ) );
      verify( lm->historyLenght( h ) == a );
      ContextLookaheadReference lah = getLookahead( h, false, a != 0 );
      backOffLookAheads[a] = std::make_pair( lah, backOffScore );
      backOffScore += backOffLookAheads[a].first->backOffScore();
    }
  }

  std::map<LookaheadId, int> nonBackoffEndsForNode;   //Only filled if considerBackOffInMaximization_ is true
#endif

  for( u32 d = 0; d < waitingLookaheadNodesByDepth_.size(); ++d )
    waitingLookaheadNodesByDepth_[d].clear();

  {
    // Special tokens like 'silence' that have a score of zero in any context
    Core::ConstantVector<std::pair<LookaheadId, Score> >::const_iterator nodeEnd = nodeForToken_.begin() + firstNodeForToken_[invalidFirstNodeForTokenIndex_ + 1];
    for( Core::ConstantVector<std::pair<LookaheadId, Score> >::const_iterator node = nodeForToken_.begin() + firstNodeForToken_[invalidFirstNodeForTokenIndex_]; node != nodeEnd; ++node )
    {
      LookaheadId nodeIdx = ( *node ).first;
      waitingLookaheadNodesByDepth_[nodes_[nodeIdx].depth].push_back( std::make_pair( nodeIdx, (Score)0 ) );
    }
  }

  for( const Lm::BackingOffLm::WordScore* current = backoff.start; current != backoff.end; ++current )
  {
    Score score = current->score_ * scale;

    verify_( current->token() + 1 <= firstNodeForToken_.size() );

    Core::ConstantVector<std::pair<LookaheadId, Score> >::const_iterator nodeEnd = nodeForToken_.begin() + firstNodeForToken_[current->token() + 1];
    for( Core::ConstantVector<std::pair<LookaheadId, Score> >::const_iterator node = nodeForToken_.begin() + firstNodeForToken_[current->token()]; node != nodeEnd; ++node )
    {
      Score endScore = score + ( *node ).second;
      LookaheadId nodeId = ( *node ).first;

#ifdef ALLOW_CONSIDER_BACK_OFF_IN_MAXIMIZATION

      if( considerBackOffInMaximization_ ) {
        std::map<LookaheadId, int>::iterator it = nonBackoffEndsForNode.find( nodeId );
        if( it == nonBackoffEndsForNode.end() )
          nonBackoffEndsForNode.insert( std::make_pair( nodeId, 1 ) );
        else
          ++it->second;
      }
#endif

      waitingLookaheadNodesByDepth_[nodes_[nodeId].depth].push_back( std::make_pair( nodeId, endScore ) );
    }
  }

  Score invertedScale = 1 / scale;

  if( nodeRecombination_.empty() )
    nodeRecombination_.resize( nEntries_, 0 );

  for( s32 depth = waitingLookaheadNodesByDepth_.size() - 1; depth >= 0; --depth )
  {
    LanguageModelLookahead::LookAheadNodesForDepth& candidatesForDepth( waitingLookaheadNodesByDepth_[depth] );

    u32 outIdx = 0;

    // Recombine the waiting look-ahead scores from this level

    if( logSemiringFactor_ )
    {
      for( u32 candidateIdx = 0; candidateIdx < candidatesForDepth.size(); ++candidateIdx )
      {
        const std::pair<LookaheadId, Score>& candidate( candidatesForDepth[candidateIdx] );

        u32& recombination( nodeRecombination_[candidate.first] );
        if( recombination < candidateIdx && candidatesForDepth[recombination].first == candidate.first )
        {
          candidatesForDepth[recombination].second = scaledLogAdd( candidatesForDepth[recombination].second, candidate.second, scale, invertedScale );
        }
        else
        {
          recombination = outIdx;
          candidatesForDepth[outIdx] = candidate;
          ++outIdx;
        }
      }
    }else{
      for( u32 candidateIdx = 0; candidateIdx < candidatesForDepth.size(); ++candidateIdx )
      {
        const std::pair<LookaheadId, Score>& candidate( candidatesForDepth[candidateIdx] );

        u32& recombination( nodeRecombination_[candidate.first] );
        if( recombination < candidateIdx && candidatesForDepth[recombination].first == candidate.first )
        {
          if( candidate.second < candidatesForDepth[recombination].second )
            candidatesForDepth[recombination].second = candidate.second;
        }
        else
        {
          recombination = outIdx;
          candidatesForDepth[outIdx] = candidate;
          ++outIdx;
        }
      }
    }

    candidatesForDepth.shrink( outIdx );

    // Construct look-ahead nodes from the recombined look-ahead scores

    for( u32 candidateIdx = 0; candidateIdx < outIdx; ++candidateIdx )
    {
      std::pair<LookaheadId, Score>& node( candidatesForDepth[candidateIdx] );

      verify_( nodes_[node.first].depth == depth );

#ifdef ALLOW_CONSIDER_BACK_OFF_IN_MAXIMIZATION
      if( considerBackOffInMaximization_ )
      {
        int nonBackOffEnds = 0;
        std::map<LookaheadId, int>::const_iterator nbi = nonBackoffEndsForNode.find( node.first );
        if( nbi != nonBackoffEndsForNode.end() )
          nonBackOffEnds = nbi->second;

        int ends = states_[node.first + 1].firstEnd - states_[node.first].firstEnd;

        if( nonBackOffEnds != ends )
        {
          for( int e = states_[node.first].firstEnd; e != states_[node.first + 1].firstEnd; ++e )
          {
            Score s = lm_->score( history, ends_[e]->lemma()->syntacticTokenSequence()[0] ) + endOffsets_[e];
            if( s < node.second )
              node.second = s;
          }
        }

        for( u32 succIdx = states_[node.first].firstSuccessor; succIdx < states_[node.first + 1].firstSuccessor; ++succIdx )
        {
          LookaheadId succ = successors_[succIdx];
          if( lookahead.sparseScores_.contains( succ ) )
            continue;
          //Check the back-off look-ahead scores of nodes that are not part of this sparse look-ahead

          //Also consider the back-off scores in the maximization
          for( int a = historyLength - 1; a >= 0; --a )
          {
            Score score = backOffLookAheads[a].first->scoreForLookAheadId( succ );

            if( score != Core::Type<Score>::max )
            {
              score += backOffLookAheads[a].second;

              if( score < node.second )
                node.second = score;

              break;
            }
          }
        }
      }
#endif

      verify_( node.second != Core::Type<Score>::max );

//       if( accumulateNodeObservations_ )
//         nodeObservations_[node.first] += 1;

      // Insert the final score
      if( approx )
      {
        insertedSparseScoreSkips += lookahead.approxSparseScores_.insert( hashForNode_[node.first], node.second );
        u32 newSize = lookahead.approxSparseScores_.checkResize( sparseHashResizeAtFillFraction_ );
        if( newSize )
        {
          lookahead.approxSparseScores_.clear( newSize );
          ++cacheStatistics_->sparseStats.resizedTables;
          resized = true;

          // Re-insert everything inserted until now with higher depth
          for( s32 d = waitingLookaheadNodesByDepth_.size() - 1; d > depth; --d )
          {
            LanguageModelLookahead::LookAheadNodesForDepth& cands( waitingLookaheadNodesByDepth_[d] );
            for( u32 cand = 0; cand < cands.size(); ++cand )
            {
              std::pair<LookaheadId, Score>& n( cands[cand] );
              lookahead.approxSparseScores_.insert( hashForNode_[n.first], n.second );
            }
          }

          // Re-insert everything inserted until now on the same depth
          for( u32 cand = 0; cand <= candidateIdx; ++cand )
          {
            std::pair<LookaheadId, Score>& n( candidatesForDepth[cand] );
            lookahead.approxSparseScores_.insert( hashForNode_[n.first], n.second );
          }
        }
      }else{
        insertedSparseScoreSkips += lookahead.sparseScores_.insert( hashForNode_[node.first], node.second );
        if( lookahead.sparseScores_.checkResize( sparseHashResizeAtFillFraction_ ) )
        {
          ++cacheStatistics_->sparseStats.resizedTables;
          resized = true;
        }
      }

      // Propagate to parents
      Successors::const_iterator parentEnd = parents_.begin() + nodes_[node.first + 1].firstParent;
      for( Successors::const_iterator parent = parents_.begin() + nodes_[node.first].firstParent; parent != parentEnd; ++parent )
      {
        verify( nodes_[*parent].depth < depth );
        waitingLookaheadNodesByDepth_[nodes_[*parent].depth].push_back( std::make_pair( *parent, node.second ) );
      }
    }
  }

  if( resized )
    ++cacheStatistics_->sparseStats.uniqueResizedTables;

  u32 nodeCount = 0;

  if( approx )
  {
    nodeCount = lookahead.approxSparseScores_.size();
    cacheStatistics_->sparseStats.totalHashSize += lookahead.approxSparseScores_.hashSize();
  }else{
    nodeCount = lookahead.sparseScores_.size();
    cacheStatistics_->sparseStats.totalHashSize += lookahead.sparseScores_.hashSize();
  }
  cacheStatistics_->sparseStats.backOffLookaheadNodes += nodeCount;
  sparseNodesPrediction_.add( predictionKey, nodeCount );

  cacheStatistics_->sparseStats.lookAheadNodesExpectationDeviation += ( expectedNodeCount - nodeCount ) * ( expectedNodeCount - nodeCount );

  cacheStatistics_->sparseStats.backOffLookaheadNodeHashIterations += insertedSparseScoreSkips;

  lookahead.backOffScore_ = backoff.backOffScore * scale;

  verify( lookahead.scores_.empty() );

  return true;
}

LanguageModelLookahead::ContextLookahead::ContextLookahead(
  const LanguageModelLookahead *la,
  const Lm::History &_history ) :
  la_( la ),
  history_( _history ),
  freePos_( la->freeTables_.end() ),
  sparseScores_( Core::Type<Score>::max ),
  approxSparseScores_(),
  isFilled_( false ),
  backOffScore_( Core::Type<Score>::max )
{
  verify( Core::Type<Score>::max == F32_MAX );
}

LanguageModelLookahead::ContextLookahead *LanguageModelLookahead::acquireTable(
  const Lm::History &h ) const {
  ContextLookahead *t = 0;
  if ( ( nTables() < cacheSizeLowMark_ ) || !nFreeTables_ ) {
    t = new ContextLookahead( this, h );
    tables_.push_front( t ); t->pos_ = tables_.begin(); ++nTables_;
  } else {
    t = freeTables_.back(); freeTables_.pop_back(); --nFreeTables_;
    t->freePos_ = freeTables_.end();
    map_.erase( t->history_ );
    t->history_ = h;
  }
  ensure( t->history_ == h );
  ensure( t->isActive() );
  return t;
}

void LanguageModelLookahead::releaseTable( const ContextLookahead *ct ) const {
  ContextLookahead *t = const_cast<ContextLookahead*>( ct );
  require( t->isActive() );
  if ( nTables() > cacheSizeHighMark_ ) {
    if ( nFreeTables_ ) {
      freeTables_.push_front( t ); t->freePos_ = freeTables_.begin();
      t = freeTables_.back();  freeTables_.pop_back();
    }
    verify( *t->pos_ == t );
    tables_.erase( t->pos_ ); --nTables_;
    map_.erase( t->history_ );
    delete t;
  } else {
    freeTables_.push_front( t ); t->freePos_ = freeTables_.begin(); ++nFreeTables_;
  }
}

LanguageModelLookahead::ContextLookahead *LanguageModelLookahead::getCachedTable( const Lm::History &h ) const {
  ContextLookahead *t = 0;

  Map::const_iterator i = map_.find( h );
  if ( i != map_.end() ) {
    t = i->second;
    if ( t->freePos_ != freeTables_.end() ) {
      cacheStatistics_->cacheEvents += CacheStatistics::freeCacheHit;
      verify( *t->freePos_ == t );
      freeTables_.erase( t->freePos_ ); --nFreeTables_;
      t->freePos_ = freeTables_.end();
    } else {
      cacheStatistics_->cacheEvents += CacheStatistics::shareInCacheHit;
    }
  }

  return t;
}

Lm::History LanguageModelLookahead::getReducedHistory( const Lm::History& history ) const {
  if( historyLimit_ == -1 )
    return history;
  else
    return lm_->reducedHistory( history, historyLimit_ );
}

LanguageModelLookahead::ContextLookaheadReference
LanguageModelLookahead::getLookahead( const Lm::History& fh, bool noHistoryLimit ) const {
  Lm::History h( ( noHistoryLimit || historyLimit_ == -1 ) ? fh : lm_->reducedHistory( fh, historyLimit_ ) );

  ContextLookahead *t = getCachedTable( h );
  if ( !t ) {
    cacheStatistics_->cacheEvents += CacheStatistics::cacheMiss;
    map_[h] = t = acquireTable( h );

    t->isFilled_ = false;
    t->history_ = h;
  }

  ensure( t->history_ == h );
  ensure( t->isActive() );

  return ContextLookaheadReference( t );
}

void LanguageModelLookahead::fill( ContextLookaheadReference lookahead, bool sparse, bool approx ) {
  if( lookahead->isFilled_ )
    return;

  ContextLookahead *t = const_cast<ContextLookahead*>( lookahead.get() );

  if( lookahead->isFilled_ )
    return;  ///@todo If another thread is filling this table, wait

  if( sparse )
  {
    //Only really use sparse look-ahead if the history is not empty
    const Lm::BackingOffLm* lm = dynamic_cast<const Lm::BackingOffLm*>( lm_->unscaled().get() );
    verify( lm ); // Sparse look-ahead is only supported with a backing-off LM
    if( lm->historyLenght( lookahead->history_ ) == 0 )
      sparse = false;
  }

  if( not sparse or not ( ( approx && computeScoresSparse<true>( *t ) ) || ( !approx && computeScoresSparse<false>( *t ) ) ) )
  {
    t->sparseScores_.clear();
    t->approxSparseScores_.clear();
    computeScores( t->history_, t->scores_ );
  }

  t->isFilled_ = true;
}

void LanguageModelLookahead::fillZero( ContextLookaheadReference lookahead ) {
  ContextLookahead *t = const_cast<ContextLookahead*>( lookahead.get() );

  t->sparseScores_.clear();
  t->approxSparseScores_.clear();
  t->scores_.assign( nEntries_, 0 );

  t->isFilled_ = true;
}

LanguageModelLookahead::ContextLookaheadReference
LanguageModelLookahead::tryToGetLookahead( const Lm::History& fh, bool noHistoryLimit ) const {
  Lm::History h( noHistoryLimit ? fh : getReducedHistory( fh ) );

  ContextLookahead *t = getCachedTable( h );

  ensure( !t || t->history_ == h );
  ensure( !t || t->isActive() );

  if ( t && t->isFilled_ )
    return ContextLookaheadReference( t );
  else
    return ContextLookaheadReference();
}

const Core::Choice LanguageModelLookahead::CacheStatistics::cacheEventChoice(
  "cache hits on active tables  ", shareInCacheHit,
  "cache hits on inactive tables", freeCacheHit,
  "number of table calculations ", cacheMiss,
  Core::Choice::endMark() );

void SparseStatistics::write( Core::XmlWriter& w ) const {
  if ( sparseTables )
  {
    w << Core::XmlOpen( "language-model-lookahead-sparse-statistics" );
    w << " potential lookahead nodes: " << potentialLookaheadNodes << "  back off nodes: " << backOffLookaheadNodes << " number of scores: " << totalScoreCount;
    if ( backOffLookaheadNodes )
      w << " average lookahead hash clash iterations: " << ( ( (double)backOffLookaheadNodeHashIterations ) / backOffLookaheadNodes );
    if ( totalHashSize )
      w << " average hash fill: " << ( ( (double)backOffLookaheadNodes ) / totalHashSize );
    if ( sparseTables )
      w << " node-count expectation standard deviation: " << sqrt( ( (double)lookAheadNodesExpectationDeviation ) / sparseTables );
    if ( sparseTables )
      w << " computed sparse tables: " << sparseTables <<  " resized tables percentage: " << ( ( (double)resizedTables ) / sparseTables ) << " unique: " << ( ( (double)uniqueResizedTables ) / sparseTables );
    w << Core::XmlClose( "language-model-lookahead-sparse-statistics" );
  }
}

void SparseStatistics::clear() {
  totalScoreCount = 0;
  potentialLookaheadNodes = 0;
  backOffLookaheadNodes = 0;
  backOffLookaheadNodeHashIterations = 0;
  totalHashSize = 0;
  expectedLookAheadNodes = 0;
  lookAheadNodesExpectationDeviation = 0;
  sparseTables = 0;
  resizedTables = 0;
  uniqueResizedTables = 0;
}
void LanguageModelLookahead::CacheStatistics::clear() {
  cacheEvents.clear();
  nTables.clear();
  nActiveTables.clear();
  sparseStats.clear();
}

void LanguageModelLookahead::CacheStatistics::write( Core::XmlWriter &os ) const {
  os << Core::XmlOpen( "language-model-lookahead-cache-statistics" )
     << cacheEvents
     << nActiveTables
     << nTables
     << Core::XmlClose( "language-model-lookahead-cache-statistics" );
  sparseStats.write( os );
}

LanguageModelLookahead::CacheStatistics::CacheStatistics() :
  cacheEvents( "look-ahead requests", cacheEventChoice ),
  nTables( "number of tables in memory" ),
  nActiveTables( "number of active tables" ) {
  clear();
}

void LanguageModelLookahead::collectStatistics() const {
  cacheStatistics_->nTables       += nTables();
  cacheStatistics_->nActiveTables += nActiveTables();
}

void LanguageModelLookahead::logStatistics() const {
  if ( statisticsChannel_.isOpen() )
    cacheStatistics_->write( statisticsChannel_ );
}

LanguageModelLookahead::LookaheadId LanguageModelLookahead::lastNodeOnDepth( int depth ) const {
  verify( depth < 100000 );

  LanguageModelLookahead::LookaheadId ret = 0;
  for( int a = 0; a < nEntries_; ++a )
  {
    if( nodes_[a].depth == depth )
      ret = a;
  }

  if( ret == 0 )
    return lastNodeOnDepth( depth + 1 );

  return ret;
}

int LanguageModelLookahead::nodeDepth( LanguageModelLookahead::LookaheadId node ) const {
  return nodes_[node].depth;
}
