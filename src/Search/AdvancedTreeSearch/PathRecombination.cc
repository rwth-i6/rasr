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
#include "PathRecombination.hh"

using namespace Search;

const Core::ParameterString paramCache(
        "recombination-pruning-cache",
        "",
        "");

const Core::ParameterFloat paramConvergenceFactor(
        "path-recombination-pruning-convergence-factor",
        "",
        1.5);

const Core::ParameterFloat paramDelta(
        "path-recombination-pruning-delta",
        "",
        1.0);

const Core::ParameterInt paramMaxCacheSize(
        "path-recombination-max-cache-size",
        "",
        5000000);

const Core::ParameterInt paramMaxDepth(
        "path-recombination-max-depth",
        "maximum depth, starting at zero (eg. max-depth 1 equals 2 levels: level 0 and level 1)",
        5);

const Core::ParameterInt paramMaxExactInterval(
        "path-recombination-max-exact-interval",
        "maximum interval-length up to which the interval should be computed exactly",
        50);

const Core::ParameterBool paramTruncateNotPromisingPaths(
        "path-recombination-truncate-not-promising",
        "this is slow when the network is properly compressed",
        false);

const Core::ParameterInt paramPromisingApproximation(
        "path-recombination-promising-approximation",
        "",
        0);

const Core::ParameterBool paramApproximateLinearSequences(
        "path-recombination-approximate-linear-sequences",
        "",
        true);

void PathRecombination::logStatistics() {
    if (nVisits_) {
        std::cout << "average path-recombination visits: " << (totalVisits_ / nVisits_) << " expensive interval-computations " << nVisits_ << std::endl;
        Core::Application::us()->log() << "average path-recombination visits: " << (totalVisits_ / nVisits_) << " expensive interval-computations " << nVisits_;
    }
}

PathRecombination::PathRecombination(const Search::PersistentStateTree& network, const Core::Configuration& config)
        : network_(network),
          distancesPtr_(0),
          offsetAndDistanceStateForStatePtr_(0),
          currentVisits_(0),
          nVisits_(0),
          totalVisits_(0) {
    convergenceFactor_          = paramConvergenceFactor(config);
    delta_                      = paramDelta(config);
    cache_                      = paramCache(config);
    truncateNotPromising_       = paramTruncateNotPromisingPaths(config);
    approximateLinearSequences_ = paramApproximateLinearSequences(config);
    maxExactInterval_           = paramMaxExactInterval(config);
    promisingApproximation_     = paramPromisingApproximation(config);

    maxCacheSize_ = paramMaxCacheSize(config);
    maxDepth_     = paramMaxDepth(config);

    asymmetryFactor_ = delta_ * convergenceFactor_ - delta_ / convergenceFactor_;
    std::cout << "path-recombination delta " << delta_ << " convergence " << convergenceFactor_ << " asymmetry " << asymmetryFactor_ << std::endl;

    std::cout << "building recombination states" << std::endl;
    buildRecombinationStates();
    std::cout << "connecting recombination states" << std::endl;
    buildRecombinationNetwork();
    std::cout << "computing recombination distances" << std::endl;
    buildDistances();
    std::cout << "path-recombination ready" << std::endl;
}

void PathRecombination::buildRecombinationStates() {
    std::vector<u32> fanIn(network_.structure.stateCount(), 0);

    for (StateId state = 1; state < network_.structure.stateCount(); ++state)
        for (HMMStateNetwork::SuccessorIterator targ = network_.structure.successors(state); targ; ++targ)
            if (!targ.isLabel())
                fanIn[*targ] += 1;

    recombinationStateMap_.resize(network_.structure.stateCount(), 0);

    for (std::set<StateId>::iterator it = network_.coarticulatedRootStates.begin(); it != network_.coarticulatedRootStates.end(); ++it)
        for (HMMStateNetwork::SuccessorIterator targ = network_.structure.successors(*it); targ; ++targ)
            if (!targ.isLabel())
                recombinationStateMap_[*targ] = 1;

    for (HMMStateNetwork::SuccessorIterator targ = network_.structure.successors(network_.rootState); targ; ++targ)
        if (!targ.isLabel())
            recombinationStateMap_[*targ] = 1;

    for (HMMStateNetwork::SuccessorIterator targ = network_.structure.successors(network_.ciRootState); targ; ++targ)
        if (!targ.isLabel())
            recombinationStateMap_[*targ] = 1;

    recombinationStates_.push_back(RecombinationState());
    for (StateId state = 1; state < network_.structure.stateCount(); ++state) {
        if (fanIn[state] > 1)
            recombinationStateMap_[state] = 1;

        if (recombinationStateMap_[state]) {
            recombinationStateMap_[state] = recombinationStates_.size();
            recombinationStates_.push_back(RecombinationState());
            recombinationStates_.back().state = state;
        }
    }
    Core::Application::us()->log() << "recombination-states: " << recombinationStates_.size() - 1;

    visitingRecombinationState_.resize(recombinationStates_.size(), false);
}

void PathRecombination::buildRecombinationNetwork() {
#if 0
  u32 statesWithSelfLoop = 0;

  // Build the compressed network consisting only of recombination-states
  for( u32 recombinationState = 1; recombinationState < recombinationStates_.size(); ++recombinationState )
  {
    struct Propagator {
      Propagator( PathRecombination& r, u32 recombinationState ) :
        r_( r ),
        successors_( r.recombinationStates_[recombinationState].successors ),
        startState_( r.recombinationStates_[recombinationState].state ),
        visited_( 0 ) {
        propagateSuccessors( startState_, 0 );
      }

      void propagate( StateId state, u32 distance ) {
        if( state == startState_ )
          return;

        ++visited_;

        if( r_.recombinationStateMap_[state] )
        {
          if( hadSuccessors_.count( state ) == 0 )
          {
            hadSuccessors_.insert( std::make_pair<StateId, u32>( state, successors_.size() ) );
            successors_.push_back( RecombinationState::Successor() );
            successors_.back().state = state;
          }

          RecombinationState::Successor& succ = successors_[hadSuccessors_[state]];

          if( distance < succ.shortestDistance )
            succ.shortestDistance = distance;
          if( distance > succ.longestDistance )
            succ.longestDistance = distance;
        }else{
          propagateSuccessors( state, distance );
        }
      }

      void propagateSuccessors( StateId state, u32 distance ) {
        for( HMMStateNetwork::SuccessorIterator it = r_.network_.structure.successors( state ); it; ++it )
        {
          if( it.isLabel() )
          {
            verify( it.label() < r_.network_.exits.size() );

            StateId transit = r_.network_.exits[it.label()].transitState;

            for( HMMStateNetwork::SuccessorIterator it2 = r_.network_.structure.successors( transit ); it2; ++it2 )
            {
              verify( !it2.isLabel() ); // Ignore epsilon pronunciations for now
              propagate( *it2, distance + 1 );
            }
          }else{
            propagate( *it, distance + 1 );
          }
        }
      }

      PathRecombination& r_;
      std::vector<RecombinationState::Successor>& successors_;
      StateId startState_;
      std::map<StateId, u32> hadSuccessors_;
      u32 visited_;
    } propagator( *this, recombinationState );

    u32 shortestDistance = Core::Type<u32>::max, longestDistance = 0;
    for( std::vector<RecombinationState::Successor>::iterator it = recombinationStates_[recombinationState].successors.begin(); it != recombinationStates_[recombinationState].successors.end(); ++it )
    {
      longestDistance = std::max( longestDistance, it->longestDistance );
      shortestDistance = std::min( shortestDistance, it->shortestDistance );
      if( ( *it ).state == recombinationStates_[recombinationState].state )
      {
        recombinationStates_[recombinationState].loop = true;
        ++statesWithSelfLoop;
      }
    }

//     std::cout << "found successors for recombination-state " << recombinationState << " of " << recombinationStates_.size() << ": " << recombinationStates_[recombinationState].successors.size() << " longest distance " << longestDistance << " shortest distance " << shortestDistance << " visited " << propagator.visited_ << std::endl;
  }
  std::cout << "Total recombination-states: " << recombinationStates_.size() - 1 << " with self-loop: " << statesWithSelfLoop << std::endl;
#endif
}

StateId PathRecombination::getUniqueSuccessor(StateId state) const {
    HMMStateNetwork::SuccessorIterator it = network_.structure.successors(state);
    if (it.countToEnd() == 1) {
        if (it.isLabel()) {
            HMMStateNetwork::SuccessorIterator it2 = network_.structure.successors(network_.exits[it.label()].transitState);
            if (it2.countToEnd() == 1) {
                verify(!it2.isLabel());
                return *it2;
            }
        }
        else {
            return *it;
        }
    }
    return 0;
}

struct VisitManager {
    VisitManager()
            : visitingToken(0) {
    }

    void next() {
        ++visitingToken;
    }

    // Returns the depth with which the position was already visited, or Core::Type<u32>::max
    u32 visited(u32 pos) const {
        const_cast<VisitManager&>(*this).updateArray(pos);
        if (visitedArray[pos].first == visitingToken)
            return visitedArray[pos].second;
        else
            return Core::Type<u32>::max;
    }

    void visit(u32 pos, u32 depth) {
        updateArray(pos);
        if (visited(pos) <= depth)
            return;
        visitedArray[pos].first  = visitingToken;
        visitedArray[pos].second = depth;
    }

    void updateArray(u32 pos) {
        if (visitedArray.size() <= pos)
            visitedArray.resize(pos + 1000, std::make_pair<u32, u32>(0, 0));
    }

    std::vector<std::pair<u32, u32>> visitedArray;
    u32                              visitingToken;
} manager;

struct SuccessorPlan {
    SuccessorPlan(StateId _state = 0, u32 _distance = 0, bool _direct = false)
            : state(_state),
              distance(_distance),
              direct(_direct) {
    }
    StateId state;
    u32     distance;
    bool    direct;
};

void PathRecombination::computeDistancesForState(StateId state) {
#if 0
  if( offsetAndDistanceStateForState_[state].second != 0 )
    return;  // Return if already computed

  {
    // Check if there is a unique successor, if yes, re-use its distance-state with
    // an additional offset of 1
    StateId uniqueSuccessor = getUniqueSuccessor( state );

    if( uniqueSuccessor )
    {
      computeDistancesForState( uniqueSuccessor );
      offsetAndDistanceStateForState_[state] = offsetAndDistanceStateForState_[uniqueSuccessor];
      offsetAndDistanceStateForState_[state].first += 1;   // Increase offset by 1
      return;
    }
  }

//   std::cout << "computing distances for state " << state << std::endl;

  // Compute the longest and shortest distance to each recombination-state
  offsetAndDistanceStateForState_[state].first = 0;
  offsetAndDistanceStateForState_[state].second = distances_.size();
  distances_.push_back( DistanceState() );

  DistanceState& distanceState = distances_.back();
  distanceState.distances.resize( recombinationStates_.size() );

  struct Propagator {
    Propagator( PathRecombination& r, StateId startState, DistanceState& distanceState ) : r_( r ), successors_( r.recombinationStates_[startState].successors ), startState_( startState ), distanceState_( distanceState ), visits_( 0 ), directVisits_( 0 ) {
      manager.next();
      if( r_.recombinationStateMap_[startState] )
        distanceState_.distances[r_.recombinationStateMap_[startState]].shortestDistance = 0;
      propagateSuccessors( startState, 0, true );

      while( !queue_.empty() )
      {
        propagateSuccessors( queue_.front().state, queue_.front().distance, queue_.front().direct );
        queue_.pop_front();
      }

      for( std::vector<DistanceState::DistanceItem>::iterator it = ++distanceState_.distances.begin(); it != distanceState_.distances.end(); ++it )
        verify( ( *it ).shortestDistance != Core::Type<u32>::max );     // All recombination-states must have been visited

      std::cout << "total visits: " << visits_ << " direct " << directVisits_ << std::endl;
    }

    void propagate( StateId state, u32 distance, bool direct ) {
      if( state == startState_ )
        return;

      ++visits_;
      if( direct )
        ++directVisits_;

      u32 recombinationState = r_.recombinationStateMap_[state];

      verify( direct || recombinationState );

      if( recombinationState )
      {
        verify( r_.recombinationStates_[recombinationState].state == state );
        verify( recombinationState < distanceState_.distances.size() );

        DistanceState::DistanceItem& item = distanceState_.distances[recombinationState];

        if( direct )
        {
          // Manage the direct-successor list
          std::map<u32, u32>::iterator it = directSuccessorMap_.find( state );
          if( it != directSuccessorMap_.end() )
          {
            // Eventually update longest distance to the successor
            if( distance > distanceState_.directSuccessorStates[( *it ).second].first )
              distanceState_.directSuccessorStates[( *it ).second].first = distance;
          }else{
            // Insert to the successor-map
            distanceState_.directSuccessorStates.push_back( std::make_pair( distance, recombinationState ) );
            directSuccessorMap_.insert( std::make_pair<u32, u32>( state, distanceState_.directSuccessorStates.size() - 1 ) );
          }
        }

        if( distance < item.shortestDistance )
        {
          item.shortestDistance = distance;
        }else{
          return;         // No need to continue propagation
        }
      }

      queue_.push_back( SuccessorPlan( state, distance, direct && recombinationState == 0 ) );
    }

    void propagateSuccessors( StateId state, u32 distance, bool direct ) {
      u32 recombinationState = r_.recombinationStateMap_[state];

      if( recombinationState )
      {
        for( std::vector<RecombinationState::Successor>::iterator it = r_.recombinationStates_[recombinationState].successors.begin(); it != r_.recombinationStates_[recombinationState].successors.end(); ++it )
        {
          verify( r_.recombinationStateMap_[it->state] );
          propagate( it->state, distance + it->shortestDistance, direct );

          if( direct )         // If direct, we also record the longest distance
            propagate( it->state, distance + it->longestDistance, direct );
        }
      }else{
        for( HMMStateNetwork::SuccessorIterator it = r_.network_.structure.successors( state ); it; ++it )
        {
          if( it.isLabel() )
          {
            verify( it.label() < r_.network_.exits.size() );

            StateId transit = r_.network_.exits[it.label()].transitState;

            for( HMMStateNetwork::SuccessorIterator it2 = r_.network_.structure.successors( transit ); it2; ++it2 )
            {
              verify( !it2.isLabel() );         // Ignore epsilon pronunciations for now
              propagate( *it2, distance + 1, direct );
            }
          }else{
            propagate( *it, distance + 1, direct );
          }
        }
      }
    }

    PathRecombination& r_;
    std::vector<RecombinationState::Successor>& successors_;
    StateId startState_;
    DistanceState& distanceState_;
    std::map<u32, u32> directSuccessorMap_;
    std::deque<SuccessorPlan> queue_;
    u32 visits_;
    u32 directVisits_;
  } propagator( *this, state, distanceState );
#endif
}

void PathRecombination::buildDistances() {
    offsetAndDistanceStateForState_.resize(network_.structure.stateCount(), std::make_pair<u32, u32>(0, 0));
#if 0
  if( !cache_.empty() && std::ifstream( cache_.c_str(), std::ifstream::in | std::ifstream::binary ).good() )
  {
    std::cout << "reading distances from cache " << cache_ << std::endl;
    std::ifstream cache( cache_.c_str(), std::ifstream::in | std::ifstream::binary );
    cache.read( (char*)offsetAndDistanceStateForState_.data(), sizeof( offsetAndDistanceStateForState_.front() ) * network_.structure.stateCount() );
    verify( cache.good() );
    u32 numDistances = 0;
    cache >> numDistances;
    verify( cache.good() );

    mappedCacheFile_.load( cache_.c_str() );
    distancesPtr_ = mappedCacheFile_.data<const DistanceState>( cache.tellg() );
    for( u32 d = 0; d < numDistances; ++d )
    {
      distances_.push_back( DistanceState() );
      readVector( cache, distances_.back().directSuccessorStates );
      readVector( cache, distances_.back().distances );
      verify( cache.good() );
    }

    distancesPtr_ = distances_.data();
    offsetAndDistanceStateForStatePtr_ = offsetAndDistanceStateForState_.data();

    return;
  }
#endif

    distances_.push_back(DistanceState());

    for (u32 state = 1; state < network_.structure.stateCount(); ++state) {
        std::cout << "building distance for " << state << " distance-states: " << distances_.size() << std::endl;
        computeDistancesForState(state);
    }

#if 0
  if( !cache_.empty() )
  {
    std::cout << "writing distances to cache " << cache_ << std::endl;
    std::ofstream cache( cache_.c_str(), std::ofstream::out | std::ofstream::binary );
    cache.write( (char*)offsetAndDistanceStateForState_.data(), sizeof( offsetAndDistanceStateForState_.front() ) * network_.structure.stateCount() );
    verify( cache.good() );
    u32 numDistances = distances_.size();
    cache << numDistances;
    verify( cache.good() );
    for( u32 d = 0; d < numDistances; ++d )
    {
      writeVector( cache, distances_[d].directSuccessorStates );
      writeVector( cache, distances_[d].distances );
      verify( cache.good() );
    }

    distancesPtr_ = distances_.data();
    offsetAndDistanceStateForStatePtr_ = offsetAndDistanceStateForState_.data();
    return;
  }
#endif
}

u32 PathRecombination::recombinationInterval(StateId a, StateId b) const {
    verify(distancesPtr_ && offsetAndDistanceStateForStatePtr_);
    {
        // Try asymetric linear recombination
        u32 linear = linearChainLength(a, b);
        if (linear == Core::Type<u32>::max)
            linear = linearChainLength(b, a);

        if (linear != Core::Type<u32>::max)
            return t(linear, 0);
    }

    u32                  offsetA = offsetAndDistanceStateForStatePtr_[a].first, offsetB = offsetAndDistanceStateForStatePtr_[b].first;
    const DistanceState& distancesA = distancesPtr_[offsetAndDistanceStateForStatePtr_[a].second];
    const DistanceState& distancesB = distancesPtr_[offsetAndDistanceStateForStatePtr_[b].second];

    if (approximateLinearSequences_ && (offsetA || offsetB))
        return t(offsetA, offsetB) + recombinationInterval(offsetA ? getUniqueSuccessor(a) : a, offsetB ? getUniqueSuccessor(b) : b);

    {
        // Try using the cache
        IntervalCache::const_iterator cacheIt = intervalCache_.find(std::make_pair(a, b));
        if (cacheIt != intervalCache_.end())
            return cacheIt->second;

        cacheIt = intervalCache_.find(std::make_pair(b, a));
        if (cacheIt != intervalCache_.end())
            return cacheIt->second;
    }

    manager.next();

    currentVisits_ = 0;

    u32 maxInterval = 0;

    if (recombinationStateMap_[b]) {
        maxInterval = const_cast<PathRecombination&>(*this).R(offsetA, distancesA, offsetB, distancesB, recombinationStateMap_[b], 0);
    }
    else {
        for (std::vector<std::pair<u32, u32>>::const_iterator it = distancesB.directSuccessorStates.begin(); it != distancesB.directSuccessorStates.end(); ++it) {
            if (it->first > 100)
                std::cout << "distance to direct successor " << it->first << std::endl;

            u32 interval = const_cast<PathRecombination&>(*this).R(offsetA, distancesA, offsetB + it->first, distancesB, it->second, 0);

            if (interval > maxInterval)
                maxInterval = interval;
        }
    }

    if (intervalCache_.size() > maxCacheSize_) {
        std::cout << "clearing interval cache" << std::endl;
        intervalCache_.clear();
    }

    intervalCache_.insert(std::make_pair(std::make_pair(a, b), maxInterval));

    totalVisits_ += currentVisits_;
    nVisits_ += 1;

    return maxInterval;
}

bool PathRecombination::isNotPromising(u32 recombinationState, const PathRecombination::DistanceState& distancesA) {
    u32 distanceIdx = &distancesA - distancesPtr_;

    {
        PromisingCache::const_iterator it = promisingCache_.find(std::make_pair(recombinationState, distanceIdx));
        if (it != promisingCache_.end())
            return (*it).second;
    }

    verify(recombinationState);

    const DistanceState& distRec(distancesPtr_[offsetAndDistanceStateForStatePtr_[recombinationStates_[recombinationState].state].second]);
    verify(distRec.distances.size() == distancesA.distances.size());

    u32 distFromA = distancesA.distances[recombinationState].shortestDistance;

    u32 minOtherDist = Core::Type<u32>::max;
    for (u32 otherRecState = 1; otherRecState < distRec.distances.size(); ++otherRecState) {
        u32 dist = distancesA.distances[otherRecState].shortestDistance + distRec.distances[otherRecState].shortestDistance;
        if (dist < minOtherDist)
            minOtherDist = dist;
    }

    bool ret = distFromA <= minOtherDist + promisingApproximation_;

    if (promisingCache_.size() > maxCacheSize_)
        promisingCache_.clear();

    promisingCache_.insert(std::make_pair(std::make_pair(recombinationState, distanceIdx), ret));

    return ret;
}

u32 PathRecombination::R(u32 offsetA, const DistanceState& distancesA, u32 distB, const DistanceState& distancesB, u32 recombinationState, u32 depth) {
    ++currentVisits_;

    verify(recombinationState != 0);

    if (visitingRecombinationState_[recombinationState])
        return Core::Type<u32>::max;

    visitingRecombinationState_[recombinationState] = true;

    u32 distA = offsetA + distancesA.distances[recombinationState].shortestDistance;

    u32 intervalHere = t(distA, distB);

    u32 intervalNext = 0;

    const RecombinationState& recState(recombinationStates_[recombinationState]);

    if (recState.loop || depth >= maxDepth_ || (truncateNotPromising_ && isNotPromising(recombinationState, distancesA)) || distB >= maxExactInterval_) {
        intervalNext = Core::Type<u32>::max;
    }
    else {
        for (std::vector<RecombinationState::Successor>::const_iterator succ = recState.successors.begin(); succ != recState.successors.end(); ++succ) {
            u32 interval = R(offsetA, distancesA, distB + succ->longestDistance, distancesB, recombinationStateMap_[succ->state], depth + 1);
            if (interval > intervalNext)
                intervalNext = interval;
        }
    }

    visitingRecombinationState_[recombinationState] = false;

    return std::min(intervalHere, intervalNext);
}
