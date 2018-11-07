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
#include "PathRecombinationApproximation.hh"

using namespace Search;

Core::ParameterIntVector paramCliqueSizes( "path-recombination-approximation-clique-sizes",
  "",
  "," );

const int maxIterations = 3;
const int switchRandomModulo = 10;

PathRecombinationApproximation::PathRecombinationApproximation( const PersistentStateTree& network, const Core::Configuration& config, const PathRecombination& pathrec )
  : network_( network ), pathrec_( pathrec ) {
  cliqueSizes_ = paramCliqueSizes( config );
  initialize();
}

PathRecombinationApproximation::~PathRecombinationApproximation() {
  for( std::map<u32, CliquePartition*>::const_iterator it = partitionForCliqueSize_.begin(); it != partitionForCliqueSize_.end(); ++it )
    delete ( *it ).second;
}

void PathRecombinationApproximation::initialize() {
  for( std::vector<s32>::const_iterator cliqueSizeIt = cliqueSizes_.begin(); cliqueSizeIt != cliqueSizes_.end(); ++cliqueSizeIt )
    partitionForCliqueSize_.insert( std::make_pair( *cliqueSizeIt, new CliquePartition( network_, pathrec_, *cliqueSizeIt ) ) );
}

PathRecombinationApproximation::CliquePartition::CliquePartition( const PersistentStateTree& network, const PathRecombination& pathrec, u32 cliqueSize )
  : network_( network ),
  pathrec_( pathrec ),
  nCliques_( 0 ) {
  std::cout << "computing cliques for size " << cliqueSize << std::endl;
  cliqueForState_.resize( network_.structure.stateCount(), Core::Type<u32>::max );

  u32 itemsInLastClique = cliqueSize;

  for( StateId state = 1; state < network_.structure.stateCount(); ++state )
  {
    if( itemsInLastClique == cliqueSize )
    {
      ++nCliques_;
      statesForClique_.resize( nCliques_ );
      itemsInLastClique = 0;
    }

/*    if(statesForClique_[nCliques_ - 1].empty())
    {*/
    cliqueForState_[state] = nCliques_ - 1;
    statesForClique_[nCliques_ - 1].insert( state );
//     }

    ++itemsInLastClique;
  }

  recombinationIntervalForClique.resize( nCliques_, Core::Type<u32>::max );
  recombinationIntervalForCliqueWithoutState.resize( network_.structure.stateCount(), Core::Type<u32>::max );

  u32 iteration = 0;

  while( iteration < maxIterations )
  {
    ++iteration;
    std::cout << "iteration " << iteration << std::endl;

    for( StateId own = 1; own < network_.structure.stateCount(); ++own )
    {
      u32 ownClique = cliqueForState_[own];

      StateId bestOther = 0;
      s32 bestImprovement = 0;
      u32 bestOtherToOwnCliqueInterval = Core::Type<u32>::max;
      u32 OwnToOtherCliqueInterval = Core::Type<u32>::max;

      for( StateId other = 1; other < network_.structure.stateCount(); ++other )
      {
        u32 otherClique = cliqueForState_[other];

        if( ( rand() % switchRandomModulo ) != 0 || otherClique == ownClique )
          continue;

        s32 otherToOwnCliqueInterval = symmetricStateCliqueRecombinationInterval( ownClique, other );
        s32 ownToOtherCliqueInterval = symmetricStateCliqueRecombinationInterval( otherClique, own );

        s32 ownCliqueOldInterval = cliqueRecombinationInterval( ownClique );
        s32 ownCliqueReplacedInterval = std::max( (s32)cliqueWithoutStateRecombinationInterval( ownClique, own ), otherToOwnCliqueInterval );

        s32 otherCliqueOldInterval = cliqueRecombinationInterval( otherClique );
        s32 otherCliqueReplacedInterval = std::max( (s32)cliqueWithoutStateRecombinationInterval( otherClique, other ), ownToOtherCliqueInterval );

        s32 improvement = ( ownCliqueOldInterval - ownCliqueReplacedInterval ) + ( otherCliqueOldInterval - otherCliqueReplacedInterval );
        if( improvement > bestImprovement )
        {
          bestImprovement = improvement;
          bestOther = other;
          bestOtherToOwnCliqueInterval = otherToOwnCliqueInterval;
          OwnToOtherCliqueInterval = ownToOtherCliqueInterval;
        }
      }

      u64 totalRecombinationInterval = 0;
      for( u32 clique = 0; clique < nCliques_; ++clique )
        totalRecombinationInterval += cliqueRecombinationInterval( clique );

      if( !bestOther )
      {
        std::cout << "left state " << own << " in same clique " << ownClique << " clique size " << ( statesForClique_[ownClique].size() ) << std::endl;
      }else{
        u32 otherClique = cliqueForState_[bestOther];
        removeFromClique( ownClique, own );
        removeFromClique( otherClique, bestOther );
        addToClique( ownClique, bestOther, bestOtherToOwnCliqueInterval );
        addToClique( otherClique, own, OwnToOtherCliqueInterval );
        std::cout << "switched state " << own << " with state " << bestOther << " improvement " << bestImprovement << " total interval " << totalRecombinationInterval << std::endl;
      }
    }
  }

  for( u32 clique = 0; clique < nCliques_; ++clique )
    std::cout << "items in clique " << clique << ": " << statesForClique_[clique].size() << std::endl;
}

u32 PathRecombinationApproximation::CliquePartition::cliqueRecombinationInterval( u32 clique ) {
  if( recombinationIntervalForClique[clique] != Core::Type<u32>::max )
    return recombinationIntervalForClique[clique];

  const std::set<u32>& states( statesForClique_[clique] );
  u32 max = 0;
  for( std::set<u32>::iterator it = states.begin(); it != states.end(); ++it )
    for( std::set<u32>::iterator it2 = states.begin(); it2 != states.end(); ++it2 )
      max = std::max( max, pathrec_.recombinationInterval( *it, *it2 ) );

  recombinationIntervalForClique[clique] = max;

  return max;
}

u32 PathRecombinationApproximation::CliquePartition::symmetricStateCliqueRecombinationInterval( u32 clique, StateId state ) {
  u32 max = 0;
  const std::set<u32>& states( statesForClique_[clique] );
  for( std::set<u32>::iterator it = states.begin(); it != states.end(); ++it )
  {
    max = std::max( max, pathrec_.recombinationInterval( *it, state ) );
    max = std::max( max, pathrec_.recombinationInterval( state, *it ) );
  }
  return max;
}

u32 PathRecombinationApproximation::CliquePartition::cliqueWithoutStateRecombinationInterval( u32 clique, StateId state ) {
  verify( cliqueForState_[state] == clique );
  if( recombinationIntervalForCliqueWithoutState[state] != Core::Type<u32>::max )
    return recombinationIntervalForCliqueWithoutState[state];

  const std::set<u32>& states( statesForClique_[clique] );
  u32 max = 0;
  for( std::set<u32>::iterator it = states.begin(); it != states.end(); ++it )
  {
    if( *it == state )
      continue;
    for( std::set<u32>::iterator it2 = states.begin(); it2 != states.end(); ++it2 )
    {
      if( *it2 == state )
        continue;
      max = std::max( max, pathrec_.recombinationInterval( *it, *it2 ) );
    }
  }

  recombinationIntervalForCliqueWithoutState[state] = max;

  return max;
}

void PathRecombinationApproximation::CliquePartition::removeFromClique( u32 clique, StateId state ) {
  std::set<StateId>& cliqueStates( statesForClique_[clique] );

  for( std::set<StateId>::iterator it = statesForClique_[clique].begin(); it != statesForClique_[clique].end(); ++it )
    recombinationIntervalForCliqueWithoutState[*it] = Core::Type<u32>::max;      // These have to be invalidated because a member was removed

  size_t removed = cliqueStates.erase( state );
  verify( removed );
  verify( cliqueForState_[state] == clique );
  cliqueForState_[state] = Core::Type<u32>::max;
  recombinationIntervalForClique[clique] = Core::Type<u32>::max;
}

void PathRecombinationApproximation::CliquePartition::addToClique( u32 clique, StateId state, u32 symmetricLocalRecombinationInterval ) {
  verify( cliqueForState_[state] == Core::Type<u32>::max );
  verify( recombinationIntervalForCliqueWithoutState[state] == Core::Type<u32>::max );
  cliqueForState_[state] = clique;

  for( std::set<StateId>::iterator it = statesForClique_[clique].begin(); it != statesForClique_[clique].end(); ++it )
    recombinationIntervalForCliqueWithoutState[*it] = Core::Type<u32>::max;      // These have to be invalidated because a new member was added

  statesForClique_[clique].insert( state );
  if( recombinationIntervalForClique[clique] != Core::Type<u32>::max )
  {
    recombinationIntervalForCliqueWithoutState[state] = recombinationIntervalForClique[clique];
    recombinationIntervalForClique[clique] = std::max( recombinationIntervalForClique[clique], symmetricLocalRecombinationInterval );
  }
}
