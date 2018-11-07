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
#include "TraceManager.hh"

using namespace Search;

std::vector<TraceItem> TraceManager::items_;
std::vector<std::pair<u32, TraceManager::Modification> > TraceManager::modifications_;

TraceId TraceManager::getTrace( const Search::TraceItem& item ) {
  TraceId ret = items_.size();
  items_.push_back( item );
  items_.back().range = 1;
  return ret;
}

TraceId TraceManager::getTrace( const std::vector<TraceItem>& items ) {
  if( items.size() == 1 )
    return getTrace( items.front() );

  if( items.size() > 0xffff )
  {
    Core::Application::us()->log() << "Too many items in one trace: " << items.size() << " dropping " << items.size() - 0xffff;
    std::vector<TraceItem> tempItems = items;
    tempItems.resize( 0xffff );
    return getTrace( tempItems );
  }

  TraceId ret = items_.size();
  items_.insert( items_.end(), items.begin(), items.end() );
  items_[ret].range = items.size();
  verify( items_[ret].range == items.size() );
  verify( !isModified( ret ) );
  return ret;
}

std::unordered_map<TraceId, TraceId> TraceManager::cleanup( std::unordered_set<TraceId>& retain ) {
  std::unordered_map<TraceId, TraceId> ret;

  std::unordered_set<u32> retainHardModifications;
  std::unordered_set<u32> retainSoftModifications;

  for( std::unordered_set<TraceId>::const_iterator it = retain.begin(); it != retain.end(); ++it )
    if( ( ( *it ) & ModifyMask ) == ModifyMask )
      retainHardModifications.insert( ( *it ) & UnModifyMask );
    else if( ( *it ) & ModifyMask )
      retainSoftModifications.insert( ( *it ) & UnModifyMask );

  for( std::unordered_set<TraceId>::const_iterator it = retainHardModifications.begin(); it != retainHardModifications.end(); ++it )
    retain.insert( modifications_[( *it )].first );

  for( std::unordered_set<TraceId>::const_iterator it = retainSoftModifications.begin(); it != retainSoftModifications.end(); ++it )
    retain.insert( *it );

  // Now clean up the real traces

  u32 newSize = 0;

  for( TraceId a = 0; a < items_.size(); ++a )
  {
    if( retain.count( a ) )
    {
      verify( items_[a].range );
      ret.insert( std::make_pair( a, newSize ) );
//       Score lastScore = Core::Type<Score>::min;
      u32 range = items_[a].range;
      for( u32 b = 0; b < range; ++b )
      {
//         verify( range == 1 || items_[a + b].score >= lastScore );
//         lastScore = items_[a + b].score;
        verify( b == 0 || items_[a + b].range == 0 );
        items_[newSize] = items_[a + b];
        ++newSize;
      }
    }
  }

  items_.resize( newSize );

  // Now clean up the modifications

  newSize = 0;

  for( u32 a = 0; a < modifications_.size(); ++a )
  {
    if( retainHardModifications.count( a ) )
    {
      // Apply the mapping
      modifications_[newSize] = std::pair<u32, Modification> ( (u32)( ret[modifications_[a].first] ), modifications_[a].second );
      ret.insert( std::make_pair<u32, u32>( ModifyMask | a, ModifyMask | newSize ) );
      ++newSize;
    }
  }

  modifications_.resize( newSize );

  // Now map the 'soft' modified items

  for( std::unordered_set<TraceId>::const_iterator it = retain.begin(); it != retain.end(); ++it )
    if( ( ( *it ) & ModifyMask ) && ( ( ( *it ) & ModifyMask ) !=  ModifyMask ) )
      ret.insert( std::pair<u32, u32>( *it, ret[( *it ) & UnModifyMask] | ( ( *it ) & ModifyMask ) ) );

  return ret;
}
