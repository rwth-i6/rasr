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
#ifndef SEARCH_TRACEMANAGER_HH
#define SEARCH_TRACEMANAGER_HH

#include <Core/ReferenceCounting.hh>
#include "Trace.hh"

namespace Search
{
#define FAST_TRACE_MODIFICATION

typedef u32 TraceId;
static const TraceId invalidTraceId = Core::Type<TraceId>::max;

struct TraceItem final {
public:
  TraceItem( Core::Ref<Trace> t, Lm::History rch, Lm::History lah, Lm::History sch )
           : trace( t ), recombinationHistory( rch ), lookaheadHistory( lah ), scoreHistory( sch ), range( 0 ) {
  }
  TraceItem() : range( 0 ) {
  }
  ~TraceItem() {
  }

  Core::Ref<Trace> trace;
  Lm::History recombinationHistory;
  Lm::History lookaheadHistory;
  Lm::History scoreHistory;

  u16 range;   //Total number of trace-items in the range, including this one
private:
  friend class TraceManager;
};

class TraceManager
{
private:
  enum {
    ModifyMask = 0xff000000
  };
  enum {
    UnModifyMask = 0x00ffffff
  };
public:
  static void clear() {
    items_.clear();
    modifications_.clear();
  }

/// Returns a trace-id that represents the given list of items
  static TraceId getTrace( const std::vector<TraceItem>& items );

/// Returns a trace-id that represents only the given item
  static TraceId getTrace( const TraceItem& item );

/// Returns the trace-id of a trace that already is managed by the TraceManager
  static TraceId getManagedTraceId( const TraceItem* item ) {
    return ( (const char*)item - (const char*)items_.data() ) / sizeof( TraceItem );
  }

/// Returns the current number of existing trace-items
  static uint numTraceItems() {
    return items_.size();
  }

/// Returns the maximum number of trace-items
  static uint maxTraceItems() {
    return UnModifyMask;
  }

/// Returns whether a cleanup is currently strictly necessary (see cleanup())
  static bool needCleanup() {
    return numTraceItems() > maxTraceItems() / 2;
  }

/// Returns whether the given trace-id is additionally modified by a custom value
  inline static bool isModified( TraceId trace ) {
    return trace & ModifyMask;
  }

  struct Modification {
    Modification( u32 _first = 0, u32 _second = 0, u32 _third = 0 ) :
      first( _first ),
      second( _second ),
      third( _third )
    {}
    bool operator==( const Modification& rhs ) const {
      return first == rhs.first && second == rhs.second && third == rhs.third;
    }
    u32 first, second, third;
  };

/// Returns the custom modification-value that was attached to the trace-id. Must only be called if isModified(trace).
  inline static Modification getModification( TraceId trace ) {
    verify_( isModified( trace ) );
    u32 mod = ( ( trace & ModifyMask ) >> 24 );
    Modification ret;
    if( mod == 255 )
    {
      ret = modifications_[trace & UnModifyMask].second;
    }else{
      ret.first = mod;
    }

    ret.first -= 1; // Remove the offset that we have applied in modify(...)

    return ret;
  }

/// Returns the unmodified version of the given trace.
  inline static TraceId getUnmodified( TraceId trace ) {
    verify_( isModified( trace ) );
    if( ( trace & ModifyMask ) == ModifyMask )
    {
      return modifications_[trace & UnModifyMask].first;
    }else{
      return trace & UnModifyMask;
    }
  }

/// Modifies the given trace-id with a specific value. The value can later be retrieved through
/// getModification(traceid), where traceid is the returned value.
  static TraceId modify( TraceId trace, u32 value, u32 value2 = 0, u32 value3 = 0 ) {
    verify_( trace != invalidTraceId );
    verify_( !isModified( trace ) );
    TraceId ret;

    value += 1; // Offset by 1, so we can also modify with 0

#ifdef FAST_TRACE_MODIFICATION
    if( value < 255 && value2 == 0 )
    {
      ret = ( value << 24 ) | trace;
    }else{
#endif
    ret = modifications_.size() | ModifyMask;
    modifications_.push_back( std::make_pair<u32, Modification>( (u32)trace, Modification( value, value2, value3 ) ) );
#ifdef FAST_TRACE_MODIFICATION
  }
#endif

    verify_( getModification( ret ) == Modification( value - 1, value2, value3 ) );
    return ret;
  }

/// Calls the given visitor with each TraceItem that is contained by the given TraceId
  template <class Visitor>
  inline void visitItems( Visitor& visitor, TraceId trace ) {
    if( trace == invalidTraceId )
      return;

    trace = getUnmodified( trace );

    const TraceItem& item = items_[trace];
    if( !visitor.visit( item ) )
      return;

    if( item.range != 1 )
    {
      TraceId until = trace + item.range;
      for( TraceId pos = trace + 1; pos < until; ++pos )
        if( !visitor.visit( items_[pos] ) )
          return;
    }
  }

/// Returns the (first) trace-item associated to the given trace-id
/// The trace-id must be valid
  inline static TraceItem& traceItem( TraceId trace ) {
    return items_[getUnmodified( trace )];
  }

/// Returns the number of trace-items associated to the given trace-id
/// The trace-id must be valid
  inline static u32 traceCount( TraceId trace ) {
    return items_[getUnmodified( trace )].range;
  }

/// Cleans up the trace manager, retaining only the given trace lists
/// Returns a map that maps old trace-ids to new ones
/// The given set of traces will be altered for internal reasons.
  static std::unordered_map<TraceId, TraceId> cleanup( std::unordered_set<TraceId>& retain );

private:
  static std::vector<TraceItem> items_;
  static std::vector<std::pair<u32, Modification> > modifications_;
};
}

#endif // SEARCH_TRACEMANAGER_HH
