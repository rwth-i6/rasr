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
#ifndef PATHTRACE_HH
#define PATHTRACE_HH

#include <Core/ReferenceCounting.hh>
#include <Core/Component.hh>
#include <Search/Types.hh>
#include <Bliss/Lexicon.hh>
#include <map>

// #define TRACE_PATH

namespace Search
{
#ifdef TRACE_PATH

struct PathPruningDescriptor : public Core::ReferenceCounted {
  std::map<std::pair<char*, int>, Score> offsets;
  typedef Core::Ref<PathPruningDescriptor> Ref;
};

class PathTrace
{
public:

  void log( const Core::Component& component, const Bliss::LemmaPronunciation* pron )
  {
    if( !pruning )
    {
      std::cout << "pruning missing in path-trace" << std::endl;
      return;
    }

    if( pron && pron->lemma() && pron->lemma()->symbol() )
    {
      component.log() << "Word identity:" << pron->lemma()->symbol();
      component.log() << "Word pron length:" << pron->pronunciation()->length();
      if( !pron->lemma()->hasEvaluationTokenSequence() )
        return;  // Don't log tokens which are not evaluated
    }

    for( std::map<std::pair<char*, int>, Score>::iterator it = pruning->offsets.begin(); it != pruning->offsets.end(); ++it )
    {
      if( it->first.second == -1 )
        component.log() << "Word " << it->first.first << ":" << it->second;
      else
        component.log() << "Word " << it->first.first << ": [" << it->first.second << "] " << it->second;
    }
  }

  void maximizeOffset( char* desc, Score offset, int index = -1 )
  {
    if( !pruning )
      makeUnique();

    std::map<std::pair<char*, int>, Score>::iterator it = pruning->offsets.find( std::make_pair( desc, index ) );
    if( it == pruning->offsets.end() )
    {
      makeUnique();
      pruning->offsets.insert( std::make_pair( std::make_pair( desc, index ), offset ) );
      return;
    }

    if( offset > it->second )
    {
      makeUnique();
      pruning->offsets[std::make_pair( desc, index )] = offset;
    }
  }

  enum {
    Enabled = 1
  };

private:
  PathPruningDescriptor::Ref pruning;

  ///Makes sure that the given descriptor is referenced only once, else creates a copy.
  ///Use this before manipulating a descriptor.
  void makeUnique() {
    if( !pruning )
      pruning = PathPruningDescriptor::Ref( new PathPruningDescriptor );
    if( pruning->refCount() > 1 )
      pruning =  PathPruningDescriptor::Ref( new PathPruningDescriptor( *pruning ) );
  }
};

#else

class PathTrace
{
public:
  void log( const Core::Component&, const Bliss::LemmaPronunciation* pron ) { }
  enum {
    Enabled = 0
  };
  void maximizeOffset( char*, Score, int = -1 ) {
  }
};

#endif
}

#endif
