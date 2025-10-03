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
#ifndef _SEARCH_MODULE_HH
#define _SEARCH_MODULE_HH

#include <Core/Configuration.hh>
#include <Core/Singleton.hh>
#include "SearchV2.hh"

#include "TreeBuilder.hh"

namespace Search {

class SearchAlgorithm;
class LatticeHandler;

enum TreeBuilderType {
    previousBehavior,
    classicHmm,
    minimizedHmm,
    ctc,
    rna,
};

enum SearchType {
    AdvancedTreeSearch,
    LinearSearchType,
    ExpandingFsaSearchType
};

enum SearchTypeV2 {
    LexiconfreeLabelsyncBeamSearchType,
    LexiconfreeTimesyncBeamSearchType,
    TreeTimesyncBeamSearchType
};

class Module_ {
private:
    static const Core::Choice          searchTypeV2Choice;
    static const Core::ParameterChoice searchTypeV2Param;

public:
    Module_();

    std::unique_ptr<AbstractTreeBuilder> createTreeBuilder(Core::Configuration config, const Bliss::Lexicon& lexicon, const Am::AcousticModel& acousticModel, Search::PersistentStateTree& network, bool initialize = true) const;
    SearchAlgorithm*                     createRecognizer(SearchType type, const Core::Configuration& config) const;
    SearchAlgorithmV2*                   createSearchAlgorithmV2(const Core::Configuration& config) const;
    LatticeHandler*                      createLatticeHandler(const Core::Configuration& c) const;
};

typedef Core::SingletonHolder<Module_> Module;
}  // namespace Search

#endif  // _SEARCH_MODULE_HH
