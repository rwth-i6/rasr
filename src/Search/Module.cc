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
#include <Modules.hh>
#include <Search/LatticeHandler.hh>
#include <Search/Module.hh>
#include "LexiconfreeLabelsyncBeamSearch/LexiconfreeLabelsyncBeamSearch.hh"
#include "LexiconfreeRNNTTimesyncBeamSearch/LexiconfreeRNNTTimesyncBeamSearch.hh"
#include "LexiconfreeTimesyncBeamSearch/LexiconfreeTimesyncBeamSearch.hh"
#include "TreeBuilder.hh"
#include "TreeTimesyncBeamSearch/TreeTimesyncBeamSearch.hh"
#ifdef MODULE_SEARCH_WFST
#include <Search/Wfst/ExpandingFsaSearch.hh>
#include <Search/Wfst/LatticeHandler.hh>
#endif
#ifdef MODULE_SEARCH_LINEAR
#include <Search/LinearSearch.hh>
#endif
#ifdef MODULE_ADVANCED_TREE_SEARCH
#include "AdvancedTreeSearch/AdvancedTreeSearch.hh"
#endif

using namespace Search;

Module_::Module_() {
}

const Core::Choice Module_::searchTypeV2Choice(
        "lexiconfree-labelsync-beam-search", SearchTypeV2::LexiconfreeLabelsyncBeamSearchType,
        "lexiconfree-timesync-beam-search", SearchTypeV2::LexiconfreeTimesyncBeamSearchType,
        "lexiconfree-rnnt-timesync-beam-search", SearchTypeV2::LexiconfreeRNNTTimesyncBeamSearchType,
        "tree-timesync-beam-search", SearchTypeV2::TreeTimesyncBeamSearchType,
        Core::Choice::endMark());

const Core::ParameterChoice Module_::searchTypeV2Param(
        "type", &Module_::searchTypeV2Choice, "type of search", SearchTypeV2::LexiconfreeTimesyncBeamSearchType);

const Core::Choice choiceTreeBuilderType(
        "classic-hmm", static_cast<int>(TreeBuilderType::classicHmm),
        "minimized-hmm", static_cast<int>(TreeBuilderType::minimizedHmm),
        "ctc", static_cast<int>(TreeBuilderType::ctc),
        "rna", static_cast<int>(TreeBuilderType::rna),
        "aed", static_cast<int>(TreeBuilderType::aed),
        Core::Choice::endMark());

const Core::ParameterChoice paramTreeBuilderType(
        "tree-builder-type",
        &choiceTreeBuilderType,
        "which tree builder to use",
        static_cast<int>(TreeBuilderType::previousBehavior));

std::unique_ptr<AbstractTreeBuilder> Module_::createTreeBuilder(Core::Configuration config, const Bliss::Lexicon& lexicon, const Am::AcousticModel& acousticModel, Search::PersistentStateTree& network, bool initialize) const {
    switch (paramTreeBuilderType(config)) {
        case TreeBuilderType::classicHmm: {  // Use StateTree.hh
            return std::unique_ptr<AbstractTreeBuilder>(nullptr);
        } break;
        case TreeBuilderType::previousBehavior:
        case TreeBuilderType::minimizedHmm: {  // Use TreeStructure.hh
            return std::unique_ptr<AbstractTreeBuilder>(new MinimizedTreeBuilder(config, lexicon, acousticModel, network, initialize));
        } break;
        case TreeBuilderType::ctc: {
            return std::unique_ptr<AbstractTreeBuilder>(new CtcTreeBuilder(config, lexicon, acousticModel, network, initialize));
        } break;
        case Search::TreeBuilderType::rna: {
            return std::unique_ptr<AbstractTreeBuilder>(new RnaTreeBuilder(config, lexicon, acousticModel, network, initialize));
        } break;
        case Search::TreeBuilderType::aed: {
            return std::unique_ptr<AbstractTreeBuilder>(new AedTreeBuilder(config, lexicon, acousticModel, network, initialize));
        } break;
        default: defect();
    }
}

SearchAlgorithm* Module_::createRecognizer(SearchType type, const Core::Configuration& config) const {
    SearchAlgorithm* recognizer = nullptr;
    switch (type) {
        case AdvancedTreeSearch:
#ifdef MODULE_ADVANCED_TREE_SEARCH
            recognizer = new Search::AdvancedTreeSearchManager(config);
#else
            Core::Application::us()->criticalError("Module MODULE_ADVANCED_TREE_SEARCH not available!");
#endif
            break;
        case ExpandingFsaSearchType:
#ifdef MODULE_SEARCH_WFST
            recognizer = new Search::Wfst::ExpandingFsaSearch(config);
#else
            Core::Application::us()->criticalError("Module MODULE_SEARCH_WFST not available!");
#endif
            break;
        case LinearSearchType:
#ifdef MODULE_SEARCH_LINEAR
            recognizer = new Search::LinearSearch(config);
#else
            Core::Application::us()->criticalError("Module MODULE_SEARCH_LINEAR not available!");
#endif
            break;

        default:
            Core::Application::us()->criticalError("unknown recognizer type: %d", type);
            break;
    }
    return recognizer;
}

SearchAlgorithmV2* Module_::createSearchAlgorithmV2(const Core::Configuration& config) const {
    SearchAlgorithmV2* searchAlgorithm = nullptr;
    switch (searchTypeV2Param(config)) {
        case LexiconfreeLabelsyncBeamSearchType:
            searchAlgorithm = new Search::LexiconfreeLabelsyncBeamSearch(config);
            break;
        case LexiconfreeTimesyncBeamSearchType:
            searchAlgorithm = new Search::LexiconfreeTimesyncBeamSearch(config);
            break;
        case LexiconfreeRNNTTimesyncBeamSearchType:
            searchAlgorithm = new Search::LexiconfreeRNNTTimesyncBeamSearch(config);
            break;
        case TreeTimesyncBeamSearchType:
            searchAlgorithm = new Search::TreeTimesyncBeamSearch(config);
            break;
        default:
            Core::Application::us()->criticalError("Unknown search algorithm type: %d", searchTypeV2Param(config));
            break;
    }
    return searchAlgorithm;
}

LatticeHandler* Module_::createLatticeHandler(const Core::Configuration& c) const {
    LatticeHandler* handler = new LatticeHandler(c);
#ifdef MODULE_SEARCH_WFST
    handler = new Search::Wfst::LatticeHandler(c, handler);
#endif
    /**
     * @todo: add Flf::LatticeHandler?
     * This would add a dependency to module Flf without actual benefit.
     */
    return handler;
}
