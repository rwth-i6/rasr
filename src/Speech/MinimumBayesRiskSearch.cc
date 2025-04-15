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
#include "MinimumBayesRiskSearch.hh"

#include <Fsa/Arithmetic.hh>
#include <Fsa/Basic.hh>
#include <Fsa/Cache.hh>
#include <Fsa/Project.hh>
#include <Lattice/Compose.hh>
#include <Search/MinimumBayesRiskAStarSearch.hh>
#include <Search/MinimumBayesRiskNBestListSearch.hh>
#include <Search/MinimumBayesRiskSearch.hh>
#include "NBestListExtractor.hh"

namespace Speech {

const Core::ParameterChoice MinimumBayesRiskSearchNode::paramSearchMethod(
        "search-method",
        &choiceSearchMethod,
        "method for searching mbr string in lattice",
        searchDryRun);

const Core::Choice MinimumBayesRiskSearchNode::choiceSearchMethod(
        "dry-run", searchDryRun,
        "map", searchMap,
        "n-bestlist-naive", searchNBestListNaive,
        "a-star", searchAStar,
        Core::Choice::endMark());

MinimumBayesRiskSearchNode::MinimumBayesRiskSearchNode(const Core::Configuration& config)
        : Component(config),
          LatticeSetProcessor(config),
          search_(0),
          nBestListExtractor_(0) {
    switch (paramSearchMethod(config)) {
        case searchDryRun:
            search_ = 0;
            break;
        case searchMap:
            search_ = new Search::MinimumBayesRiskMapSearch(select("search"));
            break;
        case searchNBestListNaive:
            search_             = new Search::MinimumBayesRiskNBestListNaiveSearch(select("search"));
            nBestListExtractor_ = new NBestListExtractor(select("n-best-list-extraction"));
            break;
        case searchAStar:
            search_ = new Search::MinimumBayesRiskAStarSearch(select("search"));
            break;
        default:
            criticalError("search method does not exist!");
    }
}

MinimumBayesRiskSearchNode::~MinimumBayesRiskSearchNode() {
    delete search_;
    delete nBestListExtractor_;
}

Lattice::ConstWordLatticeRef MinimumBayesRiskSearchNode::mapEvalToLemmaPronunciation(Fsa::ConstAutomatonRef eval, Lattice::ConstWordLatticeRef lattice) {
    Fsa::ConstAutomatonRef lemmaPron = Fsa::cache(Fsa::projectInput(Fsa::composeMatching(lemmaPronToLemma_,
                                                                                         Fsa::composeMatching(lemmaToEval_, eval))));

    return Lattice::composeMatching(lemmaPron, lattice);
}

void MinimumBayesRiskSearchNode::initialize(Bliss::LexiconRef lexicon) {
    Precursor::initialize(lexicon);

    lexicon_          = lexicon;
    lemmaPronToLemma_ = lexicon_->createLemmaPronunciationToLemmaTransducer();
    lemmaToEval_      = Fsa::multiply(lexicon_->createLemmaToEvaluationTokenTransducer(),
                                      Fsa::Weight(0.0));

    if (nBestListExtractor_) {
        nBestListExtractor_->initialize(lexicon);
    }
}

void MinimumBayesRiskSearchNode::processWordLattice(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* s) {
    // case dry run
    if (!search_) {
        return;
    }

    if (lattice->nParts() != 1) {
        error("Lattice must consist of a single part.");
    }

    Fsa::ConstAutomatonRef fsa = lattice->mainPart();
    if (nBestListExtractor_) {
        fsa = nBestListExtractor_->getNBestList(lattice)->mainPart();
    }

    /**
     * Assumption: fsa is an n-best list or lattice with evaluation tokens
     * as input labels.
     */
    if (fsa->getInputAlphabet() != lexicon_->evaluationTokenAlphabet()) {
        error("Lattice must have the evaluation alphabet as input alphabet.");
    }
    search_->performSearch(fsa);

    Core::Ref<Lattice::WordLattice> result(new Lattice::WordLattice);
    result->setFsa(search_->getBestAutomaton(), Lattice::WordLattice::totalFsa);
    Precursor::processWordLattice(result, s);
}

}  // namespace Speech
