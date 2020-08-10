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
#include "MinimumBayesRiskAStarSearch.hh"

#include <Fsa/Arithmetic.hh>
#include <Fsa/Best.hh>
#include <Fsa/Compose.hh>
#include <Fsa/Levenshtein.hh>
#include <Fsa/Output.hh>
#include <Fsa/Prune.hh>
#include <Fsa/Rational.hh>
#include <Fsa/RemoveEpsilons.hh>

#include <algorithm>
#include <queue>

namespace Search {

const u32 MinimumBayesRiskAStarSearch::MaxLogWeights = 100;

std::vector<Fsa::Weight> MinimumBayesRiskAStarSearch::LogWeights(MaxLogWeights);

typedef MinimumBayesRiskAStarSearch::ConstSearchNodeRef ConstSearchNodeRef;
typedef MinimumBayesRiskAStarSearch::SearchNodeRef      SearchNodeRef;

const Core::ParameterBool MinimumBayesRiskAStarSearch::paramShallDumpStack(
        "dump-stack",
        "shall contents of stack be printed to clog",
        false);

const Core::ParameterBool MinimumBayesRiskAStarSearch::paramExactEstimate(
        "exact-estimate",
        "shall goal estimate be exact or overestimate",
        false);

const Core::ParameterInt MinimumBayesRiskAStarSearch::paramMaxStackSize(
        "maximum-stack-size",
        "maximum size of a single stack",
        50);

const Core::ParameterBool MinimumBayesRiskAStarSearch::paramShallPrune(
        "shall-prune",
        "flag whether evaluation and summation space shall be pruned",
        true);

const Core::ParameterFloat MinimumBayesRiskAStarSearch::paramInitialPruningThreshold(
        "initial-pruning-threshold",
        "pruning threshold to start with",
        500.0);

const Core::ParameterFloat MinimumBayesRiskAStarSearch::paramThresholdFactor(
        "threshold-factor",
        "factor the threshold gets multiplied with",
        0.9);

const Core::ParameterInt MinimumBayesRiskAStarSearch::paramMaximumNumberHypotheses(
        "maximum-number-hypotheses",
        "maximum number hypotheses the lattice shall contain",
        10000);

/**
 * Computes one column of the DP levenshtein distance algorithm.
 */
inline void computeLevenshteinColumn(const Fsa::LabelId               symbol,
                                     const std::vector<Fsa::LabelId>& hypothesis,
                                     const std::vector<u32>&          oldScores,
                                     std::vector<u32>&                newScores,
                                     const u32                        initialNewScore) {
    require(oldScores.size() == newScores.size());
    require(newScores.size() == hypothesis.size() + 1);

    newScores.at(0) = initialNewScore;
    for (u32 n = 1; n < hypothesis.size() + 1; ++n) {
        u32 del         = oldScores.at(n) + 1;
        u32 ins         = newScores.at(n - 1) + 1;
        u32 sub         = oldScores.at(n - 1) + (hypothesis.at(n - 1) == symbol ? 0 : 1);
        newScores.at(n) = (del <= ins ? (del <= sub ? del : (ins <= sub ? ins : sub)) : (ins <= sub ? ins : sub));
    }
}

inline u32 minimum(const std::vector<u32>& scores) {
    u32 minimum = Core::Type<u32>::max;
    for (u32 n = 0; n < scores.size(); ++n) {
        minimum = std::min(minimum, scores.at(n));
    }
    return minimum;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                                     */
/*                                   SearchNode                                        */
/*                                                                                     */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
MinimumBayesRiskAStarSearch::SearchNode::SearchNode() {}

MinimumBayesRiskAStarSearch::SearchNode::SearchNode(u32 longestDistance, Fsa::StateId initialStateId)
        : totalProbability_(Fsa::LogSemiring->zero()),
          finalProbability_(Fsa::LogSemiring->zero()),
          estimate_(Fsa::LogSemiring->max()),
          overestimate_(Core::Type<f32>::min),
          longestDistance_(longestDistance),
          isFinal_(false),
          isSemifinal_(false),
          isExplorable_(true) {
    logForwardProbabilities_.insert(std::make_pair(initialStateId, Fsa::LogSemiring->one()));
    logTotalProbabilities_.push_back(Fsa::LogSemiring->one());
}

MinimumBayesRiskAStarSearch::SearchNode::SearchNode(
        const std::vector<Fsa::LabelId>&     hypothesis,
        const Fsa::LabelId                   symbol,
        const Fsa::Weight&                   totalProbability,
        const std::vector<std::vector<u32>>& levenshteinColumns,
        const Core::Vector<Fsa::Weight>&     levenshteinScores,
        const Fsa::Weight&                   estimate,
        u32                                  oldNodePosition)
        : hypothesis_(hypothesis),
          totalProbability_(totalProbability),
          finalProbability_(totalProbability),
          levenshteinColumns_(levenshteinColumns),
          levenshteinScores_(levenshteinScores),
          levenshteinScoresOverestimate_(levenshteinScores),
          estimate_(estimate),
          overestimate_(Core::Type<f32>::min),
          longestDistance_(0),
          isFinal_(false),
          isSemifinal_(false),
          isExplorable_(true) {
    if (levenshteinColumns_.size() > 0 && oldNodePosition != Core::Type<u32>::max) {
        levenshteinScoresOverestimate_.erase(levenshteinScoresOverestimate_.begin() + oldNodePosition);
    }
    hypothesis_.push_back(symbol);
}

MinimumBayesRiskAStarSearch::SearchNode::~SearchNode() {}

void MinimumBayesRiskAStarSearch::SearchNode::addState(const Fsa::StateId stateId,
                                                       const u32          longestDistance,
                                                       const Fsa::Weight  logForwardProbability,
                                                       const Fsa::Weight  logBackwardProbability,
                                                       const Fsa::Weight  inverseNormalizationConstant) {
    longestDistance_                                      = std::max(longestDistance_, longestDistance);
    std::map<Fsa::StateId, Fsa::Weight>::iterator logProb = logForwardProbabilities_.find(stateId);
    if (logProb == logForwardProbabilities_.end()) {
        logForwardProbabilities_.insert(std::make_pair(stateId, logForwardProbability));
    }
    else {
        logProb->second = Fsa::LogSemiring->collect(logProb->second, logForwardProbability);
    }

    logTotalProbabilities_.push_back(Fsa::LogSemiring->extend(inverseNormalizationConstant,
                                                              Fsa::LogSemiring->extend(logForwardProbability,
                                                                                       logBackwardProbability)));
}

std::map<Fsa::StateId, Fsa::Weight>::const_iterator MinimumBayesRiskAStarSearch::SearchNode::begin() const {
    return logForwardProbabilities_.begin();
}

std::map<Fsa::StateId, Fsa::Weight>::const_iterator MinimumBayesRiskAStarSearch::SearchNode::end() const {
    return logForwardProbabilities_.end();
}

void MinimumBayesRiskAStarSearch::SearchNode::computeTotalProbability() {
    Fsa::Accumulator* collector = Fsa::LogSemiring->getCollector(Fsa::LogSemiring->zero());
    for (u32 i = 0; i < logTotalProbabilities_.size(); ++i) {
        collector->feed(logTotalProbabilities_[i]);
    }
    totalProbability_ = collector->get();
    delete collector;
}

bool MinimumBayesRiskAStarSearch::SearchNode::isFinal() const {
    return isFinal_;
}

void MinimumBayesRiskAStarSearch::SearchNode::setFinal() {
    isFinal_ = true;
}

bool MinimumBayesRiskAStarSearch::SearchNode::isExplorable() const {
    return isExplorable_;
}

void MinimumBayesRiskAStarSearch::SearchNode::setInexplorable() {
    isExplorable_ = false;
}

void MinimumBayesRiskAStarSearch::SearchNode::dump(Core::XmlWriter& out, Fsa::ConstAlphabetRef alphabet) const {}

bool operator<(SearchNodeRef& lower, SearchNodeRef& taller) {
    if (f32(lower->estimate_) == f32(taller->estimate_)) {
        return (f32(lower->totalProbability_) < f32(taller->totalProbability_));
    }
    else {
        return (f32(lower->estimate_) > f32(taller->estimate_));
    }
}

void MinimumBayesRiskAStarSearch::dump(Core::XmlWriter& out, Fsa::ConstAlphabetRef alphabet) const {
    std::cerr << " --- --- ---" << std::endl;
    std::cerr << " | Size: " << stack_->size()
              << " | Minimal: " << minimalNode_->index_;
    if (secondMinimalNode_)
        std::cerr << " | 2nd-Minimal: " << secondMinimalNode_->index_;
    if (minimalIncompleteNode_)
        std::cerr << " | minimal incomplete: " << minimalIncompleteNode_->index_;
    std::cerr << " |" << std::endl;
    std::cerr << std::endl;
    for (u32 n = 0; n < stack_->size(); ++n) {
        std::cerr << n << " --> ";
        ConstSearchNodeRef fn = stack_->at(n);
        std::cerr << fn->index_ << " --> ";

        std::cerr << f32(fn->estimate_) << "\t";
        std::cerr << f32(fn->overestimate_) << "\t";
        std::cerr << f32(fn->totalProbability_) << "\t";

        std::vector<Fsa::LabelId>::const_iterator label = fn->hypothesis_.begin();
        for (; label != fn->hypothesis_.end(); ++label) {
            std::cerr << alphabet->symbol(*label) << " ";
        }
        std::cerr << std::endl;
    }
    std::cerr << " --- --- ---" << std::endl;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                                     */
/*                                   MinimumBayesRiskAStarSearch                       */
/*                                                                                     */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
MinimumBayesRiskAStarSearch::MinimumBayesRiskAStarSearch(
        const Core::Configuration& config)
        : MinimumBayesRiskSearch(config),
          newMinimalNode_(SearchNodeRef()),
          newMinimalNodeCounter_(u32(0)),
          maxStackSize_(paramMaxStackSize(config)),
          shallPrune_(paramShallPrune(config)),
          initialPruningThreshold_(paramInitialPruningThreshold(config)),
          thresholdFactor_(paramThresholdFactor(config)),
          maximumNumberHypotheses_(paramMaximumNumberHypotheses(config)),
          dumpStack_(paramShallDumpStack(config)),
          numberOverestimates_(u32(0)),
          semiring_(Fsa::LogSemiring) {}

void MinimumBayesRiskAStarSearch::expand() {
    ++numberEvaluations_;

    // As the values of the expansion node are needed for the construction of the new grid nodes,
    // this reference serves as a temporary copy of the expansion node. Processing this way, the
    // deletion can be safely carried out without further effort.
    SearchNodeRef expansionNode = minimalIncompleteNode_;

    expansionNode->isExplorable_ = false;

    if (fsa_->getState(expansionNode->begin()->first)->isFinal()) {
        expansionNode->setFinal();
    }

    // The comparison and optimality test make use of null compares. As a result, the corresponding
    // references have to be initialized by null.
    minimalNode_           = SearchNodeRef();
    secondMinimalNode_     = SearchNodeRef();
    minimalIncompleteNode_ = SearchNodeRef();

    // In case the current expanded node is also final, it must not be deleted from the stack
    // but instead, its overestimate has to be computed. In fact, this is the only case during
    // which the initial overestimate will be computed.
    // In addition, a heuristic has been prepared to stop computation if the minimal node remains the
    // same over 50 iterations.
    if (expansionNode->isFinal()) {
        if (!newMinimalNode_ || newMinimalNode_ != expansionNode) {
            newMinimalNode_        = expansionNode;
            newMinimalNodeCounter_ = 1;
        }
        else {
            if (++newMinimalNodeCounter_ > 50) {
                minimalNode_ = newMinimalNode_;
                stack_->clear();
                return;
            }
        }

        // needed for prefix tree search.
        Fsa::Accumulator* collector = Fsa::LogSemiring->getCollector(Fsa::LogSemiring->zero());
        collector->feed(expansionNode->finalProbability_);
        std::map<Fsa::StateId, Fsa::Weight>::const_iterator stateId = expansionNode->begin();
        for (; stateId != expansionNode->end(); ++stateId) {
            collector->feed(
                    semiring_->extend(
                            semiring_->extend(
                                    stateId->second,
                                    fsa_->getState(stateId->first)->weight()),
                            inverseBackwardPotentialsNormalizationConstant_));
        }
        expansionNode->totalProbability_ = collector->get();
        delete collector;

        // one-half-criterion
        if (f32(expansionNode->totalProbability_) < -std::log(0.5)) {
            minimalNode_ = expansionNode;
            stack_->clear();
            return;
        }

        overestimateFinal(expansionNode);
    }
    else {
        // In case the current expansion node is not a final one, it has to be deleted from the stack. In addition, all
        // score entries have to be deleted for all other stack nodes. As expansionNode is a copy of the deleted node,
        // all values for creation of new nodes are available.
        // In addition, the old entry has to be deleted from the pruning stack.
        stack_->erase(stack_->begin() + expansionNode->index_);
        std::vector<SearchNodeRef>::iterator stackNode = stack_->begin();
        for (; stackNode != stack_->end(); ++stackNode) {
            (*stackNode)->levenshteinColumns_.erase((*stackNode)->levenshteinColumns_.begin() + expansionNode->index_);
            (*stackNode)->levenshteinScores_.erase((*stackNode)->levenshteinScores_.begin() + expansionNode->index_);
        }

        if (!stackEntries_.empty()) {
            std::vector<SearchNodeRef>&          oldEntries      = stackEntries_.at(expansionNode->hypothesis_.size());
            std::vector<SearchNodeRef>::iterator oldNodeToDelete = oldEntries.begin();
            for (; oldNodeToDelete != oldEntries.end(); ++oldNodeToDelete) {
                if ((*oldNodeToDelete)->index_ == expansionNode->index_) {
                    oldEntries.erase(oldNodeToDelete);
                    break;
                }
            }
        }
    }

    std::vector<SearchNodeRef> newSearchNodes = createNewSearchNodes(expansionNode);

    u32 newLength_ = expansionNode->hypothesis_.size() + 1;
    if (stackEntries_.size() < newLength_ + 1) {
        stackEntries_.resize(newLength_ + 2);
    }

    updateStack(expansionNode, newSearchNodes);
    generateNewGridNodes(expansionNode, newSearchNodes);
    numberComputations_ += stack_->size() * stack_->size();
    pruneStack();
    newMinimalNode_ = minimalNode_;
    if (dumpStack_) {
        dump(clog(), fsa_->inputAlphabet());
    }
}

std::vector<SearchNodeRef> MinimumBayesRiskAStarSearch::createNewSearchNodes(ConstSearchNodeRef expansionNode) {
    std::vector<SearchNodeRef> result;

    // This will be needed for the prefix tree search. All new created nodes will share the string of the expansion node
    // as prefix. In case the lattice is nondeterministic, there may exist several paths belonging to this prefix, so all
    // target states have to be determined for the same prefix. As the new hypothesis differ in exactly one word/label,
    // the corresponding data structure was chosen to be a map.
    std::map<Fsa::LabelId, SearchNodeRef> newNodes;

    u32 exIndex = (expansionNode->isFinal() ? Core::Type<u32>::max : expansionNode->index_);

    // generate new Stack nodes from expansion node.
    std::map<Fsa::StateId, Fsa::Weight>::const_iterator stateId = expansionNode->begin();
    for (; stateId != expansionNode->end(); ++stateId) {
        Fsa::ConstStateRef         state = fsa_->getState(stateId->first);
        Fsa::State::const_iterator arc   = state->begin();
        for (; arc != state->end(); ++arc) {
            NodeMap::iterator mapEntry = newNodes.find(arc->input());
            if (mapEntry == newNodes.end()) {
                mapEntry = newNodes.insert(mapEntry,
                                           std::make_pair(arc->input(),
                                                          SearchNodeRef(new SearchNode(expansionNode->hypothesis_,
                                                                                       arc->input(),
                                                                                       Fsa::LogSemiring->zero(),
                                                                                       expansionNode->levenshteinColumns_,
                                                                                       expansionNode->levenshteinScores_,
                                                                                       Fsa::LogSemiring->max(),
                                                                                       exIndex))));
            }

            Fsa::ConstStateRef targetState        = fsa_->getState(arc->target());
            Fsa::Weight        forwardProbability = semiring_->extend(stateId->second, arc->weight());

            mapEntry->second->addState(arc->target(),
                                       distances_.at(arc->target()),
                                       forwardProbability,
                                       backwardPotentials_.at(arc->target()),
                                       inverseBackwardPotentialsNormalizationConstant_);
        }
    }

    // In case a prefix tree search will be performed, the probability mass has to be accumulated,
    // because there may exist several paths for the string, a search node represents.
    NodeMap::iterator newSearchNode = newNodes.begin();
    for (; newSearchNode != newNodes.end(); ++newSearchNode) {
        newSearchNode->second->computeTotalProbability();
        result.push_back(newSearchNode->second);
    }

    return result;
}

void MinimumBayesRiskAStarSearch::updateStack(
        SearchNodeRef expansionNode, std::vector<SearchNodeRef>& newNodes) {
    if (stack_->size() < 1) {
        return;
    }
    for (u32 n = 0; n < stack_->size(); ++n) {
        SearchNodeRef competingSearchNode = stack_->at(n);
        competingSearchNode->index_       = n;
        if (competingSearchNode->isFinal()) {
            ++numberOverestimates_;
        }

        std::vector<u32> competingLevenshteinColumn = competingSearchNode->levenshteinColumns_.at(expansionNode->index_);

        std::vector<SearchNodeRef>::iterator newSearchNode = newNodes.begin();
        for (; newSearchNode != newNodes.end(); ++newSearchNode) {
            std::vector<u32> levenshteinColumn(competingSearchNode->hypothesis_.size() + 1);
            computeLevenshteinColumn(
                    (*newSearchNode)->hypothesis_.back(),
                    competingSearchNode->hypothesis_,
                    competingLevenshteinColumn,
                    levenshteinColumn,
                    expansionNode->hypothesis_.size() + 1);

            // compute levenshtein score
            if (competingSearchNode->isFinal()) {
                if (!exactEstimate_) {
                    u32              longestDistance      = (*newSearchNode)->longestDistance_;
                    std::vector<u32> newLevenshteinColumn = levenshteinColumn;
                    std::vector<u32> tmpLevenshteinColumn;
                    for (u32 l = 0; l < longestDistance; ++l) {
                        tmpLevenshteinColumn = newLevenshteinColumn;
                        computeLevenshteinColumn(
                                Fsa::InvalidLabelId,
                                competingSearchNode->hypothesis_,
                                tmpLevenshteinColumn,
                                newLevenshteinColumn,
                                (*newSearchNode)->hypothesis_.size() + l + 1);
                    }
                    Fsa::Weight levenshteinScoreOverestimate = Fsa::LogSemiring->extend(LogWeights.at(newLevenshteinColumn.back()),
                                                                                        (*newSearchNode)->totalProbability_);
                    competingSearchNode->levenshteinScoresOverestimate_.push_back(levenshteinScoreOverestimate);
                }
            }

            Fsa::Weight levenshteinScore = Fsa::LogSemiring->extend(LogWeights.at(minimum(levenshteinColumn)),
                                                                    (*newSearchNode)->totalProbability_);
            competingSearchNode->levenshteinScores_.push_back(levenshteinScore);

            (*newSearchNode)->levenshteinColumns_.at(n).push_back(levenshteinColumn.back());

            (*newSearchNode)->levenshteinScores_.at(n) = Fsa::LogSemiring->extend(Fsa::Weight(LogWeights.at(minimum((*newSearchNode)->levenshteinColumns_.at(n)))),
                                                                                  competingSearchNode->totalProbability_);

            competingSearchNode->levenshteinColumns_.push_back(levenshteinColumn);
        }

        if (competingSearchNode->isExplorable_ && (!minimalIncompleteNode_ || competingSearchNode < minimalIncompleteNode_)) {
            minimalIncompleteNode_ = competingSearchNode;
        }

        Fsa::Accumulator* collector = Fsa::LogSemiring->getCollector(Fsa::LogSemiring->zero());
        for (u32 i = 0; i < competingSearchNode->levenshteinScores_.size(); ++i) {
            collector->feed(competingSearchNode->levenshteinScores_[i]);
        }
        competingSearchNode->estimate_ = collector->get();
        delete collector;

        if (competingSearchNode->isFinal()) {
            Fsa::Accumulator* collector = Fsa::LogSemiring->getCollector(Fsa::LogSemiring->zero());
            for (u32 i = 0; i < competingSearchNode->levenshteinScoresOverestimate_.size(); ++i) {
                collector->feed(competingSearchNode->levenshteinScoresOverestimate_[i]);
            }
            competingSearchNode->overestimate_ = collector->get();
            delete collector;
        }

        if (!minimalNode_ || operator<(competingSearchNode, minimalNode_)) {
            secondMinimalNode_ = minimalNode_;
            minimalNode_       = competingSearchNode;
        }
        else if (!secondMinimalNode_ || competingSearchNode < secondMinimalNode_) {
            secondMinimalNode_ = competingSearchNode;
        }
    }
}

void MinimumBayesRiskAStarSearch::overestimateFinal(SearchNodeRef finalNode) {
    verify(finalNode->isFinal());
    if (exactEstimate_) {
        finalNode->estimate_ = posteriorExpectedRisk(
                createLinearAutomatonFromVector(
                        finalNode->hypothesis_,
                        Fsa::Weight(0.0),  //(*newSearchNode)->second->totalProbability_,
                        fsa_->inputAlphabet(),
                        fsa_->outputAlphabet(),
                        Fsa::TropicalSemiring),
                fsa_);
        finalNode->overestimate_ = finalNode->estimate_;
    }
    else {
        // overestimate for final node.
        {
            for (u32 n = 0; n < stack_->size(); ++n) {
                SearchNodeRef competingSearchNode = stack_->at(n);
                if (!exactEstimate_) {
                    Fsa::Weight levenshteinScoreOverestimate;
                    if (competingSearchNode != finalNode) {
                        u32              longestDistance      = competingSearchNode->longestDistance_;
                        std::vector<u32> newLevenshteinColumn = competingSearchNode->levenshteinColumns_.at(finalNode->index_);
                        std::vector<u32> tmpLevenshteinColumn;
                        for (u32 l = 0; l < longestDistance; ++l) {
                            tmpLevenshteinColumn = newLevenshteinColumn;
                            computeLevenshteinColumn(Fsa::InvalidLabelId,
                                                     competingSearchNode->hypothesis_,
                                                     tmpLevenshteinColumn,
                                                     newLevenshteinColumn,
                                                     competingSearchNode->hypothesis_.size() + l + 1);
                        }
                        levenshteinScoreOverestimate = Fsa::LogSemiring->extend(LogWeights.at(newLevenshteinColumn.back()),
                                                                                competingSearchNode->totalProbability_);
                    }
                    else {
                        levenshteinScoreOverestimate = Fsa::LogSemiring->zero();
                    }
                    finalNode->levenshteinScoresOverestimate_.push_back(levenshteinScoreOverestimate);
                }
            }
        }
    }
}

void MinimumBayesRiskAStarSearch::generateNewGridNodes(SearchNodeRef expansionNode, std::vector<SearchNodeRef>& newNodes) {
    // generate grid nodes for every pair ( new, new )
    std::vector<u32> newNodeLevenshteinEqualColumn(expansionNode->hypothesis_.size() + 2);
    std::vector<u32> newNodeLevenshteinDifferColumn(expansionNode->hypothesis_.size() + 2);

    for (u32 s = 0; s < expansionNode->hypothesis_.size() + 2; ++s) {
        newNodeLevenshteinEqualColumn.at(s)  = expansionNode->hypothesis_.size() + 1 - s;
        newNodeLevenshteinDifferColumn.at(s) = expansionNode->hypothesis_.size() + 1 - s;
    }

    newNodeLevenshteinDifferColumn.at(expansionNode->hypothesis_.size() + 1) = 1;

    // add grid node for every new stack node to every new stack node
    std::vector<SearchNodeRef>::iterator newSearchNode = newNodes.begin();
    for (; newSearchNode != newNodes.end(); ++newSearchNode) {
        std::vector<SearchNodeRef>::iterator newCompetingSearchNode = newNodes.begin();
        for (; newCompetingSearchNode != newNodes.end(); ++newCompetingSearchNode) {
            if ((*newSearchNode)->hypothesis_.back() == (*newCompetingSearchNode)->hypothesis_.back()) {
                (*newSearchNode)->levenshteinColumns_.push_back(newNodeLevenshteinEqualColumn);
                (*newSearchNode)->levenshteinScores_.push_back(semiring_->zero());
            }
            else {
                (*newSearchNode)->levenshteinColumns_.push_back(newNodeLevenshteinDifferColumn);
                (*newSearchNode)->levenshteinScores_.push_back((*newCompetingSearchNode)->totalProbability_);
            }
        }

        Fsa::Accumulator* collector = Fsa::LogSemiring->getCollector(Fsa::LogSemiring->zero());
        for (u32 i = 0; i < (*newSearchNode)->levenshteinScores_.size(); ++i) {
            collector->feed((*newSearchNode)->levenshteinScores_[i]);
        }
        (*newSearchNode)->estimate_ = collector->get();
        delete collector;

        (*newSearchNode)->index_ = stack_->size();
        stack_->push_back(*newSearchNode);
        stackEntries_.at(newLength_).push_back(*newSearchNode);

        if (!minimalNode_ || (*newSearchNode) < minimalNode_) {
            secondMinimalNode_ = minimalNode_;
            minimalNode_       = *newSearchNode;
        }
        else if (!secondMinimalNode_ || (*newSearchNode) < secondMinimalNode_) {
            secondMinimalNode_ = (*newSearchNode);
        }

        if ((*newSearchNode)->isExplorable_ && (!minimalIncompleteNode_ || (*newSearchNode) < minimalIncompleteNode_)) {
            minimalIncompleteNode_ = (*newSearchNode);
        }
    }
}

void MinimumBayesRiskAStarSearch::pruneStack() {
    if (stackEntries_.at(newLength_).size() > maxStackSize_) {
        sort(stackEntries_.at(newLength_).begin(), stackEntries_.at(newLength_).end());

        std::vector<u32>                     pruneIndices;
        bool                                 addIncomplete = 0;
        std::vector<SearchNodeRef>::iterator pruneNode     = stackEntries_.at(newLength_).begin() + maxStackSize_;
        for (; pruneNode != stackEntries_.at(newLength_).end(); ++pruneNode) {
            if ((*pruneNode)->index_ != minimalIncompleteNode_->index_) {
                pruneIndices.push_back((*pruneNode)->index_);
            }
            else {
                addIncomplete = true;
            }
        }

        sort(pruneIndices.begin(), pruneIndices.end());

        std::vector<SearchNodeRef>::iterator stackNode = stack_->begin();
        for (; stackNode != stack_->end(); ++stackNode) {
            std::vector<u32>::reverse_iterator pruneIndex = pruneIndices.rbegin();
            for (; pruneIndex != pruneIndices.rend(); ++pruneIndex) {
                (*stackNode)->levenshteinScores_.erase((*stackNode)->levenshteinScores_.begin() + *pruneIndex);
                (*stackNode)->levenshteinColumns_.erase((*stackNode)->levenshteinColumns_.begin() + *pruneIndex);
                if ((*stackNode)->isFinal()) {
                    (*stackNode)->levenshteinScoresOverestimate_.erase((*stackNode)->levenshteinScoresOverestimate_.begin() + *pruneIndex);
                }
            }
        }

        std::vector<u32>::reverse_iterator pruneIndex = pruneIndices.rbegin();
        for (; pruneIndex != pruneIndices.rend(); ++pruneIndex) {
            if ((*pruneIndex) < minimalIncompleteNode_->index_) {
                --minimalIncompleteNode_->index_;
            }
            stack_->erase(stack_->begin() + (*pruneIndex));
        }
        stackEntries_.at(newLength_).resize(maxStackSize_);
        if (addIncomplete) {
            stackEntries_.at(newLength_).push_back(minimalIncompleteNode_);
        }
    }
}

bool MinimumBayesRiskAStarSearch::stackIsOptimal() {
    if (minimalNode_->isFinal()) {
        if (f32(minimalNode_->totalProbability_) < -std::log(0.5)) {
            clog() << Core::XmlFull("stack-optimal", "0");
            return true;
        }
        else if (!secondMinimalNode_) {
            clog() << Core::XmlFull("stack-optimal", "1");
            return true;
        }
        else if (f32(minimalNode_->overestimate_) > f32(secondMinimalNode_->estimate_)) {
            clog() << Core::XmlFull("stack-optimal", "2");
            return true;
        }
        else if (!minimalIncompleteNode_) {
            clog() << Core::XmlFull("stack-optimal", "3");
            return true;
        }
    }
    return false;
}

void MinimumBayesRiskAStarSearch::performSearch(Fsa::ConstAutomatonRef fsa) {
    fsa_                 = fsa;
    evaluationSpaceSize_ = Fsa::countPaths(fsa_);
    summationSpaceSize_  = evaluationSpaceSize_;

    LogWeights.at(0) = semiring_->zero();
    for (u32 n = 1; n < MinimumBayesRiskAStarSearch::MaxLogWeights; ++n) {
        LogWeights.at(n) = Fsa::Weight(-f32(std::log(f32(n))));
    }

    exactEstimate_                                  = paramExactEstimate(config);
    backwardPotentials_                             = Fsa::sssp(Fsa::transpose(fsa_));
    inverseBackwardPotentialsNormalizationConstant_ = Fsa::LogSemiring->invert(backwardPotentials_.at(fsa_->initialStateId()));
    distances_                                      = getDistances(Fsa::transpose(fsa_));
    for (u32 n = 0; n < distances_.size(); ++n) {
        --distances_.at(n);
    }
    u32 longestDistance = distances_.at(fsa_->initialStateId()) - 1;
    fsa_                = Fsa::changeSemiring(fsa_, Fsa::LogSemiring);

    f32 threshold = initialPruningThreshold_;

    evaluationSpaceSize_ = u32(Fsa::countPaths(fsa_));
    while (threshold > 0 && evaluationSpaceSize_ > maximumNumberHypotheses_) {
        threshold *= thresholdFactor_;
        fsa_                 = Fsa::prunePosterior(fsa_, Fsa::Weight(threshold));
        evaluationSpaceSize_ = u32(Fsa::countPaths(fsa_));
    }
    summationSpaceSize_ = evaluationSpaceSize_;

    SearchNodeRef initialSearchNode(new SearchNode(longestDistance, fsa_->initialStateId()));
    if (!fsa_->getState(fsa_->initialStateId())->hasArcs()) {
        initialSearchNode->setInexplorable();
    }
    if (fsa_->getState(fsa_->initialStateId())->isFinal()) {
        initialSearchNode->setFinal();
    }

    stack_ = new std::vector<SearchNodeRef>();
    stack_->push_back(initialSearchNode);
    minimalNode_           = (*stack_->begin());
    minimalIncompleteNode_ = (*stack_->begin());

    while (!stackIsOptimal()) {
        expand();
    }

    clog() << Core::XmlFull("mbr-risk-underestimate", exp(-f32(minimalNode_->estimate_)));
    clog() << Core::XmlFull("mbr-risk-overestimate", exp(-f32(minimalNode_->overestimate_)));
    clog() << Core::XmlFull("mbr-probability", exp(-f32(minimalNode_->totalProbability_)));
    clog() << Core::XmlFull("overestimates", numberOverestimates_);

    // this has to changed to all paths in the original Fsa belonging to the best hypothesis.
    // therefore, either the interface of the search node has to be changed or the automaton
    // has to be created by a contouring algorithm.
    bestAutomaton_ = createLinearAutomatonFromVector(
            minimalNode_->hypothesis_,
            Fsa::Weight(minimalNode_->estimate_),
            fsa_->getInputAlphabet(),
            fsa_->getOutputAlphabet(),
            fsa_->semiring());

    delete stack_;
}

std::string MinimumBayesRiskAStarSearch::describe() const {
    return "stack";
}

}  //end namespace Search
