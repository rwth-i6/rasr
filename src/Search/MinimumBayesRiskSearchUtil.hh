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
#ifndef _SEARCH_MINIMUM_BAYES_RISK_UTIL_HH
#define _SEARCH_MINIMUM_BAYES_RISK_UTIL_HH

#include <Core/Types.hh>

#include <Fsa/Arithmetic.hh>
#include <Fsa/Automaton.hh>
#include <Fsa/Basic.hh>
#include <Fsa/Compose.hh>
#include <Fsa/Levenshtein.hh>
#include <Fsa/Linear.hh>
#include <Fsa/Sssp.hh>

#include <set>
#include <vector>

namespace Search {

typedef std::vector<Fsa::LabelId> Sentence;

struct StringHypothesis {
public:
    std::vector<Fsa::LabelId> sentence_;
    Fsa::Weight               probability_;

    StringHypothesis() {}  //end StringHypothesis

    StringHypothesis(std::vector<Fsa::LabelId> sentence, Fsa::Weight probability)
            : sentence_(sentence), probability_(probability) {}  //end StringHypothesis
};                                                               //end StringHypothesis

typedef std::vector<StringHypothesis> HypothesisVector;

Fsa::Weight getNbestNormalizationConstant(Fsa::ConstAutomatonRef nbestlist);

Fsa::ConstAutomatonRef normalizeNbestlist(Fsa::ConstAutomatonRef nbestlist);

Fsa::ConstAutomatonRef partialNbestlist(Fsa::ConstAutomatonRef nbestlist, u32 size);

/**
 * Compute levenshtein distance between two sentences using the dynamic programming algorithm.
 *
 * @return levenshtein distance between sentence A and sentence B
 */
u32 levenshteinDistance(const std::vector<Fsa::LabelId>& A, const std::vector<Fsa::LabelId>& B);

/**
 *
 * @return contour for all states in oldContour
 */
std::set<Fsa::StateId> getContour(std::set<Fsa::StateId> oldContour, Fsa::ConstAutomatonRef fsa);

/**
 * Computes the longest distance for a state from the source of the fsa.
 *
 * @return vector assigning each state id a distance.
 */
std::vector<Fsa::StateId> getDistances(Fsa::ConstAutomatonRef fsa);

Fsa::ConstAutomatonRef createLinearAutomatonFromVector(const std::vector<Fsa::LabelId>& sequence,
                                                       const Fsa::Weight&               score          = Fsa::Weight(),
                                                       Fsa::ConstAlphabetRef            inputAlphabet  = Fsa::ConstAlphabetRef(),
                                                       Fsa::ConstAlphabetRef            outputAlphabet = Fsa::ConstAlphabetRef(),
                                                       Fsa::ConstSemiringRef            semiring       = Fsa::ConstSemiringRef(Fsa::UnknownSemiring));

Fsa::Weight posteriorExpectedRisk(Fsa::ConstAutomatonRef center, Fsa::ConstAutomatonRef hypotheses);

Fsa::Weight collectWeights(Fsa::ConstSemiringRef sr, const Core::Vector<Fsa::Weight>& weights);

}  //end namespace Search

#endif  // _SEARCH_MINIMUM_BAYES_RISK_UTIL_HH
