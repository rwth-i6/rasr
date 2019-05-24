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
#include "MinimumBayesRiskNBestListSearch.hh"
#include <Core/ProgressIndicator.hh>
#include "MinimumBayesRiskSearchUtil.hh"

namespace Search {

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                                 */
/*                 MinimumBayesRiskNBestListSearch                                 */
/*                                                                                 */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
MinimumBayesRiskNBestListSearch::MinimumBayesRiskNBestListSearch(const Core::Configuration& config)
        : MinimumBayesRiskSearch(config) {}

MinimumBayesRiskNBestListSearch::~MinimumBayesRiskNBestListSearch() {}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                                 */
/*                 MinimumBayesRiskNBestListNaiveSearch                            */
/*                                                                                 */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
const Core::ParameterInt MinimumBayesRiskNBestListNaiveSearch::paramNumberHypothesesSummation(
        "number-hypotheses-summation",
        "number of hypothese in nbestlist for summation",
        Core::Type<u32>::max);

const Core::ParameterInt MinimumBayesRiskNBestListNaiveSearch::paramNumberHypothesesEvaluation(
        "number-hypotheses-evaluation",
        "number of hypothese in nbestlist for evaluation",
        Core::Type<u32>::max);

MinimumBayesRiskNBestListNaiveSearch::MinimumBayesRiskNBestListNaiveSearch(
        const Core::Configuration& config)
        : MinimumBayesRiskNBestListSearch(config) {}

MinimumBayesRiskNBestListNaiveSearch::~MinimumBayesRiskNBestListNaiveSearch() {}

bool MinimumBayesRiskNBestListNaiveSearch::oneHalfCriterion() const {
    return Fsa::LogSemiring->compare(mapProbability_, Fsa::Weight(-std::log(0.5))) <= 0;
}

// Sum_{l=1} p + 2 p_map >= 1 + max_{l=1} p
void MinimumBayesRiskNBestListNaiveSearch::setDistanceOneCriterion(const Fsa::Weight& distanceOne, const Fsa::Weight& maxDistanceOne) {
    distanceOneCriterion_ = Fsa::LogSemiring->compare(Fsa::LogSemiring->collect(distanceOne,
                                                                                Fsa::LogSemiring->extend(Fsa::Weight(-std::log(2.0)),
                                                                                                         mapProbability_)),
                                                      Fsa::LogSemiring->collect(Fsa::LogSemiring->one(),
                                                                                maxDistanceOne)) <= 0;
}

void MinimumBayesRiskNBestListNaiveSearch::performSearch(Fsa::ConstAutomatonRef nBestList) {
    nBestList = normalizeNbestlist(nBestList);

    evaluationSpaceSize_ = paramNumberHypothesesEvaluation(config);
    summationSpaceSize_  = paramNumberHypothesesSummation(config);

    Fsa::ConstStateRef initialState = nBestList->getState(nBestList->initialStateId());
    evaluationSpaceSize_            = std::min(evaluationSpaceSize_, initialState->nArcs());
    summationSpaceSize_             = std::min(summationSpaceSize_, initialState->nArcs());
    u32 nBestListSize               = std::max(evaluationSpaceSize_, summationSpaceSize_);

    require(initialState->nArcs() > 0);
    require(summationSpaceSize_ > 0);
    require(evaluationSpaceSize_ > 0);

    // generate vector of string hypotheses
    std::vector<StringHypothesis> hypotheses;
    u32                           n   = 0;
    Fsa::State::const_iterator    arc = initialState->begin();
    while (n < nBestListSize) {
        Fsa::ConstAutomatonRef    sentenceFsa = Fsa::partial(nBestList, arc->target());
        std::vector<Fsa::LabelId> sentence;
        Fsa::getLinearInput(sentenceFsa, sentence);
        hypotheses.push_back(StringHypothesis(sentence, arc->weight()));
        ++arc;
        ++n;
    }

    mapSentence_    = hypotheses.front().sentence_;
    mapProbability_ = hypotheses.front().probability_;
    mbrSentence_    = mapSentence_;
    mbrProbability_ = mapProbability_;

    Fsa::Weight distanceOne;
    Fsa::Weight maxDistanceOne;
    mapRisk_ = posteriorRiskNBestList(mapSentence_, hypotheses, distanceOne, maxDistanceOne);
    setDistanceOneCriterion(distanceOne, maxDistanceOne);
    mbrRisk_ = mapRisk_;

    clog() << Core::XmlFull("map-probability", exp(-f32(mapProbability_)));
    clog() << Core::XmlFull("map-risk", exp(-f32(mapRisk_)));

    Fsa::Weight risk        = Fsa::LogSemiring->zero();
    u32         mbrPosition = 0;

    Core::ProgressIndicator p("hypotheses", "");
    p.start(evaluationSpaceSize_);

    /**
     * If one of the criteria is fulfilled, then the searchcomputation does not have to be
     * carried out, but the MAP sentence is guaranteed to minimize the Bayes risk w.r.t.
     * Levenshtein loss. Only if neither of both criteria is fulfilled, the search will
     * be carried out.
     */
    if (oneHalfCriterion()) {
        clog() << Core::XmlFull("one-half-criterion", true);
        clog() << Core::XmlFull("distance-one-criterion", distanceOneCriterion());
    }
    else if (distanceOneCriterion()) {
        clog() << Core::XmlFull("one-half-criterion", false);
        clog() << Core::XmlFull("distance-one-criterion", true);
    }
    else {
        clog() << Core::XmlFull("one-half-criterion", false);
        clog() << Core::XmlFull("distance-one-criterion", false);

        // MAP hypothesis has been accomplished, so we start with the second hypothesis
        for (u32 n = 1; n < hypotheses.size(); ++n) {
            StringHypothesis& hypothesis = hypotheses[n];
            p.notify();
            Fsa::Weight risk = posteriorRiskNBestList(hypothesis.sentence_, hypotheses, mbrRisk_);
            if (Fsa::LogSemiring->compare(risk, mbrRisk_) > 0) {
                mbrSentence_    = hypothesis.sentence_;
                mbrProbability_ = hypothesis.probability_;
                mbrRisk_        = risk;
                mbrPosition     = n;
            }
        }
    }

    p.finish();

    bestAutomaton_ = Fsa::partial(nBestList,
                                  (nBestList->getState(nBestList->initialStateId())->begin() + mbrPosition)->target());

    clog() << Core::XmlFull("mbr-risk", exp(-f32(mbrRisk_)));
    clog() << Core::XmlFull("mbr-position", mbrPosition);

    MinimumBayesRiskSearch::performSearch(nBestList);
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                                 */
/*                 posteriorRiskNBestList: initialization with true sentence       */
/*                                                                                 */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
Fsa::Weight posteriorRiskNBestList(const std::vector<Fsa::LabelId>&     trueSentence,
                                   const std::vector<StringHypothesis>& hypotheses,
                                   Fsa::Weight& distanceOne, Fsa::Weight& maxDistanceOne) {
    Fsa::Weight result;

    maxDistanceOne = Fsa::LogSemiring->zero();

    // to be deleted!!!
    Fsa::Accumulator* collectorResult = Fsa::LogSemiring->getCollector(Fsa::LogSemiring->zero());
    // to be deleted!!!
    Fsa::Accumulator* collectorDistanceOne = Fsa::LogSemiring->getCollector(Fsa::LogSemiring->zero());

    std::vector<StringHypothesis>::const_iterator hypothesis = hypotheses.begin();
    for (; hypothesis != hypotheses.end(); ++hypothesis) {
        u32 lDistance = levenshteinDistance(trueSentence, hypothesis->sentence_);
        if (lDistance > 0) {
            collectorResult->feed(
                    Fsa::LogSemiring->extend(
                            hypothesis->probability_,
                            Fsa::Weight(-std::log(f32(lDistance)))));
        }
        if (lDistance == 1) {
            collectorDistanceOne->feed(hypothesis->probability_);
            if (Fsa::LogSemiring->compare(hypothesis->probability_, maxDistanceOne) < 0) {
                maxDistanceOne = hypothesis->probability_;
            }
        }
    }

    result      = collectorResult->get();
    distanceOne = collectorDistanceOne->get();

    delete collectorResult;
    delete collectorDistanceOne;

    return result;
}

Fsa::Weight posteriorRiskNBestList(const std::vector<Fsa::LabelId>&     trueSentence,
                                   const std::vector<StringHypothesis>& hypotheses,
                                   const Fsa::Weight&                   pruningThreshold) {
    Fsa::Weight result;

    // To be deleted!!!
    Fsa::Accumulator* collectorResult = Fsa::LogSemiring->getCollector(Fsa::LogSemiring->zero());

    std::vector<StringHypothesis>::const_iterator hypothesis = hypotheses.begin();
    for (; hypothesis != hypotheses.end(); ++hypothesis) {
        u32 lDistance = levenshteinDistance(trueSentence, hypothesis->sentence_);
        if (lDistance > 0) {
            collectorResult->feed(Fsa::LogSemiring->extend(hypothesis->probability_,
                                                           Fsa::Weight(-std::log(f32(lDistance)))));
        }
        if (Fsa::LogSemiring->compare(collectorResult->get(), pruningThreshold) < 0) {
            delete collectorResult;
            return Fsa::Weight(Core::Type<f32>::min);
        }
    }

    result = collectorResult->get();
    delete collectorResult;
    return result;
}

}  //end namespace Search
