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
#ifndef _SEARCH_MINIMUM_BAYES_RISK_NBESTLISTSEARCH_HH
#define _SEARCH_MINIMUM_BAYES_RISK_NBESTLISTSEARCH_HH

#include "MinimumBayesRiskSearch.hh"
#include "MinimumBayesRiskSearchUtil.hh"

namespace Search {

/**
 * Interface for search process of minimum Bayes risk using nbestlists
 * as evaluation space, i.e. the set of hypotheses over which the minimization
 * is carried out.
 *
 */
class MinimumBayesRiskNBestListSearch : public MinimumBayesRiskSearch {
protected:
    /** The overall number of hypotheses. */
    static const Core::ParameterInt paramNumberHypotheses;

    /** The string having maximum a-posteriori probability. */
    std::vector<Fsa::LabelId> mapSentence_;

    /** The negative logarithm of the probability of the MAP string. */
    Fsa::Weight mapProbability_;

    /** The negative logarithm of the Bayes risk of the MAP string. */
    Fsa::Weight mapRisk_;

    /** The string having minimal Bayes risk. */
    std::vector<Fsa::LabelId> mbrSentence_;

    /** The negative logarithm of the probability of the sentence minimizing the Bayes risk. */
    Fsa::Weight mbrProbability_;

    /** The negative logarithm of the minimal Bayes risk.*/
    Fsa::Weight mbrRisk_;

    /** to be deleted in dtor. */
    std::vector<StringHypothesis>* summationSpace_;

    /** */
    std::vector<StringHypothesis>* evaluationSpace_;

public:
    /**
     * Initialization of probs to zero, sentences to empty sentences.
     *
     */
    MinimumBayesRiskNBestListSearch(const Core::Configuration&);

    /**
     * Virtual destructor.
     */
    virtual ~MinimumBayesRiskNBestListSearch();

    /**
     * The acutal process of computing minimum Bayes risk.
     *
     */
    virtual void performSearch(Fsa::ConstAutomatonRef nBestList) = 0;
};

/**
 * Implementation of the naive approach for minimum Bayes risk using an nbestlist for both
 * the evaluation space as well as for the summation space.
 * The search can be performed on two different nbestlists depending on how much hypotheses
 * shall be used for minimization or summation. In order to stay conform to the MinimumBayesRiskSearch
 * interface, the search itself will be given a single nbestlist from which the two lists
 * for evaluation space and summation space will be extracted up to a configurable number
 * of string hypotheses.
 * In addition there exist two criteria which can abbreviate the search process. Their application
 * can be configured. If fulfilled, this information will be logged to the information channel.
 * It should be noted that the computation of the two criteria do not afford additional computation
 * time but come along with the computation process.
 *
 */
class MinimumBayesRiskNBestListNaiveSearch : public MinimumBayesRiskNBestListSearch {
private:
    /** Flag whether the distance-one criterion is fulfilled. */
    bool distanceOneCriterion_;

    /** The number of hypotheses used for summation. */
    static const Core::ParameterInt paramNumberHypothesesSummation;

    /** The number of hypotheses used for minimization.*/
    static const Core::ParameterInt paramNumberHypothesesEvaluation;

private:
    bool oneHalfCriterion() const;
    bool distanceOneCriterion() const {
        return distanceOneCriterion_;
    }
    void setDistanceOneCriterion(const Fsa::Weight& distanceOne,
                                 const Fsa::Weight& maxDistanceOne);

public:
    /**
     * Standard constructor.
     *
     * @param config  the configuration.
     */
    MinimumBayesRiskNBestListNaiveSearch(const Core::Configuration&);

    /**
     * Virtual destructor.
     */
    virtual ~MinimumBayesRiskNBestListNaiveSearch();

    /**
     * Performs the search for the string with minimal Bayes risk on the specified nbestlist.
     * @param nBestList the nbestlist on which the search will be performed.
     */
    virtual void performSearch(Fsa::ConstAutomatonRef nBestList);

    /**
     *
     * @return the actual value of the Bayes risk.
     */
    Fsa::Weight getMinimumBayesRisk() const;
};

/**
 * Naive computation of Bayes posterior risk assuming the Levenshtein loss function.
 * This method is usually used for computation of Bayes risk for the MAP hypothesis,
 * as it sets the values needed for checking whether the distance-one criterion
 * is fulfilled.
 *
 * @param trueSentence      the sentence for which the risk shall be computed.
 * @param hypotheses        vector with concurrent hypotheses
 * @param distanceOne       this parameter will store the accumulated weight of all hypotheses with Levenshtein distance = 1.
 * @param maxdistanceOne    this parameter will store the maximum weight of all hypotheses with Levenshtein distance = 1.
 *
 * @return the Bayes risk of the true sentence as Fsa::Weight.
 */
Fsa::Weight posteriorRiskNBestList(
        const std::vector<Fsa::LabelId>&     trueSentence,
        const std::vector<StringHypothesis>& hypotheses,
        Fsa::Weight&                         distanceOne,
        Fsa::Weight&                         maximumDistanceOne);

/**
 * Naive computation of Bayes posterior risk assuming the Levenshtein loss function.
 * If the present risk exceeds the pruning threshold the computation will
 * be stopped immidiately and the maximum possible value for the weight type
 * will be returned.
 *
 * @param trueSentence      the sentence for which the risk shall be computed.
 * @param hypotheses        vector with concurrent hypotheses
 * @param pruningThreshold  pruning threshold
 *
 * @return the Bayes risk of the true sentence as Fsa::Weight.
 */
Fsa::Weight posteriorRiskNBestList(
        const std::vector<Fsa::LabelId>&     trueSentence,
        const std::vector<StringHypothesis>& hypotheses,
        const Fsa::Weight&                   pruningThreshold = Fsa::LogSemiring->max());

}  // end namespace Search

#endif  //_SEARCH_MINIMUM_BAYES_RISK_NBESTLISTSEARCH_HH
