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
#ifndef _SEARCH_MINIMUM_BAYES_RISK_SEARCH_HH
#define _SEARCH_MINIMUM_BAYES_RISK_SEARCH_HH

#include <Core/Component.hh>
#include <Fsa/Automaton.hh>

namespace Search {
/**
 * Interface for performing a search for the best string in an Fsa w.r.t. minimum Bayes risk.
 * This kind of search takes only an Fsa and a configuration as an argument.
 * For statistics purposes, the number of evaluated hypotheses and the number of computations
 * performed during the search can be optionally computed by the implementing classes.
 *
 */
class MinimumBayesRiskSearch : public Core::Component {
protected:
    /** The automaton to be searched. */
    Fsa::ConstAutomatonRef fsa_;

    /** The optimal string represented as a linear Fsa. */
    Fsa::ConstAutomatonRef bestAutomaton_;

    /**
     * The number of hypotheses which can be evaluated during the search.
     * I.e. the set of hypotheses over which the minimization will be performed.
     */
    u32 evaluationSpaceSize_;

    /**
     * The number of hypotheses which are used for the summation in the Bayes risk formula.
     */
    u32 summationSpaceSize_;

    /** The number of hypotheses that were actually used for minimization. */
    u32 numberEvaluations_;

    /** The number of relevant computations depending on the kind of search. */
    u32 numberComputations_;

    Core::XmlChannel statisticsChannel_;

public:
    /**
     * The only arguments a minimum Bayes risk search takes are a configuration.
     * The automaton being searched will be a parameter of the execution of the search.
     *
     * @param config  configuration of the search.
     */
    MinimumBayesRiskSearch(const Core::Configuration&);

    /**
     * Virtual desructor without special purpose.
     */
    virtual ~MinimumBayesRiskSearch() {}

    /**
     * The actual search process will be invoked calling this function.
     *
     * @param fsa  the automaton to be searched.
     */
    virtual void performSearch(Fsa::ConstAutomatonRef fsa);

    /**
     * Getter for <code> bestAutomaton </code>.
     * Returns the sentence with minimum Bayes risk as linear Fsa wich has
     * the posterior probability as its weight.
     *
     * Note: This value is only defined after performSearch was called!
     *
     * @return the sentence having minumum Bayes risk as linear Fsa.
     */
    virtual Fsa::ConstAutomatonRef getBestAutomaton() const;

    /**
     * Getter for <code> evaluationSpace </code>.
     *
     * Note: This value is only defined after performSearch was called!
     *
     * @return the number of hypotheses used for minimization of bayes risk.
     */
    u32 getEvaluationSpaceSize() const;

    /**
     * Getter for <code> summationSpaceSize </code>.
     *
     * Note: This value is only defined after performSearch was called!
     *
     * @return  the number of hypotheses over which summation will be carried out for Bayes risk.
     */
    u32 getSummationSpaceSize() const;

    /**
     * Getter for <code> numberEvaluations </code>.
     * Hopefully, the actual number of evaluations is much lower than the number of hypotheses
     * that have to be (theoretically) used.
     *
     * Note: This value is only defined after performSearch was called!
     *
     * @return  the number of hypotheses actually used for minimization.
     */
    u32 getNumberEvaluations() const;

    /**
     * Getter for <code> numberComputations </code>.
     * The number of relevant computations (e.g. number of Levenshtein distances) strongly depends
     * on the kind of search.
     *
     * Note: This value is only defined after performSearch was called!
     *
     * @return  the number of computations performed during the search.
     *
     */
    u32 getNumberComputations() const;
};

/**
 * The standard approach to speech recognition is based on a 0-1-loss Bayes decision rule.
 * This always chooses the hypothesis with the maximum a-posteriori probability and hence
 * the best hypothesis will always be the path in the Fsa with the highest or lowest score
 * (depending on how the scores i.e. the semiring is defined).
 */
class MinimumBayesRiskMapSearch : public MinimumBayesRiskSearch {
public:
    /**
     * The only arguments a minimum Bayes risk search takes are a configuration.
     * The automaton being searched will be a parameter of the execution of the search.
     *
     * @param config  configuration of the search.
     */
    MinimumBayesRiskMapSearch(const Core::Configuration&);

    /**
     * As the optimal hypothesis is the hypothesis with the highest/lowest score,
     * only Fsa::best will be performed.
     *
     * @param fsa  the automaton to be searched.
     */
    virtual void performSearch(Fsa::ConstAutomatonRef fsa);
};

}  //end namespace Search

#endif  //_SEARCH_MINIMUM_BAYES_RISK_SEARCH_HH
