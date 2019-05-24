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
#include "MinimumBayesRiskSearch.hh"
#include <Fsa/Best.hh>

namespace Search {
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                                 */
/*                 MinimumBayesRiskSearch                                          */
/*                                                                                 */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

MinimumBayesRiskSearch::MinimumBayesRiskSearch(const Core::Configuration& config)
        : Component(config),
          fsa_(Fsa::ConstAutomatonRef()),
          bestAutomaton_(Fsa::ConstAutomatonRef()),
          evaluationSpaceSize_(0),
          summationSpaceSize_(0),
          numberEvaluations_(0),
          numberComputations_(0),
          statisticsChannel_(config, "statistics") {}

u32 MinimumBayesRiskSearch::getEvaluationSpaceSize() const {
    return evaluationSpaceSize_;
}

u32 MinimumBayesRiskSearch::getSummationSpaceSize() const {
    return summationSpaceSize_;
}

u32 MinimumBayesRiskSearch::getNumberEvaluations() const {
    return numberEvaluations_;
}

u32 MinimumBayesRiskSearch::getNumberComputations() const {
    return numberComputations_;
}

void MinimumBayesRiskSearch::performSearch(Fsa::ConstAutomatonRef fsa) {
    if (statisticsChannel_.isOpen()) {
        statisticsChannel_ << Core::XmlFull("evaluation-space-size", getEvaluationSpaceSize());
        statisticsChannel_ << Core::XmlFull("summation-space-size", getSummationSpaceSize());

        statisticsChannel_ << Core::XmlFull("number-evaluations", getNumberEvaluations());
        statisticsChannel_ << Core::XmlFull("number-computations", getNumberComputations());
    }
}

Fsa::ConstAutomatonRef MinimumBayesRiskSearch::getBestAutomaton() const {
    return bestAutomaton_;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                                 */
/*                 MinimumBayesRiskMapSearch                                       */
/*                                                                                 */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

MinimumBayesRiskMapSearch::MinimumBayesRiskMapSearch(const Core::Configuration& config)
        : MinimumBayesRiskSearch(config) {}

void MinimumBayesRiskMapSearch::performSearch(Fsa::ConstAutomatonRef fsa) {
    bestAutomaton_ = Fsa::best(fsa);
}

}  //end namespace Search
