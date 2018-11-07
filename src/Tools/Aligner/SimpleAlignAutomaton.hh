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
#ifndef SIMPLE_ALIGN_AUTOMATON
#define SIMPLE_ALIGN_AUTOMATON

#include "AlignAutomaton.hh"
#include "ConditionalLexicon.hh"


// history is not treated correctly.

namespace Fsa {

    class SimpleAlignAutomaton : public AlignAutomaton {
    private:
        Translation::ConstConditionalLexiconRef lexicon_;
        const unsigned order_;
        unsigned I_;
        unsigned J_;
        unsigned maxIndex_;

        Fsa::LabelId nullWordIndex_;
        Fsa::LabelId sentenceBeginPaddingSymbolIndex_;

        std::vector<Fsa::LabelId> mappedSourceSentence_;
        std::vector<Fsa::LabelId> mappedTargetSentence_;

        Fsa::StateId indices2StateId(const unsigned nCoveredTargetWords,
                                     const unsigned nCoveredSourceWords,
                                     const std::vector<unsigned>& sourcePositionHistory) const {
            //assert(sourcePositionHistory.size()==order_);
            //std::cerr << "indices2StateId nCoveredSourceWords=" << nCoveredSourceWords;
            #if 0
            for (unsigned j=0; j<sourcePositionHistory.size();++j) {
                std::cerr << " " << sourcePositionHistory[j];
            }
            #endif
            unsigned multiplyer=I_;
            unsigned stateNumber=nCoveredTargetWords+multiplyer*nCoveredSourceWords;
            for (unsigned n=0; n<order_; ++n) {
                multiplyer *= J_;
                stateNumber += multiplyer * sourcePositionHistory[n];
            }
            //std::cerr << " StateId: " << stateNumber << std::endl;
            return StateId(stateNumber);

        }

        void stateId2Indices(const StateId stateId, unsigned& nCoveredTargetWords, unsigned& nCoveredSourceWords, std::vector<unsigned>& sourcePositionHistory) const {
            sourcePositionHistory.resize(order_);
            unsigned stateNumber(stateId);
            unsigned divisor=I_;

            // not sure about the behaviour of this pow function for negative exponents
            if (order_>0) {
                divisor *= unsigned(::pow(J_,order_));
            }
            #if 0
            std::cerr << "divisor=" << divisor << std::endl;
            #endif
            for (int n=order_-1; n>=0; --n) {
                sourcePositionHistory[n] = stateNumber / divisor;
                //stateNumber -= (stateNumber/divisor) * divisor;
                stateNumber %= divisor;
                //if (n>0) divisor /= J_;
                divisor /= J_;
                #if 0
                std::cerr << "-divisor=" << divisor << std::endl;
                #endif
            }
            nCoveredSourceWords = stateNumber / divisor;
            nCoveredTargetWords = stateNumber % divisor;
        }

        void generateDiagonalOrHorizontalArcs_(State* s,
                                               const unsigned targetIndexIncrement,
                                               const unsigned nCoveredTargetWords,
                                               const unsigned nCoveredSourceWords,
                                               const std::vector<unsigned>& currentJ) const ;

    public:
        SimpleAlignAutomaton(Core::Configuration &config,
                                      const std::string& source,
                                      const std::string& target,
                                      const TransitionProbs& transitionProbs,
                             Translation::ConstConditionalLexiconRef lexicon,
                                      const double factorLexicon = 1.0,
                                                                const double factorTransition = 1.0,
                                      const unsigned order = 0);
        virtual ConstStateRef getState(StateId s) const;
        virtual Fsa::StateId initialStateId() const { return Fsa::StateId(0);};
        virtual std::string describe() const { return std::string("SimpleAlignAutomaton()"); }
    };
}
#endif
