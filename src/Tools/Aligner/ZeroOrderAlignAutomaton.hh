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
#ifndef ZERO_ORDER_ALIGN_AUTOMATON
#define ZERO_ORDER_ALIGN_AUTOMATON

#include "AlignAutomaton.hh"
#include "ConditionalLexicon.hh"

namespace Fsa {
class ZeroOrderAlignAutomaton : public AlignAutomaton {
private:
    typedef enum {
        diagonal   = 0,
        horizontal = 1,
        vertical   = 2
    } Transitions;

    Translation::ConstConditionalLexiconRef lexicon_;

    u32 I;
    u32 J;
    u32 M;
    u32 maxIndex;

public:
    ZeroOrderAlignAutomaton(Core::Configuration&                    config,
                            const std::string&                      source,
                            const std::string&                      target,
                            const TransitionProbs&                  transitionProbs,
                            Translation::ConstConditionalLexiconRef lex,
                            const double                            factorLexicon    = 1.0,
                            const double                            factorTransition = 1.0);

    virtual ConstStateRef getState(StateId s) const;

    virtual std::string describe() const {
        return std::string("zeroOrderAlignAutomaton()");
    }
};
}  // namespace Fsa
#endif
