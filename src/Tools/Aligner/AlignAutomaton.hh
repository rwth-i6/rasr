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
#ifndef ALIGN_AUTOMATON
#define ALIGN_AUTOMATON

#include <Core/Configuration.hh>
//#include <Fsa/Alphabet.hh>
//#include <Fsa/Automaton.hh>
//#include <Fsa/Basic.hh>
//#include <Fsa/Static.hh>
//#include <Fsa/Vector.hh>
//#include <Fsa/Types.hh>

//#include "TranslationLexicon.hh"
#include "TransitionProbs.hh"
#include <Translation/Common.hh>
//#include "ConditionalLexicon.hh"

namespace Fsa {
         typedef enum {
                  modelSimple,
                  modelZeroOrder,
                  modelZeroOrderNoEmpty,
                  modelConditional
    } models ;


         const Core::Choice modelChoice(
                  "simple", modelSimple,
                  "zeroOrder", modelZeroOrder,
                  "zeroOrderNoEmpty", modelZeroOrderNoEmpty,
                  "conditional", modelConditional,
                  Core::Choice::endMark()
                  );
}
//typedef enum {zeroorder=00,
//	      zeroordernoempty=01,
//	      zeroordersimple=02,
//	      conditional=10,
//	      hvconditional=11,
//	      firstorder=20,
//	      firstorderplus=21,
//	      pooledfirstorder=22,
//	      plain=4,
//	      simple=5,
//	      unknown=666} AlignmentModel;

namespace Fsa {

    class AlignAutomaton : public Automaton {
    protected:
                  std::vector<std::string> source_;
                  std::vector<std::string> target_;
                  const TransitionProbs transitionProbs_;
                  ConstAlphabetRef inputAlphabet_;
                  ConstAlphabetRef outputAlphabet_;
                  std::vector<LabelId> inputSentence_;
                  std::vector<LabelId> outputSentence_;
                  double factorLexicon_;
                  double factorTransition_;
                  AlignAutomaton(Core::Configuration &config,
                                                          const std::string& source,
                                                          const std::string& target,
                                                          const TransitionProbs& transitionProbs,
                                                          const double factorLexicon = 1.0,
                                const double factorTransition = 1.0);
    public:
                  virtual std::string describe() const { return std::string("AlignAutomaton()"); }
                  virtual ConstAlphabetRef getInputAlphabet() const { return inputAlphabet_; }
                  virtual ConstAlphabetRef getOutputAlphabet() const { return outputAlphabet_; }
                  virtual Type type() const { return TypeTransducer; }
                  virtual StateId initialStateId() const { return StateId(0); }
                  virtual ConstSemiringRef semiring() const { return TropicalSemiring; }

    };

}

#endif
