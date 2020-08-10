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
#include "AlignAutomaton.hh"

#include "Common.hh"

namespace Fsa {

AlignAutomaton::AlignAutomaton(Core::Configuration&   config,
                               const std::string&     source,
                               const std::string&     target,
                               const TransitionProbs& transitionProbs,
                               const double           factorLexicon,
                               const double           factorTransition)
        : source_(Core::split(source, " ")),
          target_(Core::split(target, " ")),
          transitionProbs_(transitionProbs),
          inputSentence_(),
          outputSentence_(),
          factorLexicon_(factorLexicon),
          factorTransition_(factorTransition)

{
    this->setProperties(PropertyAcyclic || PropertyLinear, PropertyAcyclic || PropertyLinear);

    std::vector<std::string> numberedSource = numberTokensVector(source_);
    StaticAlphabet*          ia             = new StaticAlphabet();
    for (std::vector<std::string>::const_iterator i = numberedSource.begin(); i != numberedSource.end(); ++i) {
        inputSentence_.push_back(ia->addSymbol(*i));
    }
    inputAlphabet_ = ConstAlphabetRef(ia);

    std::vector<std::string> numberedTarget = numberTokensVector(target_);
    StaticAlphabet*          oa             = new StaticAlphabet();
    for (std::vector<std::string>::const_iterator i = numberedTarget.begin(); i != numberedTarget.end(); ++i) {
        outputSentence_.push_back(oa->addSymbol(*i));
    }
    outputAlphabet_ = ConstAlphabetRef(oa);
}
}  // namespace Fsa
