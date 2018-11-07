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
#include "ZeroOrderAlignAutomaton.hh"

namespace Fsa {

    ZeroOrderAlignAutomaton::ZeroOrderAlignAutomaton(Core::Configuration &config,
                                                     const std::string& source,
                                                     const std::string& target,
                                                     const TransitionProbs& transitionProbs,
                                                     Translation::ConstConditionalLexiconRef lexicon,
                                                     const double factorLexicon,
                                                     const double factorTransition) :
       AlignAutomaton(config,source,target,transitionProbs, factorLexicon, factorTransition),
        lexicon_(lexicon), I(outputSentence_.size()), J(inputSentence_.size()),
        M(3), maxIndex((I+1)*(J+1))
    {}

    ConstStateRef ZeroOrderAlignAutomaton::getState(StateId s) const {
        State *sp = new State(s);

        u32 si(s);
        u32 m(si % M);
        u32 i(si / M / (J+1));
        u32 j(si / M % (J+1));

        //            std::cerr << "getState(" << s << ")"
        //                      << " m=" << m << "/" << M
        //                      << " i=" << i << "/" << I
        //                      << " j=" << j << "/" << J
        //                      << std::endl;

        bool doHorizontal(m==diagonal || m==horizontal);
        bool doVertical(m==diagonal || m==vertical);

        if (i < I && doVertical) { // when not at the TOP of lattice, make vertical movement
            std::vector<std::string> x;
            x.push_back("NULL");
            x.push_back(target_[i]);

            StateId targetState( si +(J+1)*M -m+vertical) ;

            f64 weight(lexicon_->getProb(vertical,x) * factorLexicon_);
       weight += factorTransition_ * transitionProbs_[vertical];

            sp->newArc(targetState, Fsa::Weight(weight), Epsilon, outputSentence_[i]);
            //std::cerr << si << "->" << int(targetState) << std::endl;
        }
        if (j < J && doHorizontal) { // when not at the RIGHT border of the lattice make horizontal movement
            StateId targetState( si +M -m+horizontal ) ;
            for (u32 jt=0; jt<J; ++jt) {
                std::vector<std::string> x;
                x.push_back(source_[jt]);
                x.push_back("NULL");

                f64 weight(lexicon_->getProb(horizontal,x) * factorLexicon_);
      weight += factorTransition_ * transitionProbs_[horizontal];
                //weight = -log((1-transitionProbs_.alpha) * exp(-weight) + transitionProbs_.alpha * transitionProbs_.h);

                sp->newArc(targetState, Fsa::Weight(weight), inputSentence_[jt], Epsilon);
            }
            //std::cerr << si << "->" << int(targetState) << std::endl;
        }
        if ((j < J) && (i < I)) { // when in the middle of the lattice, do diagonal movment
            StateId targetState( si +(J+1)*M +M -m+diagonal ) ;
            for (u32 jt=0; jt<J; ++jt) {
                std::vector<std::string> x;
                x.push_back(source_[jt]);
                x.push_back(target_[i]);

                f64 weight(lexicon_->getProb(diagonal,x) * factorLexicon_);
      weight += factorTransition_ * transitionProbs_[diagonal];
                //weight = -log((1-transitionProbs_.alpha) * exp(-weight)
                //              + transitionProbs_.alpha * transitionProbs_.d);

                sp->newArc(targetState, Fsa::Weight(weight), inputSentence_[jt], outputSentence_[i]);
            }
            //std::cerr << si << "->" << int(targetState) << std::endl;
        }

        if (i==I && j==J) {
            sp->setFinal(this->semiring()->one());
        }
        return ConstStateRef(sp);
    }
}
