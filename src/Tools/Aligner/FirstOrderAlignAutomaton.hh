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
#ifndef FIRST_ORDER_ALIGN_AUTOMATON
#define FIRST_ORDER_ALIGN_AUTOMATON

#include "AlignAutomaton.hh"

namespace Fsa {
class FirstOrderAlignAutomaton : public AlignAutomaton {
private:
    typedef enum {
        diagonal   = 0,
        horizontal = 1,
        vertical   = 2
    } Transitions;
    const TranslationLexicon* lexicon_;
    const TranslationLexicon* horizontal_lexicon_;
    const TranslationLexicon* vertical_lexicon_;
    u32                       I;
    u32                       J;
    u32                       M;
    u32                       maxIndex;
    bool                      context_diagonal_;

public:
    // make all this abstract
    FirstOrderAlignAutomaton(Core::Configuration&      config,
                             const std::string&        source,
                             const std::string&        target,
                             const TransitionProbs&    transitionProbs,
                             const TranslationLexicon* lex,
                             const TranslationLexicon* h_lex,
                             const TranslationLexicon* v_lex,
                             const bool                context_diagonal = false,
                             const double              factorLexicon    = 1.0)
            : AlignAutomaton(config, source, target, transitionProbs, factorLexicon),
              lexicon_(lex),
              horizontal_lexicon_(h_lex),
              vertical_lexicon_(v_lex),
              I(outputSentence_.size()),
              J(inputSentence_.size()),
              M(3),
              maxIndex((I + 1) * (J + 1)),
              context_diagonal_(context_diagonal) {}

    virtual ConstStateRef getState(StateId s) const {
        State* sp = new State(s);

        u32 si(s);
        u32 jprev(si % J);
        u32 m(si / J % M);
        u32 i(si / J / M / (J + 1));
        u32 j(si / J / M % (J + 1));

        // State id is previous_source + J*(m + M*(j + i*(J+1)))

        bool doHorizontal(m == diagonal || m == horizontal);
        bool doVertical(m == diagonal || m == vertical);

        if (i < I && doVertical) {  // when not at the TOP of lattice, make vertical movement
            if (j > 0) {
                // in the vertical case, we move just up in the
                // lattice while reading a target word
                // for weighting we keep the source word we just read

                std::vector<std::string> x;
                x.push_back(target_[i]);
                x.push_back(source_[jprev]);
                x.push_back(target_[i - 1]);

                f64 weight(vertical_lexicon_->getProb(x) * factorLexicon_);
                if (transitionProbs_)
                    weight += transitionProbs_.exponent * transitionProbs_.v;

                StateId targetState(si + J * ((J + 1) * M - m + vertical));
                sp->newArc(targetState, Fsa::Weight(weight), Epsilon, outputSentence_[i]);
            }
            else {  // j==0
                // in this case we do not know, which of the
                // source words will be read first.
                //
                // at this point we have to hypothesize all source
                // words.
                //
                // later (when going diagonal for the first time,
                // we have to ensure, that the source word we read
                // is the same that we took for the weighting of
                // the arc at this point

                StateId targetState(si + J * ((J + 1) * M - m + vertical) - jprev);
                for (u32 jt = 0; jt < J; ++jt) {
                    std::vector<std::string> x;
                    x.push_back(target_[i]);
                    x.push_back(source_[jt]);
                    x.push_back("<s>");

                    f64 weight(vertical_lexicon_->getProb(x) * factorLexicon_);
                    if (transitionProbs_)
                        weight += transitionProbs_.exponent * transitionProbs_.v;

                    sp->newArc(targetState + jt, Fsa::Weight(weight), Epsilon, outputSentence_[i]);
                }
            }
        }

        if (j < J && doHorizontal) {  // when not at the RIGHT border of the lattice make horizontal movement
            StateId targetState(si + J * (M - m + horizontal) - jprev);
            for (u32 jt = 0; jt < J; ++jt) {
                std::vector<std::string> x;
                x.push_back(source_[jt]);
                // if we are in the beginning of the sentence,
                // adjust the previous symbol
                x.push_back((j == 0) ? "<s>" : source_[jprev]);
                x.push_back((i == 0) ? "<s>" : target_[i - 1]);

                f64 weight(horizontal_lexicon_->getProb(x) * factorLexicon_);
                if (transitionProbs_)
                    weight += transitionProbs_.exponent * transitionProbs_.h;

                sp->newArc(targetState + jt, Fsa::Weight(weight), inputSentence_[jt], Epsilon);
            }
        }
        if ((j < J) && (i < I)) {  // when in the middle of the lattice, do diagonal movment
            if (j > 0) {
                StateId targetState(si + J * ((J + 1) * M + M - m + diagonal) - jprev);
                for (u32 jt = 0; jt < J; ++jt) {
                    std::vector<std::string> x;
                    x.push_back(source_[jt]);
                    x.push_back(target_[i]);
                    if (context_diagonal_) {
                        x.push_back(source_[jprev]);
                        x.push_back((i > 0) ? target_[i - 1] : "<s>");
                    }

                    f64 weight(lexicon_->getProb(x) * factorLexicon_);
                    if (transitionProbs_)
                        weight += transitionProbs_.exponent * transitionProbs_.d;

                    sp->newArc(targetState + jt, Fsa::Weight(weight), inputSentence_[jt], outputSentence_[i]);
                }
            }
            else {  // j==0
                // as mentioned above when going vertical, in the
                // beginning of the sentence we have to take care
                // that we now read the source word we initially
                // hypothized

                StateId targetState(si + J * ((J + 1) * M + M - m + diagonal));

                std::vector<std::string> x;
                x.push_back(source_[jprev]);
                x.push_back(target_[i]);

                if (context_diagonal_) {
                    x.push_back("<s>");
                    x.push_back((i > 0) ? target_[i - 1] : "<s>");
                }

                f64 weight(lexicon_->getProb(x) * factorLexicon_);
                if (transitionProbs_)
                    weight += transitionProbs_.exponent * transitionProbs_.d;

                sp->newArc(targetState, Fsa::Weight(weight), inputSentence_[jprev], outputSentence_[i]);
            }
        }

        if (i == I && j == J) {
            sp->setFinal(this->semiring()->one());
        }
        return ConstStateRef(sp);
    }
    virtual std::string describe() const {
        return std::string("simpleAlignAutomaton()");
    }
};
}  // namespace Fsa
#endif
