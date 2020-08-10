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
#include "SimpleAlignAutomaton.hh"
#include <Fsa/AlphabetUtility.hh>

namespace Fsa {

SimpleAlignAutomaton::SimpleAlignAutomaton(Core::Configuration&                    config,
                                           const std::string&                      source,
                                           const std::string&                      target,
                                           const TransitionProbs&                  transitionProbs,
                                           Translation::ConstConditionalLexiconRef lexicon,
                                           const double                            factorLexicon,
                                           const double                            factorTransition,
                                           const unsigned                          order)
        : AlignAutomaton(config, source, target, transitionProbs, factorLexicon, factorTransition),
          lexicon_(lexicon),
          order_(order)

{
    I_ = (outputSentence_.size()) + 1;
    J_ = (inputSentence_.size()) + 1;

    maxIndex_ = I_ * unsigned(::pow(J_, order_ + 1));

    ConstAlphabetRef lexiconAlphabet(lexicon_->getTokenAlphabet());

    for (unsigned j = 0; j < J_ - 1; ++j) {
        mappedSourceSentence_.push_back(lexiconAlphabet->index(source_[j]));
    }

    for (unsigned i = 0; i < I_ - 1; ++i) {
        mappedTargetSentence_.push_back(lexiconAlphabet->index(target_[i]));
    }

    //! \todo null word and sentence begin padding symbol are hardcoded
    nullWordIndex_                   = Fsa::Epsilon;
    sentenceBeginPaddingSymbolIndex_ = lexiconAlphabet->index("<s>");
}

//! targetIndexIncement
void SimpleAlignAutomaton::generateDiagonalOrHorizontalArcs_(State*                       s,
                                                             const unsigned               targetIndexIncrement,
                                                             const unsigned               nCoveredTargetWords,
                                                             const unsigned               nCoveredSourceWords,
                                                             const std::vector<unsigned>& sourcePositionHistory) const {
    // get "offset" cost defined by the transition
    double cost = factorTransition_ * (transitionProbs_[1 - targetIndexIncrement]);

    // loop over all possible input words and create arcs with the proper probability
    for (unsigned currentSourceWordIndex = 0; currentSourceWordIndex < J_ - 1; ++currentSourceWordIndex) {
        // initialize lexicon key and index vector with current source word
        std::vector<Fsa::LabelId> diagonalKey(1, mappedSourceSentence_[currentSourceWordIndex]);
        std::vector<unsigned>     diagonalJ(order_, 0);

        if (order_ > 0) {
            diagonalJ[0] = currentSourceWordIndex;
        }

        Fsa::LabelId currentTargetWordAlphabetIndex;

        // new source position j=0, ..., J_-1
        // new target position i=i+1
        if (targetIndexIncrement == 1) {
            // Diagonal movement, we increase the target word index by 1
            // nCoveredTargetWords is the number of target words covered so far.
            // the arcs leavin this state are labeled with the word that is the first
            // uncovered target word. since the target sentence vector is indexed
            // starting with 0, the index of the next target word is the same as
            // nCoveredTargetWords
            currentTargetWordAlphabetIndex = mappedTargetSentence_[nCoveredTargetWords];
        }
        else {
            currentTargetWordAlphabetIndex = nullWordIndex_;
        }
        diagonalKey.push_back(currentTargetWordAlphabetIndex);

        // loop copies the reduced source and target context
        // stops at order_-1 (dont copy oldest history elements)
        for (unsigned n = 0; n < order_; ++n) {
            if (n < nCoveredSourceWords) {
                diagonalKey.push_back(mappedSourceSentence_[sourcePositionHistory[n]]);
            }
            else {
                diagonalKey.push_back(sentenceBeginPaddingSymbolIndex_);
            }

            // next target position i+1, going back in history with each n

            // when going back in the target history, we ave to take care of the
            // sentence boundary. when we move past the first wordposition, words
            // are padded by sentence begin symbols
            if (nCoveredTargetWords > 0) {
                diagonalKey.push_back(mappedTargetSentence_[nCoveredTargetWords - n - 1]);
            }
            else {
                diagonalKey.push_back(sentenceBeginPaddingSymbolIndex_);
            }
            if (order_ > 1) {
                // update index vector
                diagonalJ[n + 1] = sourcePositionHistory[n];
            }
        }

        // get StateId
        Fsa::StateId arcTarget = indices2StateId(nCoveredTargetWords + targetIndexIncrement, nCoveredSourceWords + 1, diagonalJ);
        diagonalKey[0]         = mappedSourceSentence_[currentSourceWordIndex];

        if (targetIndexIncrement == 1) {
            // Diagonal movement, include target label
            s->newArc(arcTarget, Fsa::Weight(cost + factorLexicon_ * lexicon_->getReverseCost(1 - targetIndexIncrement, diagonalKey)), inputSentence_[currentSourceWordIndex], outputSentence_[nCoveredTargetWords]);
        }
        else {
            s->newArc(arcTarget, Fsa::Weight(cost + factorLexicon_ * lexicon_->getReverseCost(1 - targetIndexIncrement, diagonalKey)), inputSentence_[currentSourceWordIndex], Fsa::Epsilon);
        }
    }
}

ConstStateRef SimpleAlignAutomaton::getState(StateId s) const {
    //! create new state to be returned
    State*                result = new State(s);
    unsigned              nCoveredTargetWords;
    std::vector<unsigned> sourcePositionHistory(order_ + 1, sentenceBeginPaddingSymbolIndex_);
    unsigned              nCoveredSourceWords;
    this->stateId2Indices(s, nCoveredTargetWords, nCoveredSourceWords, sourcePositionHistory);

    StateId arcTarget;

    // Diagonal Movement

    if (nCoveredTargetWords < I_ - 1 && nCoveredSourceWords < J_ - 1) {
        this->generateDiagonalOrHorizontalArcs_(result, 1, nCoveredTargetWords, nCoveredSourceWords, sourcePositionHistory);
    }

    // Horizontal Movement

    if (nCoveredSourceWords < J_ - 1) {
        this->generateDiagonalOrHorizontalArcs_(result, 0, nCoveredTargetWords, nCoveredSourceWords, sourcePositionHistory);
    }

    // Vertical Movement
    // just the target index changes, this means, we just have to add 1 to the state index

    if (nCoveredTargetWords < I_ - 1) {
        arcTarget = s + 1;

        std::vector<Fsa::LabelId> verticalKey;

        verticalKey.push_back(nullWordIndex_);
        verticalKey.push_back(mappedTargetSentence_[nCoveredTargetWords]);

        // loop copies the full source context (it is not changed in a vertical movement)
        for (unsigned n = 0; n < order_; ++n) {
            // need to check if order ist too long for history
            verticalKey.push_back(mappedSourceSentence_[sourcePositionHistory[n]]);
            if (nCoveredTargetWords < n + 1) {
                verticalKey.push_back(sentenceBeginPaddingSymbolIndex_);
            }
            else {
                verticalKey.push_back(mappedTargetSentence_[nCoveredTargetWords - n - 1]);
            }
        }

        double cost = factorLexicon_ * lexicon_->getReverseCost(2, verticalKey);
        cost += factorTransition_ * transitionProbs_[2];
        result->newArc(arcTarget, Fsa::Weight(cost), Epsilon, outputSentence_[nCoveredTargetWords]);
    }

    if (nCoveredTargetWords == I_ - 1 && nCoveredSourceWords == J_ - 1) {
        result->setFinal(this->semiring()->one());
    }

    return ConstStateRef(result);
}

}  // namespace Fsa
