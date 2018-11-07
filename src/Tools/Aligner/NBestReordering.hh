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
#include <Fsa/Automaton.hh>
#include "TranslationLexicon.hh"

namespace Fsa {
class nBestAutomaton : public Automaton {
private:
   std::vector<std::string> source_;
   std::vector<std::string> target_;
   ConstAlphabetRef inputAlphabet_;
   ConstAlphabetRef outputAlphabet_;
   std::vector<LabelId> inputSentence_;
   std::vector<LabelId> outputSentence_;
   const TranslationLexicon *lexicon_;
   State *state_;
public:
   nBestAutomaton(const std::string& source, const std::string& target, const TranslationLexicon* lex) :
      source_(Core::split(source," ")), target_(Core::split(target," ")), inputSentence_(), outputSentence_(), lexicon_(lex), state_(new State) {
      //this->setProperties(PropertyAcyclic || PropertyLinear, PropertyAcyclic || PropertyLinear);

      std::vector<std::string> numberedSource = numberTokensVector(source_);
      StaticAlphabet *ia = new StaticAlphabet();
      for (std::vector<std::string>::const_iterator i=numberedSource.begin(); i!=numberedSource.end(); ++i) {
         inputSentence_.push_back(ia->addSymbol(*i));
      }
      inputAlphabet_ = ConstAlphabetRef(ia);

      std::vector<std::string> numberedTarget = numberTokensVector(target_);
      StaticAlphabet *oa = new StaticAlphabet();
      for (std::vector<std::string>::const_iterator i=numberedTarget.begin(); i!=numberedTarget.end(); ++i) {
         outputSentence_.push_back(oa->addSymbol(*i));
      }
      outputAlphabet_ = ConstAlphabetRef(oa);

      // generate state (there is only one)
      // it contains all the arcs of translations from source words to target words
      // as permitted by the lexicon and additionally all the translations to the
      // empty word from any source word

      // regular words
      for (u32 i=0; i<numberedTarget.size(); ++i) {
         for (u32 j=0; j<numberedSource.size(); ++j) {
            std::vector<std::string> x;
            x.push_back(source_[j]);
            x.push_back(target_[i]);
            state_->newArc(state_->id(), Fsa::Weight(lexicon_->getProb(x)), inputSentence_[j], outputSentence_[i]);
         }
      }

      // empty word
      // for (u32 j=0; j<numberedSource.size(); ++j) {
      //     std::vector<std::string> x;
      //     x.push_back(source_[j]);
      //     x.push_back("NULL");
      //     state_->newArc(state_->id(), Fsa::Weight(lexicon_->getProb(x)), inputSentence_[j], Epsilon);
      // }
      state_->setFinal(this->semiring()->one());
   }

public:
   virtual ConstStateRef getState(StateId s) const { return ConstStateRef(state_); }
   virtual std::string describe() const { return std::string("nBestAutomaton()"); }
   virtual ConstAlphabetRef getInputAlphabet() const { return inputAlphabet_; }
   virtual ConstAlphabetRef getOutputAlphabet() const { return outputAlphabet_; }
   virtual Type type() const { return TypeTransducer; }
   virtual StateId initialStateId() const { return 0; }
   virtual ConstSemiringRef semiring() const { return TropicalSemiring; }
};

class ReorderAutomaton : public SlaveAutomaton {
private:
   StateRef initial_;
   StateId nextFreeStateId_;
   std::map<StateId,StateRef> statemap_;
public:
   ReorderAutomaton(ConstAutomatonRef f) :
      SlaveAutomaton(f),
      initial_(new State()),
      nextFreeStateId_(1)
      {
         this->setProperties(PropertyAcyclic || PropertyLinear, PropertyAcyclic || PropertyLinear);
         //std::cerr << "entering constructor\n";
         statemap_[0]=initial_;

         ConstStateRef initial=fsa_->getState(initialStateId());
         // iterate over all hypotheses to generate nbest reordering graphs
         for (State::const_iterator i=initial->begin(); i!=initial->end(); ++i) {
            ConstStateRef ts = fsa_->getState(i->target());
            std::map< LabelId, Vector<LabelId> > sentence;
            //std::cerr << "this arc " << int(i->input()) << " - " << int(i->output()) << " - " << f32(i->weight()) << std::endl;
            // traverse this hypotheses and create mapping in target word order
            while (!(ts->isFinal())) {
               //                    while (!(ts->isFinal()) && ts->begin()->input() == Epsilon) {
               //                        ts = fsa_->getState(ts->begin()->target());
               //                    }
               LabelId out(ts->begin()->output());
               LabelId in(ts->begin()->input());
               //std::cerr << "ts arc " << in << " - " << out << std::endl;

               if (in != Epsilon && out != Epsilon) {
                  //std::cerr << in << " " << out << std::endl;
                  if (sentence.find(out) == sentence.end()) {
                     sentence[out] = Vector<LabelId>(1,in);
                  } else {
                     sentence[out].push_back(in);
                  }
               }
               ts = fsa_->getState(ts->begin()->target());
            }
            // create permutation graph in target word order
            //std::cerr << "writing: "  << std::endl;
            StateRef new_state = initial_;
            for (std::map< LabelId, Vector<LabelId> >::const_iterator i=sentence.begin(); i!=sentence.end(); ++i) {
               for (Vector<LabelId>::const_iterator j=i->second.begin(); j!=i->second.end(); ++j) {
                  LabelId in = *j;
                  //std::cerr << in << " " << out << std::endl;
                  bool foundArc(false);
                  for (State::const_iterator k=new_state->begin(); k!=new_state->end(); ++k) {
                     if (k->input() == in) {
                        //std::cerr << "traversing arc label=" << in << std::endl;
                        new_state = statemap_[k->target()];
                        foundArc=true;
                        break;
                     }
                  }
                  // this part after the for loop is only reached, when no arc exists in the current state
                  // of the permutation graph that corresponds to the current input symbol
                  //
                  // create new arc that corresponds to the current input symbol and traverse
                  if (!foundArc) {
                     new_state->newArc(nextFreeStateId_, this->semiring()->one(), in);
                     statemap_[nextFreeStateId_]=StateRef(new State(nextFreeStateId_));
                     new_state=statemap_[nextFreeStateId_];
                     ++nextFreeStateId_;
                  }
               }
            }
            new_state->setFinal(this->semiring()->one());
         }
      }
   //virtual ConstStateRef getState(StateId s) const { return ConstStateRef(statemap_[s]); }
   //virtual ConstStateRef getState(StateId s) const { return initial_; }
    virtual ConstStateRef getState(StateId s) const { return Fsa::ConstStateRef(statemap_.find(s)->second); }
   virtual std::string describe() const { return std::string("reorderAutomaton(" + fsa_->describe() + ")"); }
   virtual StateId initialStateId() const { return 0; }
   virtual Type type() const { return TypeAcceptor; }
   virtual ConstAlphabetRef getOutputAlphabet() const { return fsa_->getInputAlphabet(); }

};

ConstAutomatonRef reorderNbest(std::string sourceSentence, std::string targetSentence, const TranslationLexicon* lex, u32 n=1) {
   return projectInput(
      nbest(
         composeMatching(
            ConstAutomatonRef(
               staticCopy(
                  numberTokens(sourceSentence),
                  TropicalSemiring)
               ),
            ConstAutomatonRef(
               new nBestAutomaton(
                  sourceSentence,
                  targetSentence,
                  lex)
               )
            ),
         n,
         false
         )
      );
};

ConstAutomatonRef nbestReorder(std::string sourceSentence, std::string targetSentence, const TranslationLexicon* lex, u32 n=1) {
   //    writeXml(ConstAutomatonRef(new StaticAutomaton(numberTokens(sourceSentence), TropicalSemiring)), std::cerr);
   //    writeXml(ConstAutomatonRef(new nBestAutomaton(sourceSentence, targetSentence, lex)), std::cerr);
   //    writeXml(nbest(composeMatching(ConstAutomatonRef(new StaticAutomaton(numberTokens(sourceSentence), TropicalSemiring)), ConstAutomatonRef(new nBestAutomaton(sourceSentence, targetSentence, lex))), n, false), std::cerr);
   //    writeXml(ConstAutomatonRef(new ReorderAutomaton(nbest(composeMatching(ConstAutomatonRef(new StaticAutomaton(numberTokens(sourceSentence), TropicalSemiring)), ConstAutomatonRef(new nBestAutomaton(sourceSentence, targetSentence, lex))), n, false))), std::cerr);
   return ConstAutomatonRef(
      new ReorderAutomaton(
         nbest(
            composeMatching(
               ConstAutomatonRef(
                  staticCopy(
                     numberTokens(sourceSentence),
                     TropicalSemiring)
                  ),
               ConstAutomatonRef(
                  new nBestAutomaton(
                     sourceSentence,
                     targetSentence,
                     lex)
                  )
               ),
            n,
            false)
         )
      );
}

class AlignLinearNbestAutomaton : public SlaveAutomaton {
private:
   StateRef initial_;
   StateId nextFreeStateId_;
   std::map<StateId,StateRef> statemap_;
public:
   AlignLinearNbestAutomaton (ConstAutomatonRef sourceAutomaton, ConstAutomatonRef alignAutomaton)
      : SlaveAutomaton(sourceAutomaton)
      {
         for (State::const_iterator i=sourceAutomaton->getState(sourceAutomaton->initialStateId())->begin();
              i != sourceAutomaton->getState(sourceAutomaton->initialStateId())->end(); ++i) {

            ConstAutomatonRef bestTrace = best(composeMatching(partial(sourceAutomaton, i->target()), alignAutomaton));
            ConstStateRef s = bestTrace->getState(bestTrace->initialStateId());
            do {
               State* sp = new State(nextFreeStateId_, s->tags(), s->weight());
               statemap_[nextFreeStateId_] = StateRef(sp);
               ++nextFreeStateId_;
               sp->newArc(nextFreeStateId_,
                          s->begin()->weight(),
                          s->begin()->input(),
                          s->begin()->output());
               s = bestTrace->getState(s->begin()->target());
            } while (!s->isFinal());
            State* sp = new State(nextFreeStateId_, s->tags(), s->weight());
            sp->setFinal(this->semiring()->one());
            statemap_[nextFreeStateId_] = StateRef(sp);
            ++nextFreeStateId_;
         }
      }
    virtual ConstStateRef getState(StateId s) const { return Fsa::ConstStateRef(statemap_.find(s)->second); }
   virtual StateId initialStateId() const { return 0; }
};
}
