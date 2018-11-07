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
#include "ConditionalLexiconSri.hh"
#include <Core/StringUtilities.hh>

namespace Translation {

    Core::ParameterString ConditionalLexiconSri::paramFilename_(
                  "file", "lexicon file", "");

    Core::ParameterInt ConditionalLexiconSri::paramLmOrder_(
                  "lmOrder", "order of the language model (optional)", 0);

    PREPARE_MALLOC_OPTIMIZED_HISTORY

    void ConditionalLexiconSri::read() {
                  if (lexiconFilename_ == "")
                                criticalError() << "No file name fiven.";
                  sriTupleVocabulary_.unkIsWord() = true;
                  File fpTupleLM(lexiconFilename_.c_str(), "r");
                  srilm_.read(fpTupleLM);

                  if (userLmOrder_ > 0) {
                                lmOrder_ = userLmOrder_;
                                srilm_.setorder(lmOrder_);
                  } else
                                lmOrder_ = srilm_.setorder();
                  //history_ = new VocabIndex[lmOrder_ + 1]; // We will do a trick with position 0 (see expandHypothesis bellow)

                  std::cerr << "finished reading lexicon" << std::endl;

                  unknownIndex_ = sriTupleVocabulary_.unkIndex();
                  sentenceBeginIndex_ = sriTupleVocabulary_.ssIndex();
                  sentenceEndIndex_ = sriTupleVocabulary_.seIndex();

                  std::cerr << "unknown word index = " << unknownIndex_ << std::endl;
                  std::cerr << "sentence beginindex = " << sentenceBeginIndex_ << std::endl;
                  std::cerr << "sentence end index = " << sentenceEndIndex_ << std::endl;
                  std::cerr << "srilm_.setorder() = " << srilm_.setorder(lmOrder_) << std::endl;

                  VocabIter vocabIterator(sriTupleVocabulary_);

                  std::cerr << "extraction monolingual tokens from lm vocab" << std::endl;

                  while (VocabString bilingualWord = vocabIterator.next()) {
                                //std::cerr << "splitting word " << bilingualWord;
                                // Now we extract the source and target parts
                                std::vector<std::string> fields = Core::split(bilingualWord, "|");
                                if (fields.size() == 2) { // If it is not a special word
                                         //std::cerr  << " into " << fields[0] << " and " << fields[1];
                                         Fsa::LabelId sourceLabelId;
                                         Fsa::LabelId targetLabelId;
                                         if (fields[0]!="$")
                                             sourceLabelId=tokens_->addSymbol(fields[0]);
                                         else
                                                  sourceLabelId=Fsa::Epsilon;

                                         // std::cerr << " mapped source LabelId = " << unsigned(sourceLabelId);

                                         if (fields[1]!="$")
                                             targetLabelId=tokens_->addSymbol(fields[1]);
                                         else
                                                  targetLabelId=Fsa::Epsilon;

                                         //std::cerr << " mapped target LabelId = " << unsigned(targetLabelId)
                                         //  		  << std::endl;

                                         vocabMap[make_pair(sourceLabelId,targetLabelId)]=sriTupleVocabulary_.getIndex(bilingualWord);
                                }

                  }

    }

}
