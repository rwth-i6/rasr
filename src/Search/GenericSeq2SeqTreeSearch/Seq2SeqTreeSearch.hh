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
 *
 *  author: Wei Zhou
 */

#ifndef SEQ2SEQ_TREE_SEARCH_HH
#define SEQ2SEQ_TREE_SEARCH_HH

#include <Core/Component.hh>
#include <Search/Search.hh>

// search manager: interface between search space and high level recognizer
// - manage step-wise decoding: expansion, pruning, recombination, etc.
// - results pulling (traceback)

namespace Search {


class Seq2SeqTreeSearchManager : public SearchAlgorithm {

  public:
    Seq2SeqTreeSearchManager(const Core::Configuration &);

    // ---- from SearchAlgorithm (overwrite required) ----
    virtual bool setModelCombination(const Speech::ModelCombination& modelCombination);
    virtual void setGrammar(Fsa::ConstAutomatonRef);
    virtual void restart();

    // replaced by decode and decodeNext 
    virtual void feed(const Mm::FeatureScorer::Scorer&) {}

    // TODO partial result not supported yet

    virtual void getCurrentBestSentence(Traceback &result) const;
    virtual Core::Ref<const LatticeAdaptor> getCurrentWordLattice() const;

    virtual void resetStatistics();
    virtual void logStatistics() const;
  };

} // namespace

#endif

