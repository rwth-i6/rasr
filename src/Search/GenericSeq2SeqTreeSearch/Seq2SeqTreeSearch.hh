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
#include <Core/ReferenceCounting.hh>
#include <Speech/ModelCombination.hh>
#include <Search/Search.hh>
#include <Search/Histogram.hh>
#include "Trace.hh"

// search manager: interface between search space and high level recognizer
// - manage step-wise decoding: expansion, pruning, recombination, etc.
// - results pulling (traceback)

namespace Search {

class Seq2SeqSearchSpace;
class LabelTree;

class Seq2SeqTreeSearchManager : public SearchAlgorithm {
    typedef Core::Ref<Seq2SeqTreeSearch::Trace> TraceRef;
    typedef std::unordered_map<const Seq2SeqTreeSearch::Trace*, Fsa::State*, Core::conversion<const Seq2SeqTreeSearch::Trace*, size_t> > TraceStateMap;

  public:
    Seq2SeqTreeSearchManager(const Core::Configuration &);
    virtual ~Seq2SeqTreeSearchManager();

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
    // ---------------------------------------------------

    // interface for decoding all remaining data
    virtual void decode();

    bool hasPronunciation() const;

  protected:
    void debugPrint(std::string msg, bool newStep=false);

    // 1-step decoding: expansion, pruning, recombination, etc.
    virtual void decodeNext();

    // only interface to get result from search space
    TraceRef sentenceEnd() const;

    // trace -> traceback
    void traceback(TraceRef end, Traceback &result) const;

    // trace -> lattice
    Core::Ref<const LatticeAdaptor> buildLatticeForTrace(TraceRef trace) const;

  private:
    Bliss::LexiconRef lexicon_;
    const Bliss::Lemma *silence_;
    // legacy HMM models, maybe needed if labels are states
    Core::Ref<const Am::AcousticModel> acousticModel_;
    Core::Ref<const Lm::ScaledLanguageModel> lm_;

    // model for label scoring
    Core::Ref<Nn::LabelScorer> labelScorer_;
    Score wpScale_;
    Seq2SeqSearchSpace* ss_;
    mutable Core::XmlChannel statisticsChannel_;

    s32 cleanupInterval_;
    bool createLattice_;
    bool optimizeLattice_;

    Index decodeStep_;
    bool simpleBeamSearch_;
    bool debug_;

    mutable TraceRef sentenceEnd_;
  };

} // namespace

#endif

