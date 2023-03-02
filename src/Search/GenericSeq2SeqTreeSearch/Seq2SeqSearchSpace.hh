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

#ifndef SEQ2SEQ_SEARCH_SPACE_HH
#define SEQ2SEQ_SEARCH_SPACE_HH

#include <Search/Histogram.hh>
#include "LabelTree.hh"
#include "SearchSpaceHelpers.hh"
#include "SearchSpaceStatistics.hh"

namespace Seq2SeqTreeSearch {

class LanguageModelLookahead; 

}

namespace Search {

class Seq2SeqSearchSpace : public Core::Component {

  public:
    Seq2SeqSearchSpace(const Core::Configuration& configuration,
                     Core::Ref<const Am::AcousticModel> acousticModel,
                     Bliss::LexiconRef lexicon,
                     Core::Ref<const Lm::ScaledLanguageModel> lm,
                     Score wpScale, Core::Ref<Nn::LabelScorer> labelScorer);
    ~Seq2SeqSearchSpace();

    // statistics
    u32 nActiveTrees() const { return activeInstances_.size(); }
    u32 nLabelHypotheses() const { return labelHypotheses_.size(); }
    u32 nWordEndHypotheses() const { return wordEndHypotheses_.size(); }
    u32 nEndTraces() const { return endTraces_.size(); }

    // clean up complete search space (between segments)
    void clear();
    // clean up intermediate search space (within segment): remove useless stuff for memory
    void cleanUp();

    // must be called after creation
    virtual void initialize(bool simpleBeamSearch);
    bool isInitialized() const { return initialized_; }

    void setDecodeStep(u32 step) { decodeStep_ = step; }
    void setInputLength(u32 len) { inputLength_ = len; }

    // startup of a segment
    void addStartupWordEndHypothesis(Index step);

    // start new or update trees based on the active word end hypotheses
    void startNewTrees();

    // expand labels
    void expandLabels();
    // prune labels
    void applyLabelPruning();

    // word boundary expansion and pruning
    void findWordEndsAndPrune();
    // simple beam search with one global beam
    void findWordEndsAndPruneGlobal();

    // extend LM history
    void extendWordHistory();

    // create traces
    void createTraces();

    // optional word end recombination
    void recombineWordEnds(bool createLattice);

    // simple optimize lattice
    void optimizeLattice();

    // rescales the scores with globalScoreOffset_ for better numeric stability
    // calling only after label pruning before word boundary expansion
    void rescale(Score offset=0);
    Score bestLabelScore();
    Score bestLabelProspect();

    // special end handling for asynchronous finished paths (e.g. attention models)
    // end detection, trace pruning, max length stop ...
    bool needEndProcessing() const { return needEndProcessing_; }
    void processEnd();
    // stop search
    bool shouldStopSearch() const { return stopSearch_; }
    // check if we can stop search early (before runing the current step)
    bool mayStopEarly();

    // sentence end (should be called only after a complete decoding step)
    TraceRef getSentenceEnd(bool createLattice);
    TraceRef getSentenceEndFallBack();

    // compute lattice arc scores from trace
    SearchAlgorithm::ScoreVector computeArcTraceScore(TraceRef& arcTrace, TraceRef& preTrace);

    const Bliss::Lemma* getEndLemma() const;
    bool hasPronunciation() const { return staticLabelTree_.hasPronunciation(); }

    void resetStatistics() { statistics_.clear(); }
    void logStatistics(Core::XmlChannel& channel) const { statistics_.write(channel); }

  protected:
    // initialization
    void initializePruning(bool simpleBeamSearch);
    void initializeLanguageModel();

    // history-based tree caching
    TreeInstance* activateOrUpdateTree(const WordEndHypothesis& weh);

    // ---- within-word label expansion ----
    template <bool transitionPenalty, bool localPruning>
    void _expandLabels();

    template <bool allowBlank, bool relativePosition, bool transitionPenalty, bool localPruning>
    void expandLabelsInTree();

    template <bool allowBlank, bool relativePosition, bool transitionPenalty, bool localPruning>
    void expandLabelWithScore(const LabelHypothesis& lh, bool isRoot = false);

    // scoring and position (segmental)
    template <bool allowBlank, bool transitionPenalty, bool localPruning>
    void expandLabelWithScoreAndPosition(const LabelHypothesis& lh, bool isRoot = false);
    // ------------------------------------

    // in case of cuted audio and tight pruning (no results): record best label trace
    void recordBestLabelEndTrace(const LabelHypothesis& lh);
    // best prospect hyps and tree
    LabelHypothesesList::const_iterator bestProspectLabel();
    TreeInstance* bestProspectLabelTree(u32 bestIndex);
    WordEndHypothesesList::const_iterator bestProspectWordEnd();

    // apply lookahead to labels in each tree instance
    template <bool eos, bool wordLen>
    void applyLookaheadInInstances();
    void activateLmLookahead(TreeInstance* instance, bool compute = true);

    // label pruning and clean up
    template<bool maxInLenStop, bool removeNonExpandable, bool wordLen, bool deleteTree>
    void pruneLabels(Score threshold);
    bool mayDeactivateTree(TreeInstance *at);

    // within word label recombination (viterbi or full-sum): after pruning before history extension
    void recombineLabels();
    template <bool blankUpdateHistory, bool loopUpdateHistory, bool historyHash, bool simple>
    void recombineLabelsInTree();

    // delayed label history extension after global pruning
    void extendLabelHistory();

    // create early word end hypotheses from exiting labels
    template <bool stepReNorm, bool wordLen, bool pruneGlobal>
    void findEarlyWordEnds(bool exitPenalty);

    // prune early word end hypotheses and expand to normal word end hypotheses
    // Note: delayed label history extension is also moved here by default
    template <bool wordLen>
    void pruneAndExpandEarlyWordEnds(Score threshold, bool extendlabelHistory=true);
    // prune word end hypotheses
    void pruneWordEnds(Score threshold);

    // histogram pruning threshold
    Score quantileScore(Score minScore, Score maxScore, u32 nHyps, bool label, bool word, bool endTrace=false);

    // global pruning across labels and wordEnds
    void pruneLabelsAndWordEnds();
    // fixed beam pruning across (expandable) labels, wordEnds and endTraces
    void pruneGlobalWithFixedBeam(u32 beamSize, bool expandable=true);
    // beam insertion with score comparison
    typedef std::multimap<Score, std::pair<u32, u32>, std::greater<Score>> Beam;
    void insertBeam(Beam& beam, u32 beamSize, Score score, u32 category, u32 idx);

    // recombination of word ends
    template<bool labelHistoryHash, bool labelOtherHash>
    void _recombineWordEnds(bool createLattice);
    void recombineTwoWordEnds(WordEndHypothesis&, WordEndHypothesis&, bool createLattice);

    // detect ending traces and ending processing
    template<bool stepReNorm, bool wordLen>
    void detectEndTraces();
    template<bool stepReNorm, bool wordLen>
    void detectEndTracesFromStates(TraceList& stepEndTraces);

    // prune traces
    void pruneEndTraces(Score threshold);
    void pruneTraces(Score threshold);

    // stopping criteria
    void checkStoppingCriteria();

    // special sentence end handling based on end traces
    TraceRef getSentenceEndFromEndTraces(bool createLattice);

    // standard sentence end handling based on all current hypotheses
    TraceRef getSentenceEndFromHypotheses(bool createLattice);

    // ending traces handling for full-sum decoding
    void fullsumMergeTraces(HistoryTraceMap& historyTraceMap, size_t hash, TraceRef& t);
    TraceRef getBestTrace(const HistoryTraceMap& historyTraceMap, bool createLattice);

    // different score computation (inline)
    Score computeLabelProspect(const LabelHypothesis& lh, Score lmlaScore=0);
    Score computeWordEndProspect(const EarlyWordEndHypothesis& eWeh);
    Score computeWordEndProspect(const WordEndHypothesis& weh);
    Score computeTraceProspect(const LabelHypothesis& lh);
    Score computeTraceProspect(TraceRef& trace, bool isEnd=false);
    Score computeLengthNormalizedScore(Score acoustic, Score lm, u32 nLabels, u32 nWords);

  private:
    // statistics and performance measure
    Seq2SeqTreeSearch::SearchSpaceStatistics statistics_;

    Bliss::LexiconRef lexicon_;
    Core::Ref<const Am::AcousticModel> acousticModel_;
    Core::Ref<const Lm::ScaledLanguageModel> languageModel_; // major scoring lm
    Core::Ref<const Lm::LanguageModel> lookaheadLm_;         // use lm-lookahead.scale for scaling
    Core::Ref<const Lm::LanguageModel> recombinationLm_;     // no scaling needed

    Score wpScale_;

    // model for label scoring
    Core::Ref<Nn::LabelScorer> labelScorer_;
    u32 inputLength_;

    // static representation of the search network
    LabelTree staticLabelTree_;

    // language model lookahead
    Seq2SeqTreeSearch::LanguageModelLookahead* lmLookahead_;
    Lm::History unigramHistory_;
    LmLookahead unigramLookAhead_;

    // flag of initialization
    bool initialized_;

    // flag to turn of LM scoring
    bool useLmScore_;

    // viterbi or full-sum
    bool fullSumDecoding_;
    bool labelFullSum_;

    // pruning and thresholds
    Score localLabelPruning_;
    Score labelPruning_;
    u32 labelPruningLimit_;

    // Note: broad-level word -> actually transcription output unit (orthography)
    Score wordEndPruning_;
    u32 wordEndPruningLimit_;

    Histogram histogram_;

    u32 instanceDeletionTolerance_;
    f32 instanceLookaheadLabelThreshold_;

    // global decode step (time if time-synchronized)
    Index decodeStep_;

    // global best prospect/scores
    mutable Score bestLabelScore_, bestLabelProspect_, bestWordEndProspect_;
    f64 globalScoreOffset_; // for numerical stability 

    // TreeInstance Cache
    KeyInstanceMap activeInstanceMap_;
    InstanceList activeInstances_;
    TreeInstance* currentInstance_;

    // LabelHypothesis
    LabelHypothesesList labelHypotheses_;
    LabelHypothesesList newLabelHypotheses_;

    bool allowLabelRecombination_;
    s32 labelRecombinationLimit_;
    std::vector<size_t> labelHistoryHash_; // speed-up: cache hash for 1st-order dependency

    bool allowLabelLoop_;
    u32 minLoopOccur_;

    // allow blank labels (transducer)
    bool allowBlankLabel_;
    Index blankLabelIndex_;
    Score blankLabelPenalty_;
    Score blankLabelScoreThreshold_; // threshold to prevent label stay in unlikely blank

    // WordEndHypothesis
    WordEndHypothesesList wordEndHypotheses_;
    EarlyWordEndHypothesesList earlyWordEndHypotheses_;

    bool allowWordEndRecombination_;
    s32 wordEndRecombinationLimit_;

    // ---- asynchronous ending (need end processing) ----
    bool needEndProcessing_;
    bool positionDependent_;

    Index endLabelIndex_; // model class index
    u32 endNodeId_; // tree node id
    u32 endExitId_; // tree exit id
    TraceList endTraces_;
    TraceRef bestLabelEndTrace_; // fallback

    bool restrictWithInputLength_;

    // trace pruning for asynchronous finshed paths 
    bool pruneTrace_;
    Score tracePruning_;
    u32 tracePruningLimit_;
    Score bestEndTraceProspect_;

    // stopping criteria
    bool stopSearch_;

    // simple beam search with fixed beam size
    bool fixedBeamSearch_;

    // length normalization 
    bool lengthNorm_;
    bool normLabelOnly_; // normalize total score by label length only
    bool normWordOnly_; // only normalize lm score by word length (different interpretation)
    Score eosThreshold_; // for subword-based lemma only

    // step-wise beam-renormalization based beam search: derived length model
    bool stepReNorm_;
    bool stepEarlyStop_;
    Score stepSumScore_;
    Score stepEndScore_;

    // accumulated non-ending probability (length probability)
    bool stepLengthOnly_; // only apply additional length modeling
    Score stepLengthScale_; // tunable but preferrably not to use
    Score stepAccuLenScore_;

    // global pruning across labels and word-ends
    bool pruneWordsWithLabels_;

    // global balance of different word lengths (mainly for pruning)
    // only with stepReNorm this would have effect on final prob. for decision
    // Note: only for label-sync-search
    bool wordLenBalance_;
    Score wordLenScale_;
    std::vector<Score> wordLenBestProspect_;
    std::vector<Score> wordLenScore_;

    // segmental decoding: mainly for equivalent transducer to segmental modeling
    bool allowBlankSegment_; // allow partial segment of blank only
    Index silenceIndex_; // optional silence model index

    // relative position w.r.t. last output label (not alignment label) 
    bool useRelativePosition_;
    u32 relativePositionClip_;

    // standard RNN-T topology
    bool verticalTransition_;
    // ---- asynchronous ending (need end processing) ----
};

inline SearchAlgorithm::ScoreVector Seq2SeqSearchSpace::computeArcTraceScore(TraceRef& arcTrace, 
                                                                             TraceRef& preTrace) {
  Score arcAcoustic = arcTrace->score.acoustic,
        arcLm = arcTrace->score.lm;

  // correct possible mismatch between decision prospect and arc-wise trace scores
  // Note: just correct on the final arcs in lattice (further processing, e.g. CN-decoding, ok ?)
  if ((lengthNorm_ || stepReNorm_ || wordLenBalance_) && 
      !arcTrace->lemma && !arcTrace->pronunciation) {
    // linearly scaled in probability domain
    Score ratio = 0.5 * (arcTrace->prospect - arcTrace->score);
    arcAcoustic += ratio;
    arcLm += ratio;
  }

  return SearchAlgorithm::ScoreVector(arcAcoustic - preTrace->score.acoustic, 
                                      arcLm - preTrace->score.lm);
}

} // namespace

#endif
