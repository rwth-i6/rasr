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

#include "Seq2SeqSearchSpace.hh"
#include <Lm/Module.hh>
#include <Math/Utilities.hh>
#include "LanguageModelLookahead.hh"

using namespace Search;

const Core::ParameterBool paramUseLmScore(
  "use-lm-score",
  "whether to use lm scores in search (otherwise only history management)",
  true);

const Core::ParameterBool paramFullSumDecoding(
  "full-sum-decoding",
  "apply full sum decoding",
  false);

const Core::ParameterBool paramLabelFullSum(
  "label-full-sum",
  "apply full sum within the same word sequence as well as the same label sequence",
  false);

const Core::ParameterFloat paramLocalLabelPruning(
  "local-label-pruning",
  "threshold for locally (per tree node) pruning of label hypotheses",
  Core::Type<f32>::max, 0.0);

const Core::ParameterFloat paramLabelPruning(
  "label-pruning",
  "threshold for pruning of label hypotheses",
  Core::Type<f32>::max, 0.0);

const Core::ParameterInt paramLabelPruningLimit(
  "label-pruning-limit",
  "maximum number of active labels, enforced by histogram pruning \
   this value is important, because it sets an upper bound for the runtime.",
  50000, 1);

const Core::ParameterFloat paramWordEndPruning(
  "word-end-pruning",
  "threshold for pruning of word end hypotheses \
   If the value is below 1.0, eg. 0.7, then it is relative to label-pruning (recommended).",
  Core::Type<f32>::max, 0.0);

const Core::ParameterInt paramWordEndPruningLimit(
  "word-end-pruning-limit",
  "maximum number of word ends, enforced by histogram pruning \
   this value is important, because it sets an upper bound for the runtime \
   20000 is a good default value, reduce it more if the runtime becomes too slow for some segments.",
  5000, 1);

const Core::ParameterInt paramHistogramPruningBins(
  "histogram-pruning-bins",
  "number of bins for histogram pruning (very minor effect)",
  101, 2);

const Core::ParameterInt paramInstanceDeletionTolerance(
  "instance-deletion-tolerance",
  "label steps of inactivity before an instance is deleted",
  0);

const Core::ParameterFloat paramInstanceLookaheadLabelThreshold(
  "instance-lookahead-label-threshold",
  "apply full-order lookahead in instances that have at least this dominance",
  0.0);

const Core::ParameterBool paramEnableLmLookahead(
  "lm-lookahead",
  "enable language model lookahead (recommended)",
  true);

const Core::ParameterBool paramSeparateLookaheadLm(
  "separate-lookahead-lm",
  "use a separate lm for lookahead (one that is not provided by the main language-model)",
  false);

const Core::ParameterBool paramSeparateRecombinationLm(
  "separate-recombination-lm",
  "use a separate lm for recombination (one that is not provided by the main language-model)",
  false);

const Core::ParameterBool paramAllowLabelRecombination(
  "allow-label-recombination",
  "allow recombination of labels in the same tree instance",
  false);

const Core::ParameterInt paramLabelRecombinationLimit(
  "label-recombination-limit",
  "history length of label recombination (-1 for infinity)",
  -1);

const Core::ParameterBool paramAllowLabelLoop(
  "allow-label-loop",
  "allow loop transition of labels in the same tree instance",
  false);

// minimum duration (decoding heuristics)
const Core::ParameterInt paramMinLoopOccurance(
  "min-loop-occurance",
  "minimum occurance of label loop",
  0, 0);

const Core::ParameterBool paramAllowBlankLabel(
  "allow-blank-label",
  "allow blank label (transducer like)",
  false);

const Core::ParameterFloat paramBlankLabelPenalty(
  "blank-label-penalty",
  "score penalty for staying in blank label",
  0.0);

const Core::ParameterFloat paramBlankLabelProbabilityThreshold(
  "blank-label-probability-threshold",
  "probability threshold for label to stay in blank",
  0.0);

// word end recombination
const Core::ParameterBool paramAllowWordEndRecombination(
  "allow-word-end-recombination",
  "allow recombination of word ends with the same recombination history",
  true);

const Core::ParameterInt paramWordEndRecombinationLimit(
  "word-end-recombination-limit",
  "history length of word end recombination (-1 for infinity)",
  -1);

// ---- only when end processing is needed ----
// trace pruning
const Core::ParameterBool paramPruneTrace(
  "prune-trace",
  "whether to prune traces",
  true);

const Core::ParameterFloat paramTracePruning(
  "trace-pruning",
  "threshold for pruning traces",
  Core::Type<f32>::max, 0.0);

const Core::ParameterInt paramTracePruningLimit(
  "trace-pruning-limit",
  "maximum number of active leaf traces including previously ended ones, enforced by histogram pruning \
   this value is important, because it sets an upper bound for the runtime.",
  1000, 1);

// stopping criteria
const Core::ParameterBool paramRestrictWithInputLength(
  "restrict-with-input-length",
  "filter output sequences exceeding input (encoding) length",
  true);

// step-wise beam renormalization: derive explicit length modeling for label-sync models
// beam-pruning-dependent model modification -> more robust search and decision
const Core::ParameterBool paramStepReNormalization(
  "step-re-normalization",
  "re-normalize probability mass at each step for ending traces",
  false);

const Core::ParameterBool paramStepEarlyStop(
  "step-early-stop",
  "apply early stopping for step-re-normalization",
  true);

const Core::ParameterBool paramStepLengthOnly(
  "step-length-only",
  "apply step-re-normalization based explicit length modeling only, and not re-normalize the original sequence posterior",
  false);

const Core::ParameterFloat paramStepLengthScale(
  "step-length-scale",
  "scaling factor for step-accumulated length score",
  1.0, 0.0);

// global pruning and possible word-length balance (only for label-sync-search)
const Core::ParameterBool paramPruneWordsWithLabels(
  "prune-words-with-labels",
  "whether to apply global pruning across labels and word-ends",
  false);

const Core::ParameterBool paramWordLengthBalance(
  "word-length-balance",
  "globally balance score and pruning w.r.t. different word lengths",
  false);

const Core::ParameterFloat paramWordLengthScale(
  "word-length-scale",
  "scale for word length balance",
  1.0, 0.0);

// simple beam search + heuristics
const Core::ParameterBool paramLengthNormalization(
  "length-normalization",
  "normalize the score by length for pruning and decision",
  false);

const Core::ParameterBool paramNormalizeLabelOnly(
  "normalize-label-only",
  "normalize total score by label length only",
  true);

const Core::ParameterBool paramNormalizeWordOnly(
  "normalize-word-only",
  "only normalize lm score by word length",
  false);
 
const Core::ParameterBool paramFixedBeamSearch(
  "fixed-beam-search",
  "apply simle beam search with fixed beam size",
  false);

const Core::ParameterFloat paramEosThreshold(
  "eos-threshold",
  "factor threshold w.r.t best non-ending hypothesis to filter EOS",
  Core::Type<f32>::max, 0.0);

// segmental decoding: mainly for equivalent transducer to segmental modeling
const Core::ParameterBool paramAllowBlankSegment(
  "allow-blank-segment",
  "allow partial segment of blank only (transducer to segmental)",
  false);
// ---- only when end processing is needed ----


Seq2SeqSearchSpace::Seq2SeqSearchSpace(const Core::Configuration& config,
                                   Core::Ref<const Am::AcousticModel> acousticModel,
                                   Bliss::LexiconRef lexicon,
                                   Core::Ref<const Lm::ScaledLanguageModel> lm,
                                   Score wpScale,
                                   Core::Ref<Nn::LabelScorer> labelScorer) :
    Core::Component(config),
    lexicon_(lexicon),
    acousticModel_(acousticModel),
    languageModel_(lm),
    wpScale_(wpScale),
    labelScorer_(labelScorer),
    staticLabelTree_(select("label-tree"), acousticModel, lexicon, labelScorer),
    lmLookahead_(nullptr),
    initialized_(false),

    useLmScore_(paramUseLmScore(config)),
    fullSumDecoding_(paramFullSumDecoding(config)),
    labelFullSum_(paramLabelFullSum(config)),
  
    instanceDeletionTolerance_(paramInstanceDeletionTolerance(config)),
    instanceLookaheadLabelThreshold_(paramInstanceLookaheadLabelThreshold(config)),

    allowLabelRecombination_(paramAllowLabelRecombination(config)),
    labelRecombinationLimit_(paramLabelRecombinationLimit(config)),
    allowLabelLoop_(paramAllowLabelLoop(config)),
    minLoopOccur_(paramMinLoopOccurance(config)),
    allowBlankLabel_(paramAllowBlankLabel(config)),
    blankLabelIndex_(labelScorer->getBlankLabelIndex()),
    blankLabelPenalty_(paramBlankLabelPenalty(config)),
    allowWordEndRecombination_(paramAllowWordEndRecombination(config)),
    wordEndRecombinationLimit_(paramWordEndRecombinationLimit(config)),

    needEndProcessing_(false),
    endNodeId_(Core::Type<u32>::max),
    endExitId_(Core::Type<u32>::max),
    restrictWithInputLength_(paramRestrictWithInputLength(config)),

    fixedBeamSearch_(paramFixedBeamSearch(config)),
    lengthNorm_(paramLengthNormalization(config)),
    normLabelOnly_(paramNormalizeLabelOnly(config)),
    normWordOnly_(paramNormalizeWordOnly(config)),
    eosThreshold_(paramEosThreshold(config)),

    stepReNorm_(paramStepReNormalization(config)),
    stepEarlyStop_(paramStepEarlyStop(config)),
    stepLengthOnly_(paramStepLengthOnly(config)),
    stepLengthScale_(paramStepLengthScale(config)),

    pruneWordsWithLabels_(paramPruneWordsWithLabels(config)),
    wordLenBalance_(paramWordLengthBalance(config)),
    wordLenScale_(paramWordLengthScale(config)),

    allowBlankSegment_(paramAllowBlankSegment(config)),
    silenceIndex_(Core::Type<Index>::max),

    useRelativePosition_(false),
    verticalTransition_(false) {

  clear();

  if (!useLmScore_)
    warning() << "deactivate Languge Model scoring";

  if (fullSumDecoding_) {
    // check full history 
    if (labelRecombinationLimit_ != -1)
      warning() << "apply full-sum decoding with truncated label history " << labelRecombinationLimit_;
    if (wordEndRecombinationLimit_ != -1)
      error() << "apply full-sum decoding with truncated word history " << wordEndRecombinationLimit_;
    if (labelFullSum_)
      log() << "apply full-sum decoding based on full label sequence as well";
  }
}

Seq2SeqSearchSpace::~Seq2SeqSearchSpace() {
  clear();
  if (lmLookahead_) {
    unigramLookAhead_.reset();
    delete lmLookahead_;
  }
}

void Seq2SeqSearchSpace::clear() {
  labelHypotheses_.clear();
  newLabelHypotheses_.clear();

  currentInstance_ = nullptr;
  for (u32 t = 0; t < activeInstances_.size(); ++t)
    delete activeInstances_[t];
  activeInstances_.clear();
  activeInstanceMap_.clear();

  earlyWordEndHypotheses_.clear();
  wordEndHypotheses_.clear();

  endTraces_.clear();
  bestLabelEndTrace_.reset();

  decodeStep_ = 0;
  inputLength_ = 0;
  globalScoreOffset_ = 0.0;

  bestLabelScore_ = Core::Type<Score>::max;
  bestLabelProspect_ = Core::Type<Score>::max;
  bestWordEndProspect_ = Core::Type<Score>::max;
  bestEndTraceProspect_ = Core::Type<Score>::max; 

  wordLenBestProspect_.clear();
  wordLenScore_.clear();

  stepSumScore_ = Core::Type<Score>::max;
  stepEndScore_ = Core::Type<Score>::max;
  stepAccuLenScore_ = 0.0;

  stopSearch_ = false;
}

void Seq2SeqSearchSpace::initialize(bool simpleBeamSearch) {
  // search network (label tree)
  if (!staticLabelTree_.read()) {
    // read image failed, build it
    staticLabelTree_.build(); 
    if (staticLabelTree_.write())
      log() << "writing network image succeed";
    else
      warning() << "writing network image failed";
  }

  if (allowLabelLoop_) {
    staticLabelTree_.activateLoop();
    if (minLoopOccur_ > 0) {
      verify(!allowBlankLabel_); // not much point together with blank
      log() << "force label loop to occur at least " << minLoopOccur_ << " times";
    }
  } else {
    verify(minLoopOccur_ == 0);
  }
  staticLabelTree_.setExpansionPenalty(allowLabelLoop_);

  if (allowBlankLabel_) {
    if (blankLabelIndex_ == Core::Type<Index>::max)
      criticalError() << "no blank label found";
    Score blankProbThreshold = paramBlankLabelProbabilityThreshold(config);
    log() << "blank label penalty " << blankLabelPenalty_ << " probability threshold " << blankProbThreshold;
    blankLabelScoreThreshold_ = (blankProbThreshold == 0) ?
                                Core::Type<Score>::max : -std::log(blankProbThreshold);
    // CTC-like, but no forcing blank between label repetitions
    // this is already known from the tree (also clear in the recognition output)
    if (allowLabelLoop_)
      log() << "both blank and label loop allowed (stop loop after blank)";
  }

  // speed-up: cache hash for 1st-order recombination context dependency
  // higher-order still have redundant hash computation
  if (labelScorer_->isHistoryDependent() && labelRecombinationLimit_ == 1)
    labelHistoryHash_.resize(labelScorer_->numClasses(), 0); 

  positionDependent_ = labelScorer_->isPositionDependent();
  needEndProcessing_ = labelScorer_->needEndProcess();
  verticalTransition_ = labelScorer_->useVerticalTransition() && allowBlankLabel_;
  if (positionDependent_) {
    // segmental decoding: use position to finish
    // the underlying label topology does not matter anymore and is left to the model completely
    verify(needEndProcessing_);
    verify(!allowLabelLoop_);
    NodeId silId = staticLabelTree_.silence();
    if (silId != Core::Type<NodeId>::max) {
      // possible special duration handling for silence
      silenceIndex_ = staticLabelTree_.getLabelIndex(silId);
      log() << "silence tree node id:" << silId << " model index:" << silenceIndex_;
    }
  } else if (verticalTransition_) {
    // alignment-sync search for standard RNN-T: use position to finish
    verify(needEndProcessing_);
    verify(!allowLabelLoop_);
    // output sequence longer than encoder length possible
    if (restrictWithInputLength_)
      error() << "apply vertical transition but limit the output sequence w.r.t. input length";
  } else if (needEndProcessing_) {
    // label-sync search for attention model (no position): use end label to finish
    endLabelIndex_ = labelScorer_->getEndLabelIndex();
    if (endLabelIndex_ == Core::Type<Index>::max)
      criticalError("no end label found");
    // add sentence end score already at word end for better asynchronous endTraces
    staticLabelTree_.activateEndLabel(endLabelIndex_, getEndLemma());
    endNodeId_ = staticLabelTree_.getEndNodeId();
    endExitId_ = staticLabelTree_.getEndExitIdx();
    log() << "end label index:" << endLabelIndex_
          << " tree node id:" << endNodeId_
          << " (total number of nodes:" << staticLabelTree_.numNodes() << ")"
          << " exit id:" << endExitId_
          << " (total number of exits:" << staticLabelTree_.allExits().size() << ")";
  } else { 
    // otherwise should be strictly monotonic: time-sync search
    // additional relative position for scoring: so far only for blank-based FFNN-Transducer
    useRelativePosition_ = labelScorer_->useRelativePosition() && allowBlankLabel_;
    if (useRelativePosition_) {
      const Core::ParameterInt paramRelPosClip("relative-position-clipping", "", 32);
      relativePositionClip_ = paramRelPosClip(config);
      log() << "use relative position in decoding with clipping" << relativePositionClip_;
    }
  }

  initializePruning(simpleBeamSearch);
  initializeLanguageModel();
  initialized_ = true;
}

void Seq2SeqSearchSpace::initializePruning(bool simpleBeamSearch) {
  localLabelPruning_ = paramLocalLabelPruning(config);
  if (localLabelPruning_ != Core::Type<Score>::max)
    log() << "using local label pruning " << localLabelPruning_;

  labelPruning_ = paramLabelPruning(config);
  labelPruningLimit_ = paramLabelPruningLimit(config);
  if (!simpleBeamSearch)
    log() << "using label pruning " << labelPruning_ << " limit " << labelPruningLimit_;

  wordEndPruning_ = paramWordEndPruning(config);
  wordEndPruningLimit_ = paramWordEndPruningLimit(config);
  if (wordEndPruning_ <= 1.0)
    wordEndPruning_ *= labelPruning_;
  log() << "using word end pruning " << wordEndPruning_ << " limit " << wordEndPruningLimit_;

  // ending traces pruning: affect resulting lattice size
  pruneTrace_ = paramPruneTrace(config);
  tracePruning_ = paramTracePruning(config);
  tracePruningLimit_ = paramTracePruningLimit(config);
  if (pruneTrace_ && !simpleBeamSearch && needEndProcessing_)
    log() << "using trace pruning " << tracePruning_ << " limit " << tracePruningLimit_;

  if (simpleBeamSearch) {
    log() << "apply simple beam search with one global beam on all hyps level";
    if (fixedBeamSearch_) 
      log() << "using word end pruning limit as fixed beam size";
    else
      log() << "using word end pruning and limit for global pruning";
    pruneTrace_ = false;
    pruneWordsWithLabels_ = true; // just for cleanUp flag
    wordLenBalance_ = false;
    stepReNorm_ = false;
  }

  // histogram pruning threshold computation (shared by all levels of pruning)
  histogram_.setBins(paramHistogramPruningBins(config));

  // heuristic approaches
  lengthNorm_ = lengthNorm_ && needEndProcessing_;
  if (lengthNorm_) {
    log() << "apply length normalization for pruning and decision";
    warning() << "can not apply score offset with normalized scores";
    if (normLabelOnly_)
      log() << "normalize total score by label lenth only";
    else if (normWordOnly_)
      log() << "only normalize lm score by word length";
  }
  if (eosThreshold_ != Core::Type<Score>::max)
    log() << "apply eos threshold " << eosThreshold_;

  // ---- advanced search ----
  if (pruneWordsWithLabels_)
    log() << "apply global pruning between labels and word-ends";

  wordLenBalance_ = wordLenBalance_ && needEndProcessing_;
  if (wordLenBalance_) {
    // upon global pruning between labels and word-ends
    verify(!lengthNorm_);
    log() << "apply word-length balanced global pruning"
          << " - label pruning within same word length and skip limit"
          << " - word end pruning global";
  }

  stepReNorm_ = stepReNorm_ && needEndProcessing_;
  stepEarlyStop_ = stepEarlyStop_ && stepReNorm_;
  if (stepReNorm_) { 
    verify(!lengthNorm_);
    log() << "apply step-wise re-normalization for ending traces";
    if (stepLengthOnly_)
      log() << "explicit length modeling only";
    if (stepLengthScale_ != 1.0)
      log() << "scale length score with " << stepLengthScale_;
    if (stepEarlyStop_)
      log() << "further apply early stopping";
  }

  if (positionDependent_ && allowBlankLabel_ && allowBlankSegment_)
    log() << "allow partial segment with only blank";
}

void Seq2SeqSearchSpace::initializeLanguageModel() { 
  // TODO if full-sum-decoding, verify unlimited history of recombinationLm_ 
  if (paramSeparateRecombinationLm(config)) {
    log() << "using separate recombination lm";
    recombinationLm_ = Lm::Module::instance().createLanguageModel(select("recombination-lm"), lexicon_);
  } else if (languageModel_->recombinationLanguageModel()) {
    log() << "using recombination lm from one of the combined lms";
    recombinationLm_ = languageModel_->recombinationLanguageModel();
  } else {
    recombinationLm_ = languageModel_;
  }

  if (useLmScore_ && paramEnableLmLookahead(config)) {
    // Note: require explicit lm-lookahead.scale setting to avoid double scaling mistake
    if (paramSeparateLookaheadLm(config)) {
      log() << "using separate lookahead lm";
      lookaheadLm_ = Lm::Module::instance().createLanguageModel(select("lookahead-lm"), lexicon_);
    } else if (languageModel_->lookaheadLanguageModel()) {
      log() << "using lookahead lm from one of the combined lms";
      lookaheadLm_ = languageModel_->lookaheadLanguageModel();
    } else {
      lookaheadLm_ = languageModel_->unscaled();
    }

    lmLookahead_ = new Seq2SeqTreeSearch::LanguageModelLookahead(
        Core::Configuration(config, "lm-lookahead"), wpScale_, 
        Core::Ref<Lm::LanguageModel>(const_cast<Lm::LanguageModel*>(lookaheadLm_.get())), 
        staticLabelTree_);
    // unigram initialization for speed up
    unigramHistory_ = lookaheadLm_->reducedHistory(lookaheadLm_->startHistory(), 0);
    unigramLookAhead_ = lmLookahead_->getLookahead(unigramHistory_);
  } else {
    log() << "lm-lookahead deactivated";
  }
}

void Seq2SeqSearchSpace::addStartupWordEndHypothesis(Index step) {
  Nn::LabelHistory lbh = labelScorer_->startHistory();
  verify(lbh.isValid());

  Lm::History rch = recombinationLm_->startHistory();
  Lm::History sch = languageModel_->startHistory();
  verify(rch.isValid());
  verify(sch.isValid());

  Lm::History lah;
  if (lmLookahead_) { 
    lah = lookaheadLm_->startHistory();
    verify(lah.isValid());
  }

  SearchAlgorithm::ScoreVector score(0.0, 0.0);
  Core::Ref<Seq2SeqTreeSearch::Trace> t(new Seq2SeqTreeSearch::Trace(step, score));
  t->score.acoustic += globalScoreOffset_;
  wordEndHypotheses_.push_back(WordEndHypothesis(lbh, t, 
                                                 rch, sch, lah, score, 0,
                                                 Core::Type<NodeId>::max, Core::Type<u32>::max, 
                                                 0, 0, 0));
}

void Seq2SeqSearchSpace::startNewTrees() {
  for (const WordEndHypothesis& weh : wordEndHypotheses_) {
    TreeInstance* instance = activateOrUpdateTree(weh);
    verify(instance);
  }
  wordEndHypotheses_.clear();
}

TreeInstance* Seq2SeqSearchSpace::activateOrUpdateTree(const WordEndHypothesis& weh) {
  // treeKey always based on full-order history depending on the recombination LM
  TreeInstanceKey key(weh.recombinationHistory);
  KeyInstanceMap::iterator iter = activeInstanceMap_.find(key);
  if (iter == activeInstanceMap_.end()) {
    TreeInstance* t = new TreeInstance(key, weh.scoreHistory, weh.lookaheadHistory);
    iter = activeInstanceMap_.insert(std::make_pair(key, t)).first;
    activeInstances_.push_back(t);
  }
  TreeInstance* t = iter->second;
  NodeId transitRoot = (weh.exitId == Core::Type<u32>::max) ? 
                       staticLabelTree_.root() : (staticLabelTree_.getExit(weh.exitId)).transitRoot;
  t->enter(transitRoot, weh);
  // compute hash also for roots (possible to stay in blank)
  LabelHypothesis& lh = t->entryLabelHypotheses.back();
  lh.hash = lh.labelHistory.reducedHashKey(labelRecombinationLimit_);
  return t;
}

void Seq2SeqSearchSpace::expandLabels() {
  bestLabelScore_ = Core::Type<Score>::max;
  bestLabelProspect_ = Core::Type<Score>::max;
  wordLenBestProspect_.clear();

  // label expansion + local scoring
  if (staticLabelTree_.useExpansionPenalty()) {
    if (localLabelPruning_ == Core::Type<Score>::max)
      _expandLabels<true, false>();
    else
      _expandLabels<true, true>();
  } else {
    if (localLabelPruning_ == Core::Type<Score>::max)
      _expandLabels<false, false>();
    else
      _expandLabels<false, true>();
  }

  // further LM-lookahead scoring + record best
  // after all trees' label expansion (maybe beneficial for multi-threading lm scoring)
  if (eosThreshold_ == Core::Type<Score>::max) {
    if ( wordLenBalance_ )
      applyLookaheadInInstances<false, true>();
    else 
      applyLookaheadInInstances<false, false>();
  } else {
    if (wordLenBalance_)
      applyLookaheadInInstances<true, true>();
    else
      applyLookaheadInInstances<true, false>();
  }
}

// label expansion is the most expensive steps: use template flags to avoid redundant check
// especially for those in the successors loop
template <bool transitionPenalty, bool localPruning>
void Seq2SeqSearchSpace::_expandLabels() {
  if (allowBlankLabel_ ) {
    if (useRelativePosition_)
      expandLabelsInTree<true, true, transitionPenalty, localPruning>();
    else
      expandLabelsInTree<true, false, transitionPenalty, localPruning>();
  } else {
    expandLabelsInTree<false, false, transitionPenalty, localPruning>();
  }
}

template <bool allowBlank, bool relativePosition, bool transitionPenalty, bool localPruning>
void Seq2SeqSearchSpace::expandLabelsInTree() {
  newLabelHypotheses_.reserve(labelHypotheses_.size() + activeInstances_.size());
  for (TreeInstance* instance : activeInstances_) {
    currentInstance_ = instance;
    LabelHypothesesList::const_iterator lhBegin = labelHypotheses_.begin() + instance->labels.begin,
                                        lhEnd = labelHypotheses_.begin() + instance->labels.end;
    instance->labels.begin = newLabelHypotheses_.size();

    if (!positionDependent_) {
      // entry roots
      for (const LabelHypothesis& lh : instance->entryLabelHypotheses)
        expandLabelWithScore<allowBlank, relativePosition, transitionPenalty, localPruning>(lh, true);
      instance->entryLabelHypotheses.clear();
      // prev hyps
      for (LabelHypothesesList::const_iterator lh = lhBegin; lh != lhEnd; ++lh)
        expandLabelWithScore<allowBlank, relativePosition, transitionPenalty, localPruning>(*lh);
    } else { // segmental expansion
      // entry roots
      for (const LabelHypothesis& lh : instance->entryLabelHypotheses)
        expandLabelWithScoreAndPosition<allowBlank, transitionPenalty, localPruning>(lh, true);
      instance->entryLabelHypotheses.clear();
      // prev hyps
      for (LabelHypothesesList::const_iterator lh = lhBegin; lh != lhEnd; ++lh) 
        expandLabelWithScoreAndPosition<allowBlank, transitionPenalty, localPruning>(*lh);
    }

    instance->labels.end = newLabelHypotheses_.size();
  }
  labelHypotheses_.swap(newLabelHypotheses_);
  newLabelHypotheses_.clear();
}

// label/time/alignment synchronous label hypothesis expansion without segmental aspects
template <bool allowBlank, bool relativePosition, bool transitionPenalty, bool localPruning>
inline void Seq2SeqSearchSpace::expandLabelWithScore(const LabelHypothesis& lh, bool isRoot) {
  const std::vector<NodeId>& successors = staticLabelTree_.getSuccessors(lh.treeNodeId);
  if (successors.empty())
    return;
  bool isEnd = (successors.size() == 1 && lh.treeNodeId == successors.back());
  if (lh.isBlank && isEnd)
    return;
  if (isRoot)
    verify(lh.treeNodeId != successors.back()); // root has no loop

  const std::vector<Score>& scores = labelScorer_->getScores(lh.labelHistory, false);
  Score localThreshold = Core::Type<Score>::max;
  if (localPruning)
    localThreshold = *std::min_element(scores.begin(), scores.end()) + localLabelPruning_; 

  for (auto iter = successors.begin(); iter != successors.end(); ++iter) {
    bool isLoop = lh.treeNodeId == *iter; // forward or loop
    if (lh.isBlank && isLoop)
      continue; // no more loop after blank
    if (!isRoot && lh.nLoop < minLoopOccur_ && !isLoop)
      continue; // not forward before minimum duration (non-root)

    Index classId = staticLabelTree_.getLabelIndex(*iter);
    Score localScore = scores[classId];

    // loop may have different scoring mechanism
    if (isLoop) {
      // so far mainly posterior HMM
      // 1. (re)normalized forward|loop joint modeling (nClass+1 output)
      // 2. label-dependent loop (2*nClass output)
      if (scores.size() == labelScorer_->numClasses() + 1) {
        localScore = scores.back();
      } else if (scores.size() == labelScorer_->numClasses() * 2) {
        const std::vector<Score>& loopScores = labelScorer_->getScores(lh.labelHistory, true);
        localScore = loopScores[classId + labelScorer_->numClasses()];
      }
    }

    // local pruning without actually creating the label hypothesis
    // especially useful for open-vocab subword-based system (1 root followed by all labels)
    // Note: use safer threshold, since lookahead is not yet included here
    if (localPruning)
      if (localScore > localThreshold)
        continue;

    newLabelHypotheses_.push_back(lh);
    LabelHypothesis& nlh = newLabelHypotheses_.back();
    nlh.treeNodeId = *iter; nlh.isBlank = false; nlh.isLoop = isLoop;
    nlh.score.local = localScore;
    nlh.score.acoustic += localScore;
    if (isLoop) {
      ++nlh.nLoop;
    } else {
      ++nlh.nLabels;
      nlh.nLoop = 0;
    }
    if (transitionPenalty)
      nlh.score.acoustic += staticLabelTree_.getTransitionPenalty(lh.treeNodeId, nlh.treeNodeId);
    if (relativePosition)
      nlh.position = 0; 
  }

  if (allowBlank && !isEnd) {
    // blank (not loop and no transition)
    Score blankScore = scores[blankLabelIndex_];
    if (blankScore < blankLabelScoreThreshold_ && (!localPruning || blankScore <= localThreshold)) {
      // copy the same label into hypotheses and mark it blank
      newLabelHypotheses_.push_back(lh);
      LabelHypothesis& nlh = newLabelHypotheses_.back();
      nlh.isBlank = true; nlh.isLoop = false;
      nlh.score.local = blankScore;
      nlh.score.acoustic += blankScore;
      // penalize staying in blank
      nlh.score.acoustic += blankLabelPenalty_;
      if (relativePosition) {
        // increase relative position w.r.t. last non-blank label
        if (++nlh.position > relativePositionClip_)
          nlh.position = relativePositionClip_;
      } else if (verticalTransition_) {
        ++nlh.position;
      }
    }
  }
}

// segmental label hypothesis expansion: (label, pos, pos+len)
// allow blank: transducer -> segmental model
template <bool allowBlank, bool transitionPenalty, bool localPruning>
inline void Seq2SeqSearchSpace::expandLabelWithScoreAndPosition(const LabelHypothesis& lh, bool isRoot) {
  const std::vector<NodeId>& successors = staticLabelTree_.getSuccessors(lh.treeNodeId);
  if (successors.empty())
    return;

  bool reachEnd = true;
  Score localThreshold = Core::Type<Score>::max;

  for (auto sucIt = successors.begin(); sucIt != successors.end(); ++sucIt) {
    bool isLoop = lh.treeNodeId == *sucIt;
    verify(!isLoop); // no loop for segmental approach
    Index classId = staticLabelTree_.getLabelIndex(*sucIt);

    // given segment start and label -> (length, score: joint prob. of label and duration)
    const Nn::SegmentScore& segScores = labelScorer_->getSegmentScores(lh.labelHistory, classId, lh.position);
    if (localPruning && !segScores.empty())
      localThreshold = std::min(localThreshold, std::min_element(segScores.begin(), segScores.end(),
          [](const std::pair<u32, Score>& lhs, const std::pair<u32, Score>& rhs) { 
            return lhs.second < rhs.second;
          })->second + localLabelPruning_);

    for (Nn::SegmentScore::const_iterator iter = segScores.begin(); iter != segScores.end(); ++iter) {
      // local pruning without actually creating the label hypothesis
      if (localPruning)
        if (iter->second > localThreshold)
          continue;
      newLabelHypotheses_.push_back(lh);
      LabelHypothesis& nlh = newLabelHypotheses_.back();
      nlh.treeNodeId = *sucIt; ++nlh.nLabels; nlh.isBlank = false; nlh.isLoop = isLoop;
      nlh.score.local = iter->second;
      nlh.score.acoustic += nlh.score.local;
      if (allowBlank) {
        // optionally penalize staying in blank (exclude loop)
        nlh.score.acoustic += blankLabelPenalty_ * (iter->first - 1);
      } else if (transitionPenalty) {
        // tdp for length penalty: forward + (len-1) * loop
        nlh.score.acoustic += staticLabelTree_.getTransitionPenalty(lh.treeNodeId, nlh.treeNodeId) +
            staticLabelTree_.getTransitionPenalty(nlh.treeNodeId, nlh.treeNodeId) * (iter->first - 1);
      }
      nlh.position += iter->first; // t_s + 1: start position of the next segment
      reachEnd = false;
    }
  }

  // if not allow partial segment, then only tailing blank segment possible
  if (allowBlank && (allowBlankSegment_ || (isRoot && labelScorer_->maybeFinalSegment(lh.position)))) {
    // full sub-segment of only blanks: account for very long segment with several partial segments
    const Nn::SegmentScore& segScores = labelScorer_->getSegmentScores(lh.labelHistory, blankLabelIndex_, lh.position);
    if (!segScores.empty() && (!localPruning || segScores.back().second <= localThreshold)) {
      // copy the same label into hypotheses and mark it blank
      newLabelHypotheses_.push_back(lh);
      LabelHypothesis& nlh = newLabelHypotheses_.back();
      nlh.isBlank = true; nlh.isLoop = false;
      nlh.score.local = segScores.back().second;
      nlh.score.acoustic += nlh.score.local;
      nlh.score.acoustic += blankLabelPenalty_ * segScores.back().first;
      nlh.position += segScores.back().first; // t_s + 1
      reachEnd = false;
    }
  }

  // in case of cuted audio and tight pruning (no results): record best fallback label trace
  // reachEnd: mainly for segmental with remaining frames less than minimum
  if (reachEnd && endTraces_.empty())
    recordBestLabelEndTrace(lh);
}

void Seq2SeqSearchSpace::recordBestLabelEndTrace(const LabelHypothesis& lh) {
  Score traceProspect = computeTraceProspect(lh);
  if (!bestLabelEndTrace_ || traceProspect < bestLabelEndTrace_->prospect) {
    TraceRef& preTrace = currentInstance_->entryTraces[lh.traceId];    
    bestLabelEndTrace_ = Core::ref(new Seq2SeqTreeSearch::Trace(preTrace, 0, 0, 
                                                                decodeStep_+1, lh.score,
                                                                lh.nLabels, lh.nWords, lh.position));
    bestLabelEndTrace_->scoreHistory = currentInstance_->scoreHistory;
    bestLabelEndTrace_->score.acoustic += globalScoreOffset_;
    bestLabelEndTrace_->prospect = traceProspect;
  }
}

// all label prospect computed here
template <bool eos, bool wordLen>
void Seq2SeqSearchSpace::applyLookaheadInInstances() {
  for (TreeInstance* instance : activeInstances_) {
    instance->bestNonEndLocal = Core::Type<Score>::max;
    if (instance->labels.empty())
      continue;

    if (wordLen) {
      verify(!instance->entryNWords.empty());
      u32 maxLen = *(std::max_element(instance->entryNWords.begin(), instance->entryNWords.end()));
      if (maxLen + 1 > wordLenBestProspect_.size())
        wordLenBestProspect_.resize(maxLen + 1, bestLabelProspect_);
    }

    LabelHypothesesList::iterator lhBegin = labelHypotheses_.begin() + instance->labels.begin,
                                  lhEnd = labelHypotheses_.begin() + instance->labels.end;

    if (!lmLookahead_) {
      for (LabelHypothesesList::iterator lh = lhBegin; lh != lhEnd; ++lh) {
        // record best for pruning
        lh->prospect = computeLabelProspect(*lh);
        if (wordLen) {
          // word length dependent best prospect 
          Score& best = wordLenBestProspect_[lh->nWords];
          if (lh->prospect < best)
            best = lh->prospect;
        } else if (lh->prospect < bestLabelProspect_) {
          bestLabelProspect_ = lh->prospect;
        }
        // eos threshold: bestNonEndLocal per tree (not for word-based)
        if (eos && lh->treeNodeId != endNodeId_ && lh->score.local < instance->bestNonEndLocal)
          instance->bestNonEndLocal = lh->score.local;
      }
    } else { // add lm-lookahead scores
      if (instance->labels.size() >= labelHypotheses_.size() * instanceLookaheadLabelThreshold_)
        activateLmLookahead(instance, true);
      LmLookahead la = instance->lookahead ? instance->lookahead : unigramLookAhead_;

      for (LabelHypothesesList::iterator lh = lhBegin ; lh != lhEnd; ++lh) { 
        Score lmlaScore = la->score(lh->treeNodeId);
        lh->prospect = computeLabelProspect(*lh, lmlaScore);
        if (wordLen) {
          // word length dependent best prospect 
          Score& best = wordLenBestProspect_[lh->nWords];
          if (lh->prospect < best)
            best = lh->prospect;
        } else if (lh->prospect < bestLabelProspect_) {
          bestLabelProspect_ = lh->prospect;
        }
        lh->score.local += lmlaScore;
        if (eos && lh->treeNodeId != endNodeId_ && lh->score.local < instance->bestNonEndLocal)
          instance->bestNonEndLocal = lh->score.local;
      }
    }
  }
}

void Seq2SeqSearchSpace::activateLmLookahead(TreeInstance* instance, bool compute) {
  if (instance->lookahead)
    return;
  if (instance->lookaheadHistory == unigramHistory_)
    instance->lookahead = unigramLookAhead_;
  else if (compute)
    instance->lookahead = lmLookahead_->getLookahead(instance->lookaheadHistory);
}

// Note: label pruning is only within labels (word-ends and traces are not yet expanded)
// TODO if history independent (or limited ?), apply errorless lm-state-pruning ? 
//      more for time-sync and need efficient implementaion !
void Seq2SeqSearchSpace::applyLabelPruning() { 
  // word length dependent pruning
  if (wordLenBalance_) {
    // one bestProspect of each word length + same threshold
    if (restrictWithInputLength_ && decodeStep_ > inputLength_)
      pruneLabels<true, false, true, true>(labelPruning_);
    else
      pruneLabels<false, false, true, true>(labelPruning_);

    // within-word label recombination
    recombineLabels();

    // Note: histogram-like global upper limit is not safe at word length dependent pruning, 
    // but also not crucial with delayed label history extension => apply at global pruning later
    statistics_.customStatistics("label pruning") += labelPruning_;
    statistics_.customStatistics("label hypotheses") += labelHypotheses_.size();
    if (labelHypotheses_.size() > labelPruningLimit_)
      statistics_.customStatistics("label histogram saturation") += 1.0;
    else
      statistics_.customStatistics("label histogram saturation") += 0.0;
    return;
  }

  // simple score-based pruning
  verify(bestLabelProspect_ != Core::Type<Score>::max || labelHypotheses_.size() <= 1);
  Score threshold = bestLabelProspect_ + labelPruning_;
  if (restrictWithInputLength_ && decodeStep_ > inputLength_)
    pruneLabels<true, false, false, true>(threshold);
  else
    pruneLabels<false, false, false, true>(threshold);

  // within-word label recombination: much cheaper after pruning
  recombineLabels();  

  // histogram pruning
  if (labelHypotheses_.size() > labelPruningLimit_) {
    Score hpThreshold = quantileScore(bestLabelProspect_, threshold, labelPruningLimit_, true, false);
    pruneLabels<false, false, false, true>(hpThreshold);
    // add threshold and saturation statistics
    statistics_.customStatistics("label pruning") += hpThreshold - bestLabelProspect_; 
    statistics_.customStatistics("label hypotheses") += labelHypotheses_.size(); 
    statistics_.customStatistics("label histogram saturation") += 1.0;
  } else {
    statistics_.customStatistics("label pruning") += labelPruning_;
    statistics_.customStatistics("label hypotheses") += labelHypotheses_.size();
    statistics_.customStatistics("label histogram saturation") += 0.0;
  }
}

// clean up non-expandable labels (e.g. free memory immediately)
void Seq2SeqSearchSpace::cleanUp() {
  if (allowLabelLoop_)
    return; // there are always successors

  // do not clean up for the last step: might be needed for fallback
  if (needEndProcessing_ && restrictWithInputLength_) {
    if (verticalTransition_) {
      if (decodeStep_ == (2 * inputLength_ - 2))
        return;
    } else if (decodeStep_ == inputLength_)
      return;
  }

  pruneLabels<false, true, false, true>(Core::Type<Score>::max);
}

template<bool maxInLenStop, bool removeNonExpandable, bool wordLen, bool deleteTree>
void Seq2SeqSearchSpace::pruneLabels(Score threshold) {
  bool eos = eosThreshold_ != Core::Type<Score>::max;
  if (threshold == Core::Type<Score>::max && !eos && !maxInLenStop && !removeNonExpandable)
    return;

  u32 instIn, instOut;
  LabelHypothesesList::iterator hypIn, hypOut, hypBegin, instHypEnd;
  hypIn = hypOut = hypBegin = labelHypotheses_.begin();

  for (instIn = instOut = 0; instIn < activeInstances_.size(); ++instIn) {
    TreeInstance* at = activeInstances_[instIn];
    verify(hypIn == hypBegin + at->labels.begin);
    at->labels.begin = hypOut - hypBegin;

    Score eosThreshold = Core::Type<Score>::max;
    if (eos && at->bestNonEndLocal != Core::Type<Score>::max)
      eosThreshold = at->bestNonEndLocal * eosThreshold_;

    for (instHypEnd = hypBegin + at->labels.end; hypIn < instHypEnd; ++hypIn) {
      verify_(hypIn < labelHypotheses_.end());
      if (removeNonExpandable) {
        // remove label hypothesis without successors (only call after word-end expansion)
        // clean up + optional global pruning
        if (staticLabelTree_.hasSuccessors(hypIn->treeNodeId)) {
          // additional global pruning (optional word length balance)
          if (wordLen)
            hypIn->prospect = hypIn->score + wordLenScore_[hypIn->nWords];
          if (hypIn->prospect <= threshold)
            *(hypOut++) = *hypIn;
        }
      } else {
        if (maxInLenStop && (hypIn->nLabels > inputLength_ || hypIn->nWords > inputLength_))
          continue;
        if (eos && hypIn->treeNodeId == endNodeId_ && hypIn->score.local > eosThreshold)
          continue;
        if (wordLen) {
          // label pruning within the same word length
          if (hypIn->prospect <= wordLenBestProspect_[hypIn->nWords] + threshold)
            *(hypOut++) = *hypIn;
        } else if (hypIn->prospect <= threshold) {
          *( hypOut++ ) = *hypIn;
        }
      }
    }

    at->labels.end = hypOut - hypBegin;
    if (!deleteTree || !mayDeactivateTree(at))
      activeInstances_[instOut++] = at;
  }

  labelHypotheses_.erase(hypOut, labelHypotheses_.end());
  activeInstances_.resize(instOut);
}

inline bool Seq2SeqSearchSpace::mayDeactivateTree(TreeInstance *at) {
  if (at->labels.empty() && ++(at->inactive) > instanceDeletionTolerance_) {
    KeyInstanceMap::iterator iter = activeInstanceMap_.find(at->key);
    if (iter != activeInstanceMap_.end())
      activeInstanceMap_.erase(iter);
    delete at;
    return true;
  }
  return false;
}

void Seq2SeqSearchSpace::recombineLabels() { 
  if (!allowLabelRecombination_ && !fullSumDecoding_)
    return;

  // use template flags to avoid redundant check
  bool historyHash = labelScorer_->isHistoryDependent() && labelRecombinationLimit_ != 0;
  if (historyHash) {
    // history update scheme only relevant for historyHash needed
    if (labelScorer_->blankUpdateHistory()) {
      if (labelScorer_->loopUpdateHistory())
        recombineLabelsInTree<true, true, true, false>();
      else
        recombineLabelsInTree<true, false, true, false>();
    } else {
      if (labelScorer_->loopUpdateHistory())
        recombineLabelsInTree<false, true, true, false>();
      else
        recombineLabelsInTree<false, false, true, false>();
    }
  } else {
    // no history nor position dependency: simple recombine at the same node (speed up)
    if (positionDependent_)
      recombineLabelsInTree<false, false, false, false>();
    else
      recombineLabelsInTree<false, false, false, true>();
  }
}

// within word(tree) recombination: same expansion and scoring afterwards (viterbi or full-sum)
template <bool blankUpdateHistory, bool loopUpdateHistory, bool historyHash, bool simple>
void Seq2SeqSearchSpace::recombineLabelsInTree() { 
  // Note: all prospect computed already
  LabelHypothesesList::iterator hypIn, hypOut, hypBegin, instHypEnd;
  hypIn = hypOut = hypBegin = labelHypotheses_.begin();

  for (TreeInstance* at : activeInstances_) {
    verify(hypIn == hypBegin + at->labels.begin);
    at->labels.begin = hypOut - hypBegin;

    LabelHypothesesMap lhMap;
    LabelHashMap simpleMap;
    LabelHashMap::iterator iter;

    for (instHypEnd = hypBegin + at->labels.end; hypIn < instHypEnd; ++hypIn) {
      // update label history hash if applicable
      if (historyHash && (!hypIn->isBlank || blankUpdateHistory) && (!hypIn->isLoop || loopUpdateHistory)) {
        Index cId = hypIn->isBlank ? blankLabelIndex_ : 
                                     staticLabelTree_.getLabelIndex(hypIn->treeNodeId);
        // further avoid redundant computation: 1st-order only
        if (labelRecombinationLimit_ == 1) {
          if (labelHistoryHash_[cId] == 0)
            labelHistoryHash_[cId] = hypIn->labelHistory.reducedExtendedHashKey(labelRecombinationLimit_, cId); 
          hypIn->hash = labelHistoryHash_[cId];
        } else {
          hypIn->hash = hypIn->labelHistory.reducedExtendedHashKey(labelRecombinationLimit_, cId);
        }
      }
      // same tree node might still have different label-history and/or position dependency (hash)
      if (historyHash || !simple) {
        size_t hashKey = updateHashKey(hypIn->hash, hypIn->position);
        LabelHashMap& map = lhMap[hypIn->treeNodeId];
        iter = map.find(hashKey);
        if (iter == map.end()) {
          map.insert(std::make_pair(hashKey, hypOut));
          *(hypOut++) = *hypIn;
          continue;
        }
      } else { // no dependency: simple recombine at the same node
        iter = simpleMap.find(hypIn->treeNodeId);
        if (iter == simpleMap.end()) {
          simpleMap.insert(std::make_pair(hypIn->treeNodeId, hypOut));
          *(hypOut++) = *hypIn;
          continue;
        }
      }
      // recombine: Viterbi or full-sum
      // all properties are taken from the better path (based on prospect)
      // scores should be comparable for the same decode steps
      LabelHypothesis& keep = *(iter->second);
      LabelHypothesis& remove = *hypIn;
      if (fullSumDecoding_) {
        // full-sum requires full lm history (same lm score) !
        Score sumAcoustic = Math::scoreSum<Score>(keep.score.acoustic, remove.score.acoustic);
        if (remove.prospect < keep.prospect)
          keep = remove;
        keep.score.acoustic = sumAcoustic;
        // Note: no update of label prospect
      } else {
        // if full label history, lm history is not important
        // if truncated label history, full lm history should be used to ensure similar behaviour
        // special case for no lm or lm-independent output, possibly re-enter the same tree 
        // and recombine on the same label: favor the shorter sequence determinstically
        if (remove.prospect < keep.prospect || 
            (remove.prospect == keep.prospect && remove.nLabels < keep.nLabels))
          keep = remove;
      }
    }
    at->labels.end = hypOut - hypBegin;
  }
  labelHypotheses_.erase(hypOut, labelHypotheses_.end());
}

// Note: hidden states may be hard copied, thus more efficient after all label pruning
void Seq2SeqSearchSpace::extendLabelHistory() {
  u32 minPos = Core::Type<u32>::max;
  if (positionDependent_)
    for (const LabelHypothesis& lh : labelHypotheses_)
      if (lh.position < minPos)
        minPos = lh.position;
  // optional clean up before new accumulation during history extension
  labelScorer_->cleanUpBeforeExtension(minPos);

  if (labelScorer_->isHistoryDependent()) {
    for (LabelHypothesis& lh : labelHypotheses_) {
      Index cId = lh.isBlank ? blankLabelIndex_ : staticLabelTree_.getLabelIndex(lh.treeNodeId);
      labelScorer_->extendLabelHistory(lh.labelHistory, cId, lh.position, lh.isLoop);
    }
  }
}

void Seq2SeqSearchSpace::findWordEndsAndPrune() {
  verify(wordEndHypotheses_.empty());
  verify(earlyWordEndHypotheses_.empty());
  bestWordEndProspect_ = Core::Type<Score>::max;

  // reuse word length dependent best prospect
  u32 size = wordLenBestProspect_.size();
  wordLenBestProspect_.clear();
  wordLenBestProspect_.resize(size+1, bestWordEndProspect_);
  wordLenScore_.clear();
  wordLenScore_.resize(size+1, Core::Type<Score>::max);

  // step-wise beam renormalization
  stepSumScore_ = Core::Type<Score>::max;
  stepEndScore_ = Core::Type<Score>::max;

  bool exitPenalty = staticLabelTree_.useTransitionPenalty();
  if (wordLenBalance_) {
    findEarlyWordEnds<false, true, false>(exitPenalty);
  } else if (pruneWordsWithLabels_) {
    findEarlyWordEnds<false, false, true>(exitPenalty);
  } else { // no more global label pruning: avoid redundant extension
    extendLabelHistory();
    if (stepReNorm_)
      findEarlyWordEnds<true, false, false>(exitPenalty);
    else 
      findEarlyWordEnds<false, false, false>(exitPenalty);
  }

  if (wordLenBalance_ || pruneWordsWithLabels_) {
    // global pruning across labels and word-ends
    pruneLabelsAndWordEnds();
    return;
  }

  // prune word-ends only with simple absolute score threshold
  Score threshold = bestWordEndProspect_ + wordEndPruning_;
  pruneAndExpandEarlyWordEnds<false>(threshold, false);

  if (wordEndHypotheses_.size() > wordEndPruningLimit_) {
    // histogram pruning 
    Score hpThreshold = quantileScore(bestWordEndProspect_, threshold, wordEndPruningLimit_, false, true);
    pruneWordEnds(hpThreshold);
    // add threshold and saturation statistics
    statistics_.customStatistics("word-end pruning") += hpThreshold - bestWordEndProspect_;
    statistics_.customStatistics("word-end hypotheses") += wordEndHypotheses_.size();
    statistics_.customStatistics("word-end histogram saturation") += 1.0;
  } else {
    statistics_.customStatistics("word-end pruning") += wordEndPruning_;
    statistics_.customStatistics("word-end hypotheses") += wordEndHypotheses_.size();
    statistics_.customStatistics("word-end histogram saturation") += 0.0;
  }
}

// Note: full-sum merging pronunciation variants is put to next tree for simplicity
//       since reduced label history (labelRecombinationLimit_) anyway cannot be too small 
//       but need to check if word-end pruning make errors for them ?
template <bool stepReNorm, bool wordLen, bool pruneGlobal>
void Seq2SeqSearchSpace::findEarlyWordEnds(bool exitPenalty) {
  for (TreeInstance* instance : activeInstances_) {
    instance->earlyWehBegin = instance->earlyWehEnd = earlyWordEndHypotheses_.size();
    if (instance->labels.empty())
      continue;

    LabelHypothesesList::iterator lhBegin = labelHypotheses_.begin() + instance->labels.begin,
                                  lhEnd = labelHypotheses_.begin() + instance->labels.end;
    for (LabelHypothesesList::iterator lh = lhBegin; lh != lhEnd; ++lh) {
      if (wordLen || pruneGlobal || stepReNorm) {
        // expandable labels are individual hyps in the beam regardless of exit or not
        // non-expandable should always have exits, thus carried over in weh
        bool expandable = staticLabelTree_.hasSuccessors(lh->treeNodeId);
        if (wordLen && expandable) {
          Score& best = wordLenBestProspect_[lh->nWords];
          if (lh->score < best)
            best = lh->score;
          Score& sum = wordLenScore_[lh->nWords];
          sum = Math::scoreSum<Score>(sum, lh->score + globalScoreOffset_);
        } else if (pruneGlobal && expandable) {
          // Note: no lm-lookahead here
          lh->prospect = computeLabelProspect(*lh);
          if (lh->prospect < bestWordEndProspect_)
            bestWordEndProspect_ = lh->prospect;
        } else if (stepReNorm && expandable) { 
          // collect probability mass for renormalization
          stepSumScore_ = Math::scoreSum<Score>(stepSumScore_, lh->score + globalScoreOffset_);
        }
      }
 
      // blank label does not exit anymore: exit only on immediate label expansion
      // blank status can be carried over in next root
      if (!staticLabelTree_.hasExit(lh->treeNodeId) || lh->isBlank)
        continue;
      // length constraint
      if (restrictWithInputLength_ && lh->nLabels > inputLength_)
        continue;
      // forbid exit if not loop at least n times
      if (lh->nLoop < minLoopOccur_)
        continue;

      const std::vector<u32>& exitIds = staticLabelTree_.getExits(lh->treeNodeId);
      for (auto eIt = exitIds.begin(); eIt != exitIds.end(); ++eIt) {
        const LabelTree::Exit& exit = staticLabelTree_.getExit(*eIt);
        // early word end hypothesis
        u32 nWords = lh->nWords;
        if (exit.lemma && exit.lemma->syntacticTokenSequence().length() > 0)
          ++nWords;
        earlyWordEndHypotheses_.emplace_back(lh->labelHistory, lh->treeNodeId, lh->traceId, *eIt, lh->isLoop, lh->score, lh->nLabels, nWords, lh->position);
        EarlyWordEndHypothesis& eWeh = earlyWordEndHypotheses_.back();

        // add pronunciation score to acoustic
        if (exit.pronunciation)
          eWeh.score.acoustic += wpScale_ * exit.pronunciation->pronunciationScore();
        // exit penalty
        if (exitPenalty)
          eWeh.score.acoustic += staticLabelTree_.getExitPenalty(lh->treeNodeId); 
        // add lm score
        if (useLmScore_)
          eWeh.score.lm += instance->getLmScore(languageModel_, exit.lemma);

        if (wordLen) {
          // exited words: individual hyps in the beam (compute prospect later)
          Score& best = wordLenBestProspect_[eWeh.nWords];
          if (eWeh.score < best)
            best = eWeh.score;
          Score& sum = wordLenScore_[eWeh.nWords];
          sum = Math::scoreSum<Score>(sum, eWeh.score + globalScoreOffset_);
        } else { // prospect for pruning
          eWeh.prospect = computeWordEndProspect(eWeh);
          if (eWeh.prospect < bestWordEndProspect_)
            bestWordEndProspect_ = eWeh.prospect;
        }
      }
    }
    instance->earlyWehEnd = earlyWordEndHypotheses_.size();
  }
}

template <bool wordLen>
void Seq2SeqSearchSpace::pruneAndExpandEarlyWordEnds(Score threshold, bool extendlabelHistory) {
  // delayed label history extension after global pruning
  if (extendlabelHistory)
    extendLabelHistory();

  for (const TreeInstance* instance : activeInstances_) {
    EarlyWordEndHypothesesList::iterator eWehBegin = earlyWordEndHypotheses_.begin() + instance->earlyWehBegin,
                                         eWehEnd = earlyWordEndHypotheses_.begin() + instance->earlyWehEnd;
    for (EarlyWordEndHypothesesList::iterator eWeh = eWehBegin; eWeh != eWehEnd; ++eWeh) {
      if (wordLen)
        eWeh->prospect = eWeh->score + wordLenScore_[eWeh->nWords]; 
      if (eWeh->prospect > threshold)
        continue;
      // only non-blank labels can exit
      if (extendlabelHistory)
        labelScorer_->extendLabelHistory(eWeh->labelHistory, 
                                         staticLabelTree_.getLabelIndex(eWeh->treeNodeId),
                                         eWeh->position, eWeh->isLoop);
      const TraceRef& trace = instance->entryTraces[eWeh->traceId];
      // expand exit already here for later efficiency (but memeory increases) ?
      wordEndHypotheses_.emplace_back(eWeh->labelHistory, trace, instance->key.history,
                                      instance->scoreHistory, instance->lookaheadHistory,
                                      eWeh->score, eWeh->prospect, eWeh->treeNodeId, eWeh->exitId,
                                      eWeh->nLabels, eWeh->nWords, eWeh->position);
    } 
  }
  earlyWordEndHypotheses_.clear();
}

void Seq2SeqSearchSpace::pruneWordEnds(Score threshold) {
  if (threshold == Core::Type<Score>::max)
    return;
  WordEndHypothesesList::iterator in, out, end;
  for (in = out = wordEndHypotheses_.begin(), end = wordEndHypotheses_.end(); in != end; ++in)
    if (in->prospect <= threshold)
      *(out++) = *in;
  wordEndHypotheses_.erase(out, wordEndHypotheses_.end());
}

void Seq2SeqSearchSpace::extendWordHistory() {
  for (WordEndHypothesis& weh : wordEndHypotheses_) {
    const LabelTree::Exit& exit = staticLabelTree_.getExit(weh.exitId);
    if (!exit.lemma)
      continue;

    const Bliss::SyntacticTokenSequence tokenSequence(exit.lemma->syntacticTokenSequence());
    for (u32 t = 0, len = tokenSequence.length(); t < len; ++t) {
      const Bliss::SyntacticToken *st = tokenSequence[t];
      weh.recombinationHistory = recombinationLm_->extendedHistory(weh.recombinationHistory, st);
      weh.scoreHistory = languageModel_->extendedHistory(weh.scoreHistory, st);
      if (lmLookahead_)
        weh.lookaheadHistory = lmLookahead_->getReducedHistory(
            lookaheadLm_->extendedHistory(weh.lookaheadHistory, st));
    }
  }
}

Score Seq2SeqSearchSpace::quantileScore(Score minScore, Score maxScore, u32 nHyps, 
                                      bool label, bool word, bool endTrace) {
  histogram_.clear();
  histogram_.setLimits(minScore, maxScore);

  if (label)
    for (const LabelHypothesis& lh : labelHypotheses_)
      histogram_ += lh.prospect;
  
  if (word)
    for (const WordEndHypothesis& weh : wordEndHypotheses_)
      histogram_ += weh.prospect;
    
  if (endTrace) {
    if (label || word)
      verify(globalScoreOffset_ == 0); // otherwise not comparable
    for (const TraceRef& et : endTraces_)
      histogram_ += et->prospect;
  }

  return histogram_.quantile(nHyps, true);
}

// joint pruning across labels and word-ends
void Seq2SeqSearchSpace::pruneLabelsAndWordEnds() {
  if (wordLenBalance_) {
    // renormalized weighting for each word length at current label position (majority voting)
    // TODO apply word-length dependent pruning first ?
    Score sum = Nn::LabelScorer::computeScoreSum(wordLenScore_);
    std::transform(wordLenScore_.begin(), wordLenScore_.end(), wordLenScore_.begin(), 
                   std::bind(std::minus<Score>(), std::placeholders::_1, sum));
    if (wordLenScale_ != 1.0)
      std::transform(wordLenScore_.begin(), wordLenScore_.end(), wordLenScore_.begin(), 
                     std::bind(std::multiplies<Score>(), std::placeholders::_1, wordLenScale_));
    for (u32 idx = 0, size = wordLenBestProspect_.size(); idx < size; ++idx) {
      if (wordLenBestProspect_[idx] == Core::Type<Score>::max)
        continue;
      // word length balanced global best prospect
      wordLenBestProspect_[idx] += wordLenScore_[idx];
      if (wordLenBestProspect_[idx] < bestWordEndProspect_)
        bestWordEndProspect_ = wordLenBestProspect_[idx];
    }
  }

  // (mis)use wordend pruning for global pruning
  Score threshold = bestWordEndProspect_ + wordEndPruning_;
  if (wordLenBalance_) {
    // compute word-length balanced prospect in the meanwhile
    pruneLabels<false, true, true, false>(threshold);
    pruneAndExpandEarlyWordEnds<true>(threshold);
  } else {
    pruneLabels<false, true, false, false>(threshold);
    pruneAndExpandEarlyWordEnds<false>(threshold);
  }

  // histogram pruning (upper limit for memory)
  if (labelHypotheses_.size() + wordEndHypotheses_.size() > wordEndPruningLimit_) {
    Score hpThreshold = quantileScore(bestWordEndProspect_, threshold, wordEndPruningLimit_, true, true);
    pruneLabels<false, false, false, true>(hpThreshold);
    pruneWordEnds(hpThreshold); 
    // add threshold and saturation statistics
    statistics_.customStatistics("word-end pruning") += hpThreshold - bestWordEndProspect_;
    statistics_.customStatistics("word-end hypotheses") += wordEndHypotheses_.size();
    statistics_.customStatistics("word-end histogram saturation") += 1.0;
  } else {
    statistics_.customStatistics("word-end pruning") += wordEndPruning_;
    statistics_.customStatistics("word-end hypotheses") += wordEndHypotheses_.size();
    statistics_.customStatistics("word-end histogram saturation") += 0.0;
  }

  if (stepReNorm_ && !labelHypotheses_.empty()) {
    // probability mass of remained label hyps
    LabelHypothesesList::const_iterator lh = labelHypotheses_.begin();
    verify(lh->prospect != Core::Type<Score>::max && stepSumScore_ == Core::Type<Score>::max);
    stepSumScore_ = lh->prospect + globalScoreOffset_;
    for (++lh; lh != labelHypotheses_.end(); ++lh)
      stepSumScore_ = Math::scoreSum<Score>(stepSumScore_, lh->prospect + globalScoreOffset_);
  }
}

// simple beam search with global pruning across labels, word-ends and endTraces
void Seq2SeqSearchSpace::findWordEndsAndPruneGlobal() {
  // scores are comparable at all levels (label prospect computed already)
  verify(globalScoreOffset_ == 0);
  verify(wordEndHypotheses_.empty());
  verify(earlyWordEndHypotheses_.empty());
  bestWordEndProspect_ = Core::Type<Score>::max;

  // filter out invalid labels + apply safe pruning if score-based search
  Score threshold = fixedBeamSearch_ ? Core::Type<Score>::max : bestLabelProspect_ + wordEndPruning_;
  if (!fixedBeamSearch_ || eosThreshold_ != Core::Type<Score>::max) {
    if (restrictWithInputLength_ && decodeStep_ > inputLength_)
      pruneLabels<true, false, false, true>(threshold);
    else
      pruneLabels<false, false, false, true>(threshold);
    if (!fixedBeamSearch_ && labelHypotheses_.size() > wordEndPruningLimit_) {
      Score hpThreshold = quantileScore(bestLabelProspect_, threshold, wordEndPruningLimit_, true, false);
      pruneLabels<false, false, false, true>(hpThreshold);
    }
  }

  if (fixedBeamSearch_) {
    // expand word-ends for joint pruning only if different scoring due to LM
    if (!useLmScore_ || (lmLookahead_ && lookaheadLm_ == languageModel_->unscaled())) {
      pruneGlobalWithFixedBeam(wordEndPruningLimit_, false);
      recombineLabels();
      findEarlyWordEnds<false, false, false>(staticLabelTree_.useTransitionPenalty());
    } else {
      findEarlyWordEnds<false, false, false>(staticLabelTree_.useTransitionPenalty());
      pruneGlobalWithFixedBeam(wordEndPruningLimit_);
      recombineLabels();
    }
    pruneAndExpandEarlyWordEnds<false>(Core::Type<Score>::max);
    return;
  }

  // expand word-ends upon pruned labels
  findEarlyWordEnds<false, false, false>(staticLabelTree_.useTransitionPenalty());

  // misuse wordend pruning for global pruning
  Score bestProspect = bestWordEndProspect_;
  if (bestLabelProspect_ < bestProspect)
    bestProspect = bestLabelProspect_;
  if (!endTraces_.empty() && bestEndTraceProspect_ < bestProspect)
    bestProspect = bestEndTraceProspect_;
  threshold = bestProspect + wordEndPruning_;

  // non-expandable labels can be removed now
  pruneLabels<false, true, false, false>(threshold);
  recombineLabels();
  pruneAndExpandEarlyWordEnds<false>(threshold);
  pruneEndTraces(threshold);

  u32 size = labelHypotheses_.size() + wordEndHypotheses_.size() + endTraces_.size();
  if (size > wordEndPruningLimit_) {
    // histogram pruning
    Score hpThreshold = quantileScore(bestProspect, threshold, wordEndPruningLimit_, true, true, true);
    pruneLabels<false, false, false, true>(hpThreshold);
    pruneWordEnds(hpThreshold);
    pruneEndTraces(hpThreshold);
    // add threshold and saturation statistics
    statistics_.customStatistics("word-end pruning") += hpThreshold - bestProspect;
    statistics_.customStatistics("word-end hypotheses") += wordEndHypotheses_.size();
    statistics_.customStatistics("word-end histogram saturation") += 1.0;
  } else {
    statistics_.customStatistics("word-end pruning") += wordEndPruning_;
    statistics_.customStatistics("word-end hypotheses") += wordEndHypotheses_.size();
    statistics_.customStatistics("word-end histogram saturation") += 0.0;
  }
}

// global fixed beam pruning (only for simple beam search)
void Seq2SeqSearchSpace::pruneGlobalWithFixedBeam(u32 beamSize, bool expandable) {
  // Note: wordends are not expanded yet
  u32 size = labelHypotheses_.size() + earlyWordEndHypotheses_.size() + endTraces_.size();
  if (size <= beamSize)
    return;

  // beam category: (expandable) label = 0, wordend = 1, trace = 2
  Beam beam;
  for (u32 idx = 0, size = labelHypotheses_.size(); idx < size; ++idx) {
    if (expandable && !staticLabelTree_.hasSuccessors(labelHypotheses_[idx].treeNodeId))
      continue;
    // length constraint
    if (restrictWithInputLength_ && labelHypotheses_[idx].nLabels > inputLength_)
      continue;
    insertBeam(beam, beamSize, labelHypotheses_[idx].prospect, 0, idx);
  }
  for (u32 idx = 0, size = earlyWordEndHypotheses_.size(); idx < size; ++idx)
    insertBeam(beam, beamSize, earlyWordEndHypotheses_[idx].prospect, 1, idx);
  for (u32 idx = 0, size = endTraces_.size(); idx < size; ++idx)
    insertBeam(beam, beamSize, endTraces_[idx]->prospect, 2, idx);

  if (beam.size() < beamSize)
    return;
  require_eq(beam.size(), beamSize);

  std::vector<u32> beamLabel, beamWord;
  TraceList beamTrace;
  for (Beam::iterator iter = beam.begin(); iter != beam.end(); ++iter) {
    if (iter->second.first == 0)
      beamLabel.push_back(iter->second.second);
    else if (iter->second.first == 1)
      beamWord.push_back(iter->second.second);
    else
      beamTrace.push_back(endTraces_[iter->second.second]);
  }
  endTraces_.swap(beamTrace);

  // label hyps and early word-end hyps are both tree-based (all sorted)
  std::sort(beamLabel.begin(), beamLabel.end());
  std::sort(beamWord.begin(), beamWord.end());
  u32 beamLabelIdx = 0, beamWordIdx = 0;
  EarlyWordEndHypothesesList earlyWeh; 
  newLabelHypotheses_.clear();
  for (TreeInstance* at : activeInstances_) {
    u32 labelsize = newLabelHypotheses_.size();
    u32 wehsize = earlyWeh.size();
    while (beamLabelIdx < beamLabel.size() && at->labels.contains(beamLabel[beamLabelIdx])) {
      newLabelHypotheses_.push_back(labelHypotheses_[beamLabel[beamLabelIdx]]);
      ++beamLabelIdx;
    }
    while (beamWordIdx < beamWord.size() && at->earlyWehContains(beamWord[beamWordIdx])) {
      earlyWeh.push_back(earlyWordEndHypotheses_[beamWord[beamWordIdx]]);
      ++beamWordIdx;
    }
    at->labels.begin = labelsize;
    at->labels.end = newLabelHypotheses_.size();
    at->earlyWehBegin = wehsize;
    at->earlyWehEnd = earlyWeh.size();
    // no tree deletion here
  }
  verify(beamLabelIdx == beamLabel.size() && beamWordIdx == beamWord.size()); // all found
  labelHypotheses_.swap(newLabelHypotheses_);
  newLabelHypotheses_.clear();
  earlyWordEndHypotheses_.swap(earlyWeh);
}

inline void Seq2SeqSearchSpace::insertBeam(Beam& beam, u32 beamSize, Score score, 
                                           u32 category, u32 idx) {
  if (beam.size() < beamSize) {
    beam.insert(std::make_pair(score, std::make_pair(category, idx)));
  } else if (score < beam.begin()->first) {
    beam.erase(beam.begin());
    beam.insert(std::make_pair(score, std::make_pair(category, idx)));
  } 
}

void Seq2SeqSearchSpace::createTraces() {
  for (WordEndHypothesis& weh : wordEndHypotheses_) {
    const LabelTree::Exit& exit = staticLabelTree_.getExit(weh.exitId);
    verify(exit.lemma || exit.pronunciation || weh.exitId == endExitId_);
    weh.trace = Core::ref(new Seq2SeqTreeSearch::Trace(weh.trace, exit.pronunciation, exit.lemma,
                                                       decodeStep_, weh.score,
                                                       weh.nLabels, weh.nWords, weh.position));
    weh.trace->score.acoustic += globalScoreOffset_;
    weh.trace->prospect = computeTraceProspect(weh.trace);
  }
}

void Seq2SeqSearchSpace::recombineWordEnds(bool createLattice) {
  if (!allowWordEndRecombination_ && !fullSumDecoding_)
    return;

  bool labelHistoryHash = labelScorer_->isHistoryDependent() && labelRecombinationLimit_ != 0;
  if (labelHistoryHash) {
    if (positionDependent_ || staticLabelTree_.isHMMTree())
      _recombineWordEnds<true, true>(createLattice);
    else
      _recombineWordEnds<true, false>(createLattice);
  } else {
    if (positionDependent_ || staticLabelTree_.isHMMTree())
      _recombineWordEnds<false, true>(createLattice);
    else
      _recombineWordEnds<false, false>(createLattice);
  }
}

template<bool labelHistoryHash, bool labelOtherHash>
void Seq2SeqSearchSpace::_recombineWordEnds(bool createLattice) {
  WordEndHypothesesMap wehMap;
  WordEndLabelMap simpleMap;
  WordEndLabelMap::iterator iter;

  WordEndHypothesesList::iterator in, out, end;
  for (in = out = wordEndHypotheses_.begin(), end = wordEndHypotheses_.end(); in != end; ++in) {
    // no reduction on original recombination history for exact treeKey
    size_t wordHash = wordEndRecombinationLimit_ >= 0 ? 
        recombinationLm_->reducedHistory(in->recombinationHistory, 
                                         wordEndRecombinationLimit_).hashKey() :
        in->recombinationHistory.hashKey();
    if (labelHistoryHash || labelOtherHash) {
      // same word hash might still have different label hash
      // - (reduced) label history
      // - possible position and coarticulation roots (e.g. triphone models)
      size_t labelHash = 0;
      if (labelHistoryHash) {
        if (labelRecombinationLimit_ == 1) {
          Index cId = in->labelHistory.getLastLabel();
          labelHash = labelHistoryHash_.at(cId);
        } else {
          labelHash = in->labelHistory.reducedHashKey(labelRecombinationLimit_);
        }
      }
      if (labelOtherHash) {
        NodeId transitRoot = staticLabelTree_.getExit(in->exitId).transitRoot;
        labelHash = updateHashKey(updateHashKey(labelHash, transitRoot), in->position);
      }
      WordEndLabelMap& map = wehMap[wordHash];
      iter = map.find(labelHash);
      if (iter == map.end()) {
        map.insert(std::make_pair(labelHash, out));
        *(out++) = *in;
        continue;
      }
    } else { // simpler case: word hash only
      iter = simpleMap.find(wordHash);
      if (iter == simpleMap.end()) {
        simpleMap.insert(std::make_pair(wordHash, out));
        *(out++) = *in;
        continue;
      }
    }
    // recombine: Viterbi or full-sum
    recombineTwoWordEnds(*(iter->second), *in, createLattice);
  }
  wordEndHypotheses_.erase(out, wordEndHypotheses_.end());
  statistics_.customStatistics("word-end hypotheses after recombination") += wordEndHypotheses_.size();
}

// full-sum can already be done here instead of next tree for each initial label
// full-sum should be with full LM history, thus more for pronunciation/spelling variants here
inline void Seq2SeqSearchSpace::recombineTwoWordEnds(WordEndHypothesis& keep, 
                                                     WordEndHypothesis& remove, bool createLattice) {
  // replace keep by remove if better score or some deterministic order
  bool replace = keep.prospect > remove.prospect;
  if (!replace && keep.prospect == remove.prospect) {
    replace = keep.nLabels > remove.nLabels ||
              ( keep.nLabels == remove.nLabels &&
                (staticLabelTree_.getExit(keep.exitId)).lemma->id() > (staticLabelTree_.getExit(remove.exitId)).lemma->id() );
  }

  if (fullSumDecoding_) {
    Score sumAcoustic = Math::scoreSum<Score>(keep.score.acoustic, remove.score.acoustic);
    if (replace)
      keep = remove; // take property from better path
    keep.score.acoustic = sumAcoustic;
    keep.prospect = computeWordEndProspect(keep);
    keep.trace->score.acoustic = sumAcoustic + globalScoreOffset_;
    keep.trace->prospect = computeTraceProspect(keep.trace);
  } else { // Viterbi recombination
    if (replace) {
      if (createLattice) {
        verify(!remove.trace->sibling);
        remove.trace->sibling = keep.trace; 
      }
      keep = remove;
    } else if (createLattice) {
      verify(!remove.trace->sibling);
      remove.trace->sibling = keep.trace->sibling;
      keep.trace->sibling = remove.trace;
    }
  }
}

// simply remove siblings without lm token (anyway no chance to change)
void Seq2SeqSearchSpace::optimizeLattice() {
  for (WordEndHypothesis& weh : wordEndHypotheses_) {
    TraceRef trace = weh.trace;
    while (trace->sibling) {
      const Bliss::Lemma* lemma = trace->sibling->lemma; 
      if (lemma && lemma->syntacticTokenSequence().size() == 0)
        trace->sibling = trace->sibling->sibling;
      else
        trace = trace->sibling;
    }
  }
}

// only call after label pruning, apply offset only on true scores (not on prospect)
void Seq2SeqSearchSpace::rescale(Score offset) {
  if (lengthNorm_)
    return; // length normalization becomes incorrect with the offset

  if (offset == 0)
    offset = bestLabelScore();
  verify(wordEndHypotheses_.empty() && earlyWordEndHypotheses_.empty());

  for (LabelHypothesis& lh : labelHypotheses_)
    lh.score.acoustic -= offset;
  globalScoreOffset_ += offset;
}

Score Seq2SeqSearchSpace::bestLabelScore() {
  if (bestLabelScore_ == Core::Type<Score>::max) {
    verify(!labelHypotheses_.empty());
    for (const LabelHypothesis& lh : labelHypotheses_)
      if (lh.score < bestLabelScore_)
        bestLabelScore_ = lh.score;
  }
  return bestLabelScore_;
}

Score Seq2SeqSearchSpace::bestLabelProspect() {
  if (bestLabelProspect_ == Core::Type<Score>::max) {
    LabelHypothesesList::const_iterator bestHyp = bestProspectLabel();
    bestLabelProspect_ = bestHyp->prospect;
  }
  return bestLabelProspect_;
}

LabelHypothesesList::const_iterator Seq2SeqSearchSpace::bestProspectLabel() {
  verify(!labelHypotheses_.empty());
  Score bestProspect = Core::Type<Score>::max;
  LabelHypothesesList::const_iterator ret, lh;
  for (lh = labelHypotheses_.begin(); lh != labelHypotheses_.end(); ++lh)
    if (lh->prospect < bestProspect)
      bestProspect = (ret = lh)->prospect;
  return ret;
}

TreeInstance* Seq2SeqSearchSpace::bestProspectLabelTree(u32 bestIndex) {
  for (TreeInstance* at : activeInstances_)
    if (at->labels.contains(bestIndex))
      return at;
  return nullptr;
}

WordEndHypothesesList::const_iterator Seq2SeqSearchSpace::bestProspectWordEnd() {
  verify(!wordEndHypotheses_.empty());
  Score bestProspect = Core::Type<Score>::max;
  WordEndHypothesesList::const_iterator ret, weh;
  for (weh = wordEndHypotheses_.begin(); weh != wordEndHypotheses_.end(); ++weh)
    if (weh->prospect < bestProspect)
      bestProspect = (ret = weh)->prospect;
  return ret;
}

// asynchronous ending traces processing (e.g. attention, segmental, RNN-T) ----
void Seq2SeqSearchSpace::processEnd() {
  if (!needEndProcessing_)
    return;
  if (endTraces_.empty())
    bestEndTraceProspect_ = Core::Type<Score>::max;

  if (stepReNorm_) {
    if (wordLenBalance_)
      detectEndTraces<true, true>();
    else
      detectEndTraces<true, false>();
  } else {
    detectEndTraces<false, false>();
  }

  if (pruneTrace_ && !endTraces_.empty()) {
    Score threshold = bestEndTraceProspect_ + tracePruning_;
    pruneEndTraces(threshold);
    if (endTraces_.size() > tracePruningLimit_) {
      // histogram limit and statistics 
      Score hpThreshold = quantileScore(bestEndTraceProspect_, threshold, tracePruningLimit_, false, false, true);
      pruneEndTraces(hpThreshold);
      statistics_.customStatistics("trace pruning") += hpThreshold - bestEndTraceProspect_;
      statistics_.customStatistics("trace hypotheses") += endTraces_.size();
      statistics_.customStatistics("trace histogram saturation") += 1.0;
    } else {
      statistics_.customStatistics("trace pruning") += tracePruning_;
      statistics_.customStatistics("trace hypotheses") += endTraces_.size();
      statistics_.customStatistics("trace histogram saturation") += 0.0;
    }
  } 

  checkStoppingCriteria();

  // record fallback trace if about to stop but still no ending traces
  bool lastStep = false; // search will stop at the next step
  if (restrictWithInputLength_) {
    if (verticalTransition_)
      lastStep = decodeStep_ == (2 * inputLength_ - 2);
    else
      lastStep = decodeStep_ == inputLength_;
  }
  bool needFallBack = endTraces_.empty() && (stopSearch_ || lastStep);
  if (needFallBack && !bestLabelEndTrace_) {
    if (!wordEndHypotheses_.empty()) {
      WordEndHypothesesList::const_iterator weh = bestProspectWordEnd();
      bestLabelEndTrace_ = Core::ref(new Seq2SeqTreeSearch::Trace(weh->trace, 0, 0, 
          decodeStep_+1, weh->trace->score, weh->nLabels, weh->nWords, weh->position));
      bestLabelEndTrace_->scoreHistory = weh->scoreHistory;
      bestLabelEndTrace_->prospect = weh->trace->prospect;
    } else {
      LabelHypothesesList::const_iterator bestHyp = bestProspectLabel();
      u32 bestIndex = bestHyp - labelHypotheses_.begin();
      currentInstance_ = bestProspectLabelTree(bestIndex);
      recordBestLabelEndTrace(*bestHyp);
    }
  }
}

// ending traces 
// TODO for stepReNorm: use logSumExp to improve efficiency
template<bool stepReNorm, bool wordLen>
void Seq2SeqSearchSpace::detectEndTraces() {
  TraceList stepEndTraces;

  // tailing blank segment is also valid end (blank root states)
  if (verticalTransition_ || (positionDependent_ && allowBlankLabel_))
    detectEndTracesFromStates<stepReNorm, wordLen>(stepEndTraces);

  if (!verticalTransition_ || stepReNorm) {
    // ending in weh only possible if not vertical transition (root blank)
    WordEndHypothesesList::iterator in, out;
    Score score;
    bool isEnd = false;
    for (in = out = wordEndHypotheses_.begin(); in != wordEndHypotheses_.end(); ++in) {
      if (!verticalTransition_) {
        // Note: input length + 1 for maxLenStop including end label (match t_n+1)
        isEnd = positionDependent_ && in->position >= inputLength_-1;
        if (isEnd) {
          // segmental ending traces + </s> score (Note: not considered for word length balance)
          in->trace = Core::ref(new Seq2SeqTreeSearch::Trace(in->trace, 0, 0,
              decodeStep_+1, in->trace->score, in->nLabels, in->nWords+1, in->position));
          if (useLmScore_)
            in->trace->score.lm += languageModel_->sentenceEndScore(in->scoreHistory);
        } else {
          isEnd = in->exitId == endExitId_;
        }
      }
      if (stepReNorm) {
        // probability mass of remained word-end hyps
        score = in->trace->score;
        if (wordLen) // global balance for more reliable renormalization ?
          score += wordLenScore_[in->nWords];
        stepSumScore_ = Math::scoreSum<Score>(stepSumScore_, score);
      }
      // ending traces (sentence-end score already included)
      // move to end traces and no more expansion
      if (isEnd) {
        // not appear in transcription
        in->trace->pronunciation = 0;
        in->trace->lemma = 0;
        if (fullSumDecoding_) {
          // need to merge in the end
          in->trace->recombinationHistory = in->recombinationHistory;
          in->trace->labelHistory = in->labelHistory;
        }
        stepEndTraces.push_back(in->trace);
        // probability mass of ending hyps
        if (stepReNorm)
          stepEndScore_ = Math::scoreSum<Score>(stepEndScore_, score);
      } else {
        *(out++) = *in;
      }
    }
    wordEndHypotheses_.erase(out, wordEndHypotheses_.end());
  }

  // only ending traces need prospect for pruning and decision
  for (TraceRef& trace : stepEndTraces) {
    trace->prospect = computeTraceProspect(trace, true);
    if (trace->prospect < bestEndTraceProspect_)
      bestEndTraceProspect_ = trace->prospect;
  }
  endTraces_.insert(endTraces_.end(), stepEndTraces.begin(), stepEndTraces.end());

  if (stepReNorm && !stepEndTraces.empty()) {
    // accumulate non-ending probability (for next step)
    verify(stepEndScore_ >= stepSumScore_);
    Score endScore = stepEndScore_ - stepSumScore_;
    if (endScore == 0) // all ended
      stepAccuLenScore_ = Core::Type<Score>::max;
    else
      stepAccuLenScore_ += -std::log1p(-std::exp(-endScore));
  }
}

// tailing blank segment (blank root states)
// position reaching input(encoder) length always no more expansion
template<bool stepReNorm, bool wordLen>
void Seq2SeqSearchSpace::detectEndTracesFromStates(TraceList& stepEndTraces) {
  if (labelHypotheses_.empty())
    return;

  LabelHypothesesList::iterator hypIn, hypOut, hypBegin, instHypEnd;
  hypIn = hypOut = hypBegin = labelHypotheses_.begin();

  for (TreeInstance* at : activeInstances_) {
    verify(hypIn == hypBegin + at->labels.begin);
    at->labels.begin = hypOut - hypBegin;

    for (instHypEnd = hypBegin + at->labels.end; hypIn < instHypEnd; ++hypIn) {
      // valid end: blank root with position reaching input(encoder) length
      bool validEnd = hypIn->isBlank && staticLabelTree_.isRoot(hypIn->treeNodeId) &&
                      hypIn->position >= inputLength_ - 1;
      if (validEnd) {
        // ending traces + </s> score (Note: not considered for word length balance)
        TraceRef t(new Seq2SeqTreeSearch::Trace(at->entryTraces[hypIn->traceId], 0, 0,
                                                decodeStep_+1, hypIn->score,
                                                hypIn->nLabels, hypIn->nWords+1, hypIn->position));
        t->score.acoustic += globalScoreOffset_;
        if (useLmScore_)
          t->score.lm += languageModel_->sentenceEndScore(at->scoreHistory);
        if (fullSumDecoding_) {
          // need to merge in the end 
          t->recombinationHistory = at->key.history;
          t->labelHistory = hypIn->labelHistory;
        }
        if (stepReNorm) {
          // Note: already in the stepSumScore_, just add to the probability mass of ending hyps
          Score score = t->score;
          if (wordLen) // global balance for more reliable renormalization ?
            score += wordLenScore_[hypIn->nWords];
          stepEndScore_ = Math::scoreSum<Score>(stepEndScore_, score);
        }
        // move to end traces and no more expansion 
        stepEndTraces.push_back(t);
      } else if (hypIn->position < inputLength_ - 1) {
        *(hypOut++) = *hypIn;
      }
    }
    // no empty tree deletion here: will be done in the next step if still empty
    at->labels.end = hypOut - hypBegin;
  }
  labelHypotheses_.erase(hypOut, labelHypotheses_.end());
}

// prune ending traces
void Seq2SeqSearchSpace::pruneEndTraces(Score threshold) {
  if (threshold == Core::Type<Score>::max)
    return;
  TraceList::iterator in, out, end;
  for (in = out = endTraces_.begin(), end = endTraces_.end(); in != end; ++in)
    if ((*in)->prospect <= threshold) 
      *(out++) = *in;
  endTraces_.erase(out, endTraces_.end());
}

// prune ongoin traces (likely w.r.t. best ending traces): not used
void Seq2SeqSearchSpace::pruneTraces(Score threshold) {
  if (threshold == Core::Type<Score>::max)
    return;
  // on-going traces are carried out by word end hypotheses
  WordEndHypothesesList::iterator in, out, end;
  for (in = out = wordEndHypotheses_.begin(), end = wordEndHypotheses_.end(); in != end; ++in)
    if (in->trace->prospect <= threshold)
      *(out++) = *in;
  wordEndHypotheses_.erase(out, wordEndHypotheses_.end());
}

void Seq2SeqSearchSpace::checkStoppingCriteria() {
  if (wordEndHypotheses_.empty() && labelHypotheses_.empty())
    stopSearch_ = true; // no more hypotheses -> stop

  // early stoping for step-re-normalization: all future paths can not be better anymore
  if (!endTraces_.empty() && stepEarlyStop_ && 
      stepLengthScale_ * stepAccuLenScore_ >= bestEndTraceProspect_)
    stopSearch_ = true;

  if (stopSearch_)
    log() << "stop search at step: " << decodeStep_ 
          << " input length: " << labelScorer_->getEncoderLength()
          << " number of LabelHypotheses: " << labelHypotheses_.size()
          << " number of WordEndHypotheses: " << wordEndHypotheses_.size()
          << " number of endTraces: " << endTraces_.size();
}
// ---- asynchronous ending traces processing 


bool Seq2SeqSearchSpace::mayStopEarly() {
  if (needEndProcessing_ && !verticalTransition_) {
    bool stop = restrictWithInputLength_ && decodeStep_ > inputLength_;
    if (!stop && !endTraces_.empty() && !lengthNorm_ && !stepReNorm_ && !wordLenBalance_) {
      // early stopping for deterministic scoring: no further hyps can be better anymore
      stop = bestEndTraceProspect_ < bestLabelProspect_ + globalScoreOffset_ &&
             bestEndTraceProspect_ < bestWordEndProspect_ + globalScoreOffset_;
    }
    if (stop) {
      // all labels will be pruned away anyway
      labelHypotheses_.clear();
      wordEndHypotheses_.clear();
      return true;
    }
  } 
  return false;
}

TraceRef Seq2SeqSearchSpace::getSentenceEnd(bool createLattice) {
  if(needEndProcessing_) 
    return getSentenceEndFromEndTraces(createLattice);
  else
    return getSentenceEndFromHypotheses(createLattice);   
}

// full-sum merging of ending traces: same full lm history (same lm score)
inline void Seq2SeqSearchSpace::fullsumMergeTraces(HistoryTraceMap& historyTraceMap, 
                                                   size_t hash, TraceRef& t) {
  HistoryTraceMap::iterator htIter = historyTraceMap.find(hash);
  if (htIter == historyTraceMap.end()) {
    historyTraceMap.insert(std::make_pair(hash, t));
    return;
  }

  TraceRef& ht = htIter->second;
  Score sumAcoustic = Math::scoreSum<Score>(t->score.acoustic, ht->score.acoustic);
  Score sumProspect = Math::scoreSum<Score>(t->prospect, ht->prospect);
  // property from better path (assume similar length)
  if (t->prospect < ht->prospect) 
    ht = t;
  ht->score.acoustic = sumAcoustic;
  if (lengthNorm_)
    sumProspect = computeTraceProspect(ht, true);
  ht->prospect = sumProspect;
}

TraceRef Seq2SeqSearchSpace::getBestTrace(const HistoryTraceMap& historyTraceMap, bool createLattice) {
  verify(!historyTraceMap.empty());
  HistoryTraceMap::const_iterator htIter = historyTraceMap.begin();
  TraceRef best, last, bestParent;
  best = last = htIter->second;

  // record best and link siblings if createLattice
  for (++htIter; htIter != historyTraceMap.end(); ++htIter) {
    const TraceRef& current = htIter->second;
    if (createLattice)
      last->sibling = current;
    if (current->prospect < best->prospect || 
        (!useLmScore_ && current->prospect == best->prospect && current->nLabels < best->nLabels)) {
      best = current;
      bestParent = last;
    }
    last = current;
  }
  if (createLattice && best != historyTraceMap.begin()->second) {
    bestParent->sibling = best->sibling;
    best->sibling = historyTraceMap.begin()->second;
  }
  return best;
}

// asynchronous ending case: decision based on prospect score
TraceRef Seq2SeqSearchSpace::getSentenceEndFromEndTraces(bool createLattice) {
  TraceRef best;
  if (endTraces_.empty()) {
    // no more hypothesis (maxLenStop), but no end traces found
    // this is possible when audio is cuted and pruning is tight
    warning() << "no end traces found !";
    return best;
  }

  // sentence end already included in end traces
  if (fullSumDecoding_) {
    // merge traces supporting the same word sequence
    HistoryTraceMap historyTraceMap;
    for (TraceRef& t : endTraces_) {
      size_t hash = t->recombinationHistory.hashKey();
      if (labelFullSum_)
        hash = updateHashKey(hash, t->labelHistory.hashKey());
      fullsumMergeTraces(historyTraceMap, hash, t); 
    }
    best = getBestTrace(historyTraceMap, createLattice);
  } else {
    best = endTraces_.front();
    TraceRef bestParent, last = best;
    for (TraceRef& t : endTraces_) {
      if (createLattice)
        last->sibling = t;
      if (t->prospect < best->prospect) {
        best = t;
        bestParent = last;
      }
      last = t;
    }
    if (createLattice && best != endTraces_.front()) {
      bestParent->sibling = best->sibling;
      best->sibling = endTraces_.front();
    }
  }
  return best;
}

// only allow boundary cases: (uncoarticulated) word end or blank root
TraceRef Seq2SeqSearchSpace::getSentenceEndFromHypotheses(bool createLattice) {
  TraceRef best;

  // full-sum: merge traces supporting the same word sequence
  HistoryTraceMap historyTraceMap;

  // check if record uncoarticulated word end label in labelTree
  const std::unordered_set<NodeId>& endNodes = staticLabelTree_.forceEndNodes();
  bool forceEndLabel = !endNodes.empty();

  for (const WordEndHypothesis& weh : wordEndHypotheses_) {
    if (weh.score >= Core::Type<Score>::max)
      continue;
    if (forceEndLabel) {
      NodeId transitRoot = staticLabelTree_.getExit(weh.exitId).transitRoot;
      if (endNodes.count(transitRoot) == 0)
        continue;
    }
    // Note: no history extension to sentenceEnd (only score), but regarded as + </s>
    TraceRef t(new Seq2SeqTreeSearch::Trace(weh.trace, 0, 0,
                                            decodeStep_+1, weh.trace->score,
                                            weh.nLabels, weh.nWords+1, weh.position));
    if (useLmScore_)
      t->score.lm += languageModel_->sentenceEndScore(weh.scoreHistory);
    t->prospect = computeTraceProspect(t, true);

    if (fullSumDecoding_) {
      size_t hash = weh.recombinationHistory.hashKey();
      if (labelFullSum_)
        hash = updateHashKey(hash, weh.labelHistory.hashKey());
      fullsumMergeTraces(historyTraceMap, hash, t);
      continue; 
    }

    if (!best || t->prospect < best->prospect || 
        (!useLmScore_ && t->prospect == best->prospect && t->nLabels < best->nLabels)) {
      if (createLattice)
        t->sibling = best;
      best = t;
    } else if (createLattice) {
      t->sibling = best->sibling;
      best->sibling = t;
    }
  }

  // label ending cases: 
  // - blank root for CTC, Transducer, etc.
  // - uncoarticulated word ends for hybrid-HMM, etc.
  if (allowBlankLabel_ || forceEndLabel) {
    for (const TreeInstance* instance : activeInstances_) {
      if (instance->labels.empty())
        continue;

      size_t treeHash = instance->key.history.hashKey();
      LabelHypothesesList::iterator lhBegin = labelHypotheses_.begin() + instance->labels.begin,
                                    lhEnd = labelHypotheses_.begin() + instance->labels.end;

      for (LabelHypothesesList::iterator lh = lhBegin; lh != lhEnd; ++lh) {
        bool isValidEnd = (allowBlankLabel_ && staticLabelTree_.isRoot(lh->treeNodeId)) ||
                          (forceEndLabel && endNodes.count(lh->treeNodeId)>0);
        if (!isValidEnd || lh->score >= Core::Type<Score>::max)
          continue;

        TraceRef t(new Seq2SeqTreeSearch::Trace(instance->entryTraces[lh->traceId], 0, 0,
                                                decodeStep_+1, lh->score, 
                                                lh->nLabels, lh->nWords+1, lh->position));
        t->score.acoustic += globalScoreOffset_;
        if (useLmScore_)
          t->score.lm += languageModel_->sentenceEndScore(instance->scoreHistory);
        t->prospect = computeTraceProspect(t, true);

        if (fullSumDecoding_) {
          size_t hash = treeHash;
          if (labelFullSum_)
            hash = updateHashKey(hash, lh->labelHistory.hashKey());
          fullsumMergeTraces(historyTraceMap, hash, t);
          continue;
        }

        if (!best || t->prospect < best->prospect ||
            (!useLmScore_ && t->prospect == best->prospect && t->nLabels < best->nLabels)) {
          if (createLattice)
            t->sibling = best;
          best = t;
        } else if (createLattice) {
          t->sibling = best->sibling;
          best->sibling = t;
        }
      }
    }
  }

  if (fullSumDecoding_ && !historyTraceMap.empty())
    best = getBestTrace(historyTraceMap, createLattice);

  if (best && pruneTrace_) {
    // additional ending traces pruning (relaxed)
    Score threshold = best->prospect + labelPruning_ + wordEndPruning_;
    TraceRef t = best;
    while (t->sibling) {
      if (t->sibling->prospect > threshold)
        t->sibling = t->sibling->sibling;
      else
        t = t->sibling;
    }
  }

  return best;
}

// no boundary hypotheses found: take the best with-in word label hypothesis
// this can happen when the recording is truncated in the middle of a word and pruning is tight.
TraceRef Seq2SeqSearchSpace::getSentenceEndFallBack() {
  log() << "get fallback sentence end";
 
  // no more labels, use fall back trace
  if (needEndProcessing_) {
    verify(bestLabelEndTrace_);
    Score lmScore = 0;
    if (useLmScore_)
      lmScore = languageModel_->sentenceEndScore(bestLabelEndTrace_->scoreHistory);
    bestLabelEndTrace_->score.lm += lmScore;
    ++(bestLabelEndTrace_->nWords);
    if (lengthNorm_)
      bestLabelEndTrace_->prospect = computeTraceProspect(bestLabelEndTrace_, true);
    else // simply add sentence end score as length penalty (anyway the only one)
      bestLabelEndTrace_->prospect += lmScore; 
    return bestLabelEndTrace_;  
  }

  // best label hypothesis
  verify(!labelHypotheses_.empty());
  LabelHypothesesList::const_iterator bestHyp = bestProspectLabel();
  u32 bestIndex = bestHyp - labelHypotheses_.begin();
  TreeInstance* instance = bestProspectLabelTree(bestIndex);
  TraceRef best(new Seq2SeqTreeSearch::Trace(instance->entryTraces[bestHyp->traceId], 0, 0,
                                             decodeStep_+1, bestHyp->score,
                                             bestHyp->nLabels, bestHyp->nWords+1, bestHyp->position));
  best->score.acoustic += globalScoreOffset_;
  if (useLmScore_)
    best->score.lm += languageModel_->sentenceEndScore(instance->scoreHistory);
  best->prospect = computeTraceProspect(best, true);
  return best;
}

const Bliss::Lemma* Seq2SeqSearchSpace::getEndLemma() const {
  const Bliss::Lemma* lemma = lexicon_->specialLemma("sentence-boundary");
  if (!lemma)
    lemma = lexicon_->specialLemma("sentence-end");
  return lemma;
}

// length normalized score (label and optional word sequence length)
// TODO also support length reward here
inline Score Seq2SeqSearchSpace::computeLengthNormalizedScore(Score acoustic, Score lm, 
                                                              u32 nLabels, u32 nWords) {
  nLabels = std::max(nLabels, (u32)1);
  nWords = std::max(nWords, (u32)1);
  if (normLabelOnly_)
    return (acoustic + lm) / nLabels;
  else if (normWordOnly_)
    return acoustic + lm/nWords;
  else
    return acoustic/nLabels + lm/nWords;
}

inline Score Seq2SeqSearchSpace::computeLabelProspect(const LabelHypothesis& lh, Score lmlaScore) {
  Score prospect = lh.score + lmlaScore;
  if (lengthNorm_) {
    u32 nWords = lh.nWords;
    if (lmlaScore != 0)
      nWords += 1;
    prospect = computeLengthNormalizedScore(lh.score.acoustic, lh.score.lm+lmlaScore, 
                                            lh.nLabels, nWords);
  } 
  return prospect;
}

inline Score Seq2SeqSearchSpace::computeWordEndProspect(const EarlyWordEndHypothesis& eWeh) {
  Score prospect = eWeh.score;
  if (lengthNorm_)
    prospect = computeLengthNormalizedScore(eWeh.score.acoustic, eWeh.score.lm, 
                                            eWeh.nLabels, eWeh.nWords);
  return prospect;
}

inline Score Seq2SeqSearchSpace::computeWordEndProspect(const WordEndHypothesis& weh) {
  Score prospect = weh.score;
  if (lengthNorm_)
    prospect = computeLengthNormalizedScore(weh.score.acoustic, weh.score.lm, 
                                            weh.nLabels, weh.nWords);
  return prospect;
}

// fallback endTrace (can not expand but not fully end yet)
inline Score Seq2SeqSearchSpace::computeTraceProspect(const LabelHypothesis& lh) {
  Score prospect = lh.score + globalScoreOffset_;
  if (lengthNorm_) {
    prospect = computeLengthNormalizedScore(lh.score.acoustic+globalScoreOffset_, lh.score.lm, 
                                            lh.nLabels, lh.nWords);
  } else {
    if (wordLenBalance_)
      prospect += wordLenScore_[lh.nWords];
    if (stepReNorm_)
      prospect -= stepSumScore_;
  }
  return prospect;
}

// properly finished endTrace (prospect is used for both pruning and decision making)
inline Score Seq2SeqSearchSpace::computeTraceProspect(TraceRef& trace, bool isEnd) {
  Score prospect = trace->score;
  if (lengthNorm_) {
    prospect = computeLengthNormalizedScore(trace->score.acoustic, trace->score.lm, 
                                            trace->nLabels, trace->nWords);  
  } else if (isEnd) {
    if (wordLenBalance_) {
      if (positionDependent_)
        prospect += wordLenScore_[trace->nWords-1];
      else
        prospect += wordLenScore_[trace->nWords];
    }
    if (stepReNorm_) {
      // reformulated final probability with explicit length modeling
      prospect += stepLengthScale_ * (stepEndScore_ - stepSumScore_ + stepAccuLenScore_);
      if (!stepLengthOnly_) 
        prospect -= stepEndScore_; // also renormalize the sequence posterior
    }
  }
  return prospect;
}

