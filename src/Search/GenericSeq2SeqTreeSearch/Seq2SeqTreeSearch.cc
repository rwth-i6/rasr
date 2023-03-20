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

#include "Seq2SeqTreeSearch.hh"
#include <Lattice/Lattice.hh>
#include <Lattice/LatticeAdaptor.hh>
#ifdef MODULE_LM_FSA
#include <Lm/FsaLm.hh>
#endif
#include "Seq2SeqSearchSpace.hh"

// maybe needed for extrememly long utterances 
const Core::ParameterInt paramCleanupInterval(
  "cleanup-interval",
  "apply score offset at intger-multiple of steps (-1 deactivated)",
  -1);

const Core::ParameterBool paramCreateLattice(
  "create-lattice",
  "enable generation of word lattice",
  true);

const Core::ParameterBool paramOptimizeLattice(
  "optimize-lattice",
  "simple optimize lattice",
  false);

const Core::ParameterBool paramSimpleBeamSearch(
  "simple-beam-search",
  "apply simple beam search with one global beam for all levels of hyps, \
   otherwise apply hyp-level-individual beam search",
  false);

const Core::ParameterBool paramDebug(
  "debug",
  "print debug msg for each search step",
  false);

Seq2SeqTreeSearchManager::Seq2SeqTreeSearchManager(const Core::Configuration &c) :
    Core::Component(c),
    SearchAlgorithm(c),
    silence_(0),
    wpScale_(0),
    ss_(nullptr),
    statisticsChannel_(c, "statistics"),
    cleanupInterval_(paramCleanupInterval(c)),
    createLattice_(paramCreateLattice(c)),
    optimizeLattice_(paramOptimizeLattice(c)),
    simpleBeamSearch_(paramSimpleBeamSearch(c)),
    debug_(paramDebug(c)) {
  if (!createLattice_)
    optimizeLattice_ = false;
}

Seq2SeqTreeSearchManager::~Seq2SeqTreeSearchManager() {
  // end trace may contain separate recombineLM history sometimes, free before delete
  sentenceEnd_.reset();
  if (ss_)
    delete ss_;
}

bool Seq2SeqTreeSearchManager::setModelCombination(const Speech::ModelCombination& modelCombination) {
  lexicon_ = modelCombination.lexicon();
  silence_ = lexicon_->specialLemma("silence");
  acousticModel_ = modelCombination.acousticModel();
  lm_ = modelCombination.languageModel();
  wpScale_ = modelCombination.pronunciationScale();

  labelScorer_ = modelCombination.labelScorer();

  verify(!ss_);
  // initialize the search space
  restart();
  return true;
}

bool Seq2SeqTreeSearchManager::hasPronunciation() const {
  verify(ss_ && ss_->isInitialized());
  return ss_->hasPronunciation();
}

void Seq2SeqTreeSearchManager::setGrammar(Fsa::ConstAutomatonRef g) {
  log("Set grammar");
#ifdef MODULE_LM_FSA
  require(lm_);
  const Lm::FsaLm *constFsaLm = dynamic_cast<const Lm::FsaLm*>(lm_->unscaled().get());
  require(constFsaLm);
  Lm::FsaLm *fsaLm = const_cast<Lm::FsaLm*>(constFsaLm);
  fsaLm->setFsa(g);
#else
  criticalError("Module LM_FSA is not available");
#endif
  delete ss_; 
  ss_ = nullptr;
}

void Seq2SeqTreeSearchManager::resetStatistics() {
  ss_->resetStatistics();
}

void Seq2SeqTreeSearchManager::logStatistics() const {
  if (statisticsChannel_.isOpen())
    ss_->logStatistics(statisticsChannel_);
}

void Seq2SeqTreeSearchManager::restart() { 
  if (!ss_) {
    verify(lexicon_); // setModelCombination must have been called already
    ss_ = new Seq2SeqSearchSpace(config, acousticModel_, lexicon_, lm_, wpScale_, labelScorer_);
    ss_->initialize(simpleBeamSearch_);
  } else {
    verify(ss_->isInitialized());
    ss_->clear();
  }

  decodeStep_ = 0;
  ss_->addStartupWordEndHypothesis(decodeStep_);
  sentenceEnd_.reset();
}

void Seq2SeqTreeSearchManager::debugPrint(std::string msg, bool newStep) {
  if (newStep)
    std::cout << "# " << msg << " "<< decodeStep_ 
              << " inputLength:" << labelScorer_->getEncoderLength() - 1 << std::endl;
  else
    std::cout << "  # " << msg 
              << " numTrees:" << ss_->nActiveTrees() 
              << " numLabelHyps:" << ss_->nLabelHypotheses() 
              << " numWehs:"<< ss_->nWordEndHypotheses() 
              << " numEndTraces:" << ss_->nEndTraces() << std::endl;
}

void Seq2SeqTreeSearchManager::decodeNext() {
  sentenceEnd_.reset();
  ++decodeStep_;
  ss_->setDecodeStep(decodeStep_);
  ss_->setInputLength(labelScorer_->getEncoderLength());

  if (debug_)
    debugPrint("decodeStep", true);

  if (!ss_->mayStopEarly()) {
    ss_->startNewTrees();
    if (debug_)
      debugPrint("startNewTrees");
    ss_->expandLabels();
    if (debug_)
      debugPrint("expandLabels");

    if (simpleBeamSearch_) {
      // one global beam
      ss_->findWordEndsAndPruneGlobal();
      if (debug_)
        debugPrint("prune");
    } else {
      ss_->applyLabelPruning();
      if (debug_)
        debugPrint("pruneLabels");
      if (cleanupInterval_ > 0 && decodeStep_ % cleanupInterval_ == 0)
        ss_->rescale();
      ss_->findWordEndsAndPrune();
      if (debug_)
        debugPrint("findWordEndsAndPrune");
    }

    ss_->extendWordHistory();
    ss_->createTraces();
    ss_->recombineWordEnds(createLattice_);
    if (debug_)
      debugPrint("recombineWordEnds");
    if (optimizeLattice_)
      ss_->optimizeLattice();
    // clean up search space if needed (mainly non-expandable labels and trees)
    ss_->cleanUp();
    if (debug_)
      debugPrint("cleanUp");
  } else {
    // if need ending processing: simple stopping criteria reached (e.g. length)
    if (debug_)
      debugPrint("stopEarly");
  }

  // if need ending processing: asynchronous finished paths + additional stopping criteria
  ss_->processEnd(); 
  if (debug_)
    debugPrint("processEnd");
}

void Seq2SeqTreeSearchManager::decode() {
  while(labelScorer_->bufferFilled() && !labelScorer_->reachEnd()) {
    if (ss_->shouldStopSearch())
      break;
    decodeNext();
    labelScorer_->increaseDecodeStep();
    if (debug_ && labelScorer_->reachEnd())
      debugPrint("labelScorer reachEnd" , true);
  }
  if (labelScorer_->reachEnd())
    labelScorer_->clearBuffer();
}

// Note: partial traceback should not call this
TraceRef Seq2SeqTreeSearchManager::sentenceEnd() const {
  if (!sentenceEnd_) {
    sentenceEnd_ = ss_->getSentenceEnd(createLattice_);
    if (!sentenceEnd_)
      sentenceEnd_ = ss_->getSentenceEndFallBack();
    verify(sentenceEnd_);

    // post processing: remove sentence-end lemma for output
    if (sentenceEnd_->lemma == ss_->getEndLemma())
      sentenceEnd_->lemma = nullptr;
  }
  return sentenceEnd_; 
}

void Seq2SeqTreeSearchManager::traceback(TraceRef end, Traceback &result) const {
  result.clear();
  for (; end; end = end->predecessor)
    result.push_back(*end); // upcasting to TracebackItem
  std::reverse(result.begin(), result.end());
}

void Seq2SeqTreeSearchManager::getCurrentBestSentence(Traceback &result) const {
  TraceRef t = sentenceEnd();
  if (!t) {
    error("Cannot determine sentence hypothesis: No active ending hypothesis.");
    result.clear();
    return;
  }
  traceback(t, result);
}

Core::Ref<const LatticeAdaptor> Seq2SeqTreeSearchManager::getCurrentWordLattice() const {
  return buildLatticeForTrace(sentenceEnd());
}

Core::Ref<const LatticeAdaptor> Seq2SeqTreeSearchManager::buildLatticeForTrace(TraceRef trace) const {
  if (!trace) // just return empty
    return Core::ref(new Lattice::WordLatticeAdaptor());

  // graphemic systems w/o pronunciation
  // switch input alphabet to lemmaAlphabet (arc also use lemma id)
  bool useLemmaAlphabet = !(ss_->hasPronunciation());

  Core::Ref<Lattice::StandardWordLattice> result(new Lattice::StandardWordLattice(lexicon_, useLemmaAlphabet));
  Core::Ref<Lattice::WordBoundaries> wordBoundaries(new Lattice::WordBoundaries);
  TraceRef initialTrace;

  // avoid invalid interval for the final state
  if (ss_->needEndProcessing() && trace->step < decodeStep_)
    trace->step = decodeStep_;

  TraceStateMap traceStateMap;
  traceStateMap[trace.get()] = result->finalState();
  std::stack<TraceRef> stack;
  stack.push(trace);

  Fsa::State *previousState, *currentState;
  while (!stack.empty()) { 
    trace = stack.top(); 
    stack.pop();
    currentState = traceStateMap[trace.get()];
    // if not time-sync decoding, misuse step to be time for word boundary
    // still ok for confusion network ?
    wordBoundaries->set(currentState->id(), Lattice::WordBoundary(trace->step));

    // all siblings share the same current state
    for (TraceRef arcTrace = trace; arcTrace; arcTrace = arcTrace->sibling) {
      TraceRef preTrace = arcTrace->predecessor;
      if (preTrace->predecessor) {
        if (traceStateMap.find(preTrace.get()) == traceStateMap.end()) {
          previousState = traceStateMap[preTrace.get()] = result->newState();
          stack.push(preTrace);
        } else {
          previousState = traceStateMap[preTrace.get()];
        }
      } else {
        previousState = result->initialState();
        initialTrace = preTrace;
      }

      SearchAlgorithm::ScoreVector scores = ss_->computeArcTraceScore(arcTrace, preTrace);
      if (useLemmaAlphabet)
        result->newArc(previousState, currentState, arcTrace->lemma, scores.acoustic, scores.lm);
      else
        result->newArc(previousState, currentState, arcTrace->pronunciation, scores.acoustic, scores.lm);
    }
  }

  verify(initialTrace);
  wordBoundaries->set(result->initialState()->id(), Lattice::WordBoundary(initialTrace->step));
  result->setWordBoundaries(wordBoundaries);
  result->addAcyclicProperty();

  return Core::ref(new Lattice::WordLatticeAdaptor(result));  
}

