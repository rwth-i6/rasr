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

#include "Seq2SeqAligner.hh"
#include <Speech/Alignment.hh>
#include <Speech/ModelCombination.hh>

using namespace Search;

const Core::ParameterFloat paramLabelPruning(
  "label-pruning",
  "threshold for pruning of label hypotheses",
  Core::Type<f32>::max, 0.0);

const Core::ParameterInt paramLabelPruningLimit(
  "label-pruning-limit",
  "maximum number of active labels, enforced by histogram pruning \
   this value is important, because it sets an upper bound for the runtime.",
  100000, 1);

const Core::ParameterInt paramHistogramPruningBins(
  "histogram-pruning-bins",
  "number of bins for histogram pruning (very minor effect)",
  101, 2);

const Core::ParameterInt paramLabelRecombinationLimit(
  "label-recombination-limit",
  "history length of label recombination (-1 for infinity)",
  -1);

const Core::ParameterBool paramDebug("debug", "", false);

Seq2SeqAligner::Seq2SeqAligner(const Core::Configuration& c):
    Core::Component(c),
    statisticsChannel_(c, "statistics"),
    labelPruning_(paramLabelPruning(c)),
    labelPruningLimit_(paramLabelPruningLimit(c)),
    labelRecombinationLimit_(paramLabelRecombinationLimit(c)),
    debug_(paramDebug(c)) {
  histogram_.setBins(paramHistogramPruningBins(c));
}

void Seq2SeqAligner::initialize(const Speech::ModelCombination& modelCombination) {
  acousticModel_ = modelCombination.acousticModel();
  labelScorer_ = modelCombination.labelScorer();
  blankLabelIndex_ = labelScorer_->getBlankLabelIndex();

  // still time-sync but with implicit length modeling via relative position
  useRelativePosition_ = labelScorer_->useRelativePosition();
  if (useRelativePosition_) {
    verify(blankLabelIndex_ != Core::Type<Index>::max);
    const Core::ParameterInt paramRelPosClip("relative-position-clipping", "", 32);
    relativePositionClip_ = paramRelPosClip(config);
    log() << "use relative position with clipping" << relativePositionClip_;
  }
}

void Seq2SeqAligner::restart(Fsa::ConstAutomatonRef model) {
  require(model);
  require(model->initialStateId() != Fsa::InvalidStateId);
  model_ = Fsa::staticCopy(model);

  stateDepth_.clear();
  stateDepth_.resize(model_->size(), Core::Type<u32>::max);
  if (getStateDepth(model_->initialStateId()) == 0)
    warning() << "initial state is also final state";

  step_ = 0;
  labelHypotheses_.clear();
  newLabelHypotheses_.clear();
  bestEndTrace_.reset();
  statistics_.clear();

  addStartupHypothesis();
}

u32 Seq2SeqAligner::getStateDepth(Fsa::StateId sId) {
  verify(sId < stateDepth_.size());
  if (stateDepth_[sId] == Core::Type<u32>::max) {
    const Fsa::State* state = model_->fastState(sId);
    for (Fsa::State::const_iterator arc = state->begin(); arc != state->end(); ++arc) {
      Fsa::StateId target = arc->target_;
      if (target == sId)
        continue;
      u32 depth = getStateDepth(target) + 1;
      if (depth < stateDepth_[sId])
        stateDepth_[sId] = depth;
    }
    if (state->isFinal())
      stateDepth_[sId] = 0;
  }
  return stateDepth_[sId];
}

void Seq2SeqAligner::addStartupHypothesis() {
  Nn::LabelHistory lbh = labelScorer_->startHistory();
  verify(lbh.isValid());
  labelHypotheses_.emplace_back(model_->initialStateId(), lbh, 0.0);
}

// Note: no score caching and computaton can be expensive 
// therefore, forward once only with safe pruning
void Seq2SeqAligner::align() {
  verify(model_ && !bestEndTrace_);
  while(labelScorer_->bufferFilled() && !labelScorer_->reachEnd()) {
    // alignment is always time-synchronous
    alignNext();
    // inform laber scorer to increase decoding step
    labelScorer_->increaseDecodeStep();
  }
  if (labelScorer_->reachEnd())
    labelScorer_->clearBuffer();
  if (labelScorer_->reachEOS())
    getBestEndTrace();
}

void Seq2SeqAligner::debugPrint(std::string msg, bool newStep) {
  if (newStep) {
    std::cout << "# " << msg << " "<< step_
              << " inputLength:" << labelScorer_->getEncoderLength() - 1 << std::endl;
  } else {
    u32 nBlank = 0;
    for (const AlignLabelHypothesis& hyp : labelHypotheses_)
      if (hyp.isBlank)
        ++nBlank;
    std::cout << "  # " << msg
              << " numLabelHyps:" << labelHypotheses_.size() 
              << " numBlankHyps:" << nBlank << std::endl;
  }
}

void Seq2SeqAligner::alignNext() {
  ++step_;
  if (debug_)
    debugPrint("labelStep", true);

  expand();
  if (debug_)
    debugPrint("expand and recombine");

  prune();
  if (debug_)
    debugPrint("prune");

  extendLabelHistory();
  createTrace();
  if (debug_)
    debugPrint("extend history and create Trace");
}

void Seq2SeqAligner::expand()
{
  newLabelHypotheses_.clear();
  labelHypothesesMap_.clear();

  for (const AlignLabelHypothesis& lh : labelHypotheses_) {
    const Fsa::State* state = model_->fastState(lh.stateId);
    for (Fsa::State::const_iterator arc = state->begin(); arc != state->end(); ++arc) {
      Score arcScore = Score(arc->weight_);
      if (arcScore >= Core::Type<Score>::max)
        continue; // probably dis-allowed path (pruned anyway)

      Fsa::StateId target = arc->target_;
      Index label = acousticModel_->emissionIndex(arc->input_);
      bool isBlank = label == blankLabelIndex_; 
      bool isLoop = !isBlank && target == lh.stateId;
      newLabelHypotheses_.push_back(lh);
      AlignLabelHypothesis& nlh = newLabelHypotheses_.back();
      nlh.stateId = target; nlh.labelId = arc->input_; 
      nlh.isBlank = isBlank; nlh.isLoop = isLoop;

      // loop may have different scoring mechanism
      // so far mainly posterior HMM
      // 1. (re)normalized forward|loop joint modeling (nClass+1 output)
      // 2. label-dependent loop (2*nClass output)
      const std::vector<Score>& scores = labelScorer_->getScores(lh.labelHistory, isLoop);
      if (isLoop && scores.size() == labelScorer_->numClasses() + 1)
        nlh.score += arcScore + scores.back();
      else if (isLoop && scores.size() == labelScorer_->numClasses() * 2)
        nlh.score += arcScore + scores[label + labelScorer_->numClasses()];
      else
        nlh.score += arcScore + scores[label];

      if (useRelativePosition_)
        nlh.position = isBlank ? std::min(nlh.position+1, relativePositionClip_) : 0;

      activateOrUpdate(nlh, label);
    }
  }
  labelHypotheses_.swap(newLabelHypotheses_);
  newLabelHypotheses_.clear();
}

void Seq2SeqAligner::activateOrUpdate(const AlignLabelHypothesis& lh, Index label) {
  bool simple = !labelScorer_->isHistoryDependent() && !useRelativePosition_;
  size_t key1, key2;

  if (simple) {
    key1 = 0;
    key2 = lh.stateId;
  } else {
    // same FSA state might still have different label history (hash)
    // either updated or old history according to labelScorer
    key1 = lh.stateId;
    if ((lh.isBlank && !labelScorer_->blankUpdateHistory()) ||
        (lh.isLoop && !labelScorer_->loopUpdateHistory()))
      key2 = lh.labelHistory.reducedHashKey(labelRecombinationLimit_);
    else
      key2 = lh.labelHistory.reducedExtendedHashKey(labelRecombinationLimit_, label);
    // optional further relative position dependent
    if (useRelativePosition_)
      key2 = updateHashKey(key2, lh.position);
  }

  LabelHashMap& map = labelHypothesesMap_[key1];
  LabelHashMap::iterator iter = map.find(key2);
  if (iter == map.end()) {
    map.insert(std::make_pair(key2, newLabelHypotheses_.size() - 1));
  } else {
    AlignLabelHypothesis& rh = newLabelHypotheses_[iter->second];
    if (lh.score < rh.score)
      rh = lh;
    newLabelHypotheses_.pop_back();
  }
}

void Seq2SeqAligner::prune() {
  bestScore_ = Core::Type<Score>::max;
  u32 remainLength = labelScorer_->getEncoderLength() - step_ - 1;
  // record best and filter out non-finishable pathes
  for (AlignLabelHypothesis& lh : labelHypotheses_) {
    if (stateDepth_.at(lh.stateId) > remainLength)
      lh.score = Core::Type<Score>::max;
    else if (lh.score < bestScore_)
      bestScore_ = lh.score;
  }

  // score and histogram pruning
  Score threshold = bestScore_ + labelPruning_;
  pruneLabel(threshold);
  if (labelHypotheses_.size() > labelPruningLimit_) {
    Score hpThreshold = quantileScore(bestScore_, threshold, labelPruningLimit_);
    pruneLabel(hpThreshold);
    statistics_.customStatistics("label pruning") += hpThreshold - bestScore_;
    statistics_.customStatistics("label hypotheses") += labelHypotheses_.size();
    statistics_.customStatistics("label histogram saturation") += 1.0;
  } else {
    statistics_.customStatistics("label pruning") += labelPruning_;
    statistics_.customStatistics("label hypotheses") += labelHypotheses_.size();
    statistics_.customStatistics("label histogram saturation") += 0.0;
  }
}

void Seq2SeqAligner::pruneLabel(Score threshold) {
  LabelHypothesesList::iterator in, out, end;
  for (in = out = labelHypotheses_.begin(), end = labelHypotheses_.end(); in != end; ++in)
    if (in->score <= threshold)
      *(out++) = *in;
  labelHypotheses_.erase(out, labelHypotheses_.end());
}

Score Seq2SeqAligner::quantileScore(Score minScore, Score maxScore, u32 nHyps) {
  histogram_.clear();
  histogram_.setLimits(minScore, maxScore);
  for (const AlignLabelHypothesis& lh : labelHypotheses_)
    histogram_ += lh.score;
  return histogram_.quantile(nHyps);
}

void Seq2SeqAligner::extendLabelHistory() {
  labelScorer_->cleanUpBeforeExtension(Core::Type<u32>::max);
  if (!labelScorer_->isHistoryDependent())
    return;

  for (AlignLabelHypothesis& lh : labelHypotheses_) {
    Index label = acousticModel_->emissionIndex(lh.labelId);
    labelScorer_->extendLabelHistory(lh.labelHistory, label, lh.position, lh.isLoop);
  }
}

void Seq2SeqAligner::createTrace() {
  // likely no score offset needed for numerical stability ?
  for (AlignLabelHypothesis& lh : labelHypotheses_)
    lh.trace = Core::ref(new AlignTrace(lh.trace, lh.labelId, step_, lh.score)); 
}

void Seq2SeqAligner::getBestEndTrace() {
  if (bestEndTrace_)
    return;
  Score best = Core::Type<Score>::max;
  u32 valid = 0;
  for (const AlignLabelHypothesis& lh : labelHypotheses_) {
    const Fsa::State* state = model_->fastState(lh.stateId);
    if (!state->isFinal())
      continue;
    ++valid;
    Score finalScore = lh.trace->score + Score(state->weight_);
    if (finalScore < best) {
      bestEndTrace_ = lh.trace;
      best = finalScore;
    }
  }
  if (statisticsChannel_.isOpen())
    statistics_.write(statisticsChannel_);
  if (debug_)
    debugPrint("get endTrace (valid:" + std::to_string(valid) + ")");
}

// for Viterbi path: no need to convert to alignmentFSA and use Dfs to pass alignment
void Seq2SeqAligner::setAlignment(Speech::Alignment& alignment, bool outputLabelId) {
  if (!bestEndTrace_)
    return;

  alignment.clear();
  for (Core::Ref<AlignTrace> t = bestEndTrace_; t; t = t->predecessor) {
    // Note: alignment weight usually for BW alignment and affect further writing behavior
    verify(t->step - 1 >= 0); // time is 0-based
    alignment.emplace_back(t->step - 1, t->labelId, 1.0);
    if (outputLabelId)
      alignment.back().emission = acousticModel_->emissionIndex(t->labelId);
  }
  std::reverse(alignment.begin(), alignment.end());
  verify(alignment.size() == step_);
}


