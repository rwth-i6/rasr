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

#include "LabelScorer.hh"
#include "Prior.hh"

#include <functional> 

using namespace Nn;

const Core::ParameterString LabelScorer::paramLabelFile(
  "label-file",
  "label index mapping file",
  "" );

const Core::ParameterInt LabelScorer::paramNumOfClasses(
  "number-of-classes",
  "number of classes (network output)",
  0 );

const Core::ParameterInt LabelScorer::paramBufferSize(
  "buffer-size",
  "buffer-wise encoding/decoding (online fashion)",
  Core::Type<u32>::max );

const Core::ParameterFloat LabelScorer::paramScale(
  "scale",
  "scaling for the label scores",
  1.0);

const Core::ParameterBool LabelScorer::paramUsePrior(
  "use-prior",
  "whether to use prior",
  false);

const Core::ParameterInt LabelScorer::paramPriorContextSize(
  "prior-context-size",
  "label context size for prior",
  0, 0);

const Core::ParameterBool LabelScorer::paramLoopUpdateHistory(
  "loop-update-history",
  "whether label loop should update label sequence of history (dependency)",
  false);

const Core::ParameterBool LabelScorer::paramBlankUpdateHistory(
  "blank-update-history",
  "whether blank label should update label sequence of history (dependency)",
  false);

const Core::ParameterBool LabelScorer::paramPositionDependent(
  "position-dependent",
  "whether model is position dependent",
  false);

const Core::ParameterIntVector LabelScorer::paramReductionFactors(
  "reduction-factors",
  "input (time) reduction factors of each downsampling layer to compute the maximum length",
  ",", 1);

const Core::ParameterBool LabelScorer::paramUseStartLabel(
  "use-start-label",
  "force start label to present for start history",
  false);

// only for segmental decoding
const Core::ParameterFloat LabelScorer::paramSegmentLengthScale(
  "segment-length-scale",
  "scaling for the segment length score",
  1.0);

const Core::ParameterInt LabelScorer::paramMinSegmentLength(
  "min-segment-length",
  "minimum segment length in frames (encodings)",
  1);

const Core::ParameterInt LabelScorer::paramMaxSegmentLength(
  "max-segment-length",
  "maximum segment length in frames (encodings)",
  20);


LabelScorer::LabelScorer(const Core::Configuration& config) :
    Core::Component(config),
    dependency_(paramLabelFile(config)),
    redFactors_(paramReductionFactors(config)),
    scale_(paramScale(config)),
    numClasses_(paramNumOfClasses(config)),
    usePrior_(paramUsePrior(config)),
    priorContextSize_(paramPriorContextSize(config)),
    loopUpdateHistory_(paramLoopUpdateHistory(config)),
    blankUpdateHistory_(paramBlankUpdateHistory(config)),
    needEndProcessing_(false),
    isPositionDependent_(paramPositionDependent(config)),
    useStartLabel_(paramUseStartLabel(config)),
    startLabelIndex_(Core::Type<LabelIndex>::max),
    startPosition_(0), // not configurable, but rather model specific
    segLenScale_(paramSegmentLengthScale(config)),
    minSegLen_(paramMinSegmentLength(config)),
    maxSegLen_(paramMaxSegmentLength(config)),
    bufferSize_(paramBufferSize(config)) {
  init();
  reset();
}

void LabelScorer::init() {
  labelHistoryManager_ = new LabelHistoryManager();

  if (numClasses_ == 0) {
    log() << "no number-of-classes given, try to get it from label-file";
    getLabelIndexMap();
  }
  log() << "number of classes: " << numClasses_;

  if (usePrior_ && priorContextSize_ == 0) {
    // Note: prior scale independent of posterior scale
    log() << "use context-independent label pirors";
    Prior<f32> prior(config);
    if (!prior.fileName().empty())
      prior.read();
    else
      criticalError() << "no prior file provided";
    u32 size = prior.size();
    verify(size >= numClasses_);
    logPriors_.reserve(size);
    for (u32 idx = 0; idx < size; ++idx)
      logPriors_.push_back(prior.scale() * prior.at(idx));
    log() << "logPrior scale: " << prior.scale();
  } 
}

void LabelScorer::reset() {
  inputBuffer_.clear();
  nInput_ = 0;
  eos_ = false;
  decodeStep_ = 0;
  segmentScore_.clear();

  labelHistoryManager_->reset();
}

const LabelIndexMap& LabelScorer::getLabelIndexMap() { 
  if (!labelIndexMap_.empty()) {
    verify(numClasses_ > 0);
    return labelIndexMap_;
  }

  std::string labelFile = paramLabelFile(config);
  if (labelFile.empty())
    criticalError() << "no label file provided";
  else 
    log() << "load label and index from file " << labelFile;

  u32 nClasses = 0;
  std::ifstream input(labelFile, std::ios::in);
  std::string line;
  while (input.good()) {
    std::getline(input, line);
    if (line.empty())
      continue;
    std::stringstream ss(line);
    std::string label;
    LabelIndex idx;
    ss >> label;
    ss >> idx;
    labelIndexMap_[label] = idx;
    nClasses = std::max(nClasses, idx);
  }
  if (numClasses_ > 0)
    verify(nClasses + 1 == numClasses_);
  else
    numClasses_ = nClasses + 1;

  return labelIndexMap_;
}

LabelIndex LabelScorer::getSpecialLabelIndex(const std::string& label, const std::string& name) const {
  if (labelIndexMap_.count(label) > 0) {
    return labelIndexMap_.at(label);
  } else {
    Core::ParameterInt paramLabelIndex(name.c_str(), "", Core::Type<LabelIndex>::max);
    LabelIndex index = paramLabelIndex(config);
    return index;
  }
}

LabelIndex LabelScorer::getNoContextLabelIndex() const {
  LabelIndex index = getEndLabelIndex();
  if (index == Core::Type<LabelIndex>::max)
    index = getBlankLabelIndex();
  if (index == Core::Type<LabelIndex>::max) {
    // if neither eos nor blank, then probably silence (need to specify)
    Core::ParameterInt paramLabelIndex("no-context-label-index", "", Core::Type<LabelIndex>::max);
    index = paramLabelIndex(config);
  }
  return index;
}

u32 LabelScorer::getReducedLength(u32 len) const {
  for (u32 idx = 0; idx < redFactors_.size(); ++idx)
    len = (len + redFactors_[idx] - 1) / redFactors_[idx];
  return len;
}

bool LabelScorer::reachEnd() const {
  if (needEndProcessing_ || !bufferFilled()) {
    return false;
  } else {
    u32 len = inputBuffer_.size();
    // adjust to downsampled input length (including 0-padding)
    if (!redFactors_.empty())
      len = getReducedLength(len);
    return decodeStep_ >= len;
  }
}

u32 LabelScorer::getEncoderLength() const {
  // more to come
  if (!eos_)
    return Core::Type<u32>::max;
  u32 len = nInput_;
  // adjust to downsampled input length (including 0-padding)
  if (!redFactors_.empty())
    len = getReducedLength(len);
  return len + 1; // plus 1 for ending
}

bool LabelScorer::maybeFinalSegment(u32 startPos) const {
  if (!isPositionDependent_)
    return false;
  u32 remainLen = getEncoderLength() - 1 - startPos;
  return remainLen >= minSegLen_ && remainLen <= maxSegLen_;
}

// input: vector of log(p) => output: log( sum_p )
Score LabelScorer::logSumExp(const std::vector<Score>& scores) {
  Score max = *(std::max_element(scores.begin(), scores.end()));
  verify(!std::isinf(max));
  Score sum = 0.0;
  for (std::vector<Score>::const_iterator iter = scores.begin(); iter != scores.end(); ++iter)
    sum += std::exp(*iter - max);
  return std::log(sum) + max;
}

// logSumExp in -log() domain: more efficient for more than 2 terms
Score LabelScorer::computeScoreSum(const std::vector<Score>& scores) {
  Score best = *(std::min_element(scores.begin(), scores.end()));
  verify(best < Core::Type<Score>::max); // 0-prob defined in RASR
  Score expSum = 0.0;
  for (std::vector<Score>::const_iterator iter = scores.begin(); iter != scores.end(); ++iter)
    if (*iter != Core::Type<Score>::max) // filter invalid ones
      expSum += std::exp(best - *iter);
  return -std::log(expSum) + best;
}

// ---------------------------- PrecomputedScorer -----------------------------
const Core::ParameterBool PrecomputedScorer::paramFirstOrder("first-order", "", false);

PrecomputedScorer::PrecomputedScorer(const Core::Configuration& config) :
    Core::Component(config),
    Precursor(config),
    firstOrder_(paramFirstOrder(config)) {
  log() << "use precomputed scorer (log-posterior)";
  redFactors_.clear(); // input is already reduced
  isPositionDependent_ = false;

  if (firstOrder_) {
    log() << "as 1st-order model score caching";
    useStartLabel_ = true;
    startLabelIndex_ = getStartLabelIndex();
    verify(startLabelIndex_ != Core::Type<LabelIndex>::max);
    log() << "use start label index " << startLabelIndex_;

    cachedScore_.resize(numClasses_);
    cachedHistory_.resize(numClasses_, nullptr);
  }

  blankLabelIndex_ = getBlankLabelIndex();
}

void PrecomputedScorer::addInput(Core::Ref<const Speech::Feature> f) {
  Precursor::addInput(f);
  if (inputBuffer_.size() == 1) {
    if (firstOrder_)
      verify(inputBuffer_.front().size() >= numClasses_ * numClasses_);
    else 
      verify(inputBuffer_.front().size() >= numClasses_);
  }

  // log(p)
  std::vector<f32>& scores = inputBuffer_.back();
  // -alpha * log(p) + optional beta * log(prior)
  std::transform(scores.begin(), scores.end(), scores.begin(), 
                 std::bind(std::multiplies<f32>(), std::placeholders::_1, -scale_));
  if (usePrior_ && priorContextSize_ == 0) {
    verify(scores.size() == logPriors_.size());
    std::transform(scores.begin(), scores.end(), logPriors_.begin(), scores.begin(), 
                   std::plus<f32>());
  }
}

LabelHistory PrecomputedScorer::startHistory() {
  if (!firstOrder_)
    return labelHistoryManager_->history(0);

  LabelHistoryDescriptor* lhd = getHistory(startLabelIndex_);
  return labelHistoryManager_->history(lhd);
}

void PrecomputedScorer::extendLabelHistory(LabelHistory& h, LabelIndex idx, 
                                           u32 position, bool isLoop) {
  if (firstOrder_) {
    if ((idx == blankLabelIndex_ && !blankUpdateHistory_) || (isLoop && !loopUpdateHistory_))
      return;
    LabelHistoryDescriptor* lhd = getHistory(idx);
    h = labelHistoryManager_->history(lhd);
  }
}

PrecomputedScorer::LabelHistoryDescriptor* PrecomputedScorer::getHistory(LabelIndex idx) {
  LabelHistoryDescriptor* lhd = cachedHistory_.at(idx);
  if (lhd == nullptr) {
    lhd = new LabelHistoryDescriptor();
    lhd->labelSeq.push_back(idx);
    CacheUpdateResult result = labelHistoryManager_->updateCache(lhd, 0);
    verify(result.second);
    ++(lhd->ref_count); // always kept in cache
    cachedHistory_[idx] = lhd;
  }
  return lhd;
}

const std::vector<Score>& PrecomputedScorer::getScores(const LabelHistory& h, bool isLoop) {
  const std::vector<Score>& scores = inputBuffer_.at(decodeStep_);
  if (!firstOrder_)
    return scores;

  LabelIndex idx = h.getLastLabel();
  std::vector<Score>& cs = cachedScore_[idx];
  if (cs.empty()) {
    cs.resize(numClasses_);
    u32 start = idx * numClasses_;
    std::copy(scores.begin() + start, scores.begin() + start + numClasses_, cs.begin()); 
  }
  return cs;
}

void PrecomputedScorer::cleanUpBeforeExtension(u32 minPos) {
  if (firstOrder_) {
    cachedScore_.clear();
    cachedScore_.resize(numClasses_);
  }
}
