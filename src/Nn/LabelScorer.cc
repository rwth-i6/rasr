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

// Define all the parameters that this class needs
// Each parameter is initialized with its name, description, and default value

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

/**
 * @brief Constructs a new LabelScorer object.
 *
 * This constructor initializes the LabelScorer object using the specified configuration.
 *
 * @param config The configuration object which is used to initialize the following members:
 * - "labelFile": Sets the file used for labels (dependency_).
 * - "reductionFactors": Sets the reduction factors (redFactors_).
 * - "scale": Sets the scale factor (scale_).
 * - "numOfClasses": Sets the number of classes (numClasses_).
 * - "usePrior": Sets whether to use prior or not (usePrior_).
 * - "priorContextSize": Sets the size of the prior context (priorContextSize_).
 * - "loopUpdateHistory": Sets whether to update the loop history (loopUpdateHistory_).
 * - "blankUpdateHistory": Sets whether to update the blank history (blankUpdateHistory_).
 * - "positionDependent": Sets whether the scorer is position dependent or not (isPositionDependent_).
 * - "useStartLabel": Sets whether to use a start label (useStartLabel_).
 * - "segmentLengthScale": Sets the segment length scale (segLenScale_).
 * - "minSegmentLength": Sets the minimum segment length (minSegLen_).
 * - "maxSegmentLength": Sets the maximum segment length (maxSegLen_).
 * - "bufferSize": Sets the buffer size (bufferSize_).
 *
 * The constructor also sets the following members directly:
 * - `needEndProcessing_` is set to `false`.
 * - `startLabelIndex_` is set to the maximum value of `LabelIndex`.
 * - `startPosition_` is set to `0`.
 */
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

/**
 * Initialize the LabelScorer.
 *
 * This method:
 * - Instantiates a new LabelHistoryManager
 * - Checks the number of classes, and if zero, fetches from label-file
 * - Logs the number of classes
 * - If priors are to be used and the context size is zero, it uses context-independent label priors
 *
 * @throws If no prior file is provided, or if the size of the priors is not greater than or equal to the number of classes.
 */
void LabelScorer::init() {
  labelHistoryManager_ = new LabelHistoryManager();

  // If the number of classes is not defined (i.e., zero), attempt to get it from the label file
  if (numClasses_ == 0) {
    log() << "no number-of-classes given, try to get it from label-file";
    getLabelIndexMap();
  }
  log() << "number of classes: " << numClasses_;

  // If priors are to be used and the context size is zero, use context-independent label priors
  if (usePrior_ && priorContextSize_ == 0) {
    // Note: prior scale independent of posterior scale
    log() << "use context-independent label pirors";
    Prior<f32> prior(config);
    // If a filename is provided for the priors, read the priors from it
    if (!prior.fileName().empty())
      prior.read();
    else
      criticalError() << "no prior file provided";
    u32 size = prior.size();
    verify(size >= numClasses_);
    logPriors_.reserve(size);
    // For each index, push back the scaled value of the prior to the logPriors vector
    for (u32 idx = 0; idx < size; ++idx)
      logPriors_.push_back(prior.scale() * prior.at(idx));
    log() << "logPrior scale: " << prior.scale();
  } 
}

/**
 * @brief Resets the LabelScorer by clearing buffers, resetting counts, and
 *        setting relevant flags to their initial states.
 */
void LabelScorer::reset() {
  inputBuffer_.clear();
  nInput_ = 0;
  eos_ = false;
  decodeStep_ = 0;
  segmentScore_.clear();

  labelHistoryManager_->reset();
}

/**
 * @brief Returns the mapping between labels and indices.
 *
 * This function will read the mapping from a file if it hasn't been read before. The label file is
 * provided through the program's configuration. The mapping is stored internally and is returned.
 *
 * @return A reference to the internal label-to-index map.
 *
 * @throws If the number of classes doesn't match with the indices in the label file, or if no label file is provided.
 */
const LabelIndexMap& LabelScorer::getLabelIndexMap() {
  // If the label to index map has already been filled, validate that numClasses_ is set and return the map
  if (!labelIndexMap_.empty()) {
    verify(numClasses_ > 0);
    return labelIndexMap_;
  }

  std::string labelFile = paramLabelFile(config);
  // Ensure that a label file has been provided
  if (labelFile.empty())
    criticalError() << "no label file provided";
  else 
    log() << "load label and index from file " << labelFile;

  u32 nClasses = 0;
  // Open the label file and begin reading lines
  std::ifstream input(labelFile, std::ios::in);
  std::string line;
  while (input.good()) {
    std::getline(input, line);
    if (line.empty())
      continue;
    // Extract the label and its index from the line
    std::stringstream ss(line);
    std::string label;
    LabelIndex idx;
    ss >> label;
    ss >> idx;
    // Store the label and index in the map, and update the number of classes
    labelIndexMap_[label] = idx;
    nClasses = std::max(nClasses, idx);
  }
  // Validate that the number of classes matches the indices in the label file
  if (numClasses_ > 0)
    verify(nClasses + 1 == numClasses_);
  else
    // If numClasses_ was not set, set it now
    numClasses_ = nClasses + 1;

  return labelIndexMap_;
}

/**
 * @brief Retrieves the label index for a given label.
 * @param label The label whose index is to be retrieved.
 * @param name The name of the parameter.
 * @return Label index. If label is not in the map, it returns the label index from config.
 */
LabelIndex LabelScorer::getSpecialLabelIndex(const std::string& label, const std::string& name) const {
  if (labelIndexMap_.count(label) > 0) {
    return labelIndexMap_.at(label);
  } else {
    Core::ParameterInt paramLabelIndex(name.c_str(), "", Core::Type<LabelIndex>::max);
    LabelIndex index = paramLabelIndex(config);
    return index;
  }
}

/**
 * @brief Retrieves the label index with no context.
 * @return Label index. If end label or blank label index is not set, it retrieves the label index from config.
 */
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

/**
 * @brief Reduces the input length using a series of reduction factors.
 * @param len The original length.
 * @return The reduced length after applying all reduction factors.
 */
u32 LabelScorer::getReducedLength(u32 len) const {
  for (u32 idx = 0; idx < redFactors_.size(); ++idx)
    len = (len + redFactors_[idx] - 1) / redFactors_[idx];
  return len;
}

/**
 * @brief Determines if the end of the scoring process is reached.
 * @return True if the end is reached, false otherwise.
 */
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

/**
 * @brief Retrieves the encoder length. If the end of the stream is not reached, it returns the max value.
 * @return The encoder length adjusted to downsampled input length and incremented by one for the ending.
 */
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

/**
 * @brief Determines if the current segment is possibly the final segment.
 * @param startPos The start position of the segment.
 * @return True if it is possibly the final segment, false otherwise.
 */
bool LabelScorer::maybeFinalSegment(u32 startPos) const {
  if (!isPositionDependent_)
    return false;
  u32 remainLen = getEncoderLength() - 1 - startPos;
  return remainLen >= minSegLen_ && remainLen <= maxSegLen_;
}

/**
 * @brief Calculates the logarithm of the sum of the exponentials of the input scores.
 * @param scores The input scores.
 * @return The result of the log-sum-exp operation.
 */
Score LabelScorer::logSumExp(const std::vector<Score>& scores) {
  Score max = *(std::max_element(scores.begin(), scores.end()));
  verify(!std::isinf(max));
  Score sum = 0.0;
  for (std::vector<Score>::const_iterator iter = scores.begin(); iter != scores.end(); ++iter)
    sum += std::exp(*iter - max);
  return std::log(sum) + max;
}

/**
 * @brief Computes the sum of scores in -log() domain. More efficient for more than 2 terms.
 * @param scores The input scores.
 * @return The result of the operation.
 */
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
//! Parameter to determine if the PrecomputedScorer is using a first order model.
const Core::ParameterBool PrecomputedScorer::paramFirstOrder("first-order", "", false);

/**
 * @brief Constructs a new PrecomputedScorer.
 *
 * @param config Configuration for the scorer.
 */
PrecomputedScorer::PrecomputedScorer(const Core::Configuration& config) :
    Core::Component(config),
    Precursor(config),
    firstOrder_(paramFirstOrder(config)) {
  log() << "use precomputed scorer (log-posterior)";
  redFactors_.clear(); // input is already reduced
  isPositionDependent_ = false;

  if (firstOrder_) {
    log() << "as 1st-order model score caching";
    // Ensure there's a start label and set its index.
    useStartLabel_ = true;
    startLabelIndex_ = getStartLabelIndex();
    verify(startLabelIndex_ != Core::Type<LabelIndex>::max);
    log() << "use start label index " << startLabelIndex_;

    // Initialize caches for score and history.
    cachedScore_.resize(numClasses_);
    cachedHistory_.resize(numClasses_, nullptr);
  }

  blankLabelIndex_ = getBlankLabelIndex();
}

/**
 * @brief Adds input to the scorer.
 *
 * @param f Feature input.
 */
void PrecomputedScorer::addInput(Core::Ref<const Speech::Feature> f) {
  Precursor::addInput(f);
  // Ensure the input buffer size is as expected based on the first-order configuration.
  if (inputBuffer_.size() == 1) {
    if (firstOrder_)
      verify(inputBuffer_.front().size() >= numClasses_ * numClasses_);
    else 
      verify(inputBuffer_.front().size() >= numClasses_);
  }

  // Scale the scores and adjust them with prior values if required.
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

/**
 * @brief Retrieves the starting history for the labels.
 *
 * @return Label history for starting labels.
 */
LabelHistory PrecomputedScorer::startHistory() {
  if (!firstOrder_)
    return labelHistoryManager_->history(0);

  LabelHistoryDescriptor* lhd = getHistory(startLabelIndex_);
  return labelHistoryManager_->history(lhd);
}

/**
 * @brief Extends label history with a given label index.
 *
 * @param h Current label history.
 * @param idx Index of the label.
 * @param position Current position.
 * @param isLoop If this extension is a loop.
 */
void PrecomputedScorer::extendLabelHistory(LabelHistory& h, LabelIndex idx, 
                                           u32 position, bool isLoop) {
  if (firstOrder_) {
    if ((idx == blankLabelIndex_ && !blankUpdateHistory_) || (isLoop && !loopUpdateHistory_))
      return;
    LabelHistoryDescriptor* lhd = getHistory(idx);
    h = labelHistoryManager_->history(lhd);
  }
}

/**
 * @brief Retrieves the label history for a given label index.
 *
 * @param idx Index of the label.
 * @return Pointer to the label history descriptor.
 */
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

/**
 * @brief Retrieves scores for a given label history.
 *
 * @param h Label history.
 * @param isLoop If the retrieval is in a loop context.
 * @return Vector of scores.
 */
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

/**
 * @brief Cleans up data structures before extending the scorer.
 *
 * @param minPos Minimum position for cleanup.
 */
void PrecomputedScorer::cleanUpBeforeExtension(u32 minPos) {
  if (firstOrder_) {
    cachedScore_.clear();
    cachedScore_.resize(numClasses_);
  }
}
