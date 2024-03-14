/* Copyright 2020 RWTH Aachen University. All rights reserved.
 *
 * Licensed under the RWTH ASR License (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "OnnxLabelScorer.hh"
#include "Prior.hh"

using namespace Nn;

const Core::ParameterBool OnnxModelBase::paramTransformOuputLog(
        "transform-output-log",
        "apply log to tensorflow output",
        false);

const Core::ParameterBool OnnxModelBase::paramTransformOuputNegate(
        "transform-output-negate",
        "negate tensorflow output (after log)",
        false);

const Core::ParameterInt OnnxModelBase::paramMaxBatchSize(
        "max-batch-size",
        "maximum number of histories forwarded in one go",
        64, 1);

OnnxModelBase::OnnxModelBase(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config),
          encoderSession_(select("encoder-session")),
          encoderMapping_(select("encoder-io-map"), encoderIoSpec_),
          decoderSession_(select("decoder-session")),
          decoderMapping_(select("decoder-io-map"), decoderIoSpec_),
          validator_(select("validator")),
          encoder_features_name_(encoderMapping_.getOnnxName("features")),
          encoder_features_size_name_(encoderMapping_.getOnnxName("features-size")),
          encoder_output_name_(encoderMapping_.getOnnxName("encoder-output")),
          encoder_output_size_name_(encoderMapping_.getOnnxName("encoder-output-size")),
          decoder_input_name_(decoderMapping_.getOnnxName("encoder-output")),
          decoder_input_size_name_(decoderMapping_.getOnnxName("encoder-output-size")),
//          decoder_hidden_input_name_(decoderMapping_.getOnnxName("hidden-input")),
          decoder_feedback_name_(decoderMapping_.getOnnxName("feedback")),
//          decoder_hidden_output_name_(decoderMapping_.getOnnxName("hidden-output")),
          decoder_output_name_(decoderMapping_.getOnnxName("output")),
          maxBatchSize_(paramMaxBatchSize(config)) {
    scores.resize(numClasses_, 0.0f);
    bool valid = validator_.validate(encoderIoSpec_, encoderMapping_, encoderSession_);
    if (not valid) {
        warning("Failed to validate encoder model.");
    }

    valid = validator_.validate(decoderIoSpec_, decoderMapping_, decoderSession_);
    if (not valid) {
        warning("Failed to validate encoder model.");
    }

    bool transform_output_log    = paramTransformOuputLog(config);
    bool transform_output_negate = paramTransformOuputNegate(config);
    if (transform_output_log && transform_output_negate) {
        decoding_output_transform_function_ = [](Score v, Score scale) { return -scale * std::log(v); };
        log() << "apply -log(.) to model output";
    }
    else if (transform_output_log) {
        decoding_output_transform_function_ = [](Score v, Score scale) { return scale * std::log(v); };
        log() << "apply log(.) to model output";
    }
    else if (transform_output_negate) {
        decoding_output_transform_function_ = [](Score v, Score scale) { return -scale * v; };
        log() << "apply -(.) to model output";
    }
    else if (scale_ != 1.0) {
        decoding_output_transform_function_ = [](Score v, Score scale) { return scale * v; };
    }

    // unique start history handle
    initStartHistory();

    // optional static context-dependent prior
    if (usePrior_ && priorContextSize_ > 0)
        loadPrior();

    reset();
}

OnnxModelBase::~OnnxModelBase() {
    reset();
    delete startHistoryDescriptor_;
}

const std::vector<Onnx::IOSpecification> OnnxModelBase::encoderIoSpec_ = {
        Onnx::IOSpecification{
                "features",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -1, -2}, {1, -1, -2}}},
        Onnx::IOSpecification{
                "features-size",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT64},
                {{-1}}},
        Onnx::IOSpecification{
                "encoder-output",
                Onnx::IODirection::OUTPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -1, -2}, {1, -1, -2}}},
        Onnx::IOSpecification{
                "encoder-output-size",
                    Onnx::IODirection::OUTPUT,
                    true,
                    {Onnx::ValueType::TENSOR},
                    {Onnx::ValueDataType::INT64},
                    {{-1}}}};

const std::vector<Onnx::IOSpecification> OnnxModelBase::decoderIoSpec_ = {
        Onnx::IOSpecification{
                "encoder-output",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -1, -2}, {1, -1, -2}}},
        Onnx::IOSpecification{
                "encoder-output-size",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT64},
                {{-1}}},
//        Onnx::IOSpecification{
//                "hidden-input",
//                Onnx::IODirection::INPUT,
//                false,
//                {Onnx::ValueType::TENSOR},
//                {Onnx::ValueDataType::FLOAT},
//                {{-1, -1, -2}, {1, -1, -2}}}, //??????
        Onnx::IOSpecification{
                "feedback",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT64},
                {{-1, -1, -2}, {1, -1, -2}}}, //??????
//        Onnx::IOSpecification{
//                "hidden-output",
//                Onnx::IODirection::OUTPUT,
//                false,
//                {Onnx::ValueType::TENSOR},
//                {Onnx::ValueDataType::FLOAT},
//                {{-1, -1, -2}, {1, -1, -2}}}, //??????
        Onnx::IOSpecification{
                "output",
                Onnx::IODirection::OUTPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -1, -2}, {1, -1, -2}}}};

void OnnxModelBase::reset() {
    Precursor::reset();
    batch_.clear();
    cacheHashQueue_.clear();
}

// also allow (truncated) context-dependent prior (prior scale independent of posterior scale)
void OnnxModelBase::loadPrior() {
    if (!usePrior_ || priorContextSize_ == 0)
        return;

    log() << "use context-dependent label pirors (context-size:" << priorContextSize_ << ")";
    Prior<f32> prior(config);
    if (prior.fileName().empty())
        error() << "no prior file provided";
    log() << "logPrior scale: " << prior.scale();
    std::string baseName = prior.fileName();

    // sentence begin context: replace invalid context instead of append new
    // always assume useStartLabel_: all-0 embedding can also be achieved with safe embedding
    verify(useStartLabel_);
    LabelIndex noCtxId = getNoContextLabelIndex();
    if (startLabelIndex_ >= numClasses_)
        verify(noCtxId < numClasses_);

    // theoretically any context size: generate all permutations of label sequence (column-wise)
    // Note: memory cost for higher order context (speed is not crucial for init)
    std::vector<std::vector<u32>> context(priorContextSize_);
    u32                           size = std::pow(numClasses_, priorContextSize_);
    for (u32 ctx = 0; ctx < priorContextSize_; ++ctx) {
        // repeat each label within a block and fill in the column with repeating block
        u32              labelRepeat = std::pow(numClasses_, priorContextSize_ - ctx - 1);
        std::vector<u32> block;
        block.reserve(labelRepeat * numClasses_);
        for (u32 cId = 0; cId < numClasses_; ++cId) {
            std::vector<u32> vec(labelRepeat, cId);
            if (cId == noCtxId)
                vec.assign(labelRepeat, startLabelIndex_);
            block.insert(block.end(), vec.begin(), vec.end());
        }
        context[ctx].reserve(size);
        while (context[ctx].size() < size)
            context[ctx].insert(context[ctx].end(), block.begin(), block.end());
        verify(context[ctx].size() == size);
    }

    // loop over all unique context: load context-dependent prior
    for (u32 idx = 0; idx < size; ++idx) {
        // Note: fixed format for simplicity (e.g. path/prior.3-2-1.xml) right-most latest
        LabelSequence labelSeq;
        std::string   name  = baseName + ".";
        bool          valid = true;
        for (u32 ctx = 0; ctx < priorContextSize_; ++ctx) {
            u32 cId = context[ctx][idx];
            if (cId == noCtxId)
                valid = false;
            labelSeq.push_back(cId);
            name += std::to_string(cId) + "-";
        }
        if (!valid)
            continue;
        name.pop_back();
        name += ".xml";
        if (!prior.read(name)) {
            // actually may be skipped on purose for impossible context
            warning() << "failed to read " << name << " : skip this prior";
            continue;
        }
        verify(prior.size() == numClasses_);
        std::vector<Score>& logPrior = contextLogPriors_[label_sequence_hash(labelSeq)];
        verify(logPrior.empty());
        logPrior.reserve(numClasses_);
        for (u32 cId = 0; cId < numClasses_; ++cId)
            logPrior.push_back(prior.scale() * prior.at(cId));
    }

    log() << "successfully loaded " << contextLogPriors_.size() << " context-dependent label pirors";
}

// compute encoding and initialize prev_state_vars in the graph
void OnnxModelBase::encode() {
    if (inputBuffer_.empty()) {
        warning() << "no features to feed to encoder ?!";
        return;
    }

    log() << "encode input features (" << inputBuffer_[0].size() << ", "
          << inputBuffer_.size() << ")";

    MappedValueList                    inputs;
    std::vector<Math::FastMatrix<f32>> batchMat;  // single sequence: D * T
    batchMat.emplace_back(inputBuffer_[0].size(), inputBuffer_.size());
    for (u32 idx = 0, size = inputBuffer_.size(); idx < size; ++idx) {
        const std::vector<f32>& f = inputBuffer_[idx];
        std::copy(f.begin(), f.end(), &(batchMat.front().at(0, idx)));
    }
    inputs.emplace_back(std::make_pair(encoder_features_name_,
                                       Onnx::Value::create(batchMat, true)));
    if (encoderMapping_.hasOnnxName("features-size")) {
        std::vector<s64> seq_length({static_cast<s32>(inputBuffer_.size())});
        inputs.emplace_back(std::make_pair(encoder_features_size_name_,
                                           Onnx::Value::create(seq_length)));
    }

    encoderSession_.run(std::move(inputs), {encoder_output_name_, encoder_output_size_name_}, encoder_outputs);

    initComputation();
}

void OnnxModelBase::initComputation() {
    LabelHistoryDescriptor* lhd = static_cast<LabelHistoryDescriptor*>(startHistory().handle());
    verify(lhd->scores.empty());
    if (useStartLabel_) {
        // not using makeBatch, still need to compute scores later with start label input
        batch_.push_back(lhd);
    }
    else {
        makeBatch(lhd);
        verify(batch_.size() == 1);
        // compute the first score based on default initialized states
        //computeBatchScores(); ??????
    }
    // obtain initialized/updated states to startHistory (type/size all hidden in Tensor)
    //fetchBatchVariables();???????
    batch_.clear();
}

void OnnxModelBase::initStartHistory() {
    startLabelIndex_ = getStartLabelIndex();
    if (useStartLabel_) {
        verify(startLabelIndex_ != Core::Type<LabelIndex>::max);
        log() << "use start label index " << startLabelIndex_;
    }
    startHistoryDescriptor_ = new LabelHistoryDescriptor();
    startHistoryDescriptor_->labelSeq.push_back(startLabelIndex_);
//     startHistoryDescriptor_->variables->resize(var_fetch_names_.size()); //???????
    //  + other possible unified operations (if always the same)
}

LabelHistory OnnxModelBase::startHistory() {
    LabelHistoryDescriptor* lhd    = new LabelHistoryDescriptor(*startHistoryDescriptor_);
    CacheUpdateResult       result = labelHistoryManager_->updateCache(lhd, startPosition_);
    if (result.second) {
        cacheHashQueue_.push_back(lhd->cacheHash);
    }
    else {
        verify_(labelHistoryManager_->isEqualSequence(lhd, result.first->second));
        delete lhd;
        lhd = static_cast<LabelHistoryDescriptor*>(result.first->second);
    }
    return labelHistoryManager_->history(lhd);
}

void OnnxModelBase::extendLabelHistory(LabelHistory& h, LabelIndex idx, u32 position, bool isLoop) {
    LabelHistoryDescriptor* lhd = static_cast<LabelHistoryDescriptor*>(h.handle());
    // check without creating new (avoid lots of copying)
    CacheUpdateResult result = labelHistoryManager_->checkCache(lhd, idx, position);
    LabelHistoryDescriptor* nlhd;
    if (result.second) {
        // existing one: ensure no hash colision w.r.t. position
        verify_(labelHistoryManager_->isEqualSequence(lhd, idx, result.first->second));
        nlhd = static_cast<LabelHistoryDescriptor*>(result.first->second);
    } else { // creating new (keep parent's states for next computation)
        nlhd = new LabelHistoryDescriptor(*lhd);
        nlhd->labelSeq.push_back(idx);
        nlhd->isBlank = false;
        nlhd->scores.clear();
        nlhd->position = position;

        result = labelHistoryManager_->updateCache(nlhd, position);
        if (result.second) {
            // caching newly extended label history for batch scoring
            cacheHashQueue_.push_back(nlhd->cacheHash);
        } else { // this should not happen ?!
            if (position != 0)
                verify(labelHistoryManager_->isEqualSequence(nlhd, result.first->second));
            delete nlhd;
            nlhd = static_cast<LabelHistoryDescriptor*>(result.first->second);
        }
    }
    h = labelHistoryManager_->history(nlhd);
}

const std::vector<Score>& OnnxModelBase::getScores(const LabelHistory& h, bool isLoop) {
    LabelHistoryDescriptor* lhd = static_cast<LabelHistoryDescriptor*>(h.handle());
    if (!lhd->scores.empty())
        return lhd->scores;

    makeBatch(lhd);
    verify(batch_.size() > 0);
    decodeBatch();

    // results: maybe have more scores than numClasses for some special cases
    verify(lhd->scores.size() >= numClasses_);
    return lhd->scores;
}

// oldest first, still active, uniq, not-scored
void OnnxModelBase::makeBatch(LabelHistoryDescriptor* targetLhd) {
    batch_.push_back(targetLhd);
    const HistoryCache&        cache = labelHistoryManager_->historyCache();
    std::unordered_set<size_t> batchHash;
    while (batch_.size() < maxBatchSize_ && !cacheHashQueue_.empty()) {
        size_t hash = cacheHashQueue_.front();
        cacheHashQueue_.pop_front();
        if (cache.count(hash) == 0 || batchHash.count(hash) > 0)
            continue;
        LabelHistoryDescriptor* lhd = static_cast<LabelHistoryDescriptor*>(cache.at(hash));
        if (lhd == targetLhd || !lhd->scores.empty())
            continue;
        batch_.push_back(lhd);
        batchHash.insert(hash);
    }
}

void OnnxModelBase::decodeBatch() {
    //feedBatchVariables();
    //computeBatchScores();
    //fetchBatchVariables();
    batch_.clear();
}

// mainly label feedback
void OnnxModelBase::feedDecodeInput(MappedValueList& inputs) {
        u32 len = 1; // Note: no multi-step feedback yet
        Math::FastMatrix<s32> mat(batch_.size(), len);
        for (u32 bIdx = 0, bSize = batch_.size(); bIdx < bSize; ++bIdx) {
            // Note: no mask handling, all has to be evaluated for len
            verify(batch_[bIdx]->labelSeq.size() >= len);
            u32 idx = batch_[bIdx]->labelSeq.size() - len;
            for (u32 tIdx = 0; tIdx < len; ++tIdx)
            mat.at(bIdx, tIdx) = batch_[bIdx]->labelSeq[idx+tIdx];
        }
        inputs.emplace_back(std::make_pair(decoder_feedback_name_, Onnx::Value::create(mat)));
}

// assign scores to batch
void OnnxModelBase::processBatchOutput(const ValueList& decoder_outputs) {
    u32 len = 1; // no multi-step computation
    bool spacial = decoder_outputs[0].numDims() == 3;
    verify_(spacial || decoder_outputs[0].numDims() == 2);

    for (u32 bIdx = 0, bSize = batch_.size(); bIdx < bSize; ++bIdx) {
        // scores always first
        LabelHistoryDescriptor* lhd = batch_[bIdx];
        if (spacial)
            decoder_outputs[0].get(bIdx, len-1, lhd->scores);
        else
            decoder_outputs[0].get(bIdx, lhd->scores);
        if (decoding_output_transform_function_)
            std::transform(lhd->scores.begin(), lhd->scores.end(), lhd->scores.begin(),
                           std::bind(decoding_output_transform_function_, std::placeholders::_1, scale_));
    }
}

void OnnxModelBase::addPriorToBatch() {
    for (u32 bIdx = 0, bSize = batch_.size(); bIdx < bSize; ++bIdx) {
        LabelHistoryDescriptor* lhd = batch_[bIdx];
        if (priorContextSize_ == 0) { // context-independent prior
            std::transform(logPriors_.begin(), logPriors_.end(), lhd->scores.begin(), lhd->scores.begin(), std::plus<Score>());
        } else { // (truncated) context-dependent prior
            size_t hash = labelHistoryManager_->reducedHashKey(lhd, priorContextSize_);
            ScoreCache::iterator iter = contextLogPriors_.find(hash);
            verify(iter != contextLogPriors_.end());
            std::transform(iter->second.begin(), iter->second.end(), lhd->scores.begin(),
                           lhd->scores.begin(), std::plus<Score>());
        }
    }
}

// --- FFNN Transducer ---
const Core::ParameterInt OnnxFfnnTransducer::paramContextSize(
        "context-size",
        "label context size (min 1: otherwise use precomputed label scorer)",
        1, 1);

const Core::ParameterBool OnnxFfnnTransducer::paramCacheHistory(
        "cache-history",
        "cache appeared ngram history to avoid redundant computation (memory for high order !)",
        true);

// HMM-topology: implicit transition
const Core::ParameterBool OnnxFfnnTransducer::paramImplicitTransition(
        "implicit-transition",
        "derived implicit transition from label posterior: p(forward) = 1 - p(loop)",
        false);

// HMM-topology: explicit transition
const Core::ParameterBool OnnxFfnnTransducer::paramExplicitTransition(
        "explicit-transition",
        "explicit transition modeling: p(loop) appended as the last score element (|V|+1)",
        false);

const Core::ParameterBool OnnxFfnnTransducer::paramRenormTransition(
        "renorm-transition",
        "renormalize model over forward+loop (only for explicit-transition)",
        true);

const Core::ParameterBool OnnxFfnnTransducer::paramUseRelativePosition(
        "use-relative-position",
        "use (1st order) relative-position dependency",
        false);

OnnxFfnnTransducer::OnnxFfnnTransducer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          contextSize_(paramContextSize(config)),
          cacheHistory_(paramCacheHistory(config)),
          implicitTransition_(paramImplicitTransition(config)),
          explicitTransition_(paramExplicitTransition(config)),
          renormTransition_(paramRenormTransition(config)),
          useRelativePosition_(paramUseRelativePosition(config)) {

    log() << "feedforward neural transducer with label context size " << contextSize_;
    log() << "Note: decoder_input_vars order must be oldest first"; // add code to verify ?
    if (cacheHistory_)
        log() << "apply history caching (memory for high order !)";
    verify(startPosition_ == 0);

    blankLabelIndex_ = getBlankLabelIndex();
    hmmTopology_ = blankLabelIndex_ == Core::Type<LabelIndex>::max;
    if (!hmmTopology_) {
        log() << "RNA topology with blank label index " << blankLabelIndex_;
        if (blankUpdateHistory_)
            log() << "blank label updates history";
        else
            log() << "blank label does not updates history";
    } else { // loop and blank is mutual exclusive so far
        log() << "HMM topology: label loop without blank";
        verify(!useRelativePosition_);
        if (isPositionDependent_)
            criticalError() << "segmental scoring for HMM topology not supported yet !";
        if (loopUpdateHistory_) {
            verify(!isPositionDependent_); // can't be segmental
            log() << "label loop updates history";
        } else {
            log() << "label loop does not update history";
        }
    }

    if (implicitTransition_ || explicitTransition_) {
        verify(hmmTopology_ && !loopUpdateHistory_);
        verify(!(implicitTransition_ && explicitTransition_));
        if (usePrior_) // TODO need to separate
            criticalError() << "implicit/explicit transition + prior not supported yet";
        if (implicitTransition_) {
            log() << "apply implicit transition derived from label posterior";
        } else if (explicitTransition_) {
            log() << "apply explicit transition from the model (last score element for loop)";
            if (renormTransition_)
                log() << "renormalize model over forward+loop";
        }
    }

    //    // size check
    //    u32 nInput = decoding_input_tensor_names_.size();
    //    if (useRelativePosition_) {
    //        verify(nInput == contextSize_+1); // also relative position
    //        verify(!blankUpdateHistory_);
    //        verify(!isPositionDependent_); // not explicit segmental
    //        log() << "use first order relative position";
    //    } else {
    //        verify(nInput == contextSize_);
    //    }
    //
    //    for (u32 vIdx = 0; vIdx < nInput; ++vIdx)
    //        verify(decoding_input_ndims_[vIdx] == 1); // all scalars
    //    // verify(var_feed_ops_.size() == nInput); // there should be no hidden states
    //    verify(decoding_ops_.size() == 1);
    //    verify(decoding_output_tensor_names_.size() == 1);
    //    verify(decoding_output_ndims_[0] == 2);
}

OnnxFfnnTransducer::~OnnxFfnnTransducer() {
    if (cacheHistory_) {
        // free cache expicitly
        const HistoryCache cache = labelHistoryManager_->historyCache();
        for (HistoryCache::const_iterator iter = cache.begin(); iter != cache.end(); ++iter)
            delete iter->second;
        labelHistoryManager_->reset();
    }
}

void OnnxFfnnTransducer::reset() {
    inputBuffer_.clear();
    nInput_     = 0;
    eos_        = false;
    decodeStep_ = 0;

    scoreCache_.clear();
    batchHashQueue_.clear();
    batchHash_.clear();
    scoreTransitionCache_.clear();
    positionScoreCache_.clear();

    if (!cacheHistory_) {
        labelSeqCache_.clear();
        labelHistoryManager_->reset();
    }
}

void OnnxFfnnTransducer::cleanUpBeforeExtension(u32 minPos) {
    scoreCache_.clear();
    batchHashQueue_.clear();
    scoreTransitionCache_.clear();

    if (isPositionDependent_) {
        // cache clean up w.r.t min position among all hypotheses (otherwise memory expensive ?)
        for (std::pair<const u32, ScoreCache>& kv : positionScoreCache_)
            if (kv.first < minPos)
                kv.second.clear();
    }
}

LabelHistory OnnxFfnnTransducer::startHistory() {
    LabelHistoryDescriptor* lhd = new LabelHistoryDescriptor();
    if (hmmTopology_ & !loopUpdateHistory_) // keep previous segment label for loop history
        lhd->labelSeq.resize(contextSize_+1, startLabelIndex_);
    else
        lhd->labelSeq.resize(contextSize_, startLabelIndex_);

    CacheUpdateResult result = labelHistoryManager_->updateCache(lhd, startPosition_);
    if (!result.second) {
        delete lhd;
        lhd = static_cast<LabelHistoryDescriptor*>(result.first->second);
    } else {
        if (cacheHistory_)
            lhd->ref_count += 1; // always kept in cache
        if (hmmTopology_ & !loopUpdateHistory_) {
            LabelSequence labelSeq(contextSize_, startLabelIndex_);
            lhd->forwardHash = label_sequence_hash(labelSeq);
            lhd->loopHash = lhd->forwardHash;
            labelSeqCache_.insert(std::make_pair(lhd->forwardHash, labelSeq));
        }
    }
    if (decodeStep_ == 0) {
        if (hmmTopology_ & !loopUpdateHistory_)
            batchHashQueue_.insert(lhd->forwardHash);
        else
            batchHashQueue_.insert(lhd->cacheHash);
    }
    return labelHistoryManager_->history(lhd);
}

// need further speed up ?
void OnnxFfnnTransducer::extendLabelHistory(LabelHistory& h, LabelIndex idx, u32 position, bool isLoop) {
    LabelHistoryDescriptor* lhd = static_cast<LabelHistoryDescriptor*>(h.handle());
    LabelHistoryDescriptor* nlhd;

    if (!useRelativePosition_) {
        if (idx == blankLabelIndex_ && !blankUpdateHistory_) {
            // RNA topology: blank does not update history and no loop
            batchHashQueue_.insert(lhd->cacheHash);
            return;
        } else if (hmmTopology_ && !loopUpdateHistory_ && isLoop) {
            // HMM topology: loop does not update history and no blank
            batchHashQueue_.insert(lhd->forwardHash);
            batchHashQueue_.insert(lhd->loopHash);
            return;
        }
        // unless relative position: history cache is only label-seq dependent
        position = 0;
        nlhd = new LabelHistoryDescriptor(lhd->labelSeq, idx);
    } else {
        // position-aware ffnn-transducer: only for RNA topology
        // cache hash: both label-seq and rel-position dependent
        if (idx == blankLabelIndex_)
            nlhd = new LabelHistoryDescriptor(*lhd);
        else
            nlhd = new LabelHistoryDescriptor(lhd->labelSeq, idx);
        nlhd->position = position;
    }

    CacheUpdateResult result = labelHistoryManager_->updateCache(nlhd, position);
    if (!result.second) {
        delete nlhd;
        nlhd = static_cast<LabelHistoryDescriptor*>(result.first->second);
    } else { // new one: compute hash and cache label sequence
        if (cacheHistory_)
            nlhd->ref_count += 1; // always kept in cache
        if (hmmTopology_ & !loopUpdateHistory_) {
            LabelSequence fSeq(nlhd->labelSeq.begin()+1, nlhd->labelSeq.end());
            LabelSequence lSeq(nlhd->labelSeq.begin(), nlhd->labelSeq.end()-1);
            nlhd->forwardHash = label_sequence_hash(fSeq);
            nlhd->loopHash = label_sequence_hash(lSeq);
            labelSeqCache_.insert( std::make_pair(nlhd->forwardHash, fSeq) );
            labelSeqCache_.insert( std::make_pair(nlhd->loopHash, lSeq) );
        }
    }

    if (hmmTopology_ & !loopUpdateHistory_) {
        batchHashQueue_.insert(nlhd->forwardHash);
        if (!isPositionDependent_)
            batchHashQueue_.insert(nlhd->loopHash);
    } else {
        batchHashQueue_.insert(nlhd->cacheHash);
    }
    h = labelHistoryManager_->history(nlhd);
}

const std::vector<Score>& OnnxFfnnTransducer::getScores(const LabelHistory& h, bool isLoop) {
    // hmmTopology_ && !loopUpdateHistory_: special handling to include transition scores
    // p(forward) = 1 at the first frame (decodeStep_ = 0)
    if (explicitTransition_ || (implicitTransition_ && !isLoop && decodeStep_ > 0))
        return getScoresWithTransition(h, isLoop);

    LabelHistoryDescriptor* lhd = static_cast<LabelHistoryDescriptor*>(h.handle());
    size_t hash;
    if (hmmTopology_ && !loopUpdateHistory_) {
        // segment label dependent scoring: differs for loop and forward
        hash = isLoop ? lhd->loopHash : lhd->forwardHash;
    } else {
        hash = lhd->cacheHash;
    }
    const std::vector<Score>& scores = scoreCache_[hash];
    if (!scores.empty())
        return scores;

    // batch computation
    makeBatch(lhd);
    verify(batchHash_.size() > 0);
    decodeBatch(scoreCache_);

    // results
    verify(!scores.empty());
    return scores;
}

void OnnxFfnnTransducer::makeBatch(LabelHistoryDescriptor* targetLhd) {
    if (hmmTopology_ && !loopUpdateHistory_) {
        if (batchHashQueue_.erase(targetLhd->forwardHash) > 0)
            batchHash_.push_back(targetLhd->forwardHash);
        if (batchHashQueue_.erase(targetLhd->loopHash) > 0)
            batchHash_.push_back(targetLhd->loopHash);
    } else if (batchHashQueue_.erase(targetLhd->cacheHash) > 0)
        batchHash_.push_back(targetLhd->cacheHash);

    const HistoryCache& cache = labelHistoryManager_->historyCache();
    std::unordered_set<size_t>::const_iterator iter = batchHashQueue_.begin();
    while (batchHash_.size() < maxBatchSize_ && iter != batchHashQueue_.end()) {
        if (!cacheHistory_ && cache.count(*iter) == 0)
            ++iter;
        else
            batchHash_.push_back(*(iter++));
    }
    batchHashQueue_.erase(batchHashQueue_.begin(), iter);
}

void OnnxFfnnTransducer::decodeBatch(ScoreCache& scoreCache) {
    // feed in label context: left to right (right-most latest)
    MappedValueList inputs;
    std::vector<Math::FastMatrix<s64>> mat;
    mat.emplace_back(contextSize_, batchHash_.size());
    if (hmmTopology_ && !loopUpdateHistory_) {
        for (u32 bIdx = 0, bSize = batchHash_.size(); bIdx < bSize; ++bIdx) {
            const LabelSequence& seq = labelSeqCache_[batchHash_[bIdx]];
            std::copy(seq.begin(), seq.end(), &(mat.front().at(0, bIdx)));
        }
    } else {
        const HistoryCache &cache = labelHistoryManager_->historyCache();
        std::vector <s32> pos(batchHash_.size()); // optional first-order relative position
        for (u32 bIdx = 0, bSize = batchHash_.size(); bIdx < bSize; ++bIdx) {
            LabelHistoryDescriptor *lhd = static_cast<LabelHistoryDescriptor *>(cache.at(batchHash_[bIdx]));
            std::copy(lhd->labelSeq.begin(), lhd->labelSeq.end(), &(mat.front().at(0, bIdx)));
            pos[bIdx] = lhd->position;
        }
//        if (useRelativePosition_) {
//            inputs.emplace_back(std::make_pair(var_feed_names_[0], Onnx::Value::create(pos)));
//            offset = 1; // first input is always relative position
//        }
        pos.clear();
    }
    inputs.emplace_back(std::make_pair(decoder_feedback_name_, Onnx::Value::create(mat, true))); // transpose??????

    inputs.emplace_back(std::make_pair(decoder_input_name_, std::move(encoder_outputs[0])));
    inputs.emplace_back(std::make_pair(decoder_input_size_name_, std::move(encoder_outputs[1])));

    computeBatchScores(scoreCache, inputs);
    batchHash_.clear();
}

void OnnxFfnnTransducer::computeBatchScores(ScoreCache& scoreCache, MappedValueList& inputs) {
    // compute batch scores (optional prior)
    ValueList outputs;
    decoderSession_.run(std::move(inputs), {decoder_output_name_}, outputs);
    verify(outputs.size() == 1);

    for (u32 bIdx = 0, bSize = batchHash_.size(); bIdx < bSize; ++bIdx) {
        // cache score to reuse
        std::vector<Score>& score = scoreCache[batchHash_[bIdx]];
        verify(score.empty());
        log() << "1";
        outputs[0].get(bIdx, score);
        log() << "2";

        // -scale * log(posterior)
        if (decoding_output_transform_function_)
            std::transform(score.begin(), score.end(), score.begin(),
                           std::bind(decoding_output_transform_function_, std::placeholders::_1, scale_));

        // optional adding static log priors
        if (usePrior_) {
            if (priorContextSize_ == 0) { // context-independent prior
                std::transform(logPriors_.begin(), logPriors_.end(), score.begin(), score.begin(),
                               std::plus<Score>());
            } else { // (truncated) context-dependent prior
                size_t hash;
                if (hmmTopology_ && !loopUpdateHistory_) {
                    const LabelSequence& seq = labelSeqCache_[batchHash_[bIdx]];
                    hash = labelHistoryManager_->reducedHashKey(seq, priorContextSize_);
                } else {
                    const LabelSequence& seq = labelHistoryManager_->historyCache().at(batchHash_[bIdx])->labelSeq;
                    hash = labelHistoryManager_->reducedHashKey(seq, priorContextSize_);
                }
                ScoreCache::iterator iter = contextLogPriors_.find(hash);
                verify(iter != contextLogPriors_.end());
                std::transform(iter->second.begin(), iter->second.end(), score.begin(), score.begin(),
                               std::plus<Score>());
            }
        }
    }
}

// Transducer w/o blank - HMM topology: p(label|...) p(transition|...)
const std::vector<Score>& OnnxFfnnTransducer::getScoresWithTransition(const LabelHistory& h, bool isLoop) {
    // need both forward and loop scores
    // cacheHash defines the label sequence, thus everything
    LabelHistoryDescriptor* lhd = static_cast<LabelHistoryDescriptor*>(h.handle());
    std::vector<Score>& scores = scoreTransitionCache_[lhd->cacheHash];
    if (!scores.empty())
        return scores;

    const std::vector<Score>& forwardScores = scoreCache_[lhd->forwardHash];
    const std::vector<Score>& loopScores = scoreCache_[lhd->loopHash];
    if (forwardScores.empty() || loopScores.empty()) {
        // batch computation
        makeBatch(lhd);
        verify(batchHash_.size() > 0);
        decodeBatch(scoreCache_);
    }

    if (implicitTransition_) {
        // e.g. p(y_t | a_{s_t - 1}, h_1^T) only
        // - forward transition scores at segment begin
        // - derived from label posterior p(forward) = 1 - p(loop_label)
        verify(forwardScores.size() == numClasses_ && loopScores.size() == numClasses_);
        scores.resize(numClasses_, 0);
        Score forward = getExclusiveScore(loopScores.at(lhd->labelSeq.back()));
        std::transform(forwardScores.begin(), forwardScores.end(), scores.begin(),
                       std::bind(std::plus<Score>(), std::placeholders::_1, forward));
    } else { // explicitTransition_
        // e.g. p(y_t | a_{s_t - 1}, h_1^T) * p(delta_t | y_{t-1}, h_1^T)
        // - transition score at each frame: |V|+1 -th output for p(loop | y_{t-1}, h_1^T)
        // - forward: y_{t-1} = a_{s_t - 1} only feed forwardHash needed
        //   => p(y_t | a_{s_t - 1}, h_1^T) * p(forward) = 1 - p(loop)
        // - loop: feed loopHash for p(y_t=y_{t-1}| ...)
        //   => p(y_t=y_{t-1} | a_{s_t - 1}, h_1^T) * p(loop)
        // put all to model graph ? then a lot of redundant computation

        // appended ILM for forward labels only
        bool forwardILM = forwardScores.size() == 2 * numClasses_ + 1;
        if (forwardILM)
            verify(loopScores.size() == 2 * numClasses_ + 1);
        else
            verify(forwardScores.size() == numClasses_ + 1 && loopScores.size() == numClasses_ + 1);

        scores.resize(numClasses_ + 1, 0);
        Score loop = forwardScores.at(numClasses_);
        Score forward = getExclusiveScore(loop);
        std::transform(forwardScores.begin(), forwardScores.begin() + numClasses_, scores.begin(),
                       std::bind(std::plus<Score>(), std::placeholders::_1, forward));

        if (decodeStep_ > 0)
            scores.back() = loopScores.at(lhd->labelSeq.back()) + loop;
        else
            scores.back() = Core::Type<Score>::max; // no loop for the 1st frame

        // optional renormalization over forward + loop
        if (renormTransition_) {
            Score sum = computeScoreSum(scores);
            std::transform(scores.begin(), scores.end(), scores.begin(),
                           std::bind(std::plus<Score>(), std::placeholders::_1, -sum));
        }
        // ILM on output sequence level: all forward positions
        if (forwardILM)
            std::transform(scores.begin(), scores.end() - 1, forwardScores.begin() + numClasses_ + 1,
                           scores.begin(), std::minus<Score>());
    }
    return scores;
}

// -scale * log(p) => -scale * log(1 - p)
Score OnnxFfnnTransducer::getExclusiveScore(Score score) {
    // note: possible nan or inf when use prior
    return -scale_ * std::log1p( -std::exp(score / (-scale_)) );
}

// label-sync segmental decoding (expensive)
// RNA topology only: equivalence of segmental and transducer modeling
const SegmentScore& OnnxFfnnTransducer::getSegmentScores(const LabelHistory& h, LabelIndex segIdx, u32 startPos) {
    verify(isPositionDependent_);
    segmentScore_.clear();

    u32 totalLen = getEncoderLength() - 1;
    verify(totalLen >= startPos);
    u32 remainLen = totalLen - startPos;
    if (remainLen < minSegLen_)
        return segmentScore_; // empty

    LabelHistoryDescriptor* lhd = static_cast<LabelHistoryDescriptor*>(h.handle());
    size_t hash = lhd->cacheHash;
    u32 maxLen = std::min(remainLen, maxSegLen_);
    u32 minLen = std::min(u32(1), minSegLen_); // 0-frame segment also possible

    Score score = 0;
    for (u32 len = minLen; len <= maxLen; ++len) {
        u32 pos = startPos + len - 1;
        const std::vector<Score>& scores = getPositionScores(hash, pos);
        // regard label peak as segment end for scoring (simplicity: same history)
        if (len >= minSegLen_)
            segmentScore_.push_back(std::make_pair(len, score + scores[segIdx]));
        score += scores[blankLabelIndex_];
    }

    return segmentScore_;
}

const std::vector<Score>& OnnxFfnnTransducer::getPositionScores(size_t hash, u32 pos) {
    ScoreCache& scoreCache = positionScoreCache_[pos];
    const std::vector<Score>& scores = scoreCache[hash];
    if (scores.empty()) {
        makePositionBatch(hash, scoreCache);
        //setDecodePosition(pos);
        decodeBatch(scoreCache);
    }
    verify(!scores.empty());
    return scores;
}

// input scoreCache is position dependent
void OnnxFfnnTransducer::makePositionBatch(size_t hash, const ScoreCache& scoreCache) {
    verify(batchHashQueue_.count(hash) > 0);
    batchHash_.push_back(hash);

    std::unordered_set<size_t>::const_iterator iter = batchHashQueue_.begin();
    while (batchHash_.size() < maxBatchSize_ && iter != batchHashQueue_.end()) {
        // target hash is already in scoreCache with empty scores
        if (scoreCache.count(*iter) == 0)
            batchHash_.push_back(*iter);
        ++iter;
    }
    // Note: there might be a little waste of batch computation if at this step for this position,
    // only a few context is remained for scoring, but a few more new context appear at the next step
    // to be scored for this position (maybe only for low order context and only at beginning ?)
    // For higher order context, leave it as on demand
    if (decodeStep_ > 0 && contextSize_ == 1 && batchHash_.size() < maxBatchSize_/2) {
        // also cacheHash ? anyway not major use case
        LabelSeqCache::const_iterator iter = labelSeqCache_.begin();
        while (batchHash_.size() < maxBatchSize_ && iter != labelSeqCache_.end()) {
            // fill other possible context
            if (batchHashQueue_.count(iter->first) == 0 && scoreCache.count(iter->first) == 0)
                batchHash_.push_back(iter->first);
            ++iter;
        }
    }
}
