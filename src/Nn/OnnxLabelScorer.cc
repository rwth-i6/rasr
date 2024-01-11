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
          session_(select("session")),
          mapping_(select("io-map"), ioSpec_),
          validator_(select("validator")),
          features_onnx_name_(mapping_.getOnnxName("features")),
          features_size_onnx_name_(mapping_.getOnnxName("features-size")),
          output_onnx_names_({mapping_.getOnnxName("output")}),
          maxBatchSize_(paramMaxBatchSize(config)) {
    bool valid = validator_.validate(ioSpec_, mapping_, session_);
    if (not valid) {
        warning("Failed to validate input model.");
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
    //initStartHistory();

    // optional static context-dependent prior
    if (usePrior_ && priorContextSize_ > 0)
        loadPrior();

    reset();

    // debug
    Core::ParameterBool paramDebug("debug", "", false);
    debug_ = paramDebug(config);
}

OnnxModelBase::~OnnxModelBase() {
    reset();
    // delete startHistoryDescriptor_;
}

const std::vector<Onnx::IOSpecification> OnnxModelBase::ioSpec_ = {
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
                {Onnx::ValueDataType::INT32},
                {{-1}}},
        Onnx::IOSpecification{
                "output",
                Onnx::IODirection::OUTPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -1, -2}, {1, -1, -2}}}};

void OnnxModelBase::reset() {
    Precursor::reset();
    // batch_.clear();
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

// void OnnxModelBase::initStartHistory() {
//     startLabelIndex_ = getStartLabelIndex();
//     if (useStartLabel_) {
//         verify(startLabelIndex_ != Core::Type<LabelIndex>::max);
//         log() << "use start label index " << startLabelIndex_;
//     }
//     startHistoryDescriptor_ = new LabelHistoryDescriptor();
//     startHistoryDescriptor_->labelSeq.push_back(startLabelIndex_);
//     startHistoryDescriptor_->variables.resize(var_fetch_names_.size());
//     // + other possible unified operations (if always the same)
// }

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
    log() << "Note: decoder_input_vars order must be oldest first";  // add code to verify ?
    if (cacheHistory_)
        log() << "apply history caching (memory for high order !)";
    verify(startPosition_ == 0);

    blankLabelIndex_ = getBlankLabelIndex();
    hmmTopology_     = blankLabelIndex_ == Core::Type<LabelIndex>::max;
    if (!hmmTopology_) {
        log() << "RNA topology with blank label index " << blankLabelIndex_;
        if (blankUpdateHistory_)
            log() << "blank label updates history";
        else
            log() << "blank label does not updates history";
    }
    else {  // loop and blank is mutual exclusive so far
        log() << "HMM topology: label loop without blank";
        verify(!useRelativePosition_);
        if (isPositionDependent_)
            criticalError() << "segmental scoring for HMM topology not supported yet !";
        if (loopUpdateHistory_) {
            verify(!isPositionDependent_);  // can't be segmental
            log() << "label loop updates history";
        }
        else {
            log() << "label loop does not update history";
        }
    }

    if (implicitTransition_ || explicitTransition_) {
        verify(hmmTopology_ && !loopUpdateHistory_);
        verify(!(implicitTransition_ && explicitTransition_));
        if (usePrior_)  // TODO need to separate
            criticalError() << "implicit/explicit transition + prior not supported yet";
        if (implicitTransition_) {
            log() << "apply implicit transition derived from label posterior";
        }
        else if (explicitTransition_) {
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

LabelHistory OnnxFfnnTransducer::startHistory() {
    return labelHistoryManager_->history(new LabelHistoryDescriptor());
}

const std::vector<Score>& OnnxFfnnTransducer::getScores(const LabelHistory& h, bool isLoop) {
    return inputBuffer_.at(decodeStep_);
}

/*void OnnxFfnnTransducer::cleanUpBeforeExtension(u32 minPos) {
        scoreCache_.clear();
        batchHashQueue_.clear();
        scoreTransitionCache_.clear();

        if (isPositionDependent_) {
                // cache clean up w.r.t min position among all hypotheses (otherwise memory expensive ?)
                for (std::pair<const u32, ScoreCache>& kv : positionScoreCache_)
                        if (kv.first < minPos)
                                kv.second.clear();
        }
}*/
