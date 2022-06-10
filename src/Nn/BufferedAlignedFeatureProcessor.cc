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
 */
#include "BufferedAlignedFeatureProcessor.hh"

#include <limits>

#include <Math/CudaVector.hh>
#include <Math/Module.hh>
#include <Modules.hh>
#include <Speech/ModelCombination.hh>

#include "FeedForwardTrainer.hh"
#include "NeuralNetworkTrainer.hh"

using namespace Nn;

//=============================================================================

template<typename T>
const Core::ParameterFloat BufferedAlignedFeatureProcessor<T>::paramSilenceWeight(
    "silence-weight", "weight for silence state", 1.0);

template<typename T>
const Core::ParameterString BufferedAlignedFeatureProcessor<T>::paramClassWeightsFile(
    "class-weights-file", "file with class-weights-vector");

template<typename T>
const Core::ParameterBool BufferedAlignedFeatureProcessor<T>::paramWeightedAlignment(
    "weighted-alignment", "use weights from alignment", false);

template<typename T>
const Core::ParameterInt BufferedAlignedFeatureProcessor<T>::paramReduceAlignmentFactor(
    "reduce-alignment-factor", "downsample alignment (only for peaky alignment)", 1);

template<typename T>
BufferedAlignedFeatureProcessor<T>::BufferedAlignedFeatureProcessor(const Core::Configuration& config, bool loadFromFile)
        : Core::Component(config),
          BufferedFeatureExtractor<T>(config, loadFromFile),
          Speech::AlignedFeatureProcessor(config),
          silence_(0),
          acousticModelNeedInit_(true),
          classLabelWrapper_(0),
          alignmentBuffer_(0),
          alignmentWeightsBuffer_(0),
          weightedAlignment_(paramWeightedAlignment(config)),
          reduceAlignFactor_(paramReduceAlignmentFactor(config)),
          alignmentReduced_(false) {}

template<typename T>
BufferedAlignedFeatureProcessor<T>::~BufferedAlignedFeatureProcessor() {
    delete classLabelWrapper_;
}

template<typename T>
void BufferedAlignedFeatureProcessor<T>::initAcousticModel() {
    /* acoustic model to identify labels */
    Speech::ModelCombination modelCombination(select("model-combination"),
                                              Speech::ModelCombination::useAcousticModel,
                                              Am::AcousticModel::noEmissions | Am::AcousticModel::noStateTransition);
    modelCombination.load();
    acousticModel_ = modelCombination.acousticModel();
    /* set silence */
    Am::Allophone silenceAllophone(acousticModel_->silence(), Am::Allophone::isInitialPhone | Am::Allophone::isFinalPhone);
    silence_ = classIndex(acousticModel_->allophoneStateAlphabet()->index(&silenceAllophone, 0));
    this->log("silence index is ") << silence_;
    this->log("silence accumulation weight is ") << paramSilenceWeight(config);
    this->log("use alignment weights: ") << weightedAlignment_;

    u32 nClasses = acousticModel_->nEmissions();
    this->log("number of classes of acoustic model: ") << nClasses;

    if (classLabelWrapper_)
        delete classLabelWrapper_;
    classLabelWrapper_ = new ClassLabelWrapper(select("class-labels"), nClasses);
    require(classLabelWrapper_->nClassesToAccumulate() > 0);

    /* initialize class weights */
    setClassWeights();

    acousticModelNeedInit_ = false;
}

template<typename T>
void BufferedAlignedFeatureProcessor<T>::setClassWeights() {
    classWeights_.resize(classLabelWrapper_->nClassesToAccumulate(), 1.0);
    std::string classWeightsFilename(paramClassWeightsFile(config));
    if (classWeightsFilename != "") {
        if (paramSilenceWeight(config) != 1.0)
            this->error("Can not use both silence weight and class weights file");
        this->log("reading class weights file ") << classWeightsFilename;
        Math::Module::instance().formats().read(classWeightsFilename, classWeights_);
        if (classWeights_.size() != classLabelWrapper_->nClassesToAccumulate()) {
            this->error("dimension mismatch: class weights vs number of classes to accumulate")
                    << classWeights_.size() << " != " << classLabelWrapper_->nClassesToAccumulate();
        }
    }
    if (classLabelWrapper_->isClassToAccumulate(silence_))
        classWeights_.at(classLabelWrapper_->getOutputIndexFromClassIndex(silence_)) = paramSilenceWeight(config);
}

template<typename T>
void BufferedAlignedFeatureProcessor<T>::initBuffer(Core::Ref<const Speech::Feature> f) {
    alignmentBuffer_.resize(PrecursorBuffer::maxBufferSize_);
    if (weightedAlignment_)
        alignmentWeightsBuffer_.resize(PrecursorBuffer::maxBufferSize_);
    PrecursorBuffer::initBuffer(f);
}

template<typename T>
void BufferedAlignedFeatureProcessor<T>::resetBuffer() {
    std::fill(alignmentBuffer_.begin(), alignmentBuffer_.end(), 0);
    if (weightedAlignment_)
        std::fill(alignmentWeightsBuffer_.begin(), alignmentWeightsBuffer_.end(), 0.0);
    PrecursorBuffer::resetBuffer();
}

template<typename T>
void BufferedAlignedFeatureProcessor<T>::processAlignedFeature(Core::Ref<const Speech::Feature> f, Am::AllophoneStateIndex e) {
    processAlignedFeature(f, e, 1.0);
}

template<typename T>
void BufferedAlignedFeatureProcessor<T>::processAlignedFeature(Core::Ref<const Speech::Feature> f, Am::AllophoneStateIndex e, Mm::Weight w) {
    if (acousticModelNeedInit_)
        initAcousticModel();
    if (PrecursorBuffer::needInit_)
        initBuffer(f);
    u32 labelIndex = classIndex(e);
    if (classLabelWrapper_->isClassToAccumulate(labelIndex)) {
        // check for buffer overflow
        if (PrecursorBuffer::checkIsTooLongSegment())
            return;
        // check consistency
        verify_eq(alignmentBuffer_.size(), BufferedFeatureExtractor<T>::featureBuffer_.at(0).nColumns());
        alignmentBuffer_.at(PrecursorBuffer::nBufferedFeatures_) = classLabelWrapper_->getOutputIndexFromClassIndex(labelIndex);
        if (weightedAlignment_)
            alignmentWeightsBuffer_.at(PrecursorBuffer::nBufferedFeatures_) = w;
        // collect the feature -> buffer (use BufferedFeatureExtractor)
        BufferedFeatureExtractor<T>::processFeature(f);
    }
}

template<typename T>
void BufferedAlignedFeatureProcessor<T>::processExtraFeature(Core::Ref<const Speech::Feature> f, u32 size) { 
    // extra features longer than alignment due to down-sampling
    if ( !alignmentReduced_ )
    {   // store size for later verification
        alignmentReduced_ = true;
        reducedSize_ = size;
    }
    BufferedFeatureExtractor<T>::processFeature(f);
}

template<typename T>
void BufferedAlignedFeatureProcessor<T>::generateMiniBatch(std::vector<NnMatrix>& miniBatch,
                                                           Math::CudaVector<u32>& miniBatchAlignment,
                                                           std::vector<f64>&      miniBatchAlignmentWeights,
                                                           u32                    batchSize) {
    // optional downsample alignment (batchSize is feature length)
    u32 targetSize = batchSize;
    if ( reduceAlignFactor_ > 1 )
        targetSize = ceil( float(batchSize) / reduceAlignFactor_ );
    verify( targetSize <= batchSize );
    std::vector<u32> keepIdx;
    if ( targetSize == batchSize || alignmentReduced_ )
    {   // keep all: original alignment
        keepIdx.resize(targetSize, 0);
        std::iota(keepIdx.begin(), keepIdx.end(), 0);
        // already reduced input alignment should have the same size as targeted output
        if ( alignmentReduced_ )
            verify( reducedSize_ == targetSize );
    } else {
      if ( peakyAlignment_ )
          reducePeakyAlignment(targetSize, batchSize, keepIdx);
      else
          reduceAlignment(targetSize, batchSize, keepIdx);
    }

    // resize mini batch alignment
    miniBatchAlignment.resize(batchSize, 0, true);
    // fill mini batch alignment
    miniBatchAlignment.finishComputation(false);

    if (weightedAlignment_)
        miniBatchAlignmentWeights.resize(batchSize, 0);

    u32 idx = 0;
    for (u32 i = 0; i < batchSize; i++) {
        if ( idx >= targetSize || i != keepIdx[idx] )
            continue;
        u32 alignmentIndex = PrecursorBuffer::nProcessedFeatures_ + i;
        if (PrecursorBuffer::shuffle_) {
            alignmentIndex = PrecursorBuffer::shuffledIndices_.at(alignmentIndex);
        }
        miniBatchAlignment.at(idx) = alignmentBuffer_.at(alignmentIndex);
        if (weightedAlignment_)
            miniBatchAlignmentWeights.at(idx) = alignmentWeightsBuffer_.at(alignmentIndex);
        ++idx;
    }
    verify( idx == targetSize );
    // features are not changed (downsample in the network if applied)
    PrecursorBuffer::generateMiniBatch(miniBatch, batchSize);
}

// subsample alignment containing label loops (no blank)
template<typename T>
void BufferedAlignedFeatureProcessor<T>::reduceAlignment(u32 targetSize, u32 batchSize, std::vector<u32>& keepIdx) 
{
    verify( targetSize < batchSize );
    u32 silId = classLabelWrapper_->getOutputIndexFromClassIndex(silence_);
    u32 start = 0, end = std::min(reduceAlignFactor_, batchSize);
    u32 lastBlockLabel = Core::Type<u32>::max;
    u32 conflict = 0; bool hasConflict = false;
    std::vector<u32> loops; u32 nPreLoops = 0;
    while ( end <= batchSize && start < end )
    {   // block-wise processing
        std::deque<u32> labelIdx;
        u32 lastLabel = Core::Type<u32>::max;
        for (u32 i = start; i < end; ++i)
        {   // might have batch offset
            u32 label = alignmentBuffer_.at(PrecursorBuffer::nProcessedFeatures_ + i);
            if ( label != lastLabel )
            {   // supress silence to reduce block conflict ?
                //if ( !labelIdx.empty() )
                //{ if ( label == silId )
                //    continue;
                //  if ( lastLabel == silId )
                //    labelIdx.pop_back();
                //}
                labelIdx.push_back(i);
                lastLabel = label;
            }
        }
        if ( labelIdx.size() == 1 )
        {   // single label block
            if ( lastBlockLabel == lastLabel )
            {   // merge loop to solve conflicts
                if ( conflict > 0 ) {
                    labelIdx.pop_front();
                    --conflict;
                } else {
                    loops.push_back(keepIdx.size());
                    ++nPreLoops; // continous loop
                }
            } else
                nPreLoops = 0;
        } else {
            // multi labels in one block (solve conflict by removing closest loop)
            u32 firstLabel = alignmentBuffer_.at(PrecursorBuffer::nProcessedFeatures_ + labelIdx.front());
            if ( firstLabel == lastBlockLabel )
                labelIdx.pop_front();
            if ( labelIdx.size() > 1 )
            {
                hasConflict = true;
                conflict += labelIdx.size() - 1;
                while ( nPreLoops > 0 && conflict > 0 )
                {
                    keepIdx.pop_back();
                    loops.pop_back();
                    --nPreLoops;
                    --conflict;
                }
            }
            nPreLoops = 0;
        }
        keepIdx.insert(keepIdx.end(), labelIdx.begin(), labelIdx.end());
        lastBlockLabel = lastLabel;
        start = end;
        end += reduceAlignFactor_;
        end = std::min(end, batchSize);
    }

    if ( loops.size() < conflict )
        criticalError() << "can not resolve label conflict (too much reduction !)";

    if ( hasConflict )
    {
        warning() << "multiple labels in one reduced block (bad alignment with shift behaviour)";
        // still not solved conflict, just remove remaining loops
        verify( keepIdx.size() - targetSize == conflict );
        while ( keepIdx.size() > targetSize && !loops.empty() )
        {   // backwards so that previous index still valid
            u32 idx = loops.back(); loops.pop_back();
            keepIdx.erase(keepIdx.begin() + idx);
        }
    }
    verify( keepIdx.size() == targetSize );
}

// subsample alignment containing label peaks and blank elsewhere (on-the-fly: fast enough)
template<typename T>
void BufferedAlignedFeatureProcessor<T>::reducePeakyAlignment(u32 targetSize, u32 batchSize, std::vector<u32>& keepIdx)
{
    verify( targetSize < batchSize );
    u32 silId = classLabelWrapper_->getOutputIndexFromClassIndex(silence_);
    u32 nLabels = 0, nPreBlank = 0;
    u32 start = 0, end = std::min(reduceAlignFactor_, batchSize);
    std::vector<u32> blanks;
    // multi labels in one block (solve conflict by removing closest blank block)
    u32 conflict = 0; bool hasConflict = false;
    while ( end <= batchSize && start < end )
    {   // block-wise processing
        std::vector<u32> labelIdx;
        for (u32 i = start; i < end; ++i)
        {   // might have batch offset
            u32 alignmentIndex = PrecursorBuffer::nProcessedFeatures_ + i;
            if ( alignmentBuffer_.at(alignmentIndex) != silId )
            {   // all labels have to be kept
                ++nLabels;
                labelIdx.push_back(i);
            }
        }
        if ( labelIdx.empty() ) {
            if ( conflict > 0 )
                --conflict;
            else {
                blanks.push_back(keepIdx.size());
                keepIdx.push_back(start);
                ++nPreBlank; // continuous blank blocks
            }
        } else {
            if ( labelIdx.size() > 1 )
            {
                conflict += labelIdx.size() - 1;
                hasConflict = true;
                while ( nPreBlank > 0 && conflict > 0 )
                {
                    keepIdx.pop_back();
                    blanks.pop_back();
                    --nPreBlank;
                    --conflict;
                }
            }
            keepIdx.insert(keepIdx.end(), labelIdx.begin(), labelIdx.end());
            nPreBlank = 0;
        }
        start = end;
        end += reduceAlignFactor_;
        end = std::min(end, batchSize);
    }

    if ( nLabels > targetSize )
        criticalError() << "number of labels " << nLabels
                        << " is larger than target reduced size " << targetSize
                        << " (too much reduction !)";

    if ( hasConflict )
    {
        warning() << "multiple labels in one reduced block (bad alignment with shift behaviour)";
        // still not solved conflict, just remove remaining blanks
        verify( keepIdx.size() - targetSize == conflict );
        while ( keepIdx.size() > targetSize && !blanks.empty() )
        {   // backwards so that previous index still valid
            u32 idx = blanks.back(); blanks.pop_back();
            keepIdx.erase(keepIdx.begin() + idx);
        }
    }
    verify( keepIdx.size() == targetSize );
}

template<typename T>
Fsa::LabelId BufferedAlignedFeatureProcessor<T>::getSilenceAllophoneStateIndex()
{
    if (acousticModelNeedInit_)
        initAcousticModel();
    return acousticModel_->silenceAllophoneStateIndex();
}

template<typename T>
void BufferedAlignedFeatureProcessor<T>::initTrainer(const std::vector<NnMatrix>& miniBatch) {
    std::vector<u32> streamSizes;
    for (u32 stream = 0; stream < miniBatch.size(); stream++) {
        streamSizes.push_back(miniBatch.at(stream).nRows());
    }
    require(PrecursorBuffer::trainer_);
    PrecursorBuffer::trainer_->initializeTrainer(PrecursorBuffer::batchSize_, streamSizes);
    PrecursorBuffer::trainer_->setClassWeights(&classWeights_);
    if (PrecursorBuffer::trainer_->hasClassLabelPosteriors()) {
        require(classLabelWrapper_);
        if (PrecursorBuffer::trainer_->getClassLabelPosteriorDimension() != classLabelWrapper_->nClassesToAccumulate()) {
            this->warning("mismatch in number of trainer class labels (e.g. NN output layer dim) and number of classes to accumulate: ")
                    << PrecursorBuffer::trainer_->getClassLabelPosteriorDimension()
                    << " vs. "
                    << classLabelWrapper_->nClassesToAccumulate();
        }
    }
}

template<typename T>
void BufferedAlignedFeatureProcessor<T>::processBuffer() {
    PrecursorBuffer::prepareProcessBuffer();
    timeval               startBatch, endBatch, end;
    bool                  measureTime = PrecursorBuffer::trainer_->measuresTime();
    std::vector<NnMatrix> miniBatch;
    Math::CudaVector<u32> miniBatchAlignment;
    std::vector<f64>      miniBatchAlignmentWeights;
    NnVector              weights;
    while (PrecursorBuffer::nProcessedFeatures_ + PrecursorBuffer::batchSize_ <= PrecursorBuffer::nBufferedFeatures_) {
        log("Process mini-batch ") << PrecursorBuffer::nProcessedMiniBatches_ + 1 << " with " << PrecursorBuffer::batchSize_
                                   << " features";
        f64 timeMinibatch = 0, timeGenerateMiniBatch = 0;
        TIMER_START(startBatch);
        generateMiniBatch(miniBatch, miniBatchAlignment, miniBatchAlignmentWeights, PrecursorBuffer::batchSize_);
        // determine weights for the mini batch features
        weights.resize(miniBatchAlignment.size(), 0, true);
        weights.finishComputation(false);
        // weight vectors according to class membership
        for (u32 index = 0; index < weights.size(); index++) {
            weights.at(index) = classWeights_.at(miniBatchAlignment.at(index));
        }
        // additionally weight vectors according to alignment weights
        if (weightedAlignment_) {
            for (u32 index = 0; index < weights.size(); index++) {
                weights.at(index) *= miniBatchAlignmentWeights.at(index);
            }
        }
        TIMER_GPU_STOP(startBatch, end, measureTime, timeGenerateMiniBatch)
        // initialize trainer (trainer checks if initialization is needed)
        if (!PrecursorBuffer::trainer_->isInitialized())
            initTrainer(miniBatch);

        // process mini batch
        PrecursorBuffer::trainer_->processBatch_feedInput(miniBatch, &weights, PrecursorBuffer::getCurSegment());
        PrecursorBuffer::trainer_->processBatch_finishWithAlignment(miniBatchAlignment);
        PrecursorBuffer::nProcessedMiniBatches_++;
        PrecursorBuffer::nProcessedFeatures_ += PrecursorBuffer::batchSize_;
        TIMER_GPU_STOP(startBatch, endBatch, measureTime, timeMinibatch)
        if (measureTime) {
            log("time for generating mini-batch: ") << timeGenerateMiniBatch;
            log("overall processing time for mini-batch: ") << timeMinibatch;

            PrecursorBuffer::trainer_->logBatchTimes();
        }
    }
    // process the remaining feature with a smaller mini batch
    // only done for algorithms where the mini batch size is not critical
    u32 nRemainingFeatures = this->nBufferedFeatures_ - this->nProcessedFeatures_;
    if (this->processRemainingFeatures_ && nRemainingFeatures > 0) {
        log("Process mini-batch ") << this->nProcessedMiniBatches_ + 1 << " with " << nRemainingFeatures << " features.";
        generateMiniBatch(miniBatch, miniBatchAlignment, miniBatchAlignmentWeights, nRemainingFeatures);
        // determine weights for the mini batch features
        weights.resize(miniBatchAlignment.size(), 0, true);
        weights.finishComputation(false);
        // weight vectors according to class membership
        for (u32 index = 0; index < weights.size(); index++) {
            weights.at(index) = classWeights_.at(miniBatchAlignment.at(index));
        }
        // additionally weight vectors according to alignment weights
        if (weightedAlignment_) {
            for (u32 index = 0; index < weights.size(); index++) {
                weights.at(index) *= miniBatchAlignmentWeights.at(index);
            }
        }

        this->trainer_->setBatchSize(nRemainingFeatures);
        // initialize trainer
        if (!this->trainer_->isInitialized())
            initTrainer(miniBatch);
        // process mini batch
        PrecursorBuffer::trainer_->processBatch_feedInput(miniBatch, &weights, PrecursorBuffer::getCurSegment());
        PrecursorBuffer::trainer_->processBatch_finishWithAlignment(miniBatchAlignment);
        this->nProcessedMiniBatches_++;
        this->nProcessedFeatures_ += nRemainingFeatures;
        // reset to old batch size
        this->trainer_->setBatchSize(this->batchSize_);
    }
    PrecursorBuffer::finalizeProcessBuffer();
}

template<typename T>
Mm::EmissionIndex BufferedAlignedFeatureProcessor<T>::classIndex(Am::AllophoneStateIndex e) const {
    if (acousticModel_)
        return acousticModel_->emissionIndex(e);
    else {
        warning("no acoustic model available, using allophone state index as class index!");
        return e;
    }
}

template<typename T>
void BufferedAlignedFeatureProcessor<T>::enterSegment(Bliss::Segment* segment) {
    // Note: We are calling this as Speech::AlignedFeatureProcessor,
    // *not* as BufferedFeatureExtractor. We must *not* call
    // PrecursorBuffer::enterSegment() because that would also try
    // to do feature extraction (via Speech::FeatureExtractor)
    // and that would fail because it has already been done.
    // Speech::AlignedFeatureProcessor::enterSegment() gets called
    // by an underlying feature extractor.
    PrecursorAligned::enterSegment(segment);
    PrecursorBuffer::setEnteredSegment(segment);
    // treat each segment individually: allow mixed input (some sub-sampled already, some not) 
    alignmentReduced_ = false;
    reducedSize_ = 0;
}

template<typename T>
void BufferedAlignedFeatureProcessor<T>::leaveSegment(Bliss::Segment* segment) {
    // We must *not* call PrecursorBuffer::leaveSegment().
    // See comment in enterSegment().
    PrecursorBuffer::setEnteredSegment(NULL);
    PrecursorAligned::leaveSegment(segment);
}

template<typename T>
void BufferedAlignedFeatureProcessor<T>::leaveCorpus(Bliss::Corpus* corpus) {
    if (corpus->level() == 0) {
        PrecursorBuffer::processCorpus();
        Core::Component::log("Total number of processed mini-batches: ") << PrecursorBuffer::totalNumberOfProcessedMiniBatches_;
        if (!PrecursorBuffer::trainer_ || !PrecursorBuffer::trainer_->isInitialized()) {
            warning("BufferedAlignedFeatureProcessor.leaveCorpus: Trainer was not initalized. "
                    "The trainer is lazily initalized usually. "
                    "If this happens, maybe the corpus is empty or we skipped everything. "
                    "We are now initializing the trainer and directly finalizing it.");
            if (!PrecursorBuffer::trainer_)
                PrecursorBuffer::trainer_ = createTrainer(config);
            if (acousticModelNeedInit_)
                initAcousticModel();               // needed for initTrainer(), classLabelWrapper_
            initTrainer(std::vector<NnMatrix>());  // should be ok without stream-sizes
        }
        PrecursorBuffer::trainer_->finalize();
        PrecursorAligned::leaveCorpus(corpus);
    }
}

//=============================================================================

// create the specific type of NeuralNetworkTrainer
template<typename T>
NeuralNetworkTrainer<T>* BufferedAlignedFeatureProcessor<T>::createTrainer(const Core::Configuration& config) {
    return NeuralNetworkTrainer<T>::createSupervisedTrainer(config);
}

//=============================================================================
// explicit template instantiation
namespace Nn {
template class BufferedAlignedFeatureProcessor<f32>;
template class BufferedAlignedFeatureProcessor<f64>;
}  // namespace Nn
