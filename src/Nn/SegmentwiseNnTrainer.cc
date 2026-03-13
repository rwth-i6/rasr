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
#include "SegmentwiseNnTrainer.hh"

#include <sys/time.h>

#include <Am/ClassicAcousticModel.hh>
#include <Fsa/Basic.hh>
#include <Fsa/Cache.hh>
#include <Fsa/Sssp.hh>
#include <Lattice/Best.hh>
#include <Math/Module.hh>
#include <Speech/AuxiliarySegmentwiseTrainer.hh>

#include "ClassLabelWrapper.hh"
#include "LinearAndActivationLayer.hh"
#include "MeSegmentwiseNnTrainer.hh"
#include "MmiSegmentwiseNnTrainer.hh"
#include "SharedNeuralNetwork.hh"

namespace Nn {

template<typename T>
const Core::ParameterString SegmentwiseNnTrainer<T>::paramStatisticsFilename(
        "statistics-filename", "filename to write statistics to", "");

template<typename T>
const Core::ParameterFloat SegmentwiseNnTrainer<T>::paramSilenceWeight(
        "silence-weight", "weight for silence state", -1.0);

template<typename T>
const Core::ParameterString SegmentwiseNnTrainer<T>::paramClassWeightsFile(
        "class-weights-file", "file with class-weights-vector");

template<typename T>
const Core::ParameterFloat SegmentwiseNnTrainer<T>::paramCeSmoothingWeight(
        "ce-smoothing-weight", "weight for cross-entropy criterion smoothing", 0.0, 0.0, 1.0);

template<typename T>
const Core::ParameterFloat SegmentwiseNnTrainer<T>::paramFrameRejectionThreshold(
        "frame-rejection-threshold", "weight for silence state", 0.0, 0.0, 1.0);

template<typename T>
const Core::ParameterBool SegmentwiseNnTrainer<T>::paramAccumulatePrior(
        "accumulate-prior", "accumulate state prior", false);

template<typename T>
const Core::ParameterBool SegmentwiseNnTrainer<T>::paramEnableFeatureDescriptionCheck(
        "enable-feature-description-check", "check if the feature dimensions match "
                                            "the size of the input (does not work reliably on models with multiple input streams)",
        true);

template<typename T>
SegmentwiseNnTrainer<T>::SegmentwiseNnTrainer(const Core::Configuration& c)
        : Core::Component(c),
          NeuralNetworkTrainer<f32>(c),
          Speech::AbstractAcousticSegmentwiseTrainer(c),
          statisticsFilename_(paramStatisticsFilename(c)),
          ceSmoothingWeight_(paramCeSmoothingWeight(c)),
          frameRejectionThreshold_(paramFrameRejectionThreshold(c)),
          accumulatePrior_(paramAccumulatePrior(c)),
          singlePrecision_(false),  // is set later
          statistics_(0),
          priorStatistics_(0),
          numberOfProcessedSegments_(0),
          numberOfObservations_(0),
          numberOfRejectedObservations_(0),
          ceObjectiveFunction_(0),
          localObjectiveFunction_(0),
          localCeObjectiveFunction_(0),
          localClassificationErrors_(0),
          accumulator_(0),
          segmentNeedsInit_(true),
          sequenceLength_(0),
          topLayer_(0),
          maxoutLayer_(0),
          priorScale_(0),
          featureDescription_(*this),
          featureDescriptionNeedInit_(true),
          enableFeatureDescriptionCheck_(paramEnableFeatureDescriptionCheck(c)),
#ifdef MODULE_PYTHON
          pythonControl_(c, "SegmentwiseNnTrainer", true),
#endif
          timeMemoryAllocation_(0),
          timeNumeratorExtraction_(0),
          timeAlignmentVector_(0),
          timeErrorSignal_(0),
          timeCESmoothing_(0),
          timeBackpropagation_(0),
          timeGradient_(0),
          timeBaseStatistics_(0),
          timeEstimationStep(0),
          timeSync_(0) {
    setPrecision();
    logProperties();
}

template<typename T>
SegmentwiseNnTrainer<T>::~SegmentwiseNnTrainer() {
    if (statistics_)
        delete statistics_;
    if (priorStatistics_)
        delete priorStatistics_;
    if (accumulator_)
        delete accumulator_;
}

template<typename T>
void SegmentwiseNnTrainer<T>::processWordLattice(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* segment) {
    SpeechPrecursor::processWordLattice(lattice, segment);
    timeval start_all, end_all;
    timeval start, end;
    TIMER_START(start_all)

    // initialization
    initSegment(lattice, segment);
#ifdef MODULE_PYTHON
    pythonControl_.run_custom("init_segment", "{s:s}", "segment_name", segment->fullName().c_str());
#endif

    // create numerator lattice (required for alignment vector).
    // this is a lattice containing only one word seq, the reference orthography.
    TIMER_START(start)
    Lattice::ConstWordLatticeRef numeratorLattice = Speech::AbstractSegmentwiseTrainer::extractNumerator(segment->orth(), lattice);
    TIMER_GPU_STOP(start, end, measureTime_, timeNumeratorExtraction_)

    // extract alignment vector from numerator lattice.
    // we need it for basic prior/FER statistics and for the CE-smoothing.
    // if it fails, skip segment.
    bool alignmentOK = getAlignmentVector(numeratorLattice);

    if (!alignmentOK) {
        this->warning("Computing alignment vector failed."
                      "\nThis is probably caused by inf-scores in the lattice - check whether the learning rate is set too high!\n"
                      "Skipping segment.");
        errorSignal_.back().finishComputation(false);
        network().getTopLayerOutput().initComputation(false);
        LatticeSetProcessor::processWordLattice(lattice, segment);
        return;
    }

    // compute initial error signal and sync it to GPU
    // the error signal is defined by the criterion, e.g. MMI or MPE
    // assume network is already forwarded
    TIMER_START(start)
    bool errorSignalOk = true;
    if (ceSmoothingWeight_ < 1)
        errorSignalOk = computeInitialErrorSignal(lattice, numeratorLattice, segment, localObjectiveFunction_, !statistics_->hasGradient());
    else
        errorSignal_.back().setToZero();
    TIMER_GPU_STOP(start, end, measureTime_, timeErrorSignal_)

    if (!errorSignalOk) {
        this->warning("could not compute error signal (bad lattice?), skipping segment");
        errorSignal_.back().finishComputation(false);
        network().getTopLayerOutput().initComputation(false);
        LatticeSetProcessor::processWordLattice(lattice, segment);
        return;
    }

    if (maxoutLayer_) {  // if mixture expand the errorsignal
        TIMER_START(start)
        errorSignal_.back().initComputation();  // move data to GPU
        TIMER_GPU_STOP(start, end, measureTime_, timeSync_)
        errorSignal_[maxoutLayer_->getPredecessor(0)].maxoutErrorExpand(maxoutLayer_->getMixture(), maxoutLayer_->getOffset(), maxoutLayer_->getMaxindex(), errorSignal_.back());  // expand
    }

    // sync to GPU
    TIMER_START(start)
    alignment_.initComputation();  // move to GPU
    if (NeuralNetworkTrainer<f32>::weightedAccumulation_)
        weights_.initComputation();                   // move to GPU
    network().getTopLayerOutput().initComputation();  // EmissionLatticeRescorerAutomaton::forwardNetwork() sets isComputing_ to false
    prior_.initComputation();
    TIMER_GPU_STOP(start, end, measureTime_, timeSync_)

    // apply Cross-entropy smoothing
    // side effects: log-priors are added to scores, softmax is applied
    localCeObjectiveFunction_ = smoothErrorSignalWithCE();

#ifdef MODULE_PYTHON
    pythonControl_.run_custom(
            "notify_segment_loss", "{s:s,s:f}",
            "segment_name", segment->fullName().c_str(),
            "loss", (float)localObjectiveFunction_);
#endif

    // compute gradient
    if (statistics_->hasGradient()) {
        // sync error signal to GPU
        TIMER_START(start)
        errorSignal_.back().initComputation();  // move to GPU if not there
        TIMER_GPU_STOP(start, end, measureTime_, timeSync_)

        // weight error signal
        if (NeuralNetworkTrainer<f32>::weightedAccumulation_) {
            TIMER_START(start)
            if (!maxoutLayer_) {
                errorSignal_.back().multiplyColumnsByScalars(weights_);  // weight error signal
            }
            else {
                errorSignal_[maxoutLayer_->getPredecessor(0)].multiplyColumnsByScalars(weights_);  // weight error signal
            }
            TIMER_GPU_STOP(start, end, measureTime_, timeErrorSignal_)
        }

        // backpropate error and collect gradient
        backpropagateError();
        collectGradient();
    }

    // update base statistics
    accumulateBaseStatistics();
    accumulatePrior();

    if (!estimator().fullBatchMode())
        statistics_->finalize(false);

    // apply regularizer
    // TODO this doesn't work with double precision yet
    statistics_->addToObjectiveFunction(regularizer().objectiveFunction(network(), 1));
    regularizer().addGradient(network(), *statistics_, 1.0);

    // update model (different code for single and double precision)
    if (!estimator().fullBatchMode())
        updateModel();

    // relase lock on CPU memory for error signal of top layer
    errorSignal_.back().finishComputation(false);

    // we are done ..
    TIMER_STOP(start_all, end_all, this->timeProcessSegment_)

    logSegmentStatistics();

    segmentNeedsInit_ = true;
    LatticeSetProcessor::processWordLattice(lattice, segment);
}

template<typename T>
void SegmentwiseNnTrainer<T>::setFeatureDescription(const Mm::FeatureDescription& description) {
    if (!enableFeatureDescriptionCheck_)
        return;
    if (featureDescriptionNeedInit_) {
        if (needInit_)
            initializeTrainer();
        featureDescription_ = description;
        size_t nFeatures;
        featureDescription_.mainStream().getValue(Mm::FeatureDescription::nameDimension, nFeatures);
        if (network().getLayer(0).getInputDimension(0) != nFeatures) {
            this->error("mismatch in dimension: ") << network().getLayer(0).getInputDimension(0)
                                                   << " (neural network input dimension) vs. "
                                                   << nFeatures
                                                   << " (dimension of flow features)";
        }
        featureDescriptionNeedInit_ = true;
    }
    else {
        if (featureDescription_ != description) {
            this->criticalError("change of features is not allowed");
        }
    }
}

template<typename T>
void SegmentwiseNnTrainer<T>::leaveCorpus(Bliss::Corpus* corpus) {
    if (corpus->level() == 0)
        finalize();
    SpeechPrecursor::leaveCorpus(corpus);
}

template<typename T>
void SegmentwiseNnTrainer<T>::initializeTrainer() {
    if (needInit_) {
        // do some checks

        // network must have at least one layer
        require_gt(network().nLayers(), 0);
        // training only makes sense, if there is at least one trainable layer
        require_gt(network().nTrainableLayers(), 0);

        if (ceSmoothingWeight_ > 0.0) {
            // require a linear+softmax layer (softmax is not evaluated) optionally followed by maxoutvar
            topLayer_ = dynamic_cast<LinearAndSoftmaxLayer<f32>*>(&network().getTopLayer());
            if (!topLayer_) {  // if not LinearAndSoftmaxLayer maybe mixture
                maxoutLayer_ = dynamic_cast<MaxoutVarLayer<f32>*>(&network().getTopLayer());
                require(maxoutLayer_);
                topLayer_ = dynamic_cast<LinearAndSoftmaxLayer<f32>*>(&network().getLayer(maxoutLayer_->getPredecessor(0)));
            }
            require(topLayer_);
            require(!topLayer_->evaluatesSoftmax());
        }

        // initialize statistics
        u32 statisticsType = estimator().requiredStatistics() | Statistics<T>::BASE_STATISTICS;
        statistics_        = new Statistics<T>(network().nLayers(), statisticsType);
        statistics_->initialize(network());
        statistics_->initComputation(false);
        statistics_->reset();
        if (accumulatePrior_) {
            u32 priorStatisticsType = Statistics<T>::CLASS_COUNTS;
            priorStatistics_        = new Statistics<f32>(network().nLayers(), priorStatisticsType);
            priorStatistics_->initialize(network());
            priorStatistics_->initComputation(false);
            priorStatistics_->reset();
        }

        // initialize error signal (sequence length unknown yet, therefore size of matrices not set)
        errorSignal_.resize(network().nLayers(), NnMatrixf32());
        for (u32 layer = 0; layer < errorSignal_.size() - 1; layer++)
            errorSignal_.at(layer).initComputation(false);

        // create ErrorSignalAccumulator
        accumulator_ = new ErrorSignalAccumulator<f32>(&(errorSignal_.back()), &labelWrapper());

        // initialize class weights
        // need to set weightedAccumulation when using frame rejection heuristic
        NeuralNetworkTrainer<f32>::weightedAccumulation_ |= frameRejectionThreshold_ > 0;
        setClassWeights();

        // get prior if cross-entropy smoothing is used
        if (ceSmoothingWeight_ > 0) {
            Prior<f32> prior(config);
            if (prior.fileName() != "") {
                prior.read();
                prior_.resize(prior.size());
                prior.getVector(prior_);
                prior_.initComputation();
                priorScale_ = prior.scale();
                if (priorScale_ > 0)
                    this->log("prior is subtracted from scores in cross-entropy smoothing");
            }
        }

        needInit_ = false;
    }
}

template<typename T>
void SegmentwiseNnTrainer<T>::finalize() {
    // logging
    logTrainingStatistics();

    // stochastic mode : write network parameters
    if (!estimator().fullBatchMode()) {
        network().finalize();
        network().saveNetworkParameters();
    }

    // batch mode : write statistics
    if (statisticsFilename_ != "" && statistics_) {
        statistics_->finishComputation();
        statistics_->write(statisticsFilename_);
    }

    // write prior
    if (accumulatePrior_) {
        Prior<f32>  prior(this->config);
        std::string priorFilename = prior.fileName();
        if (priorFilename != "") {
            prior.setFromClassCounts(*priorStatistics_, this->classWeights_);
            prior.write();
        }
    }

    logProfilingStatistics();
}

template<typename T>
inline NeuralNetwork<f32>& SegmentwiseNnTrainer<T>::network() const {
    return SharedNeuralNetwork::network();
}

template<typename T>
inline const ClassLabelWrapper& SegmentwiseNnTrainer<T>::labelWrapper() const {
    return SharedNeuralNetwork::labelWrapper();
}
template<typename T>
void SegmentwiseNnTrainer<T>::setClassWeights() {
    classWeights_.resize(0);
    classWeights_.resize(labelWrapper().nClassesToAccumulate(), 1.0);

    std::string classWeightsFilename(paramClassWeightsFile(config));
    f32         silenceWeight = paramSilenceWeight(config);

    if (classWeightsFilename != "" && silenceWeight != -1.0)
        this->error("Can not use both silence weight and class weights file");
    else if (classWeightsFilename != "") {
        this->log("reading class weights file ") << classWeightsFilename;
        Math::Module::instance().formats().read(classWeightsFilename, classWeights_);
        if (classWeights_.size() != labelWrapper().nClassesToAccumulate())
            this->error("dimension mismatch: class weights vs number of classes to accumulate")
                    << classWeights_.size() << " != " << labelWrapper().nClassesToAccumulate();
        NeuralNetworkTrainer<f32>::weightedAccumulation_ = true;
    }
    else if (silenceWeight != -1.0) {
        verify(this->acousticModel()->silence() != Bliss::Phoneme::invalidId);
        Am::Allophone silenceAllophone(this->acousticModel()->silence(), Am::Allophone::isInitialPhone | Am::Allophone::isFinalPhone);
        u32           silence = this->acousticModel()->emissionIndex(this->acousticModel()->allophoneStateAlphabet()->index(&silenceAllophone, 0));
        this->log("silence index is ") << silence;
        if (labelWrapper().isClassToAccumulate(silence)) {
            classWeights_.at(labelWrapper().getOutputIndexFromClassIndex(silence)) = silenceWeight;
            this->log("using silence weight ") << silenceWeight;
        }
        else
            this->warning("silence weight has no effect, because silence is not accumulated");
        NeuralNetworkTrainer<f32>::weightedAccumulation_ = true;
    }
}

template<typename T>
void SegmentwiseNnTrainer<T>::resizeErrorSignal() {
    if (statistics_->hasGradient()) {
        for (u32 layer = 0; layer < network().nLayers(); layer++) {
            errorSignal_.at(layer).resize(network().getLayer(layer).getOutputDimension(), sequenceLength_);
            errorSignal_.at(layer).setToZero();
        }
    }
}

template<typename T>
void SegmentwiseNnTrainer<T>::backpropagateError() {
    timeval start, end;
    TIMER_START(start)

    u32 lastLayer;
    if (maxoutLayer_) {
        require_eq(network().nLayers() - 2, maxoutLayer_->getPredecessor(0));  // ugly, TODO
        lastLayer = network().nLayers() - 2;
    }
    else {
        lastLayer = network().nLayers() - 1;
    }

    // reset all error signals except of the last one
    for (u32 layer = 0; layer < lastLayer; layer++)
        errorSignal_.at(layer).setToZero();

    // error backpropagation
    for (s32 layer = (s32)lastLayer; layer > (s32)network().lowestTrainableLayerIndex(); layer--) {
        std::vector<NnMatrixf32*> errorSignalOut;
        for (u32 i = 0; i < network().getLayer(layer).nPredecessors(); i++)
            errorSignalOut.push_back(&(errorSignal_[network().getLayer(layer).getPredecessor(i)]));
        network().getLayer(layer).backpropagateWeights(errorSignal_.at(layer), errorSignalOut);
        network().getLayer(layer - 1).backpropagateActivations(
                errorSignal_.at(layer - 1),
                errorSignal_.at(layer - 1),
                network().getLayerOutput(layer - 1));
    }

    TIMER_GPU_STOP(start, end, measureTime_, timeBackpropagation_)
}

template<typename T>
void SegmentwiseNnTrainer<T>::collectGradient() {
    // just like FeedForwardTrainer<T>::collectGradient()
    timeval start, end;
    TIMER_START(start)

    // compute gradient from error signal and activations
    for (s32 layer = (s32)network().nLayers() - 1; layer >= (s32)network().lowestTrainableLayerIndex(); layer--) {
        /* update the gradient, if layer has weights */
        if (network().getLayer(layer).isTrainable()) {
            for (u32 stream = 0; stream < statistics_->gradientWeights(layer).size(); stream++) {
                NnMatrixf32& layerInputStream = *(network().getLayerInput(layer)[stream]);
                NnMatrixf32& gradientWeights  = statistics_->gradientWeights(layer)[stream];
                NnVectorf32& gradientBias     = statistics_->gradientBias(layer);

                // let every layer update the gradients
                network().getLayer(layer).addToWeightsGradient(layerInputStream, errorSignal_.at(layer), stream, gradientWeights);
                network().getLayer(layer).addToBiasGradient(layerInputStream, errorSignal_.at(layer), stream, gradientBias);
            }
        }
    }
    TIMER_GPU_STOP(start, end, measureTime_, timeGradient_)
}

template<typename T>
void SegmentwiseNnTrainer<T>::initSegment(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* segment) {
    if (!lattice) {
        this->log("no lattice found, skipping segment");
        return;
    }
    if (needInit_)
        initializeTrainer();
    if (!estimator().fullBatchMode()) {
        statistics_->reset();
        ceObjectiveFunction_ = 0;
    }
    // objective function of segment
    localObjectiveFunction_    = 0;
    localCeObjectiveFunction_  = 0;
    localClassificationErrors_ = 0;

    // resizing
    // activations: assume that resizing of has already happened in rescoring step
    sequenceLength_ = this->features()->size();
    require_eq(network().activationsSize(), sequenceLength_);
    timeval start, end;
    TIMER_START(start)
    if (statistics_->hasGradient())
        resizeErrorSignal();
    TIMER_GPU_STOP(start, end, measureTime_, timeMemoryAllocation_)
    segmentNeedsInit_ = false;
}

template<typename T>
void SegmentwiseNnTrainer<T>::accumulatePrior() {
    verify(!segmentNeedsInit_);
    alignment_.finishComputation(true);
    if (accumulatePrior_) {
        for (u32 i = 0; i < alignment_.size(); i++)
            priorStatistics_->incClassCount(alignment_.at(i));
    }
}

template<typename T>
void SegmentwiseNnTrainer<T>::logProperties() const {
    if (singlePrecision_)
        this->log("using single precision accumulator");
    else
        this->log("using double precision accumulator");
    if (ceSmoothingWeight_ > 0.0)
        this->log("use smoothing with cross-entropy criterion with weight: ") << ceSmoothingWeight_;
    else
        this->log("do not smooth with cross-entropy criterion");
    if (frameRejectionThreshold_ > 0)
        this->log("using frame rejection threshold ") << frameRejectionThreshold_;
    if (accumulatePrior_)
        this->log("accumulating prior");
}

template<typename T>
void SegmentwiseNnTrainer<T>::logSegmentStatistics() const {
    if (statisticsChannel_.isOpen()) {
        statisticsChannel_ << Core::XmlOpen("sequence-statistics")
                           << Core::XmlFull("sequence-length", sequenceLength_);
        statisticsChannel_ << Core::XmlFull("frame-classification-error-rate", T(localClassificationErrors_) / sequenceLength_);
        statisticsChannel_ << Core::XmlFull("MMI-objective-function", localObjectiveFunction_);
        if (ceSmoothingWeight_ > 0)
            statisticsChannel_ << Core::XmlFull("avg-ce-objective-function", localCeObjectiveFunction_ / sequenceLength_);
        T totalObjectiveFunction = (1.0 - ceSmoothingWeight_) * localObjectiveFunction_ + ceSmoothingWeight_ * localCeObjectiveFunction_;
        statisticsChannel_ << Core::XmlFull("objective-function", totalObjectiveFunction);
        statisticsChannel_ << Core::XmlClose("sequence-statistics");
    }
}

template<typename T>
void SegmentwiseNnTrainer<T>::logTrainingStatistics() const {
    this->log("number-of-processed-segments: ") << numberOfProcessedSegments_;
    this->log("number-of-observations: ") << numberOfObservations_;
    if ((frameRejectionThreshold_ > 0) && statistics_ && statistics_->hasGradient())
        this->log("number-of-rejected-observations: ") << numberOfRejectedObservations_
                                                       << " ( of " << numberOfObservations_ << ") , " << 100.0 * numberOfRejectedObservations_ / numberOfObservations_ << "%";

    if (estimator().fullBatchMode() && statistics_) {
        this->log("total-frame-classification-error: ") << statistics_->classificationError();
        this->log("total-MMI-objective-function: ") << statistics_->objectiveFunction() / numberOfProcessedSegments_;
        if (ceSmoothingWeight_ > 0)
            this->log("total-avg-ce-objective-function: ") << ceObjectiveFunction_ / numberOfObservations_;
        T totalObjectiveFunction = (1.0 - ceSmoothingWeight_) * statistics_->objectiveFunction() / numberOfProcessedSegments_ + ceSmoothingWeight_ * ceObjectiveFunction_ / numberOfProcessedSegments_;
        this->log("total-objective-function: ") << totalObjectiveFunction;
    }
}

// profiling information
template<typename T>
void SegmentwiseNnTrainer<T>::logProfilingStatistics() const {
    if (measureTime_) {
        this->log() << Core::XmlOpen("time-sequence-discriminative-nn-trainer")
                    << Core::XmlFull("sync", timeSync_)
                    << Core::XmlFull("memory-allocation", timeMemoryAllocation_)
                    << Core::XmlFull("numerator-extraction", timeNumeratorExtraction_)
                    << Core::XmlFull("alignment-vector", timeAlignmentVector_)
                    << Core::XmlFull("initial-error-signal", timeErrorSignal_)
                    << Core::XmlFull("ce-smoothing", timeCESmoothing_)
                    << Core::XmlFull("backward-pass", timeBackpropagation_)
                    << Core::XmlFull("gradient", timeGradient_)
                    << Core::XmlFull("base-statistics", timeBaseStatistics_)
                    << Core::XmlFull("estimation-step", timeEstimationStep)
                    << Core::XmlClose("time-sequence-discriminative-nn-trainer");
    }
}

template<typename T>
void SegmentwiseNnTrainer<T>::accumulateBaseStatistics() {
    verify(!segmentNeedsInit_);
    timeval start, end;
    TIMER_START(start)
    if (statistics_->hasBaseStatistics()) {
        localClassificationErrors_ = network().getTopLayerOutput().nClassificationErrors(alignment_);
        statistics_->incClassificationErrors(localClassificationErrors_);
        statistics_->incObservations(sequenceLength_);
        if (this->weightedAccumulation_)
            statistics_->addToTotalWeight(weights_.asum());
        else
            statistics_->addToTotalWeight(sequenceLength_);
        statistics_->addToObjectiveFunction(localObjectiveFunction_);
        ceObjectiveFunction_ += localCeObjectiveFunction_;
    }
    numberOfProcessedSegments_++;
    numberOfObservations_ += sequenceLength_;
    TIMER_GPU_STOP(start, end, measureTime_, timeBaseStatistics_)
}

template<typename T>
void SegmentwiseNnTrainer<T>::accumulateStatisticsOnLattice(Fsa::ConstAutomatonRef                   posteriorFsa,
                                                            Core::Ref<const Lattice::WordBoundaries> wordBoundaries,
                                                            Mm::Weight                               factor) {
    NnAccumulator* acc = createAccumulator(factor, this->weightThreshold());
    acc->setWordBoundaries(wordBoundaries);
    acc->setFsa(posteriorFsa);
    acc->work();
    delete acc;
}

template<typename T>
T SegmentwiseNnTrainer<T>::smoothErrorSignalWithCE() {
    require(prior_.isComputing());
    verify(!segmentNeedsInit_);
    verify_ge(ceSmoothingWeight_, 0.0);
    verify_le(ceSmoothingWeight_, 1.0);
    timeval start, end;
    TIMER_START(start)

    T ceObjectiveFunction = 0;
    if (ceSmoothingWeight_ > 0.0) {
        if (!maxoutLayer_) {  // simple linear+softmax
            require(topLayer_);
            require(network().getTopLayerOutput().isComputing());

            if (priorScale_ > 0)
                network().getTopLayerOutput().addToAllColumns(prior_, priorScale_);  // add prior, previously assumed: priors was merged in bias, output was ~likelihood
            topLayer_->applySoftmax(network().getTopLayerOutput());                  // and apply softmax

            if (statistics_->hasGradient()) {
                errorSignal_.back().initComputation();  // move data to GPU
                errorSignal_.back().scale(1.0 - ceSmoothingWeight_);
                // softmax - kronecker delta (minimization problem)
                errorSignal_.back().add(network().getTopLayerOutput(), ceSmoothingWeight_);  // add the CE error signal
                errorSignal_.back().addKroneckerDelta(alignment_, -ceSmoothingWeight_);
            }
        }
        else {  // linear+softmax followed by maxoutvar = mixture layer
            require(topLayer_);
            require(maxoutLayer_);
            require(network().getTopLayerOutput().isComputing());
            require(network().getLayerOutput(maxoutLayer_->getPredecessor(0)).isComputing());
            require(maxoutLayer_->getOffset().isComputing());
            require(maxoutLayer_->getMixture().isComputing());
            require(maxoutLayer_->getMaxindex().isComputing());

            if (priorScale_ > 0)
                network().getLayerOutput(maxoutLayer_->getPredecessor(0)).expandAddToAllColumns(maxoutLayer_->getMixture(), maxoutLayer_->getOffset(), prior_,
                                                                                                priorScale_);  // add state prior to all hidden variable per state (maximum approx)
            topLayer_->applySoftmax(network().getLayerOutput(maxoutLayer_->getPredecessor(0)));                // and apply softmax
            network().getTopLayerOutput().maxoutvar(maxoutLayer_->getMixture(),
                                                    maxoutLayer_->getOffset(),
                                                    network().getLayerOutput(maxoutLayer_->getPredecessor(0)), maxoutLayer_->getMaxindex());  // and redo the maxoutvar (but index should remain the same)

            if (statistics_->hasGradient()) {
                errorSignal_[maxoutLayer_->getPredecessor(0)].scale(1.0 - ceSmoothingWeight_);  // rescale the expanded seq.err.signal
                // softmax with maxout - kronecker delta (minimization problem)
                errorSignal_[maxoutLayer_->getPredecessor(0)].add(network().getLayerOutput(maxoutLayer_->getPredecessor(0)),
                                                                  ceSmoothingWeight_);  // add the CE error signal
                errorSignal_[maxoutLayer_->getPredecessor(0)].addKroneckerDelta(alignment_,
                                                                                maxoutLayer_->getOffset(),
                                                                                maxoutLayer_->getMaxindex(),
                                                                                -ceSmoothingWeight_);  // hard target is the (maximal) hidden variable of the target state, maximum approximation
            }
        }

        if (this->weightedAccumulation_)
            ceObjectiveFunction = network().getTopLayerOutput().weightedCrossEntropyObjectiveFunction(alignment_, weights_);
        else
            ceObjectiveFunction = network().getTopLayerOutput().crossEntropyObjectiveFunction(alignment_);
    }
    TIMER_GPU_STOP(start, end, measureTime_, timeCESmoothing_)
    return ceObjectiveFunction;
}

template<typename T>
NnAccumulator* SegmentwiseNnTrainer<T>::createAccumulator(Mm::Weight factor, Mm::Weight weightThreshold) const {
    NnAccumulator* result = new NnAccumulator(features(),
                                              alignmentGenerator(), accumulator_, weightThreshold,
                                              acousticModel(), factor);

    result->setAccumulationFeatures(accumulationFeatures());
    return result;
}

template<typename T>
bool SegmentwiseNnTrainer<T>::getAlignmentVector(Lattice::ConstWordLatticeRef numeratorLattice) {
    verify(!segmentNeedsInit_);
    timeval start, end;
    TIMER_START(start)

    Core::Ref<Lattice::WordLattice> numeratorLatticeMainPart(new Lattice::WordLattice);
    Lattice::ConstWordLatticeRef    bestNumeratorLattice;

    numeratorLatticeMainPart->setWordBoundaries(numeratorLattice->wordBoundaries());
    numeratorLatticeMainPart->setFsa(numeratorLattice->part(this->part_), Lattice::WordLattice::totalFsa);

    bestNumeratorLattice = Lattice::best(Lattice::ConstWordLatticeRef(numeratorLatticeMainPart));
    alignment_.finishComputation(false);
    alignment_.resize(0);
    alignment_.resize(sequenceLength_, Core::Type<u32>::max);

    // pass over lattice and collect alignment
    AlignmentAccumulator*                            alignmentAcc = new AlignmentAccumulator(&alignment_, &labelWrapper());
    CachedAcousticAccumulator<AlignmentAccumulator>* acc          = new CachedAcousticAccumulator<AlignmentAccumulator>(
            features(), alignmentGenerator(),
            alignmentAcc, Core::Type<Mm::Weight>::min, acousticModel(), 1.0);
    acc->setWordBoundaries(bestNumeratorLattice->wordBoundaries());
    acc->setFsa(bestNumeratorLattice->part(this->part_));
    acc->work();
    delete acc;
    delete alignmentAcc;

    // check that alignment is set completely
    bool alignmentOK = true;
    for (u32 i = 0; i < alignment_.size(); i++)
        alignmentOK &= (alignment_.at(i) != Core::Type<u32>::max);

    if (!alignmentOK)
        return false;

    if (NeuralNetworkTrainer<f32>::weightedAccumulation_) {
        weights_.finishComputation(false);
        weights_.resize(sequenceLength_, 0, true);
        for (u32 index = 0; index < weights_.size(); index++)
            weights_.at(index) = classWeights_.at(alignment_.at(index));
    }

    TIMER_GPU_STOP(start, end, measureTime_, timeAlignmentVector_)
    return true;
}

/**
 * factory
 */
template<typename T>
SegmentwiseNnTrainer<T>* SegmentwiseNnTrainer<T>::createSegmentwiseNnTrainer(const Core::Configuration& config) {
    switch (paramCriterion(config)) {
        case maximumMutualInformation:
            return new MmiSegmentwiseNnTrainer<T>(config);
            break;
        case minimumError:
            return new MinimumErrorSegmentwiseNnTrainer<T>(config);
            break;
        default:
            defect();
            break;
    }
    return 0;
}

template class SegmentwiseNnTrainer<f32>;
// template class SegmentwiseNnTrainer<f64>;

}  // namespace Nn
