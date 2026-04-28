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
#include "FeedForwardTrainer.hh"

#include <Math/Module.hh>

#include "NeuralNetworkTrainer.hh"
#include "Prior.hh"

using namespace Nn;

//=============================================================================

template<typename T>
const Core::ParameterString FeedForwardTrainer<T>::paramStatisticsFilename(
        "statistics-filename", "filename to write statistics to", "");

template<typename T>
const Core::ParameterBool FeedForwardTrainer<T>::paramDoublePrecisionAccumulator(
        "double-precision-accumulator", "use double precision for accumulated statistics", false);

template<typename T>
const Core::ParameterBool FeedForwardTrainer<T>::paramNormalizeByNOfObservations(
        "normalize-by-num-observations", "normalize by number of observations", true);

template<typename T>
const Core::ParameterFloat FeedForwardTrainer<T>::paramErrorSignalClip(
        "error-signal-clip", "clip error signal matrix by this value", Core::Type<T>::max, 0);

template<typename T>
const Core::ParameterBool FeedForwardTrainer<T>::paramLogFrameEntropy(
        "log-frame-entropy", "log frame entropy for each minibatch", false);

template<typename T>
FeedForwardTrainer<T>::FeedForwardTrainer(const Core::Configuration& c)
        : Core::Component(c),
          NeuralNetworkTrainer<T>(c),
          statisticsFilename_(paramStatisticsFilename(c)),
          useDoublePrecisionAccumulator_(paramDoublePrecisionAccumulator(c)),
          statistics_(0),
          doublePrecisionStatistics_(0),
          normalizeByNOfObservations_(paramNormalizeByNOfObservations(c)),
          errorSignalClip_(paramErrorSignalClip(c)),
          logFrameEntropy_(paramLogFrameEntropy(c)),
          lowestTrainableLayerIndex_(0),
          weights_(0),
          timeSync_(0),
          timeForwardPass_(0),
          timeInitialErrorSignal_(0),
          timeBackwardPass_(0),
          timeGradient_(0),
          timeBaseStatistics_(0),
          timeRegularization_(0),
          timeEstimation_(0),
          timeSyncBatch_(0),
          timeForwardPassBatch_(0),
          timeInitialErrorSignalBatch_(0),
          timeBackwardPassBatch_(0),
          timeGradientBatch_(0),
          timeBaseStatisticsBatch_(0),
          timeRegularizationBatch_(0),
          timeEstimationBatch_(0),
          minibatchCount_(0),
          discardedMinibatchCount_(0)
#ifdef MODULE_PYTHON
          ,
          pythonControl_(c, "FeedForwardTrainer", true)
#endif
{
    if (statisticsFilename_ != "")
        this->log("writing statistics to ") << statisticsFilename_;
    if (useDoublePrecisionAccumulator_) {
        if (!estimator().fullBatchMode())
            this->error("double precision accumulator only possible for batch optimization");
        else
            this->log("using double precision accumulator");
    }

    if (errorSignalClip_ < Core::Type<T>::max)
        this->log("using error signal matrix clip: ") << errorSignalClip_;
}

template<typename T>
FeedForwardTrainer<T>::~FeedForwardTrainer() {
    if (statistics_)
        delete statistics_;
    if (doublePrecisionStatistics_)
        delete doublePrecisionStatistics_;
}

template<typename T>
void FeedForwardTrainer<T>::initializeTrainer(u32 batchSize, std::vector<u32>& streamSizes) {
    if (Precursor::needInit_) {
        Precursor::initializeTrainer(batchSize, streamSizes);
        if (this->hasNetwork()) {
            lowestTrainableLayerIndex_ = -1;
        }
        // get trainable layer with lowest index
        lowestTrainableLayerIndex_ = (s32)this->nLayers() - 1;
        for (s32 layer = (s32)this->nLayers() - 1; layer >= 0; layer--) {
            if (network().getLayer(layer).isTrainable()) {
                lowestTrainableLayerIndex_ = layer;
            }
        }
        // initialize error signal
        errorSignal_.resize(this->nLayers());
        for (u32 layer = 0; layer < this->nLayers(); layer++) {
            errorSignal_[layer].resize(network().getLayer(layer).getOutputDimension(), batchSize);
        }
        errorSignalOut_.resize(this->nLayers());
        for (s32 layer = (s32)this->nLayers() - 1; layer > lowestTrainableLayerIndex_; layer--) {
            std::vector<NnMatrix*> errorSignalOut;
            for (u32 i = 0; i < network().getLayer(layer).nPredecessors(); i++) {
                errorSignalOut.push_back(&(errorSignal_[network().getLayer(layer).getPredecessor(i)]));
            }
            errorSignalOut_.at(layer) = errorSignalOut;
        }
        // initialize statistics
        u32 statisticsType = estimator().requiredStatistics();
        if (this->hasNetwork() && (statisticsChannel_.isOpen() || statisticsFilename_ != ""))  // otherwise this information would be lost anyway
            statisticsType |= Statistics<T>::BASE_STATISTICS;

        statistics_ = new Statistics<T>(this->nLayers(), statisticsType);
        if (this->hasNetwork())
            statistics_->initialize(network());
        if (useDoublePrecisionAccumulator_)
            initializeDoublePrecisionStatistics();
        // initialize computation
        statistics_->initComputation();
        statistics_->reset();
        for (u32 layer = 0; layer < errorSignal_.size(); layer++)
            errorSignal_[layer].initComputation();
        Precursor::needInit_ = false;
    }
}

template<typename T>
void FeedForwardTrainer<T>::finalize() {
    if (estimator().fullBatchMode() && statistics_) {
        Prior<T>    prior(this->config);
        std::string priorFilename = prior.fileName();
        if (priorFilename != "" && statistics().hasClassCounts()) {
            require(this->classWeights_);
            prior.setFromClassCounts(statistics(), *this->classWeights_);
            prior.write();
        }

        if (statistics().hasGradient() || statistics().hasBaseStatistics()) {
            if (statisticsFilename_ != "") {
                if (doublePrecisionStatistics_) {
                    statistics_->reset();
                    // convert double precision to single precision statistics
                    statistics_->add(*doublePrecisionStatistics_);
                }
                // write statistics
                statistics().finishComputation();
                statistics().write(statisticsFilename_);
            }
            else if (statistics().hasGradient())
                this->warning("statistics-filename not set, do not write statistics");
        }

        Core::Component::log("total-number-of-observations: ") << statistics_->nObservations();
        if (statistics().hasBaseStatistics()) {
            Core::Component::log("total-frame-classification-error: ") << (T)statistics_->classificationError();
            Core::Component::log("total-objective-function: ") << statistics_->objectiveFunction();
        }
    }
    if (!estimator().fullBatchMode() && estimator().accumulateMultipleBatches() > 1) {
        u32 latestRecentBatchesCount = minibatchCount_ % estimator().accumulateMultipleBatches();
        if (latestRecentBatchesCount > 0) {
            // This means that we did not do a model-update with the latest recent batches.
            Core::Component::log(
                    "The last %i batches were ignored because of accumulate-multiple-batches=%i",
                    latestRecentBatchesCount, estimator().accumulateMultipleBatches());
            if (minibatchCount_ < estimator().accumulateMultipleBatches())
                Core::Component::warning("We did not use any batches. (because accumulate-multiple-batches)");
        }
    }
    else if (minibatchCount_ == 0)
        Core::Component::warning("We did not use any batches.");
    Precursor::finalize();
    if (this->measureTime_) {
        this->log() << Core::XmlOpen("time-feed-forward-nn-trainer")
                    << Core::XmlFull("forwarding", timeForwardPass_)
                    << Core::XmlFull("initial-error-signal", timeInitialErrorSignal_)
                    << Core::XmlFull("backward-pass", timeBackwardPass_)
                    << Core::XmlFull("gradient", timeGradient_)
                    << Core::XmlFull("base-statistics", timeBaseStatistics_)
                    << Core::XmlFull("regularization", timeRegularization_)
                    << Core::XmlFull("estimation", timeEstimation_)
                    << Core::XmlClose("time-feed-forward-nn-trainer");
    }
    {
        this->log() << Core::XmlOpen("counts")
                    << Core::XmlFull("minibatches", minibatchCount_)
                    << Core::XmlFull("discarded-minibatches", discardedMinibatchCount_)
                    << Core::XmlClose("counts");
    }
}

template<typename T>
void FeedForwardTrainer<T>::processBatch_feedInput(std::vector<NnMatrix>& features, NnVector* weights, Bliss::Segment* segment) {
    minibatchCount_++;

    if (weightedAccumulation_) {
        weights_ = weights;
        if (!weights_)
            Core::Component::warning("weightedAccumulation without weights.");
    }
    else {
        weights_ = weights = NULL;
    }

    // for profiling
    resetBatchTimes();
    timeval start, end;

    // initialization
    if (doublePrecisionStatistics_)
        // Always accumulate full-batch in double precision stats - reset single precision stats.
        statistics_->reset();
    else if (!estimator().fullBatchMode()) {
        if ((minibatchCount_ - 1) % estimator().accumulateMultipleBatches() == 0)
            statistics_->reset();
    }

#ifdef MODULE_PYTHON
    pythonControl_.run_custom("init_segment", "{s:s}", "segment_name", segment ? segment->fullName().c_str() : NULL);
#endif

    u32 batchSize = features[0].nColumns();
    setBatchSize(batchSize);
    statistics_->incObservations(batchSize);
    if (weightedAccumulation_ && weights) {
        TIMER_START(start);
        weights->initComputation();
        TIMER_GPU_STOP_SUM2(start, end, this->measureTime_, timeSyncBatch_, timeSync_)
        statistics_->addToTotalWeight(weights->asum());
    }
    else {
        statistics_->addToTotalWeight(batchSize);
    }

    // forward network
    if (statistics_->hasBaseStatistics() || statistics_->hasGradient()) {
        TIMER_START(start);
        // sync required here only to include it in time measurement
        for (u32 i = 0; i < features.size(); ++i)
            features.at(i).initComputation();
        TIMER_GPU_STOP_SUM2(start, end, this->measureTime_, timeSyncBatch_, timeSync_)

        TIMER_START(start);
        network().forward(features);
        TIMER_GPU_STOP_SUM2(start, end, this->measureTime_, timeForwardPassBatch_, timeForwardPass_)
    }
}

template<typename T>
void FeedForwardTrainer<T>::processBatch_finishWithError_naturalPairing(T error, NnMatrix& errorSignal) {
    timeval start, end;

#ifdef MODULE_PYTHON
    pythonControl_.run_custom(
            "notify_segment_loss", "{s:s,s:f}",
            "segment_name", NULL,
            "loss", (float)error);
#endif

start:

    // calculate number of classification errors and objective function
    if (statistics_->hasBaseStatistics()) {
        TIMER_START(start);
        statistics_->addToObjectiveFunction(error);
        TIMER_GPU_STOP_SUM2(start, end, this->measureTime_, timeBaseStatisticsBatch_, timeBaseStatistics_)

        // apply regularization only when not in batch mode
        if (!estimator().fullBatchMode()) {
            TIMER_START(start);
            u32 batchSize        = network().getLayerInput(0)[0]->nColumns();
            T   regularizerError = regularizer().objectiveFunction(network(), T(batchSize));
            statistics_->addToObjectiveFunction(regularizerError);
            error += regularizerError;

            if (logFrameEntropy_) {
                Math::FastVector<T> entropy(batchSize);
                // not necessary once FastVector::columnEntropy() is implemented on GPU
                network().getTopLayerOutput().finishComputation(true);
                Math::FastMatrix<T>& output = network().getTopLayerOutput().asWritableCpuMatrix();
                entropy.columnEntropy(output);
                network().getTopLayerOutput().initComputation(false);
                statistics_->addToEntropy(entropy.sum());
            }
            TIMER_GPU_STOP_SUM2(start, end, this->measureTime_, timeRegularizationBatch_, timeRegularization_)
        }
    }

    if (statistics_->hasGradient()) {
        // reset error signals
        if (!errorSignal_.empty()) {
            for (u32 layer = 0; layer < errorSignal_.size() - 1; layer++)
                errorSignal_.at(layer).setToZero();

            // Special case: we have filled the matrix already inplace.
            // In that case, nothing needs to be done. Otherwise, copy it over.
            NnMatrix& topErrorSignal = errorSignal_[errorSignal_.size() - 1];
            if (&errorSignal != &topErrorSignal) {
                topErrorSignal.copy(errorSignal);
            }
        }

        // error backprop
        errorBackpropagation();

        // collect gradient
        collectGradient();

        // apply regularization only when not in batch mode
        if (!estimator().fullBatchMode()) {
            TIMER_START(start);
            regularizer().addGradient(network(), statistics(), T(statistics().nObservations()));
            TIMER_GPU_STOP_SUM2(start, end, this->measureTime_, timeRegularizationBatch_, timeRegularization_)
        }
    }

    // update (only if has gradient)
    if (statistics_->hasGradient() && !estimator().fullBatchMode()) {
        if (minibatchCount_ % estimator().accumulateMultipleBatches() == 0) {
            // maybe normalize statistics by batch size
            statistics_->finalize(normalizeByNOfObservations_);
            // update model
            TIMER_START(start);
            estimator().estimate(network(), statistics());
            TIMER_GPU_STOP_SUM2(start, end, this->measureTime_, timeEstimationBatch_, timeEstimation_);
        }
    }

    if (doublePrecisionStatistics_ && estimator().fullBatchMode())
        doublePrecisionStatistics_->add(*statistics_);

    // logging
    if (statisticsChannel_.isOpen() && statistics_->hasBaseStatistics() && !(estimator().fullBatchMode())) {
        if (minibatchCount_ % estimator().accumulateMultipleBatches() == 0) {
            // Note that we called statistics_->finalize() above, which might have normalized these.
            statisticsChannel_ << Core::XmlOpen("batch-statistics")
                               << Core::XmlFull("mini-batch-number", minibatchCount_)
                               << Core::XmlFull("frame-classification-error-rate-on-batch", statistics_->classificationError())
                               << Core::XmlFull("objective-function-on-batch", statistics_->objectiveFunction());
            if (estimator().accumulateMultipleBatches() > 1)
                statisticsChannel_ << Core::XmlFull("batch-total-time-frames", statistics_->nObservations());
            if (logFrameEntropy_)
                statisticsChannel_ << Core::XmlFull("batch-average-entropy", statistics_->entropy());
            statisticsChannel_ << Core::XmlClose("batch-statistics");
        }
        else {
            statisticsChannel_ << Core::XmlOpen("batch-statistics-accumulated-so-far")
                               << Core::XmlFull("mini-batch-number", minibatchCount_)
                               << Core::XmlFull("accumulated-frame-classification-error-rate", statistics_->classificationError())
                               << Core::XmlFull("accumulated-objective-function", statistics_->objectiveFunction());
            if (estimator().accumulateMultipleBatches() > 1)
                statisticsChannel_ << Core::XmlFull("accumulated-time-frames", statistics_->nObservations());
            statisticsChannel_ << Core::XmlClose("batch-statistics-accumulated-so-far");
        }
    }

    weights_ = NULL;
}

template<typename T>
void FeedForwardTrainer<T>::processBatch_finishDiscard() {
    minibatchCount_--;
    discardedMinibatchCount_++;

    if (estimator().fullBatchMode() || estimator().accumulateMultipleBatches() > 1) {
        // remove this batch statistics
        u32 batchSize = network().getLayerInput(0)[0]->nColumns();
        statistics_->decObservations(batchSize);
        if (weightedAccumulation_ && weights_)
            statistics_->addToTotalWeight(-weights_->asum());
        else
            statistics_->addToTotalWeight(-(T)batchSize);
    }
}

// main function to process a mini-batch with alignment.
// alignment has the NN output labels indices.
template<typename T>
void FeedForwardTrainer<T>::processBatch_finishWithAlignment(Math::CudaVector<u32>& alignment) {
    timeval start, end;

    // count classes
    if (statistics_->hasClassCounts())
        updateClassCounts(alignment, statistics());

    {
        TIMER_START(start);
        alignment.initComputation();
        TIMER_GPU_STOP_SUM2(start, end, this->measureTime_, this->timeSyncBatch_, this->timeSync_);
    }

    // calculate objective function and classification errors
    T    error   = 0;
    bool discard = false;
    if (statistics_->hasBaseStatistics() || statistics_->hasGradient()) {
        TIMER_START(start);
        Precursor::criterion_->inputAlignment(alignment, network().getTopLayerOutput(), weights_);
        discard = Precursor::criterion_->discardCurrentInput();
        if (!discard) {
            Precursor::criterion_->getObjectiveFunction(error);
            statistics_->incClassificationErrors(network().getTopLayerOutput().nClassificationErrors(alignment));
        }
        TIMER_GPU_STOP_SUM2(start, end, this->measureTime_, this->timeBaseStatisticsBatch_, this->timeBaseStatistics_);
    }

    // compute gradient
    NnMatrix  dummyErrorSignal;  // might be needed if no gradient is calculated
    NnMatrix& errorSignal = errorSignal_.empty() ? dummyErrorSignal : errorSignal_[errorSignal_.size() - 1];

    if (!discard && statistics_->hasGradient()) {
        // set error signal of top layer
        TIMER_START(start);
        Precursor::criterion_->getErrorSignal_naturalPairing(errorSignal, network().getTopLayer());
        TIMER_GPU_STOP_SUM2(start, end, this->measureTime_, this->timeInitialErrorSignalBatch_, this->timeInitialErrorSignal_);
    }

    if (!discard)
        this->processBatch_finishWithError_naturalPairing(error, errorSignal);
    else
        processBatch_finishDiscard();
}

template<typename T>
void FeedForwardTrainer<T>::processBatch_finishWithSpeechSegment(Bliss::SpeechSegment& segment) {
    timeval start, end;

    // calculate objective function
    T    error   = 0;
    bool discard = false;
    if (statistics_->hasBaseStatistics() || statistics_->hasGradient()) {
        TIMER_START(start);
        Precursor::criterion_->inputSpeechSegment(segment, network().getTopLayerOutput(), weights_);
        discard = Precursor::criterion_->discardCurrentInput();
        if (!discard)
            Precursor::criterion_->getObjectiveFunction(error);
        TIMER_GPU_STOP_SUM2(start, end, this->measureTime_, this->timeBaseStatisticsBatch_, this->timeBaseStatistics_);
    }

    // compute gradient
    NnMatrix  dummyErrorSignal;  // might be needed if no gradient is calculated
    NnMatrix& errorSignal = errorSignal_.empty() ? dummyErrorSignal : errorSignal_[errorSignal_.size() - 1];

    if (!discard && statistics_->hasGradient()) {
        // set error signal of top layer
        TIMER_START(start);
        Precursor::criterion_->getErrorSignal_naturalPairing(errorSignal, network().getTopLayer());
        TIMER_GPU_STOP_SUM2(start, end, this->measureTime_, this->timeInitialErrorSignalBatch_, this->timeInitialErrorSignal_);
    }

    if (!discard)
        this->processBatch_finishWithError_naturalPairing(error, errorSignal);
    else
        processBatch_finishDiscard();
}

template<typename T>
void FeedForwardTrainer<T>::errorBackpropagation() {
    timeval start, end;
    TIMER_START(start);

    // error backpropagation
    for (s32 layer = (s32)network().nLayers() - 1; layer > lowestTrainableLayerIndex_; layer--) {
        if (errorSignalClip_ < Core::Type<T>::max)
            errorSignal_.at(layer).clip(errorSignalClip_);

        network().getLayer(layer).backpropagateWeights(
                errorSignal_.at(layer), errorSignalOut_.at(layer));
        network().getLayer(layer - 1).backpropagateActivations(
                errorSignal_.at(layer - 1),
                errorSignal_.at(layer - 1),
                network().getLayerOutput(layer - 1));
    }

    TIMER_GPU_STOP_SUM2(start, end, this->measureTime_, timeBackwardPassBatch_, timeBackwardPass_);
}

template<typename T>
void FeedForwardTrainer<T>::collectGradient() {
    timeval start, end;
    TIMER_START(start);

    // gradient computation
    for (s32 layer = (s32)network().nLayers() - 1; layer >= lowestTrainableLayerIndex_; layer--) {
        /* update the gradient, if layer is trainable */
        if (network().getLayer(layer).isTrainable()) {
            for (u32 stream = 0; stream < statistics_->gradientWeights(layer).size(); stream++) {
                NnMatrix& layerInputStream = *(network().getLayerInput(layer)[stream]);
                NnMatrix& gradientWeights  = statistics_->gradientWeights(layer)[stream];
                NnVector& gradientBias     = statistics_->gradientBias(layer);

                // let every layer update the gradients
                network().getLayer(layer).addToWeightsGradient(layerInputStream,
                                                               errorSignal_.at(layer), stream, gradientWeights);
                network().getLayer(layer).addToBiasGradient(layerInputStream,
                                                            errorSignal_.at(layer), stream, gradientBias);
            }
        }
    }
    TIMER_GPU_STOP_SUM2(start, end, this->measureTime_, timeGradientBatch_, timeGradient_);
}

template<typename T>
void FeedForwardTrainer<T>::updateClassCounts(const Math::CudaVector<u32>& alignment, Statistics<T>& statistics) {
    for (u32 i = 0; i < alignment.size(); i++) {
        statistics.incClassCount(alignment.at(i));
    }
}

template<typename T>
void FeedForwardTrainer<T>::setBatchSize(u32 batchSize) {
    if (batchSize != this->batchSize()) {
        Precursor::setBatchSize(batchSize);
        for (u32 i = 0; i < errorSignal_.size(); i++) {
            u32 nRows = errorSignal_.at(i).nRows();
            errorSignal_.at(i).resize(nRows, batchSize);
        }
    }
}

template<typename T>
void FeedForwardTrainer<T>::resetBatchTimes() {
    timeSyncBatch_               = 0.0;
    timeForwardPassBatch_        = 0.0;
    timeInitialErrorSignalBatch_ = 0.0;
    timeBackwardPassBatch_       = 0.0;
    timeGradientBatch_           = 0.0;
    timeBaseStatisticsBatch_     = 0.0;
    timeRegularizationBatch_     = 0.0;
    timeEstimationBatch_         = 0.0;
}

template<typename T>
void FeedForwardTrainer<T>::logBatchTimes() const {
    this->log() << Core::XmlOpen("mini-batch-computation-times")
                << Core::XmlFull("sync", timeSyncBatch_)
                << Core::XmlFull("forward-pass", timeForwardPassBatch_)
                << Core::XmlFull("initial-error-signal", timeInitialErrorSignalBatch_)
                << Core::XmlFull("backward-pass", timeBackwardPassBatch_)
                << Core::XmlFull("gradient", timeGradientBatch_)
                << Core::XmlFull("base-statistics", timeBaseStatisticsBatch_)
                << Core::XmlFull("regularization", timeRegularizationBatch_)
                << Core::XmlFull("estimation", timeEstimationBatch_)
                << Core::XmlClose("mini-batch-computation-times");
}

//=============================================================================
template<typename T>
const Core::ParameterString FeedForwardAutoTrainer<T>::paramReferenceInputLayer(
        "input-layer", "use input of this layer as target output", "set-layer-name");

template<typename T>
const Core::ParameterInt FeedForwardAutoTrainer<T>::paramReferenceInputLayerPort(
        "input-layer-port", "use input of this port as target output", 0);

template<typename T>
FeedForwardAutoTrainer<T>::FeedForwardAutoTrainer(const Core::Configuration& config)
        : Core::Component(config),
          FeedForwardTrainer<T>(config),
          referenceInputLayer_(paramReferenceInputLayer(config)),
          referenceInputLayerPort_(paramReferenceInputLayerPort(config)) {
    require_eq(Precursor::criterion_->getType(), Criterion<T>::squaredError);
    this->log("autoencoder will learn the input of layer ")
            << referenceInputLayer_ << " on input port " << referenceInputLayerPort_;
}

template<typename T>
void FeedForwardAutoTrainer<T>::processBatch_feedInput(std::vector<NnMatrix>& features, NnVector* weights, Bliss::Segment* segment) {
    Precursor::processBatch_feedInput(features, weights, segment);
}

template<typename T>
void FeedForwardAutoTrainer<T>::processBatch_finishWithAlignment(Math::CudaVector<u32>& alignment) {
    NnMatrix  dummyErrorSignal;  // might be needed if no gradient is calculated
    NnMatrix& errorSignal = errorSignal_.empty() ? dummyErrorSignal : errorSignal_[errorSignal_.size() - 1];

    // calculate number of classification errors and objective function
    if (statistics_->hasBaseStatistics()) {
        // apply regularization only when not in batch mode
        if (!estimator().fullBatchMode()) {
            u32 batchSize        = network().getLayerInput(0)[0]->nColumns();
            T   regularizerError = regularizer().objectiveFunction(network(), T(batchSize));
            statistics_->addToObjectiveFunction(regularizerError);
        }
    }

    if (statistics_->hasGradient()) {
        // reset error signals
        if (!errorSignal_.empty()) {
            for (u32 layer = 0; layer < errorSignal_.size(); layer++) {
                errorSignal_.at(layer).setToZero();
            }

            // Special case: we have filled the matrix already inplace.
            // In that case, nothing needs to be done. Otherwise, copy it over.
            NnMatrix& topErrorSignal = errorSignal_[errorSignal_.size() - 1];
            if (&errorSignal != &topErrorSignal) {
                topErrorSignal.copy(errorSignal);
            }

            u32       layerId   = network().getLayerIdByName(referenceInputLayer_);
            NnMatrix& netInput  = *network().getLayerInput(layerId)[referenceInputLayerPort_];
            NnMatrix& netOutput = network().getTopLayerOutput();

            topErrorSignal.add(netOutput);
            topErrorSignal.add(netInput, (T)-1.0);

            T err = topErrorSignal.sumOfSquares() / netOutput.size();
            statistics_->addToObjectiveFunction(err * netOutput.nColumns());
        }

        // error backprop
        Precursor::errorBackpropagation();

        // collect gradient
        Precursor::collectGradient();

        // apply regularization only when not in batch mode
        if (!estimator().fullBatchMode()) {
            regularizer().addGradient(network(), statistics(), T(statistics().nObservations()));
        }
    }

    // update (only if has gradient)
    if (statistics_->hasGradient() && !estimator().fullBatchMode()) {
        if (Precursor::minibatchCount_ % estimator().accumulateMultipleBatches() == 0) {
            // maybe normalize statistics by batch size
            statistics_->finalize(Precursor::normalizeByNOfObservations_);
            // update model
            estimator().estimate(network(), statistics());
        }
    }

    if (Precursor::doublePrecisionStatistics_ && estimator().fullBatchMode())
        Precursor::doublePrecisionStatistics_->add(*statistics_);

    // logging
    if (statisticsChannel_.isOpen() && statistics_->hasBaseStatistics() && !(estimator().fullBatchMode())) {
        if (Precursor::minibatchCount_ % estimator().accumulateMultipleBatches() == 0) {
            // Note that we called statistics_->finalize() above, which might have normalized these.
            statisticsChannel_ << Core::XmlOpen("batch-statistics")
                               << Core::XmlFull("mini-batch-number", Precursor::minibatchCount_)
                               << Core::XmlFull("objective-function-on-batch", statistics_->objectiveFunction());
            if (estimator().accumulateMultipleBatches() > 1)
                statisticsChannel_ << Core::XmlFull("batch-total-time-frames", statistics_->nObservations());
            statisticsChannel_ << Core::XmlClose("batch-statistics");
        }
        else {
            statisticsChannel_ << Core::XmlOpen("batch-statistics-accumulated-so-far")
                               << Core::XmlFull("mini-batch-number", Precursor::minibatchCount_)
                               << Core::XmlFull("accumulated-objective-function", statistics_->objectiveFunction());
            if (estimator().accumulateMultipleBatches() > 1)
                statisticsChannel_ << Core::XmlFull("accumulated-time-frames", statistics_->nObservations());
            statisticsChannel_ << Core::XmlClose("batch-statistics-accumulated-so-far");
        }
    }
}

template<typename T>
void FeedForwardAutoTrainer<T>::processBatch_finish() {
}

//=============================================================================

// explicit template instantiation
namespace Nn {

template class FeedForwardTrainer<f32>;
template class FeedForwardTrainer<f64>;

template class FeedForwardAutoTrainer<f32>;
template class FeedForwardAutoTrainer<f64>;
}  // namespace Nn
