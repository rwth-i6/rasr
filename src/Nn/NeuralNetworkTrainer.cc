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
#include "NeuralNetworkTrainer.hh"
#include <Math/Blas.hh>
#include <algorithm>
#include <limits>
#include "FeedForwardTrainer.hh"
#include "MeanNormalizedSgdEstimator.hh"

#include <Flow/ArchiveWriter.hh>
#include <Math/Module.hh>  // XML I/O stuff for writing parameters

#ifndef CMAKE_DISABLE_MODULES_HH
#include <Modules.hh>
#endif
#ifdef MODULE_PYTHON
#include "PythonTrainer.hh"
#endif

using namespace Nn;

template<typename T>
const Core::Choice NeuralNetworkTrainer<T>::choiceNetworkTrainer(
        "dummy", dummy,
        "feed-forward-trainer", feedForwardTrainer,
        "frame-classification-error", frameClassificationErrorAccumulator,
        "mean-and-variance-accumulator", meanAndVarianceAccumulator,
        "autoencoder", autoencoderTrainer,
        "network-evaluator", networkEvaluator,
        "python-trainer", pythonTrainer,  // needs MODULE_PYTHON
        "python-evaluator", pythonEvaluator,
        Core::Choice::endMark());

template<typename T>
const Core::ParameterChoice NeuralNetworkTrainer<T>::paramNetworkTrainer(
        "trainer", &choiceNetworkTrainer,
        "trainer for the neural network", dummy);

template<typename T>
const Core::ParameterInt NeuralNetworkTrainer<T>::paramEpoch("epoch", "current epoch", 1);

template<typename T>
const Core::ParameterBool NeuralNetworkTrainer<T>::paramWeightedAccumulation(
        "weighted-accumulation", "use weights in training if possible and available", false);

template<typename T>
const Core::ParameterBool NeuralNetworkTrainer<T>::paramMeasureTime(
        "measure-time", "Measures time for executing methods in FeedForwardTrainer", false);

template<typename T>
NeuralNetworkTrainer<T>::NeuralNetworkTrainer(const Core::Configuration& config)
        : Core::Component(config),
          weightedAccumulation_(paramWeightedAccumulation(config)),
          classWeights_(0),
          measureTime_(paramMeasureTime(config)),
          needsNetwork_(true),
          statisticsChannel_(config, "statistics"),
          needInit_(true),
          network_(0) {
    estimator_   = Estimator<T>::createEstimator(config);
    regularizer_ = Regularizer<T>::createRegularizer(config);
    criterion_   = Criterion<T>::create(config);
    logProperties();
}

template<typename T>
NeuralNetworkTrainer<T>::~NeuralNetworkTrainer() {
    if (network_) {
        delete network_;
    }
    delete estimator_;
    delete regularizer_;
    delete criterion_;
}

template<typename T>
void NeuralNetworkTrainer<T>::initializeTrainer(u32 batchSize) {
    std::vector<u32> streamSizes;
    initializeTrainer(batchSize, streamSizes);
}

template<typename T>
void NeuralNetworkTrainer<T>::setBatchSize(u32 batchSize) {
    if (network_)
        network_->resizeActivations(batchSize);
}

template<typename T>
void NeuralNetworkTrainer<T>::initializeTrainer(u32 batchSize, std::vector<u32>& streamSizes) {
    if (needInit_) {
        if (estimator().type() == "prior-estimator")
            this->needsNetwork_ = false;
        if (needsNetwork_) {
            network_ = new NeuralNetwork<T>(config);
            // initialize the network with each layer and initialize (gpu) computation for the matrices
            network_->initializeNetwork(batchSize, streamSizes);
        }
    }
    needInit_ = false;
}

template<typename T>
void NeuralNetworkTrainer<T>::setClassWeights(const Math::Vector<T>* classWeights) {
    classWeights_ = classWeights;
}

template<typename T>
void NeuralNetworkTrainer<T>::finalize() {
    if (network_) {
        network_->finalize();
        // save only when network has been changed
        if (estimator_ && (!estimator_->fullBatchMode()))
            network_->saveNetworkParameters();
    }
}

template<typename T>
void NeuralNetworkTrainer<T>::resetHistory() {
    if (network_)
        network_->resetPreviousActivations();
}

template<typename T>
void NeuralNetworkTrainer<T>::logProperties() const {
    if (weightedAccumulation_) {
        this->log("using weighted accumulation");
    }
    if (measureTime_) {
        this->log("measuring computation time");
    }
}

// create the specific type of supervised neural network trainer
template<typename T>
NeuralNetworkTrainer<T>* NeuralNetworkTrainer<T>::createSupervisedTrainer(const Core::Configuration& config) {
    NeuralNetworkTrainer<T>* trainer = NULL;

    // get the type of the trainer
    switch ((TrainerType)paramNetworkTrainer(config)) {
        case feedForwardTrainer:
            Core::Application::us()->log("Create trainer: feed-forward trainer");
            trainer = new FeedForwardTrainer<T>(config);
            break;
        case frameClassificationErrorAccumulator:
            Core::Application::us()->log("Create trainer: frame-classification-error");
            trainer = new FrameErrorEvaluator<T>(config);
            break;
        case meanAndVarianceAccumulator:
            Core::Application::us()->log("Create trainer: mean-and-variance-estimation");
            trainer = new MeanAndVarianceTrainer<T>(config);
            break;
        case networkEvaluator:
            Core::Application::us()->log("Create trainer: network-evaluator");
            trainer = new NetworkEvaluator<T>(config);
            break;
        case autoencoderTrainer:
            Core::Application::us()->log("Create trainer: autoencoder");
            trainer = new FeedForwardAutoTrainer<T>(config);
            break;
        case pythonTrainer:
#ifdef MODULE_PYTHON
            Core::Application::us()->log("Create trainer: Python trainer");
            trainer = new PythonTrainer<T>(config);
#else
            Core::Application::us()->criticalError("Python-trainer: Python support not compiled");
#endif
        case pythonEvaluator:
#ifdef MODULE_PYTHON
            Core::Application::us()->log("Create trainer: Python evaluator");
            trainer = new PythonEvaluator<T>(config);
#else
            Core::Application::us()->criticalError("Python-evaluator: Python support not compiled");
#endif

            break;
        default:  // dummy trainer
            Core::Application::us()->warning("The given trainer is not a valid supervised trainer type. Create dummy trainer.");
            trainer = new NeuralNetworkTrainer<T>(config);
            Core::Application::us()->log("Create trainer: dummy");
            break;
    };

    return trainer;
}

// create the specific type of unsupervised neural network trainer
template<typename T>
NeuralNetworkTrainer<T>* NeuralNetworkTrainer<T>::createUnsupervisedTrainer(const Core::Configuration& config) {
    NeuralNetworkTrainer<T>* trainer = NULL;

    // get the type of the trainer
    switch ((TrainerType)paramNetworkTrainer(config)) {
        case meanAndVarianceAccumulator:
            Core::Application::us()->log("Create trainer: mean-and-variance-estimation");
            trainer = new MeanAndVarianceTrainer<T>(config);
            break;
        case networkEvaluator:
            Core::Application::us()->log("Create trainer: network-evaluator");
            trainer = new NetworkEvaluator<T>(config);
            break;
        case pythonTrainer:
#ifdef MODULE_PYTHON
            Core::Application::us()->log("Create trainer: Python trainer");
            trainer = new PythonTrainer<T>(config);
#else
            Core::Application::us()->criticalError("Python-trainer: Python support not compiled");
#endif
            break;
        default:  // dummy trainer
            Core::Application::us()->warning("The given trainer is not a valid unsupervised trainer type. Create dummy trainer.");
            trainer = new NeuralNetworkTrainer<T>(config);
            Core::Application::us()->log("Create trainer: dummy");
            break;
    };

    return trainer;
}

//=============================================================================

template<typename T>
const Core::ParameterBool FrameErrorEvaluator<T>::paramLogFrameEntropy(
        "log-frame-entropy", "log the average frame entropy", false);

template<typename T>
FrameErrorEvaluator<T>::FrameErrorEvaluator(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config),
          nObservations_(0),
          nFrameClassificationErrors_(0),
          objectiveFunction_(0),
          logFrameEntropy_(paramLogFrameEntropy(config)),
          frameEntropy_(0) {}

template<typename T>
void FrameErrorEvaluator<T>::finalize() {
    Core::Component::log("total-frame-classification-error: ") << (T)nFrameClassificationErrors_ / nObservations_;
    Core::Component::log("total-objective-function: ") << objectiveFunction_ / nObservations_;
    if (logFrameEntropy_)
        Core::Component::log("total-frame-entropy: ") << frameEntropy_ / nObservations_;
    network().finalize();
}

template<typename T>
void FrameErrorEvaluator<T>::processBatch_feedInput(std::vector<NnMatrix>& features, NnVector* weights, Bliss::Segment* segment) {
    if (Precursor::weightedAccumulation_ && weights)
        weights->initComputation();

    u32 nObservations = features[0].nColumns();
    this->setBatchSize(nObservations);

    network().forward(features);

    verify_eq(nObservations, network().getLayerInput(0)[0]->nColumns());
    verify_eq(nObservations, features[0].nColumns());

    weights_ = weights;

    if (Precursor::weightedAccumulation_) {
        if (!weights_) {
            Precursor::error("weighted FrameErrorEvaluator with no weights");
            return;
        }
    }
    else
        weights_ = NULL;
}

template<typename T>
void FrameErrorEvaluator<T>::processBatch_finishWithAlignment(Math::CudaVector<u32>& alignment) {
    alignment.initComputation();

    u32 nObservations = network().getLayerInput(0)[0]->nColumns();
    verify_eq(nObservations, alignment.size());

    Precursor::criterion_->inputAlignment(alignment, network().getTopLayerOutput(), weights_);
    if (Precursor::criterion_->discardCurrentInput()) {
        Core::Component::log("discard current mini-batch");
        return;
    }

    u32 batchFrameClassificationErrors = network().getTopLayerOutput().nClassificationErrors(alignment);
    T   batchObjectiveFunction         = 0;
    Precursor::criterion_->getObjectiveFunction(batchObjectiveFunction);

    T batchEntropy_ = 0;
    if (logFrameEntropy_) {
        Math::FastVector<T> entropy(nObservations);
        // not necessary once FastVector::columnEntropy() is implemented on GPU
        network().getTopLayerOutput().finishComputation(true);
        Math::FastMatrix<T>& output = network().getTopLayerOutput().asWritableCpuMatrix();
        entropy.columnEntropy(output);
        batchEntropy_ = entropy.sum();
        network().getTopLayerOutput().initComputation(false);
    }

    if (statisticsChannel_.isOpen()) {
        statisticsChannel_ << Core::XmlOpen("batch-statistics")
                           << Core::XmlFull("frame-classification-error-rate-on-batch", (T)batchFrameClassificationErrors / nObservations)
                           << Core::XmlFull("objective-function-on-batch", batchObjectiveFunction / nObservations);
        if (logFrameEntropy_)
            statisticsChannel_ << Core::XmlFull("average-entropy-on-batch", batchEntropy_ / nObservations);
        statisticsChannel_ << Core::XmlClose("batch-statistics");
    }
    nFrameClassificationErrors_ += batchFrameClassificationErrors;
    nObservations_ += nObservations;
    objectiveFunction_ += batchObjectiveFunction;
    frameEntropy_ += batchEntropy_;
}

template<typename T>
void FrameErrorEvaluator<T>::processBatch_finishWithSpeechSegment(Bliss::SpeechSegment& segment) {
    u32 nObservations = network().getLayerInput(0)[0]->nColumns();

    Precursor::criterion_->inputSpeechSegment(segment, network().getTopLayerOutput(), weights_);
    if (Precursor::criterion_->discardCurrentInput()) {
        Core::Component::log("discard current segment");
        return;
    }

    T batchObjectiveFunction = 0;
    Precursor::criterion_->getObjectiveFunction(batchObjectiveFunction);

    if (statisticsChannel_.isOpen()) {
        statisticsChannel_ << Core::XmlOpen("batch-statistics")
                           << Core::XmlFull("objective-function-on-batch", batchObjectiveFunction / nObservations);
        statisticsChannel_ << Core::XmlClose("batch-statistics");
    }

    nObservations_ += nObservations;
    objectiveFunction_ += batchObjectiveFunction;
}

template<typename T>
void FrameErrorEvaluator<T>::processBatch_finish() {
    u32 nObservations = network().getLayerInput(0)[0]->nColumns();

    Precursor::criterion_->input(network().getTopLayerOutput(), weights_);
    if (Precursor::criterion_->discardCurrentInput()) {
        Core::Component::log("discard current mini-batch");
        return;
    }

    T batchObjectiveFunction = 0;
    Precursor::criterion_->getObjectiveFunction(batchObjectiveFunction);

    if (statisticsChannel_.isOpen()) {
        statisticsChannel_ << Core::XmlOpen("batch-statistics")
                           << Core::XmlFull("objective-function-on-batch", batchObjectiveFunction / nObservations);
        statisticsChannel_ << Core::XmlClose("batch-statistics");
    }

    nObservations_ += nObservations;
    objectiveFunction_ += batchObjectiveFunction;
}

//=============================================================================

template<typename T>
const Core::ParameterString MeanAndVarianceTrainer<T>::paramMeanFile(
        "mean-file", "", "");

template<typename T>
const Core::ParameterString MeanAndVarianceTrainer<T>::paramStandardDeviationFile(
        "standard-deviation-file", "", "");

template<typename T>
const Core::ParameterString MeanAndVarianceTrainer<T>::paramStatisticsFile(
        "statistics-filename", "filename to write statistics to", "");

template<typename T>
MeanAndVarianceTrainer<T>::MeanAndVarianceTrainer(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config),
          statistics_(0),
          meanFile_(paramMeanFile(config)),
          standardDeviationFile_(paramStandardDeviationFile(config)),
          statisticsFile_(paramStatisticsFile(config)) {
    this->needsNetwork_ = false;
}

template<typename T>
MeanAndVarianceTrainer<T>::~MeanAndVarianceTrainer() {
    if (statistics_)
        delete statistics_;
}

template<typename T>
void MeanAndVarianceTrainer<T>::saveVector(std::string& filename, Math::Vector<T>& vector) {
    require(!filename.empty());
    // determine file suffix
    std::string suffix;
    if ((filename.length() >= 4) && (filename.substr(0, 4) == "bin:")) {
        suffix = ".bin";
    }
    else {
        suffix = ".xml";
    }
    // save the vector
    std::ostringstream type;
    if (typeid(T) == typeid(f32)) {
        type << "f32";
    }
    else if (typeid(T) == typeid(f64)) {
        type << "f64";
    }
    std::string newFilename = filename + "-" + type.str() + suffix;

    Math::Module::instance().formats().write(newFilename, vector, 20);
}

template<typename T>
void MeanAndVarianceTrainer<T>::initializeTrainer(u32 batchSize, std::vector<u32>& streamSizes) {
    Precursor::initializeTrainer(batchSize, streamSizes);
    if (streamSizes.size() != 1)
        Core::Component::criticalError("MeanAndVarianceTrainer only implemented for single input streams");

    statistics_ = new Statistics<T>(0, Statistics<T>::MEAN_AND_VARIANCE);
    statistics_->featureSum().resize(streamSizes[0]);
    statistics_->featureSum().setToZero();
    statistics_->squaredFeatureSum().resize(streamSizes[0]);
    statistics_->squaredFeatureSum().setToZero();
    statistics_->initComputation();
    tmp_.resize(streamSizes[0], batchSize);
    tmp_.initComputation();
    tmp_.setToZero();
}

template<typename T>
void MeanAndVarianceTrainer<T>::finalize() {
    statistics_->finishComputation();
    if (statisticsFile_ != "") {
        statistics_->write(statisticsFile_);
    }
}

template<typename T>
void MeanAndVarianceTrainer<T>::writeMeanAndStandardDeviation(Statistics<T>& statistics) {
    statistics.finalize(true);
    statistics.finishComputation();
    u32 dim = statistics.featureSum().size();
    mean_.resize(dim);
    standardDeviation_.resize(dim);
    for (u32 i = 0; i < dim; i++) {
        mean_.at(i)              = statistics.featureSum().at(i);
        standardDeviation_.at(i) = std::sqrt(statistics.squaredFeatureSum().at(i));
    }
    this->log("estimating mean and variance from ") << statistics.nObservations() << " observations";
    this->log("write mean vector to file: ") << meanFile_;
    saveVector(meanFile_, mean_);
    this->log("write standard deviation vector to file: ") << standardDeviationFile_;
    saveVector(standardDeviationFile_, standardDeviation_);
}

template<typename T>
void MeanAndVarianceTrainer<T>::processBatch_feedInput(std::vector<NnMatrix>& features, NnVector* weights, Bliss::Segment* segment) {
    if (Precursor::weightedAccumulation_ && weights)
        weights->initComputation();
    features[0].initComputation();
    if (features[0].nColumns() != tmp_.nColumns())
        tmp_.resize(tmp_.nRows(), features[0].nColumns());

    tmp_.copy(features[0]);

    // weight features
    if (Precursor::weightedAccumulation_ && weights)
        features[0].multiplyColumnsByScalars(*weights);

    // accumulate sum
    statistics_->featureSum().addSummedColumns(features[0]);

    // accumulate square sum
    tmp_.elementwiseMultiplication(tmp_);
    if (Precursor::weightedAccumulation_ && weights)
        tmp_.multiplyColumnsByScalars(*weights);
    statistics_->squaredFeatureSum().addSummedColumns(tmp_);

    // accumulate weight
    if (Precursor::weightedAccumulation_ && weights)
        statistics_->addToTotalWeight(weights->asum());
    else
        statistics_->addToTotalWeight(features[0].nColumns());
    statistics_->incObservations(features[0].nColumns());
}

//=============================================================================

template<typename T>
const Core::ParameterString NetworkEvaluator<T>::paramDumpPosteriors(
        "dump-posteriors", "cache file name", "");

template<typename T>
const Core::ParameterString NetworkEvaluator<T>::paramDumpBestPosteriorIndices(
        "dump-best-posterior-indices", "cache file name", "");

template<typename T>
NetworkEvaluator<T>::NetworkEvaluator(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config),
          nObservations_(0) {
    {
        std::string archiveFilename = paramDumpPosteriors(config);
        if (!archiveFilename.empty())
            dumpPosteriorsArchive_ = std::shared_ptr<Core::Archive>(Core::Archive::create(Core::Component::select(paramDumpPosteriors.name()),
                                                                                          archiveFilename,
                                                                                          Core::Archive::AccessModeWrite));
    }

    {
        std::string archiveFilename = paramDumpBestPosteriorIndices(config);
        if (!archiveFilename.empty())
            dumpBestPosterioIndicesArchive_ = std::shared_ptr<Core::Archive>(Core::Archive::create(Core::Component::select(paramDumpBestPosteriorIndices.name()),
                                                                                                   archiveFilename,
                                                                                                   Core::Archive::AccessModeWrite));
    }

    if (!dumpPosteriorsArchive_ && !dumpBestPosterioIndicesArchive_)
        Core::Component::warning("NetworkEvaluator: we don't dump anything");
}

template<typename T>
void NetworkEvaluator<T>::finalize() {
    Core::Component::log("total-observations: ") << nObservations_;
    network().finalize();
}

template<typename T>
void NetworkEvaluator<T>::processBatch_feedInput(std::vector<NnMatrix>& features, NnVector* weights, Bliss::Segment* segment) {
    u32 nObservations = features[0].nColumns();
    this->setBatchSize(nObservations);
    nObservations_ += nObservations;

    for (u32 i = 0; i < features.size(); ++i)
        features.at(i).initComputation();

    network().forward(features);

    verify_eq(nObservations, network().getLayerInput(0)[0]->nColumns());
    verify_eq(nObservations, network().getTopLayerOutput().nColumns());
}

template<typename T>
void NetworkEvaluator<T>::processBatch_finishWithSpeechSegment(Bliss::SpeechSegment& segment) {
    NnMatrix& networkOutput = network().getTopLayerOutput();
    networkOutput.finishComputation(true);

    u32 frameCount = networkOutput.nColumns();

    if (dumpPosteriorsArchive_) {
        Flow::ArchiveWriter<Math::Matrix<T>> writer(dumpPosteriorsArchive_.get());
        networkOutput.convert(writer.data_->data());
        writer.write(segment.fullName());
    }

    if (dumpBestPosterioIndicesArchive_) {
        Flow::ArchiveWriter<Math::Vector<u32>> writer(dumpBestPosterioIndicesArchive_.get());
        Math::Vector<u32>&                     bestEmissions = writer.data_->data();
        bestEmissions.resize(frameCount);
        for (u32 t = 0; t < frameCount; ++t) {
            u32 argMax   = 0;
            T   maxValue = networkOutput.at(argMax, t);
            for (u32 i = 1; i < networkOutput.nRows(); ++i) {
                T value = networkOutput.at(i, t);
                if (value > maxValue) {
                    maxValue = value;
                    argMax   = i;
                }
            }
            bestEmissions[t] = argMax;
        }
        writer.write(segment.fullName());
    }

    networkOutput.initComputation(false);
}

template<typename T>
void NetworkEvaluator<T>::processBatch_finish() {
    // The problem is that I don't know a good way to reference this.
    // The only good way is probably the segment name.
    Core::Component::error("NetworkEvaluator: not sure how to save this. use action = supervised-segmentwise-training.");
}

//=============================================================================
// explicit template instantiation
namespace Nn {

template class NeuralNetworkTrainer<f32>;
template class NeuralNetworkTrainer<f64>;

template class FrameErrorEvaluator<f32>;
template class FrameErrorEvaluator<f64>;

template class MeanAndVarianceTrainer<f32>;
template class MeanAndVarianceTrainer<f64>;

template class NetworkEvaluator<f32>;
template class NetworkEvaluator<f64>;

}  // namespace Nn
