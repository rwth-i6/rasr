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
#ifndef _NN_NEURAL_NETWORK_FEED_FORWARD_TRAINER_HH
#define _NN_NEURAL_NETWORK_FEED_FORWARD_TRAINER_HH

#include <Core/Component.hh>
#include <Math/CudaMatrix.hh>
#include <Math/CudaVector.hh>
#include <Modules.hh>
#include <cstring>

#include "BufferedAlignedFeatureProcessor.hh"
#include "NeuralNetwork.hh"
#include "NeuralNetworkTrainer.hh"
#include "Regularizer.hh"
#include "Statistics.hh"

#ifdef MODULE_PYTHON
#include <Nn/PythonControl.hh>
#endif

namespace Nn {

//=============================================================================

/**
 * Base class for all supervised feed forward trainer.
 * Note that frame-wise training will probably use FeedForwardAlignedTrainer.
 */
template<class T>
class FeedForwardTrainer : public NeuralNetworkTrainer<T> {
    typedef NeuralNetworkTrainer<T> Precursor;

protected:
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;
    using Precursor::measureTime_;
    using Precursor::statisticsChannel_;
    using Precursor::weightedAccumulation_;

public:
    using Precursor::criterion;
    using Precursor::estimator;
    using Precursor::network;
    using Precursor::regularizer;

protected:
    static const Core::ParameterString paramStatisticsFilename;
    static const Core::ParameterBool   paramDoublePrecisionAccumulator;
    static const Core::ParameterBool   paramNormalizeByNOfObservations;
    static const Core::ParameterFloat  paramErrorSignalClip;
    static const Core::ParameterBool   paramLogFrameEntropy;

protected:
    const std::string statisticsFilename_;
    const bool        useDoublePrecisionAccumulator_;
    Statistics<T>*    statistics_;
    Statistics<f64>*  doublePrecisionStatistics_;
    // }
    bool normalizeByNOfObservations_;
    T    errorSignalClip_;
    bool logFrameEntropy_;
    // for each layer an error signal
    std::vector<NnMatrix> errorSignal_;
    // for each layer l: vector of pointers to error signals of layers k which have a connection from k to l
    std::vector<std::vector<NnMatrix*>> errorSignalOut_;
    s32                                 lowestTrainableLayerIndex_;
    NnVector*                           weights_;  // weights of last feedInput call
    double                              timeSync_;
    double                              timeForwardPass_;
    double                              timeInitialErrorSignal_;
    double                              timeBackwardPass_;
    double                              timeGradient_;
    double                              timeBaseStatistics_;
    double                              timeRegularization_;
    double                              timeEstimation_;
    double                              timeSyncBatch_;
    double                              timeForwardPassBatch_;
    double                              timeInitialErrorSignalBatch_;
    double                              timeBackwardPassBatch_;
    double                              timeGradientBatch_;
    double                              timeBaseStatisticsBatch_;
    double                              timeRegularizationBatch_;
    double                              timeEstimationBatch_;
    u32                                 minibatchCount_;
    u32                                 discardedMinibatchCount_;
#ifdef MODULE_PYTHON
    Nn::PythonControl pythonControl_;
#endif
public:
    FeedForwardTrainer(const Core::Configuration& config);
    virtual ~FeedForwardTrainer();

    // initialization and finalization methods
    virtual void initializeTrainer(u32 batchSize) {
        Precursor::initializeTrainer(batchSize);
    }
    virtual void initializeTrainer(u32 batchSize, std::vector<u32>& streamSizes);
    virtual void finalize();

    /** returns reference to statistics */
    Statistics<T>& statistics() {
        require(statistics_);
        return *statistics_;
    }
    /** feed forward */
    virtual void processBatch_feedInput(std::vector<NnMatrix>& features, NnVector* weights, Bliss::Segment* segment);
    /** backprop + update model */
    virtual void processBatch_finishWithError_naturalPairing(T error, NnMatrix& errorSignal);
    /** process mini-batch of aligned features */
    virtual void processBatch_finishWithAlignment(Math::CudaVector<u32>& alignment);
    /** process segment */
    virtual void processBatch_finishWithSpeechSegment(Bliss::SpeechSegment& segment);
    void         processBatch_finishDiscard();

protected:
    // backpropagate error signal
    void errorBackpropagation();
    // compute gradient from error signal and activations
    void collectGradient();
    // count classes (only used via processBatch_finishWithAlignment)
    void updateClassCounts(const Math::CudaVector<u32>& alignment, Statistics<T>& statistics);
    // resize activations and error signal
    virtual void setBatchSize(u32 batchSize);
    // initialized double precision accumulator
    virtual void initializeDoublePrecisionStatistics();

    // per-batch time measurements
    virtual void resetBatchTimes();

public:
    virtual void logBatchTimes() const;
};

template<>
inline void FeedForwardTrainer<f32>::initializeDoublePrecisionStatistics() {
    u32 statisticsType = estimator().requiredStatistics();
    if (estimator().fullBatchMode() || statisticsChannel_.isOpen())
        statisticsType |= Statistics<f32>::BASE_STATISTICS;
    doublePrecisionStatistics_ = new Statistics<f64>(network().nLayers(), statisticsType);
    doublePrecisionStatistics_->copyStructure(statistics());
    doublePrecisionStatistics_->initComputation();
}

template<>
inline void FeedForwardTrainer<f64>::initializeDoublePrecisionStatistics() {
    this->warning("option \"double-precision-accumulator\" does not have an effect, because "
                  "double precision is already used for all neural network computations");
}

//=============================================================================

/*
 * Autoencoder.
 */

template<class T>
class FeedForwardAutoTrainer : public FeedForwardTrainer<T> {
    typedef FeedForwardTrainer<T> Precursor;

public:
    using Precursor::estimator;
    using Precursor::network;
    using Precursor::regularizer;
    using Precursor::statistics;

protected:
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;

    using Precursor::errorSignal_;
    using Precursor::statistics_;
    using Precursor::statisticsChannel_;
    using Precursor::statisticsFilename_;
    static const Core::ParameterString paramReferenceInputLayer;
    static const Core::ParameterInt    paramReferenceInputLayerPort;

    virtual void processBatch_feedInput(std::vector<NnMatrix>& features, NnVector* weights, Bliss::Segment* segment);
    // virtual void processBatch_finishWithSpeechSegment(Bliss::SpeechSegment& segment);
    virtual void processBatch_finishWithAlignment(Math::CudaVector<u32>& alignment);
    virtual void processBatch_finish();
    virtual bool needsToProcessAllFeatures() const {
        return true;
    }

    virtual void initializeTrainer(u32 batchSize, std::vector<u32>& streamSizes) {
        Precursor::initializeTrainer(batchSize, streamSizes);
    }
    virtual void finalize() {
        Precursor::finalize();
    }

public:
    FeedForwardAutoTrainer(const Core::Configuration& config);
    virtual ~FeedForwardAutoTrainer() {}

private:
    std::string referenceInputLayer_;
    int         referenceInputLayerPort_;
};

//=============================================================================

}  // namespace Nn

#endif
