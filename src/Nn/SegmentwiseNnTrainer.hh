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
#ifndef _NN_SEGMENTWISENNTRAINER_HH_
#define _NN_SEGMENTWISENNTRAINER_HH_

#include <Core/Channel.hh>
#include <Core/Hash.hh>  // for mix2phoneme map ( lattice coverage statistics)
#ifndef CMAKE_DISABLE_MODULES_HH
#include <Modules.hh>
#endif
#include <Speech/AcousticSegmentwiseTrainer.hh>

#include "ActivationLayer.hh"
#include "LatticeAccumulators.hh"
#include "NeuralNetwork.hh"
#include "NeuralNetworkTrainer.hh"
#include "Prior.hh"
#include "Types.hh"

#ifdef MODULE_PYTHON
#include <Nn/PythonControl.hh>
#endif

// forward declarations
namespace Speech {
struct PosteriorFsa;
}

namespace Nn {
template<typename T>
class Estimator;
template<typename T>
class Statistics;
template<typename T>
class Regulizer;
class ClassLabelWrapper;
template<typename T>
class LinearAndSoftmaxLayer;
}  // namespace Nn

namespace Nn {

/**
 *  SegmentwiseNnTrainer
 *
 *  performs sequence-discriminative training in lattice-based framework
 *  In order to implement a specific criterion, derive from SegmentwiseNnTrainer and implement computeInitialErrorSignal.
 *
 */
// TODO code duplication, use a common base class for SegmentwiseNnTrainer and FeedForwardTrainer (sth like AlignedNeuralNetworkTrainer)
// TODO implement double precision / T denotes the type of the statistics object, everything else is single precision
template<typename T>
class SegmentwiseNnTrainer : public NeuralNetworkTrainer<f32>, public Speech::AbstractAcousticSegmentwiseTrainer {
    typedef Speech::AbstractAcousticSegmentwiseTrainer SpeechPrecursor;
    typedef typename Types<f32>::NnVector              NnVectorf32;
    typedef typename Types<f32>::NnMatrix              NnMatrixf32;
    typedef typename Types<f64>::NnVector              NnVectorf64;
    typedef typename Types<f64>::NnMatrix              NnMatrixf64;

protected:
    // inherited from neural network trainer
    using NeuralNetworkTrainer<f32>::statisticsChannel_;
    using NeuralNetworkTrainer<f32>::needInit_;
    using NeuralNetworkTrainer<f32>::estimator_;
    using NeuralNetworkTrainer<f32>::regularizer_;
    // shared network passed from the LatticeProcessor
    using NeuralNetworkTrainer<f32>::network_;
    using NeuralNetworkTrainer<f32>::measureTime_;

public:
    using NeuralNetworkTrainer<f32>::regularizer;
    using NeuralNetworkTrainer<f32>::estimator;

protected:
    static const Core::ParameterString paramStatisticsFilename;
    static const Core::ParameterFloat  paramSilenceWeight;
    static const Core::ParameterString paramClassWeightsFile;
    static const Core::ParameterFloat  paramCeSmoothingWeight;
    static const Core::ParameterFloat  paramFrameRejectionThreshold;
    static const Core::ParameterBool   paramAccumulatePrior;
    static const Core::ParameterBool   paramEnableFeatureDescriptionCheck;
    const std::string                  statisticsFilename_;
    const f32                          ceSmoothingWeight_;
    const f32                          frameRejectionThreshold_;
    const bool                         accumulatePrior_;
    bool                               singlePrecision_;

protected:
    Statistics<T>*   statistics_;
    Statistics<f32>* priorStatistics_;
    // additional statistics
    u32 numberOfProcessedSegments_;
    u32 numberOfObservations_;
    u32 numberOfRejectedObservations_;  // frames rejected according to frame rejection heuristic
    T   ceObjectiveFunction_;           // cross entropy objective function
    T   localObjectiveFunction_;        // objective function of segment
    T   localCeObjectiveFunction_;      // cross-entropy objective function of segment
    u32 localClassificationErrors_;

    // error signals and error signal accumulator
    std::vector<NnMatrixf32>     errorSignal_;
    ErrorSignalAccumulator<f32>* accumulator_;

    // alignment of current segment etc.
    bool                  segmentNeedsInit_;
    Math::CudaVector<u32> alignment_;
    NnVectorf32           weights_;  // accumulation weights for alignment
    u32                   sequenceLength_;

    Math::Vector<f32> classWeights_;  // accumulation weights for each class

    LinearAndSoftmaxLayer<f32>* topLayer_;     // required for application of softmax
    MaxoutVarLayer<f32>*        maxoutLayer_;  // required for application of softmax with hidden variable (maximum approximation)
    NnVectorf32                 prior_;
    f32                         priorScale_;

    // feature description
    Mm::FeatureDescription featureDescription_;
    bool                   featureDescriptionNeedInit_;
    bool                   enableFeatureDescriptionCheck_;

#ifdef MODULE_PYTHON
    Nn::PythonControl pythonControl_;
#endif

private:
    f64 timeMemoryAllocation_;
    f64 timeNumeratorExtraction_;
    f64 timeAlignmentVector_;
    f64 timeErrorSignal_;
    f64 timeCESmoothing_;
    f64 timeBackpropagation_;
    f64 timeGradient_;
    f64 timeBaseStatistics_;
    f64 timeEstimationStep;
    f64 timeSync_;

public:
    SegmentwiseNnTrainer(const Core::Configuration& config);
    virtual ~SegmentwiseNnTrainer();

public:
    // called for every segment
    virtual void processWordLattice(Lattice::ConstWordLatticeRef l, Bliss::SpeechSegment* s);
    // needs to be implemented for AbstractAcousticSegmentwiseTrainer
    virtual void setFeatureDescription(const Mm::FeatureDescription&);
    // LatticeSetProcessor function: calls finalize if done
    virtual void leaveCorpus(Bliss::Corpus* corpus);

protected:
    // initialization and finalization functions
    virtual void initializeTrainer();
    virtual bool isInitialized() const {
        return !needInit_;
    }

    // finalize training epoch
    virtual void finalize();

    // initialize segment
    virtual void initSegment(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* segment);

    // this abstract function needs to be implemented for the corresponding criterion
    // error signal is computed from the lattice, objectiveFunction is set by function
    // return value specifies whether computation was successful
    virtual bool computeInitialErrorSignal(Lattice::ConstWordLatticeRef lattice,
                                           Lattice::ConstWordLatticeRef numeratorLattice,
                                           Bliss::SpeechSegment*        segment,
                                           T&                           objectiveFunction,
                                           bool                         objectiveFunctionOnly) = 0;
    // pass over lattice and collect statistics (depth first search)
    virtual void accumulateStatisticsOnLattice(Fsa::ConstAutomatonRef                   posterior,
                                               Core::Ref<const Lattice::WordBoundaries> wb,
                                               Mm::Weight                               factor);

    // create lattice accumulator (which passes over lattice)
    virtual NnAccumulator* createAccumulator(Mm::Weight factor, Mm::Weight weightThreshold) const;

    // logging
    virtual void logProperties() const;
    virtual void logSegmentStatistics() const;
    virtual void logTrainingStatistics() const;
    virtual void logProfilingStatistics() const;

    NeuralNetwork<f32>&      network() const;
    const ClassLabelWrapper& labelWrapper() const;

    virtual std::string name() const {
        return "nn-seq-accumulator";
    }

private:
    void setPrecision();
    // set class weights (either from file or silence-weight)
    void setClassWeights();  // TODO code duplication
    // get alignment in vector format, alignment is extracted from numerator lattice ( = orthography lattice)
    // note: numerator lattice is in general NOT linear -> extract best path from lattice
    // returns true if successful
    bool getAlignmentVector(Lattice::ConstWordLatticeRef numeratorLattice);
    // accumulate class counts (assumes alignment is set)
    void accumulatePrior();
    // accumulate base statistics
    void accumulateBaseStatistics();
    // backpropagation of error signal
    void backpropagateError();  // TODO code duplication
    // smooth error signal with CE criterion
    // side effects: log-priors are added to scores, softmax is applied
    // returns CE objective function
    T smoothErrorSignalWithCE();
    // compute gradient from error signals and activations
    void collectGradient();  // TODO code duplication
    // resize error signal to sequence length
    void resizeErrorSignal();
    // update model according to statistics (e.g. gradient for SGD)
    // different implementation for double and single precision statistics
    // single precision statistics: perform estimation step for stochastic optimization
    // double precision statistics: no stochastic optimization possible
    void updateModel();

public:
    static SegmentwiseNnTrainer* createSegmentwiseNnTrainer(const Core::Configuration&);
};

template<>
inline void SegmentwiseNnTrainer<f32>::setPrecision() {
    singlePrecision_ = true;
}

template<>
inline void SegmentwiseNnTrainer<f64>::setPrecision() {
    singlePrecision_ = false;
    if (!estimator().fullBatchMode()) {
        this->error("current implementation only uses double precision for storing accumulated statisitcs in batch mode!");
    }
}

template<>
inline void SegmentwiseNnTrainer<f32>::updateModel() {
    timeval start, end;
    TIMER_START(start)
    estimator().estimate(network(), *statistics_);
    TIMER_GPU_STOP(start, end, measureTime_, timeEstimationStep)
}

template<>
inline void SegmentwiseNnTrainer<f64>::updateModel() {
    this->criticalError("stochastic optimization with double precision statistics not possible");
    verify(false);
}

}  // namespace Nn

#endif
