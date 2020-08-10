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
#ifndef _NN_CRITERION_HH
#define _NN_CRITERION_HH

#include <Core/Component.hh>
#include <memory>
#include <vector>
#include "NeuralNetworkLayer.hh"
#include "Types.hh"

namespace Bliss {
class SpeechSegment;
}

/*
The NnTrainer tool will use the Buffered*Feature(Extractor|Processor),
which extracts features for a mini-batch or segment,
and also might provide the alignment for a segment.

This calls the NeuralNetworkTrainer, which has an instance of Criterion.

That could be to train the network (FeedForwardTrainer),
but also to calculate mean-and-variance (MeanAndVarianceTrainer),
or to evaluate the frame-error-eval (FrameErrorEvaluator).

The network training must forward the network for some input,
and then calculate the error signal + objective function via criterion,
and then backprop + collect gradients + estimate new network.

Some criterions must know the alignment (current implementation of CE, etc.),
some must operate on full segments (CTC).
In some cases (e.g. CTC), it needs the segment transcription.

Mean-and-variance: custom trainer, no criterion, weights (via alignment).
Frame-error-eval: custom trainer, aligned criterion.
Aligned-trainer: aligned criterion.
Segment-basic-trainer: no lattice, no alignment, segment, generic criterion.
Segment-lattice-trainer: -> MMI/ME/MPE, see SegmentwiseNnTrainer.

*/

namespace Nn {

template<typename FloatT>
class Criterion : public Core::Component {
public:
    enum Type {
        none,
        crossEntropy,
        squaredError,
        binaryDivergence,
        ctc,
    };

protected:
    typedef typename Types<FloatT>::NnVector NnVector;
    typedef typename Types<FloatT>::NnMatrix NnMatrix;

    NnMatrix*              nnOutput_;
    NnVector*              weights_;  // column weights; usually provided together and extracted from alignment
    Math::CudaVector<u32>* alignment_;
    Bliss::SpeechSegment*  segment_;  // might contain the reference transcription

    FloatT   objectiveFunction_;
    bool     needRecalc_objectiveFunction_;
    NnMatrix errorSignal_;
    bool     needRecalc_errorSignal_;
    Type     criterionType_;

public:
    Criterion(const Core::Configuration& c)
            : Core::Component(c),
              nnOutput_(NULL),
              weights_(NULL),
              alignment_(NULL),
              segment_(NULL),
              needRecalc_objectiveFunction_(true),
              needRecalc_errorSignal_(true) {}

    // Interface:
    // First call one (and only one) of the input* functions, then you can read
    // out the objective function value and error signal via
    // the get* functions.

    /** Override for an unsupervised criterion */
    virtual void input(NnMatrix& nnOutput, NnVector* weights = NULL) {
        nnOutput_                     = &nnOutput;
        weights_                      = weights;
        needRecalc_objectiveFunction_ = true;
        needRecalc_errorSignal_       = true;
    }

    /** Override for an alignment-based criterion (usually frame-wise, e.g. Cross-Entropy) */
    virtual void inputAlignment(Math::CudaVector<u32>& alignment, NnMatrix& nnOutput, NnVector* weights = NULL) {
        alignment_ = &alignment;
        input(nnOutput, weights);
    }

    // Note: Some inputWithReference() or so would also make sense here, which would be
    // a generic variant of inputAlignment().

    /** Override for a segment (without lattice) criterion (e.g. CTC) */
    virtual void inputSpeechSegment(Bliss::SpeechSegment& segment, NnMatrix& nnOutput, NnVector* weights = NULL) {
        segment_ = &segment;
        input(nnOutput, weights);
    }

    /** Override to calculate the objective function. Some input* function was called before. */
    virtual void getObjectiveFunction(FloatT& value) {
        value = 0;
    }

    /** Override to calculate the error signal. Some input* function was called before.
     * This is thus the error signal of the NN output.
     */
    virtual void getErrorSignal(NnMatrix& errorSignal) {
        errorSignal.setToZero();
    }

    /** Override to calculate the error signal with natural pairing of the last layer activation function.
     * This is thus the error signal of the linear part of the last layer.
     * This function is usually used in frame-wise training with a fixed alignment,
     * via BufferedAlignedFeatureProcessor.
     */
    virtual void getErrorSignal_naturalPairing(NnMatrix& errorSignal, NeuralNetworkLayer<FloatT>& lastLayer) {
        // This default implementation just uses the backprop implementation of the layer
        // and the standard getErrorSignal().
        // We don't do the weighting here (if we have weights) because we expect that
        // getErrorSignal() does it already, and the backprop is multiplicative, thus
        // the weighting is just passed down to us.
        verify(nnOutput_);
        NnMatrix imtmErrorSignal(errorSignal.nRows(), errorSignal.nColumns());
        imtmErrorSignal.initComputation(false);
        imtmErrorSignal.setToZero();
        getErrorSignal(imtmErrorSignal);
        lastLayer.backpropagateActivations(imtmErrorSignal, errorSignal, *nnOutput_);
    }

    /** Overrwrite if you want to signal any trainer to ignore this input. */
    virtual bool discardCurrentInput() {
        return false;
    }

    virtual void reset() {
        nnOutput_                     = NULL;
        weights_                      = NULL;
        alignment_                    = NULL;
        segment_                      = NULL;
        needRecalc_objectiveFunction_ = true;
        needRecalc_errorSignal_       = true;
    }

    // Calls the input* function again, which was called last time,
    // with all the same parameters, except a new nnOutput.
    virtual void reinputWithNewNnOutput(NnMatrix& nnOutput) {
        needRecalc_objectiveFunction_ = true;
        needRecalc_errorSignal_       = true;
        if (alignment_) {
            segment_ = NULL;
            inputAlignment(*alignment_, nnOutput, weights_);
        }
        else if (segment_) {
            alignment_ = NULL;
            inputSpeechSegment(*segment_, nnOutput, weights_);
        }
        else {
            alignment_ = NULL;
            segment_   = NULL;
            input(nnOutput, weights_);
        }
    }

    /** Some criterions will calculate some kind of pseudo targets
     * and have the gradient -\hat{y} / y w.r.t. y, and when y = softmax(a),
     * they have the gradient y - \hat{y} w.r.t. a.
     * In that case, we call \hat{y} the pseudo targets.
     * Examples: Cross-Entropy and CTC.
     * Can return NULL. Otherwise the pointer will be valid until the next input.
     */
    virtual NnMatrix* getPseudoTargets() {
        return NULL;
    }

    virtual Type getType() const {
        return criterionType_;
    }

    // --------------------------------

    static const Core::Choice          choiceCriterion;
    static const Core::ParameterChoice paramCriterion;

    static Criterion<FloatT>* create(const Core::Configuration& config);
};

template<typename FloatT>
class CrossEntropyCriterion : public Criterion<FloatT> {
    typedef Criterion<FloatT> Precursor;

protected:
    typedef typename Types<FloatT>::NnVector NnVector;
    typedef typename Types<FloatT>::NnMatrix NnMatrix;

public:
    CrossEntropyCriterion(const Core::Configuration& c)
            : Precursor(c) {
        Precursor::criterionType_ = Criterion<FloatT>::crossEntropy;
    }

    virtual void input(NnMatrix& nnOutput, NnVector* weights) {
        // Note: That is a limitation in the current implementation.
        // The current implementation only works via alignment.
        if (!Precursor::alignment_)
            this->criticalError("CrossEntropyCriterion is not unsupervised, it needs an alignment");
        Precursor::input(nnOutput, weights);
    }

    virtual void getObjectiveFunction(FloatT& value);
    virtual void getErrorSignal(NnMatrix& errorSignal);
    virtual void getErrorSignal_naturalPairing(NnMatrix& errorSignal, NeuralNetworkLayer<FloatT>& lastLayer);
};

template<typename FloatT>
class SquaredErrorCriterion : public Criterion<FloatT> {
    typedef Criterion<FloatT> Precursor;

protected:
    typedef typename Types<FloatT>::NnVector NnVector;
    typedef typename Types<FloatT>::NnMatrix NnMatrix;

public:
    SquaredErrorCriterion(const Core::Configuration& c)
            : Precursor(c) {
        Precursor::criterionType_ = Criterion<FloatT>::squaredError;
    }

    virtual void input(NnMatrix& nnOutput, NnVector* weights) {
        // Note: That is a limitation in the current implementation.
        // The current implementation only works via alignment.
        if (!Precursor::alignment_)
            this->criticalError("SquaredErrorCriterion is not unsupervised, it needs an alignment");
        Precursor::input(nnOutput, weights);
    }

    virtual void getObjectiveFunction(FloatT& value);
    virtual void getErrorSignal(NnMatrix& errorSignal);
    virtual void getErrorSignal_naturalPairing(NnMatrix& errorSignal, NeuralNetworkLayer<FloatT>& lastLayer);
};

template<typename FloatT>
class BinaryDivergenceCriterion : public Criterion<FloatT> {
    typedef Criterion<FloatT> Precursor;

protected:
    typedef typename Types<FloatT>::NnVector NnVector;
    typedef typename Types<FloatT>::NnMatrix NnMatrix;

public:
    BinaryDivergenceCriterion(const Core::Configuration& c)
            : Precursor(c) {
        Precursor::criterionType_ = Criterion<FloatT>::binaryDivergence;
    }

    virtual void input(NnMatrix& nnOutput, NnVector* weights) {
        // Note: That is a limitation in the current implementation.
        // The current implementation only works via alignment.
        if (!Precursor::alignment_)
            this->criticalError("BinaryDivergenceCriterion is not unsupervised, it needs an alignment");
        Precursor::input(nnOutput, weights);
    }

    virtual void getObjectiveFunction(FloatT& value);
    virtual void getErrorSignal(NnMatrix& errorSignal);
    virtual void getErrorSignal_naturalPairing(NnMatrix& errorSignal, NeuralNetworkLayer<FloatT>& lastLayer);
};

template<typename FloatT>
class SegmentCriterion : public Criterion<FloatT> {
    typedef Criterion<FloatT> Precursor;

protected:
    typedef typename Types<FloatT>::NnVector NnVector;
    typedef typename Types<FloatT>::NnMatrix NnMatrix;

public:
    SegmentCriterion(const Core::Configuration& c)
            : Precursor(c) {
        Precursor::criterionType_ = Criterion<FloatT>::ctc;
    }

    virtual void input(NnMatrix& nnOutput, NnVector* weights = NULL) {
        // Only allow inputSpeechSegment() calls.
        if (!Precursor::segment_)
            this->criticalError("SegmentCriterion needs a segment");
        Precursor::input(nnOutput, weights);
    }

    virtual void inputSpeechSegment(Bliss::SpeechSegment& segment, NnMatrix& nnOutput, NnVector* weights = NULL) {}
};

}  // namespace Nn

#endif  // CRITERION_HH
