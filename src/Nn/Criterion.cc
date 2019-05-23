/** Copyright 2018 RWTH Aachen University. All rights reserved.
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
#include "Criterion.hh"
#include <Core/Application.hh>
#include "CtcCriterion.hh"

namespace Nn {

template<typename FloatT>
const Core::Choice Criterion<FloatT>::choiceCriterion(
        "none", none,
        "cross-entropy", crossEntropy,
        "squared-error", squaredError,
        "binary-divergence", binaryDivergence,
        "ctc", ctc,
        Core::Choice::endMark());

template<typename FloatT>
const Core::ParameterChoice Criterion<FloatT>::paramCriterion(
        "training-criterion", &choiceCriterion,
        "training criterion for the neural network", none);

template<typename FloatT>
Criterion<FloatT>* Criterion<FloatT>::create(const Core::Configuration& config) {
    Criterion::Type criterion = (Criterion::Type)paramCriterion(config);
    switch (criterion) {
        case crossEntropy:
            Core::Application::us()->log("Create cross-entropy training criterion");
            return new CrossEntropyCriterion<FloatT>(config);
        case squaredError:
            Core::Application::us()->log("Create squared-error training criterion");
            return new SquaredErrorCriterion<FloatT>(config);
        case binaryDivergence:
            Core::Application::us()->log("Create binary-divergence training criterion");
            return new BinaryDivergenceCriterion<FloatT>(config);
        case ctc:
            Core::Application::us()->log("Create CTC training criterion");
            return new CtcCriterion<FloatT>(config);
        case none:
            Core::Application::us()->log("Create 'none' training criterion");
            return new Criterion<FloatT>(config);
    };
    Core::Application::us()->criticalError("Invalid criterion");
    return NULL;
}

template<typename FloatT>
void CrossEntropyCriterion<FloatT>::getObjectiveFunction(FloatT& value) {
    if (!Precursor::needRecalc_objectiveFunction_) {
        value = Precursor::objectiveFunction_;
        return;
    }

    require(Precursor::nnOutput_);
    require(Precursor::alignment_);
    if (Precursor::weights_)
        value = Precursor::nnOutput_->weightedCrossEntropyObjectiveFunction(*Precursor::alignment_, *Precursor::weights_);
    else
        value = Precursor::nnOutput_->crossEntropyObjectiveFunction(*Precursor::alignment_);
    Precursor::objectiveFunction_            = value;
    Precursor::needRecalc_objectiveFunction_ = false;
}

template<typename FloatT>
void CrossEntropyCriterion<FloatT>::getErrorSignal(NnMatrix& errorSignal) {
    Core::Component::criticalError("Generic CrossEntropyCriterion::getErrorSignal not implemented yet.");
}

template<typename FloatT>
void CrossEntropyCriterion<FloatT>::getErrorSignal_naturalPairing(NnMatrix& errorSignal, NeuralNetworkLayer<FloatT>& lastLayer) {
    require(Precursor::nnOutput_);
    require(Precursor::alignment_);  // not implemented yet without fixed alignment

    switch (lastLayer.getLayerType()) {
        case NeuralNetworkLayer<FloatT>::linearAndSoftmaxLayer:
        case NeuralNetworkLayer<FloatT>::softmaxLayer:
            // softmax - kronecker delta (minimization problem)
            errorSignal.setToZero();
            errorSignal.add(*Precursor::nnOutput_);
            errorSignal.addKroneckerDelta(*Precursor::alignment_, -1.0);
            break;
        default:
            Core::Component::criticalError("This layer-type is not yet implemented in training. "
                                           "Allowed types: softmax, linear+softmax.");
            break;
    }

    if (Precursor::weights_) {
        errorSignal.multiplyColumnsByScalars(*Precursor::weights_);
    }
}

template<typename FloatT>
void SquaredErrorCriterion<FloatT>::getObjectiveFunction(FloatT& value) {
    if (!Precursor::needRecalc_objectiveFunction_) {
        value = Precursor::objectiveFunction_;
        return;
    }

    require(Precursor::nnOutput_);
    require(Precursor::alignment_);
    if (Precursor::weights_)
        value = Precursor::nnOutput_->weightedSquaredErrorObjectiveFunction(
                *Precursor::alignment_, *Precursor::weights_);
    else
        value = Precursor::nnOutput_->squaredErrorObjectiveFunction(*Precursor::alignment_);
    Precursor::objectiveFunction_            = value;
    Precursor::needRecalc_objectiveFunction_ = false;
}

template<typename FloatT>
void SquaredErrorCriterion<FloatT>::getErrorSignal(NnMatrix& errorSignal) {
    Core::Component::criticalError("Generic SquaredErrorCriterion::getErrorSignal not implemented yet.");
}

template<typename FloatT>
void SquaredErrorCriterion<FloatT>::getErrorSignal_naturalPairing(NnMatrix& errorSignal, NeuralNetworkLayer<FloatT>& lastLayer) {
    require(Precursor::nnOutput_);
    require(Precursor::alignment_);  // not implemented yet without fixed alignment

    NnVector  tmp;
    NnMatrix& netOutput = *Precursor::nnOutput_;
    auto&     alignment = *Precursor::alignment_;

    switch (lastLayer.getLayerType()) {
        case NeuralNetworkLayer<FloatT>::linearLayer:
            errorSignal.setToZero();
            errorSignal.add(netOutput);
            errorSignal.addKroneckerDelta(alignment, -1.0);
            break;

        case NeuralNetworkLayer<FloatT>::linearAndSoftmaxLayer:
        case NeuralNetworkLayer<FloatT>::softmaxLayer:
            // (a) (softmax - kronecker-delta) .* softmax
            errorSignal.setToZero();
            errorSignal.add(netOutput);
            errorSignal.addKroneckerDelta(alignment, -1.0);
            errorSignal.elementwiseMultiplication(netOutput);
            // (b) store column sums in tmp vector
            tmp.initComputation();
            tmp.resize(errorSignal.nColumns(), 0, true);
            tmp.setToZero();
            tmp.addSummedRows(errorSignal);
            // (c) redefine error signal: softmax - kronecker-delta
            errorSignal.setToZero();
            errorSignal.add(netOutput);
            errorSignal.addKroneckerDelta(alignment, -1.0);
            // (d) subtract column sums and multiply with softmax
            errorSignal.addToAllRows(tmp, -1.0);
            errorSignal.elementwiseMultiplication(netOutput);
            break;

        default:
            Core::Component::criticalError(
                    "This layer-type is not yet implemented in training. "
                    "Allowed types: linear, softmax, linear+softmax.");
            break;
    }

    if (Precursor::weights_)
        errorSignal.multiplyColumnsByScalars(*Precursor::weights_);
}

template<typename FloatT>
void BinaryDivergenceCriterion<FloatT>::getObjectiveFunction(FloatT& value) {
    if (!Precursor::needRecalc_objectiveFunction_) {
        value = Precursor::objectiveFunction_;
        return;
    }

    require(Precursor::nnOutput_);
    require(Precursor::alignment_);
    if (Precursor::weights_)
        value = Precursor::nnOutput_->weightedBinaryDivergenceObjectiveFunction(
                *Precursor::alignment_, *Precursor::weights_);
    else
        value = Precursor::nnOutput_->binaryDivergenceObjectiveFunction(*Precursor::alignment_);
    Precursor::objectiveFunction_            = value;
    Precursor::needRecalc_objectiveFunction_ = false;
}

template<typename FloatT>
void BinaryDivergenceCriterion<FloatT>::getErrorSignal(NnMatrix& errorSignal) {
    Core::Component::criticalError("Generic BinaryDivergenceCriterion::getErrorSignal not implemented yet.");
}

template<typename FloatT>
void BinaryDivergenceCriterion<FloatT>::getErrorSignal_naturalPairing(NnMatrix& errorSignal, NeuralNetworkLayer<FloatT>& lastLayer) {
    require(Precursor::nnOutput_);
    require(Precursor::alignment_);  // not implemented yet without fixed alignment

    NnMatrix& netOutput = *Precursor::nnOutput_;
    auto&     alignment = *Precursor::alignment_;

    switch (lastLayer.getLayerType()) {
        case NeuralNetworkLayer<FloatT>::linearAndSigmoidLayer:
        case NeuralNetworkLayer<FloatT>::sigmoidLayer:
            errorSignal.setToZero();
            errorSignal.add(netOutput);
            errorSignal.addKroneckerDelta(alignment, -1.0);
            break;

        case NeuralNetworkLayer<FloatT>::linearAndSoftmaxLayer:
        case NeuralNetworkLayer<FloatT>::softmaxLayer:
            errorSignal.binaryDivergenceSoftmaxGradient(netOutput, alignment);
            break;

        default:
            Core::Component::criticalError(
                    "This layer-type is not yet implemented in training. "
                    "Allowed types: sigmoid, linear+sigmoid.");
            break;
    }

    if (Precursor::weights_)
        errorSignal.multiplyColumnsByScalars(*Precursor::weights_);
}

// explicit template instantiation
template class Criterion<f32>;
template class Criterion<f64>;

}  // namespace Nn
