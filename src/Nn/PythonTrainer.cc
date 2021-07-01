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
/*
The PythonTrainer is a generic Python bridge. It creates a Python interpreter
via the standard CPython API.

It reuses the NeuralNetworkTrainer interface so that we can use this as the main
trainer class in the NnTrainer tool.
You would just specify *.trainer = python-trainer.
action, buffer-type and training-criterion would be used as always,
as well as your Flow configuration.
Sprint will calculate the features, and the PythonTrainer then
gives it to Python and expects to get posteriors back
(just like from forwarding through an NN).
It then calculates the training-criterion with the error signal,
and gives this again to Python. Python now could do
back-propagation.

It is as generic as possible. We can use it for segmentwise, segmentwise+alignment
or unsupervised training - that implies what function gets called here, i.e.
processBatch_finish${training-type}. (See NeuralNetworkTrainer for reference.)
For the input, processBatch_feedInput gets called. This is forwarded to Python as-is,
and we expect that Python forwards this through a NN, and we get the NN output back
from Python, which we interpret as emission label posterior probabilities.
Then, processBatch_finish${training-type} gets called and we calculate the criterium
as it is set in the NeuralNetworkTrainer. This gives us the error and error signal,
which we again forward to Python as-is. Python is expected now to do the backprop
and the training itself, i.e. parameter update / estimation.

You can also calculate the error signal with natural pairing of a predefined
activation function. E.g. *.natural-pairing-layer.layer-type = softmax.

It loads a Python module (pymod-name; add path via pymod-path if needed).
It expects these general functions in the Python module:

    def init(inputDim, outputDim, config, targetMode, cudaEnabled, cudaActiveGpu)  # called in initializeTrainer()
    def exit()  # called in finalize()

In case that Sprint calculates the error (target-mode = criterion-by-sprint), we use this API:

    def feedInput(features, weights = None, segmentName = None) -> numpy.ndmatrix
    def finishDiscard()  # called if we shall discard this minibatch
    def finishError(error, errorSignal, naturalPairingType = None)

In the generic target case (target-mode = target-generic), we use this API:

    def feedInputAndTarget(features, weights=None, segmentName=None, **kwargs)
    # kwargs can include: alignment, orthography, speaker_name, speaker_gender, language

In case that we have target alignments (target-mode = target-alignment), we use this API:

    def feedInputAndTargetAlignment(features, targetAlignment, weights = None, segmentName = None)

In case of target segment orthography (target-mode = target-segment-orth):

    def feedInputAndTargetSegmentOrth(features, targetSegmentOrth, weights = None, segmentName = None)

In the unsupervised case (target-mode = unsupervised), we use this API:

    def feedInputUnsupervised(features, weights = None, segmentName = None)

In the forwarding-case (target-mode = forward-only), we use:

    def feedInputForwarding(features, weights = None, segmentName = None) -> numpy.ndmatrix

features and errorSignal are Numpy matrices where the colums represent the time frames.
weights is optional and can be a Numpy vector to weight each time frame.
segmentName is given if the underlying feature extractor is in segment-wise mode. Don't count on this.
targetAlignment is a Numpy int32 array and its entries represents the indices of the target.
targetSegmentOrth is a string.
feedInput() is expected to return a Numpy matrix (float or double) which should represent
the posteriors or any other output which is sane to be calculated with the training-criterium.
error is a scalar, i.e. Python float.
naturalPairingType is optional and can be a string like "softmax".

*/

#include "PythonTrainer.hh"
#include <Am/Module.hh>
#include <Bliss/Lexicon.hh>
#include <Core/Utility.hh>
#include <Math/CudaDataStructure.hh>
#include <Python/Init.hh>
#include <Python/Numpy.hh>
#include <Python/Utilities.hh>
#include <Python.h>
#include "ActivationLayer.hh"

#include <Flow/ArchiveWriter.hh>

const Core::Choice choiceTargetMode(
        "criterion-by-sprint", Nn::PythonTrainer<f32>::criterionBySprint,
        "target-generic", Nn::PythonTrainer<f32>::targetGeneric,
        "target-alignment", Nn::PythonTrainer<f32>::targetAlignment,
        "target-segment-orth", Nn::PythonTrainer<f32>::targetSegmentOrth,
        "unsupervised", Nn::PythonTrainer<f32>::unsupervised,
        "forward-only", Nn::PythonTrainer<f32>::forwardOnly,
        Core::Choice::endMark());

static const Core::ParameterChoice paramTargetMode(
        "target-mode", &choiceTargetMode,
        "Whether Sprint calculates the criterion and only passes the error signal, "
        "or if we just pass the target alignment/reference to Python.",
        Nn::PythonTrainer<f32>::targetAlignment);

static const Core::ParameterInt paramOutputDim(
        "trainer-output-dimension", "", 0);

static const Core::ParameterBool paramUseNetwork(
        "use-network",
        "Pass the features to the Sprint NeuralNetwork and use these "
        "outputs as the features for the Python feedInput(). "
        "You can use this for example to do feature-normalization.",
        false);

static const Core::ParameterString paramPyModPath(
        "pymod-path", "the path containing the Python module", "");

static const Core::ParameterString paramPyModName(
        "pymod-name", "the module-name, such that 'import x' would work", "");

static const Core::ParameterString paramPyModConfig(
        "pymod-config",
        "config-string, passed to init()",
        "");

static const Core::ParameterBool paramAllowDownsampling(
        "trainer-allow-downsampling", "the network is allowed to return less timeframes than there are feature-vectors", false);

namespace Nn {

template<typename T>
PythonTrainer<T>::PythonTrainer(const Core::Configuration& config)
        : Core::Component(config),
          NeuralNetworkTrainer<T>(config),
          targetMode_((TargetMode)paramTargetMode(config)),
          useNetwork_(paramUseNetwork(config)),
          inputDim_(0),  // The input dimension (feature dim) is set in initializeTrainer().
          outputDim_(paramOutputDim(config)),
          allowDownsampling_(paramAllowDownsampling(config)),
          pyModPath_(paramPyModPath(config)),
          pyModName_(paramPyModName(config)),
          pyMod_(NULL),
          features_(NULL),
          weights_(NULL),
          segment_(NULL),
          naturalPairingLayer_(NULL) {
    Core::Component::log("PythonTrainer with target-mode = %s",
                         choiceTargetMode[(s32)targetMode_].c_str());

    if (hasClassLabelPosteriors()) {
        // The output dimension is only needed for specific target modes.
        require_gt(outputDim_, 0);
    }

    // Natural pairing activation function.
    // Only relevant if Sprint calculates the error, because we must know it for the error gradient.
    if (targetMode_ == criterionBySprint) {
        Core::Configuration layerConfig = Core::Component::select("natural-pairing-layer");
        if (NeuralNetworkLayer<T>::paramNetworkLayerType(layerConfig) != NeuralNetworkLayer<T>::identityLayer) {
            naturalPairingLayer_ = NeuralNetworkLayer<T>::createNeuralNetworkLayer(layerConfig);
            if (!naturalPairingLayer_) {
                Core::Component::criticalError("PythonTrainer: could not create natural-pairing layer");
                return;
            }
        }
    }

    pythonInitializer_.init();

    // Get us the CPython GIL. However, when we return here,
    // it will get released and other Python threads can run.
    Python::ScopedGIL gil;

    if (!pyModPath_.empty())
        Python::addSysPath(pyModPath_);

    if (pyModName_.empty()) {
        pythonCriticalError("PythonTrainer: need Python module name (pymod-name)");
        return;
    }

    pyMod_ = PyImport_ImportModule(pyModName_.c_str());
    if (!pyMod_) {
        pythonCriticalError(
                "PythonTrainer: cannot import module '%s'",
                pyModName_.c_str());
        return;
    }
}

template<typename T>
PythonTrainer<T>::~PythonTrainer() {
    finalize();  // if not yet called
    pythonInitializer_.uninit();
}

template<typename T>
bool PythonTrainer<T>::hasClassLabelPosteriors() {
    switch (targetMode_) {
        case criterionBySprint: return true;
        case forwardOnly: return true;
        default: return false;
    }
}

// Specialized over Core::Component::criticialError():
// Handles recent Python exceptions (prints them).
// Note that Py_Finalize() is not called here but registered via
// std::atexit(). See constructor code+comment.
template<typename T>
Core::Component::Message PythonTrainer<T>::pythonCriticalError(const char* msg, ...) {
    Python::handlePythonError();

    va_list ap;
    va_start(ap, msg);
    Core::Component::Message msgHelper = Core::Component::vCriticalError(msg, ap);
    va_end(ap);
    return msgHelper;
}

template<typename T>
Python::CriticalErrorFunc PythonTrainer<T>::getPythonCriticalErrorFunc() {
    return [this]() {
        return this->pythonCriticalError("PythonTrainer: ");
    };
}

template<typename T>
void PythonTrainer<T>::initializeTrainer(u32 batchSize, std::vector<u32>& streamSizes) {
    if (!Precursor::needInit_)
        return;

    Precursor::needsNetwork_ = useNetwork_;

    // This will init the network if we need one.
    Precursor::initializeTrainer(batchSize, streamSizes);

    if (useNetwork_) {
        require(Precursor::network_);
        inputDim_ = Precursor::network_->getTopLayer().getOutputDimension();
        require_gt(inputDim_, 0);
        if (Precursor::network_->nTrainableLayers() != 0) {
            Core::Component::warning("There are %i trainable layers in the Neural network, "
                                     "however, we are not going to train them with the PythonTrainer.",
                                     Precursor::network_->nTrainableLayers());
        }
    }
    else {  // no network
        if (streamSizes.size() != 1) {
            Core::Component::criticalError("PythonTrainer only implemented for single input streams");
            return;
        }

        inputDim_ = streamSizes[0];
        require_gt(inputDim_, 0);
    }

    bool cudaEnabled = Math::CudaDataStructure::hasGpu();
    int  activeGpu   = cudaEnabled ? Math::CudaDataStructure::getActiveGpu() : -1;

    {
        Python::ScopedGIL gil;
        std::string       pyConfigStr(paramPyModConfig(Core::Configurable::config));
        PyObject*         res = Python::PyCallKw(pyMod_, "init", "{s:i,s:i,s:b,s:s,s:s,s:i,s:i}",
                                         "inputDim", inputDim_,
                                         "outputDim", outputDim_,
                                         "allowDownsampling", allowDownsampling_,
                                         "config", pyConfigStr.c_str(),
                                         "targetMode", choiceTargetMode[(s32)targetMode_].c_str(),
                                         "cudaEnabled", int(cudaEnabled),
                                         "cudaActiveGpu", activeGpu);
        if (!res) {
            pythonCriticalError("PythonTrainer: init() failed");
            return;
        }
        Py_CLEAR(res);
    }

    Precursor::needInit_ = false;
}

template<typename T>
void PythonTrainer<T>::finalize() {
    if (pyMod_) {
        require(Py_IsInitialized());  // should not happen. only via pythonInitializer_.

        Python::ScopedGIL gil;

        PyObject* res = PyObject_CallMethod(pyMod_, (char*)"exit", (char*)"");
        if (!res) {
            pythonCriticalError("PythonTrainer: exit() failed");
            return;
        }

        Py_CLEAR(res);
        Py_CLEAR(pyMod_);
    }

    if (useNetwork_)
        Precursor::network().finalize();
}

template<typename T>
void PythonTrainer<T>::processBatch_feedInput(std::vector<NnMatrix>& features, NnVector* weights, Bliss::Segment* segment) {
    if (useNetwork_) {
        require_gt(features.size(), 0);
        for (u32 i = 0; i < features.size(); ++i)
            features.at(i).initComputation();
        Precursor::network().initComputation(false /* always up-to-date, synced in initializeNetwork */);
        Precursor::network().forward(features);
        features_ = &Precursor::network().getTopLayerOutput();
        require_eq(features_->nColumns(), features[0].nColumns());  // time-frames
    }
    else {  // no network
        require_eq(features.size(), 1);
        features_ = &features[0];
    }

    require_gt(features_->nColumns(), 0);  // time-frames
    features_->finishComputation(true);

    if (Precursor::weightedAccumulation_) {
        weights_ = weights;
        if (!weights_)
            Core::Component::warning("weightedAccumulation without weights.");
        else
            // will be used later
            weights_->initComputation();
    }
    else {
        weights_ = weights = NULL;
    }

    segment_ = segment;

    // In some cases, we can directly forward the data right now.
    // In the remaining cases, we will forward the data in the processBatch_finish*() functions.
    if (targetMode_ == criterionBySprint || targetMode_ == unsupervised || targetMode_ == forwardOnly)
        python_feedInput();
}

template<typename T>
void PythonTrainer<T>::python_feedInput() {
    verify(targetMode_ == criterionBySprint || targetMode_ == unsupervised || targetMode_ == forwardOnly);

    Python::ScopedGIL gil;

    PyObject* pyFeaturesMat = NULL;
    PyObject* pyWeightsVec  = NULL;

    if (!Python::nnMatrix2numpy(getPythonCriticalErrorFunc(), pyFeaturesMat, *features_))
        return;
    if (weights_) {
        if (!Python::nnVec2numpy(getPythonCriticalErrorFunc(), pyWeightsVec, *weights_)) {
            Py_CLEAR(pyFeaturesMat);
            return;
        }
    }
    else {
        pyWeightsVec = Py_None;
        Py_INCREF(pyWeightsVec);
    }

    const char* functionName = NULL;
    switch (targetMode_) {
        case criterionBySprint: functionName = "feedInput"; break;
        case unsupervised: functionName = "feedInputUnsupervised"; break;
        case forwardOnly: functionName = "feedInputForwarding"; break;
        default: Core::Component::criticalError(); return;
    }

    PyObject* res = Python::PyCallKw(pyMod_, functionName, "{s:O,s:O,s:s}",
                                     "features", pyFeaturesMat,
                                     "weights", pyWeightsVec,
                                     "segmentName", segment_ ? segment_->fullName().c_str() : NULL);
    if (!res) {
        pythonCriticalError("PythonTrainer: %s() failed", functionName);
        Py_CLEAR(pyFeaturesMat);
        Py_CLEAR(pyWeightsVec);
        return;
    }

    Py_CLEAR(pyFeaturesMat);
    Py_CLEAR(pyWeightsVec);

    // In some cases, we expect to get a Numpy array returned.
    if (targetMode_ == criterionBySprint || targetMode_ == forwardOnly) {
        if (!Python::isNumpyArrayTypeExact(res)) {
            pythonCriticalError("PythonTrainer: %s() did not return a NumPy array but %s",
                                functionName, res->ob_type->tp_name);
            Py_CLEAR(res);
            return;
        }

        posteriors_.finishComputation(false);
        if (!Python::numpy2nnMatrix(getPythonCriticalErrorFunc(), res, posteriors_)) {
            Py_CLEAR(res);
            return;
        }

        Py_CLEAR(res);

        if ((posteriors_.nColumns() != features_->nColumns() and not allowDownsampling_) or (posteriors_.nColumns() > features_->nColumns() and allowDownsampling_) or (posteriors_.nRows() != static_cast<u32>(outputDim_))) {
            pythonCriticalError("PythonTrainer: feedInput() did return a matrix of wrong size (%i,%i), "
                                "but we expected (%i,%i)",
                                posteriors_.nRows(), posteriors_.nColumns(),
                                outputDim_, features_->nColumns());
            return;
        }

        posteriors_.initComputation(true);  // criterion expects it to be in computation mode
    }
}

template<typename T>
void PythonTrainer<T>::processBatch_finishWithAlignment(Math::CudaVector<u32>& alignment) {
    switch (targetMode_) {
        case criterionBySprint:
            alignment.initComputation();  // need to be in computation mode
            Precursor::criterion_->inputAlignment(alignment, posteriors_, weights_);
            passErrorSignalToPython();
            break;
        case targetGeneric:
            python_feedInputAndTarget(&alignment);
            break;
        case targetAlignment:
            python_feedInputAndTargetAlignment(alignment);
            break;
        default:
            Core::Component::criticalError("processBatch_finishWithAlignment with invalid target mode");
    }
}

template<typename T>
void PythonTrainer<T>::python_feedInputAndTarget(Math::CudaVector<u32>* alignment) {
    verify_eq(targetMode_, targetGeneric);

    auto*       speechSegment  = dynamic_cast<Bliss::SpeechSegment*>(segment_);
    auto*       speaker        = speechSegment ? speechSegment->speaker() : NULL;
    const char* speaker_gender = speaker ? Bliss::Speaker::genderId[speaker->gender()] : NULL;

    Python::ScopedGIL gil;

    PyObject* pyFeaturesMat = NULL;
    require_notnull(features_);
    if (!Python::nnMatrix2numpy(getPythonCriticalErrorFunc(), pyFeaturesMat, *features_))
        return;

    PyObject* pyWeightsVec = NULL;
    if (weights_) {
        if (!Python::nnVec2numpy(getPythonCriticalErrorFunc(), pyWeightsVec, *weights_)) {
            Py_CLEAR(pyFeaturesMat);
            return;
        }
    }
    else {
        pyWeightsVec = Py_None;
        Py_INCREF(pyWeightsVec);
    }

    PyObject* pyAlignmentVec = NULL;
    if (alignment) {
        if (!Python::nnVec2numpy(getPythonCriticalErrorFunc(), pyAlignmentVec, *alignment)) {
            Py_CLEAR(pyFeaturesMat);
            Py_CLEAR(pyWeightsVec);
            return;
        }
    }
    else {
        pyAlignmentVec = Py_None;
        Py_INCREF(pyAlignmentVec);
    }

    PyObject* res = Python::PyCallKw(pyMod_, "feedInputAndTarget", "{s:O,s:O,s:s,s:O,s:s,s:s,s:s}",
                                     "features", pyFeaturesMat,
                                     "weights", pyWeightsVec,
                                     "segmentName", segment_ ? segment_->fullName().c_str() : NULL,
                                     "alignment", pyAlignmentVec,
                                     "orthography", speechSegment ? speechSegment->orth().c_str() : NULL,
                                     "speaker_name", speaker ? speaker->name().c_str() : NULL,
                                     "speaker_gender", speaker_gender);
    if (!res)
        pythonCriticalError("PythonTrainer: python_feedTarget() failed");

    Py_CLEAR(pyFeaturesMat);
    Py_CLEAR(pyWeightsVec);
    Py_CLEAR(pyAlignmentVec);
    Py_CLEAR(res);
}

template<typename T>
void PythonTrainer<T>::python_feedInputAndTargetAlignment(Math::CudaVector<u32>& alignment) {
    verify_eq(targetMode_, targetAlignment);

    Python::ScopedGIL gil;

    verify(features_);
    PyObject* pyFeaturesMat = NULL;
    if (!Python::nnMatrix2numpy(getPythonCriticalErrorFunc(), pyFeaturesMat, *features_))
        return;

    PyObject* pyWeightsVec = NULL;
    if (weights_) {
        if (!Python::nnVec2numpy(getPythonCriticalErrorFunc(), pyWeightsVec, *weights_)) {
            Py_CLEAR(pyFeaturesMat);
            return;
        }
    }
    else {
        pyWeightsVec = Py_None;
        Py_INCREF(pyWeightsVec);
    }

    PyObject* pyAlignmentVec = NULL;
    if (!Python::nnVec2numpy(getPythonCriticalErrorFunc(), pyAlignmentVec, alignment)) {
        Py_CLEAR(pyFeaturesMat);
        Py_CLEAR(pyWeightsVec);
        return;
    }

    PyObject* res = Python::PyCallKw(pyMod_, "feedInputAndTargetAlignment", "{s:O,s:O,s:O,s:s}",
                                     "features", pyFeaturesMat,
                                     "targetAlignment", pyAlignmentVec,
                                     "weights", pyWeightsVec,
                                     "segmentName", segment_ ? segment_->fullName().c_str() : NULL);
    if (!res)
        pythonCriticalError("PythonTrainer: feedInputAndTargetAlignment() failed");

    Py_CLEAR(pyFeaturesMat);
    Py_CLEAR(pyWeightsVec);
    Py_CLEAR(pyAlignmentVec);
    Py_CLEAR(res);
}

template<typename T>
void PythonTrainer<T>::processBatch_finishWithSpeechSegment(Bliss::SpeechSegment& segment) {
    switch (targetMode_) {
        case criterionBySprint:
            Precursor::criterion_->inputSpeechSegment(segment, posteriors_, weights_);
            passErrorSignalToPython();
            break;
        case targetGeneric:
            // In python_feedTarget, we get the orthograhpy etc out of the current segment.
            require_eq(&segment, segment_);
            python_feedInputAndTarget(NULL);
            break;
        case targetSegmentOrth:
            python_feedInputAndTargetSegmentOrth(segment);
            break;
        default:
            Core::Component::criticalError("processBatch_finishWithSpeechSegment with invalid target mode");
    }
}

template<typename T>
void PythonTrainer<T>::python_feedInputAndTargetSegmentOrth(Bliss::SpeechSegment& segment) {
    verify_eq(targetMode_, targetSegmentOrth);

    Python::ScopedGIL gil;

    PyObject* pyFeaturesMat = NULL;
    if (!Python::nnMatrix2numpy(getPythonCriticalErrorFunc(), pyFeaturesMat, *features_))
        return;

    PyObject* pyWeightsVec = NULL;
    if (weights_) {
        if (!Python::nnVec2numpy(getPythonCriticalErrorFunc(), pyWeightsVec, *weights_)) {
            Py_CLEAR(pyFeaturesMat);
            return;
        }
    }
    else {
        pyWeightsVec = Py_None;
        Py_INCREF(pyWeightsVec);
    }

    PyObject* res = Python::PyCallKw(pyMod_, "feedInputAndTargetSegmentOrth", "{s:O,s:s,s:O,s:s}",
                                     "features", pyFeaturesMat,
                                     "targetSegmentOrth", segment.orth().c_str(),
                                     "weights", pyWeightsVec,
                                     "segmentName", segment_ ? segment_->fullName().c_str() : NULL);
    if (!res)
        pythonCriticalError("PythonTrainer: feedInputAndTargetSegmentOrth() failed");

    Py_CLEAR(pyFeaturesMat);
    Py_CLEAR(pyWeightsVec);
    Py_CLEAR(res);
}

template<typename T>
void PythonTrainer<T>::processBatch_finish() {
    // This is called if there are no target infos (alignment, segment ref, ...).
    switch (targetMode_) {
        case criterionBySprint:
            Precursor::criterion_->input(posteriors_, weights_);
            passErrorSignalToPython();
            break;
        case unsupervised:
            // nothing to do, we already called feedInputUnsupervised() via processBatch_feedInput()
            break;
        case forwardOnly:
            // nothing to do, we already called feedInputForwarding() via processBatch_feedInput()
            break;
        case targetGeneric:
            python_feedInputAndTarget(NULL);
            break;
        default:
            Core::Component::criticalError("processBatch_finish with invalid target mode");
    }
}

template<typename T>
void PythonTrainer<T>::passErrorSignalToPython() {
    verify_eq(targetMode_, criterionBySprint);

    bool discard = Precursor::criterion_->discardCurrentInput();
    if (discard) {
        Python::ScopedGIL gil;
        PyObject*         res = PyObject_CallMethod(pyMod_, (char*)"finishDiscard", (char*)"");
        if (!res) {
            pythonCriticalError("PythonTrainer: finishDiscard() failed");
            return;
        }
        Py_CLEAR(res);
        return;
    }

    T error = 0;
    Precursor::criterion_->getObjectiveFunction(error);

    require(features_);
    require_eq(features_->nColumns(), posteriors_.nColumns());
    NnMatrix errorSignal(posteriors_.nRows(), posteriors_.nColumns());
    errorSignal.initComputation(false);  // getErrorSignal expects it to be in computation mode
    errorSignal.setToZero();
    if (naturalPairingLayer_)
        Precursor::criterion_->getErrorSignal_naturalPairing(errorSignal, *naturalPairingLayer_);
    else
        Precursor::criterion_->getErrorSignal(errorSignal);
    errorSignal.finishComputation(true);

    {
        Python::ScopedGIL gil;

        PyObject* pyErrorSignal = NULL;
        if (!Python::nnMatrix2numpy(getPythonCriticalErrorFunc(), pyErrorSignal, errorSignal))
            return;

        PyObject* pyNatPairLayerTypeName = NULL;
        if (naturalPairingLayer_) {
            std::string natPairLayerTypeName = NeuralNetworkLayer<T>::choiceNetworkLayerType[(s32)naturalPairingLayer_->getLayerType()];
            require_ne(natPairLayerTypeName, "");
            pyNatPairLayerTypeName = PyUnicode_FromString(natPairLayerTypeName.c_str());
            if (!pyNatPairLayerTypeName) {
                pythonCriticalError("PythonTrainer: PyUnicode_FromString error");
                Py_CLEAR(pyErrorSignal);
                return;
            }
        }
        else {
            pyNatPairLayerTypeName = Py_None;
            Py_INCREF(pyNatPairLayerTypeName);
        }

        PyObject* res = Python::PyCallKw(pyMod_, "finishError", "{s:d,s:O,s:O}",
                                         "error", (double)error, "errorSignal", pyErrorSignal,
                                         "naturalPairingType", pyNatPairLayerTypeName);
        if (!res) {
            pythonCriticalError("PythonTrainer: finishError() failed");
            Py_CLEAR(pyErrorSignal);
            Py_CLEAR(pyNatPairLayerTypeName);
            return;
        }

        Py_CLEAR(pyErrorSignal);
        Py_CLEAR(pyNatPairLayerTypeName);
        Py_CLEAR(res);
    }
}

template<typename T>
const Core::ParameterString PythonEvaluator<T>::paramDumpPosteriors(
        "dump-posteriors", "cache file name", "");

template<typename T>
const Core::ParameterString PythonEvaluator<T>::paramDumpBestPosteriorIndices(
        "dump-best-posterior-indices", "cache file name", "");

template<typename T>
PythonEvaluator<T>::PythonEvaluator(const Core::Configuration& config)
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
        Core::Component::warning("PythonEvaluator: we don't dump anything");
}

template<typename T>
void PythonEvaluator<T>::finalize() {
    Core::Component::log("total-observations: ") << nObservations_;
    Precursor::finalize();
}

template<typename T>
void PythonEvaluator<T>::processBatch_finishWithSpeechSegment(Bliss::SpeechSegment& segment) {
    PythonTrainer<T>::posteriors_.finishComputation(true);

    u32 frameCount = PythonTrainer<T>::posteriors_.nColumns();

    if (dumpPosteriorsArchive_) {
        Flow::ArchiveWriter<Math::Matrix<T>> writer(dumpPosteriorsArchive_.get());
        PythonTrainer<T>::posteriors_.convert(writer.data_->data());
        writer.write(segment.fullName());
    }

    if (dumpBestPosterioIndicesArchive_) {
        Flow::ArchiveWriter<Math::Vector<u32>> writer(dumpBestPosterioIndicesArchive_.get());
        Math::Vector<u32>&                     bestEmissions = writer.data_->data();
        bestEmissions.resize(frameCount);
        for (u32 t = 0; t < frameCount; ++t) {
            u32 argMax   = 0;
            T   maxValue = PythonTrainer<T>::posteriors_.at(argMax, t);
            for (u32 i = 1; i < PythonTrainer<T>::posteriors_.nRows(); ++i) {
                T value = PythonTrainer<T>::posteriors_.at(i, t);
                if (value > maxValue) {
                    maxValue = value;
                    argMax   = i;
                }
            }
            bestEmissions[t] = argMax;
        }
        writer.write(segment.fullName());
    }

    PythonTrainer<T>::posteriors_.initComputation(false);
}

template<typename T>
void PythonEvaluator<T>::processBatch_finish() {
    // The problem is that I don't know a good way to reference this.
    // The only good way is probably the segment name.
    Core::Component::error("PythonEvaluator: not sure how to save this. use action = supervised-segmentwise-training.");
}

// explicit template instantiation
template class PythonTrainer<f32>;
template class PythonTrainer<f64>;

template class PythonEvaluator<f32>;
template class PythonEvaluator<f64>;

}  // namespace Nn
