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
#ifndef _NN_PYTHONTRAINER_HH
#define _NN_PYTHONTRAINER_HH

#include <Python/Init.hh>
#include <Python/Numpy.hh>
#include <string>
#include <Python.h>
#include "NeuralNetworkLayer.hh"
#include "NeuralNetworkTrainer.hh"

namespace Nn {

template<class T>
class PythonTrainer : public NeuralNetworkTrainer<T> {
public:
    typedef NeuralNetworkTrainer<T>      Precursor;
    typedef typename Types<T>::NnVector  NnVector;
    typedef typename Types<T>::NnMatrix  NnMatrix;
    typedef typename Math::FastVector<T> FastVector;
    typedef typename Math::FastMatrix<T> FastMatrix;

    // Whether Sprint calculates the criterion and passes the error signal,
    // or we pass the target alignment/reference to Python and let it do everything.
    enum TargetMode {
        criterionBySprint,
        targetGeneric,
        targetAlignment,
        targetSegmentOrth,
        unsupervised,
        forwardOnly
    };

protected:
    TargetMode             targetMode_;
    bool                   useNetwork_;
    int                    inputDim_, outputDim_;
    bool                   allowDownsampling_;
    Python::Initializer    pythonInitializer_;
    std::string            pyModPath_;
    std::string            pyModName_;
    PyObject*              pyMod_;
    NnMatrix*              features_;
    NnVector*              weights_;     // via feedInput
    Bliss::Segment*        segment_;     // via feedInput
    NnMatrix               posteriors_;  // output from Python after we feeded the features (feedInput)
    NeuralNetworkLayer<T>* naturalPairingLayer_;

    void passErrorSignalToPython();

public:
    PythonTrainer(const Core::Configuration& config);
    virtual ~PythonTrainer();

    virtual bool isNetworkOutputRepresentingClassLabels() {
        return false;
    }
    virtual bool      hasClassLabelPosteriors();
    virtual NnMatrix& getClassLabelPosteriors() {
        return posteriors_;
    }
    virtual int getClassLabelPosteriorDimension() {
        return outputDim_;
    }

    virtual bool allowsDownsampling() const {
        return allowDownsampling_;
    }

    virtual void initializeTrainer(u32 batchSize, std::vector<u32>& streamSizes);
    virtual void finalize();

    virtual void processBatch_feedInput(std::vector<NnMatrix>& features, NnVector* weights, Bliss::Segment* segment);
    virtual void processBatch_finishWithAlignment(Math::CudaVector<u32>& alignment);
    virtual void processBatch_finishWithSpeechSegment(Bliss::SpeechSegment& segment);
    virtual void processBatch_finish();

    void                      python_feedInput();
    void                      python_feedInputAndTarget(Math::CudaVector<u32>* alignment);
    void                      python_feedInputAndTargetAlignment(Math::CudaVector<u32>& alignment);
    void                      python_feedInputAndTargetSegmentOrth(Bliss::SpeechSegment& segment);
    Core::Component::Message  pythonCriticalError(const char* msg = 0, ...);
    Python::CriticalErrorFunc getPythonCriticalErrorFunc();
};

//=============================================================================
/** only forwards through the network and dumps the NN output (= emission label posteriors)
 */
template<class T>
class PythonEvaluator : public PythonTrainer<T> {
    typedef PythonTrainer<T> Precursor;

protected:
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;
    static const Core::ParameterString  paramDumpPosteriors;
    static const Core::ParameterString  paramDumpBestPosteriorIndices;
    u32                                 nObservations_;
    std::shared_ptr<Core::Archive>      dumpPosteriorsArchive_;
    std::shared_ptr<Core::Archive>      dumpBestPosterioIndicesArchive_;

public:
    PythonEvaluator(const Core::Configuration& config);
    virtual ~PythonEvaluator() {}
    NeuralNetwork<T>& network() {
        return Precursor::network();
    }
    virtual void finalize();
    virtual void processBatch_finishWithSpeechSegment(Bliss::SpeechSegment& segment);
    virtual void processBatch_finish();
    virtual bool needsToProcessAllFeatures() const {
        return true;
    }
};

}  // namespace Nn

#endif  // PYTHONTRAINER_HH
