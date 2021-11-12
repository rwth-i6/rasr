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
The PythonControl is a generic Python control interface.
It creates a Python interpreter via the standard CPython API.
It provides some generic commands which can be called through a single callback().

It loads a Python module (pymod-name; add path via pymod-path if needed).
It expects these general functions in the Python module:

    def init(name, sprint_unit, reference, config)

`name` is "Sprint.PythonControl" and `reference` is any Python object which
you can use as a reference (in case the module functions are used from multiple
sources at the same time). `sprint_unit` is whatever you used in the PythonControl constructor.
`init` should return an object which has these methods:

    def run_control_loop(callback)
    def exit()

`run_control_loop` will be called once when Sprint leaves the control to Python.
In `run_control_loop`, all the control can be done through `callback`, which is a function like:

    def callback(cmd, ...)

`cmd` has to be a string.
See the code below for reference about supported commands.
*/

#include "PythonControl.hh"
#include <Bliss/CorpusDescription.hh>
#include <Core/Application.hh>
#include <Core/Archive.hh>
#include <Core/Version.hh>
#include <Flow/DataAdaptor.hh>
#include <Flow/Module.hh>
#include <Flow/Registry.hh>
#include <Nn/ActivationLayer.hh>
#include <Nn/AllophoneStateFsaExporter.hh>
#include <Nn/ClassLabelWrapper.hh>
#include <Nn/Criterion.hh>
#include <Nn/CtcCriterion.hh>
#include <Python/Numpy.hh>
#include <Python/Utilities.hh>
#include <Speech/Alignment.hh>
#include <Speech/CorpusProcessor.hh>
#include <Speech/DataExtractor.hh>
#include <Speech/DataSource.hh>
#include <Speech/ModelCombination.hh>
#include <Speech/Module.hh>
#include <memory>
#include <sstream>

static const Core::ParameterBool paramPythonControlEnabled(
        "python-control-enabled", "whether to use PythonControl", false);

static const Core::ParameterString paramPyModPath(
        "pymod-path", "the path containing the Python module", "");

static const Core::ParameterString paramPyModName(
        "pymod-name", "the module-name, such that 'import x' would work", "");

static const Core::ParameterString paramPyModConfig(
        "pymod-config",
        "config-string, passed to init()",
        "");

// Increase this number when we add some new feature to Sprint
// and you want to check in Python whether Sprint is new enough to have that feature.
static const long versionNumber = 5;

struct AlignmentToPython {
    Core::Ref<const Am::AcousticModel>     acousticModel_;
    std::shared_ptr<Nn::ClassLabelWrapper> classLabelWrapper_;  // optional
    size_t                                 nSkippedAlignmentFrames_;
    Core::Component*                       parent_;
    Python::CriticalErrorFunc              criticalErrorFunc_;
    Math::FastMatrix<f32>*                 features_;  // optional. dim * time

    AlignmentToPython()
            : nSkippedAlignmentFrames_(0), parent_(NULL), features_(NULL) {}

    bool _alignmentLabelIndex(const Speech::Alignment& alignment, Fsa::LabelId emissionIndex, u32& index) {
        u32 labelIndex = emissionIndex;
        if (alignment.labelType() == Speech::Alignment::allophoneStateIds)
            labelIndex = acousticModel_->emissionIndex(labelIndex);
        if (classLabelWrapper_) {
            require_lt(labelIndex, classLabelWrapper_->nClasses());
            if (!classLabelWrapper_->isClassToAccumulate(labelIndex))
                return false;
            index = classLabelWrapper_->getOutputIndexFromClassIndex(labelIndex);
        }
        else
            index = labelIndex;
        return true;
    }

    void extractViterbiAlignment(const Speech::Alignment& alignment, Python::ObjRef& pyAlignment) {
        u32              time     = 0;
        u32              t_offset = 0;
        std::vector<u32> alignmentVec;
        alignmentVec.reserve(alignment.size());
        for (auto iter = alignment.begin(); iter != alignment.end(); ++iter) {
            const Speech::AlignmentItem& item = *iter;
            if (features_ && item.time >= features_->nColumns() + t_offset) {
                parent_->error("Viterbi alignment: got time frame %u but sequence length is %u", item.time, features_->nColumns() + t_offset);
                return;
            }
            if (item.time < time + t_offset) {
                parent_->error("Viterbi alignment: expected time frame %u, got %u. (maybe Baum-Welch alignment?)", time + t_offset, item.time);
                return;
            }
            while (item.time > time + t_offset) {
                parent_->warning("Viterbi alignment: skipped time frame %u, got %u", time + t_offset, item.time);
                if (features_)
                    features_->removeColumn(time);
                t_offset++;
                nSkippedAlignmentFrames_++;
            }
            require_eq(item.time, time + t_offset);
            u32 classIdx = 0;
            if (!_alignmentLabelIndex(alignment, item.emission, classIdx)) {
                if (nSkippedAlignmentFrames_ == 0)
                    parent_->log("Viterbi alignment: we skip some frames because of the class label wrapper");
                if (features_)
                    features_->removeColumn(time);
                t_offset++;
                nSkippedAlignmentFrames_++;
                continue;
            }
            alignmentVec.push_back(classIdx);
            ++time;
        }
        require_eq(time, alignmentVec.size());
        if (features_) {
            require_eq(time, features_->nColumns());
        }
        pyAlignment.clear();
        Python::stdVec2numpy(criticalErrorFunc_, pyAlignment.obj, alignmentVec);
    }

    void extractSoftAlignment(const Speech::Alignment& alignment, Python::ObjRef& pySoftAlignment) {
        u32              time = 0;
        std::vector<u32> alignmentTime;
        std::vector<u32> alignmentClassIdx;
        std::vector<f32> alignmentWeight;
        alignmentTime.reserve(alignment.size());
        alignmentClassIdx.reserve(alignment.size());
        alignmentWeight.reserve(alignment.size());
        for (auto iter = alignment.begin(); iter != alignment.end(); ++iter) {
            const Speech::AlignmentItem& item = *iter;
            if (features_ && item.time >= features_->nColumns()) {
                parent_->error("Soft alignment: got time frame %u but sequence length is %u", item.time, features_->nColumns());
                return;
            }
            if (item.time < time) {
                parent_->error("Soft alignment: expected time frame %u, got %u", time, item.time);
                return;
            }
            if (item.time > time) {
                if (item.time == time + 1)
                    ++time;
                else {
                    parent_->warning("Soft alignment: skipped time frame %u, got %u", time, item.time);
                    time = item.time;
                }
            }
            require_eq(item.time, time);
            u32 classIdx = 0;
            if (!_alignmentLabelIndex(alignment, item.emission, classIdx))
                continue;
            alignmentTime.push_back(time);
            alignmentClassIdx.push_back(classIdx);
            alignmentWeight.push_back(item.weight);  // std space in [0,1]
        }
        require_eq(alignmentTime.size(), alignmentClassIdx.size());
        require_eq(alignmentTime.size(), alignmentWeight.size());
        // like sparse matrix in COOrdinate format
        // http://docs.scipy.org/doc/scipy/reference/sparse.html
        // https://github.com/scipy/scipy/blob/master/scipy/sparse/coo.py
        Python::ObjRef pyI, pyJ, pyData;
        if (!Python::stdVec2numpy(criticalErrorFunc_, pyI.obj, alignmentTime))
            return;
        if (!Python::stdVec2numpy(criticalErrorFunc_, pyJ.obj, alignmentClassIdx))
            return;
        if (!Python::stdVec2numpy(criticalErrorFunc_, pyData.obj, alignmentWeight))
            return;
        pySoftAlignment.takeOver(PyTuple_Pack(3, pyI.obj, pyJ.obj, pyData.obj));
    }
};

struct BuildSegmentToOrthMapVisitor : public Bliss::CorpusVisitor {
    BuildSegmentToOrthMapVisitor()
            : Bliss::CorpusVisitor(), map_(new Core::StringHashMap<std::string>()) {}

    virtual void visitSpeechSegment(Bliss::SpeechSegment* s) {
        (*map_)[s->fullName()] = s->orth();
    }

    std::shared_ptr<Core::StringHashMap<std::string>> map_;
};

static std::shared_ptr<Core::StringHashMap<std::string>> build_segment_to_orth_map(Core::Configuration const& config) {
    Bliss::CorpusDescription     corpus(config);
    BuildSegmentToOrthMapVisitor visitor;
    corpus.accept(&visitor);
    return visitor.map_;
}

namespace Nn {

static const char* capsule_internal_name = "Sprint.PythonControl.Internal";
static PyObject*   callback(PyObject* self, PyObject* args, PyObject* kws);
static PyMethodDef callback_method_def = {
        "callback", (PyCFunction)callback, METH_VARARGS | METH_KEYWORDS,
        "Sprint PythonControl Callback."};

struct PythonControl::Internal : public Core::Component {
    PyObject*                                             capsule_;
    PyObject*                                             callback_;
    std::shared_ptr<Criterion<f32>>                       criterion_;
    std::shared_ptr<AllophoneStateFsaExporter>            allophoneStateFsaExporter_;
    std::shared_ptr<Core::StringHashMap<std::string>>     segmentToOrthMap_;
    std::map<std::string, std::shared_ptr<Core::Archive>> cacheArchives_;
    Core::Ref<const Am::AcousticModel>                    acousticModel_;

    Internal(const Core::Configuration& c)
            : Core::Component(c),
              capsule_(NULL),
              callback_(NULL),
              criterion_(NULL),
              allophoneStateFsaExporter_(NULL),
              segmentToOrthMap_(NULL) {
        Python::ScopedGIL gil;
        capsule_  = PyCapsule_New(this, capsule_internal_name, NULL);
        callback_ = PyCFunction_New(&callback_method_def, capsule_);
    }

    virtual ~Internal() {
        Python::ScopedGIL gil;
        Py_CLEAR(capsule_);
        Py_CLEAR(callback_);
    }

    Core::Component::Message pythonCriticalError(const char* msg, ...) {
        Python::handlePythonError();

        va_list ap;
        va_start(ap, msg);
        Core::Component::Message msgHelper = Core::Component::vCriticalError(msg, ap);
        va_end(ap);
        return msgHelper;
    }

    Python::CriticalErrorFunc getPythonCriticalErrorFunc() {
        return [this]() {
            return this->pythonCriticalError("PythonControl::Internal: ");
        };
    }

    PyObject* version() {
        std::stringstream     ss;
        Core::XmlWriter       xml(ss);
        Core::VersionRegistry vr;
        vr.reportVersion(xml);
        xml.flush();
        std::string s = ss.str();
        return PyUnicode_FromStringAndSize(s.data(), s.size());
    }

    PyObject* versionNumber() {
        return PyLong_FromLong(::versionNumber);
    }

    PyObject* help() {
        return PyUnicode_FromString(
                "Usage: callback(cmd, *args).\n"
                "callback comes via Sprint PythonControl.\n"
                "See Sprint src/Nn/PythonControl.cpp for available commands.\n");
    }

    void _initCriterion() {
        if (criterion_)
            return;
        criterion_ = std::shared_ptr<Criterion<f32>>(Criterion<f32>::create(getConfiguration()));
        require(criterion_);
    }

    void _initAllophoneStateFsaExporter() {
        if (allophoneStateFsaExporter_)
            return;
        allophoneStateFsaExporter_ = std::make_shared<AllophoneStateFsaExporter>(select("alignment-fsa-exporter"));
    }

    void _initSegmentToOrthMap() {
        if (segmentToOrthMap_)
            return;
        segmentToOrthMap_ = build_segment_to_orth_map(select("corpus"));
    }

    std::shared_ptr<Core::Archive> getCacheArchive(const std::string& cacheFilename) {
        auto f = cacheArchives_.find(cacheFilename);
        if (f != cacheArchives_.end())
            return f->second;
        std::shared_ptr<Core::Archive> a(Core::Archive::create(config, cacheFilename, Core::Archive::AccessModeRead));
        if (!a)
            return a;
        cacheArchives_[cacheFilename] = a;
        return a;
    }

    void _initAcousticModel() {
        if (acousticModel_)
            return;
        // e.g. see BufferedAlignedFeatureProcessor<T>::initAcousticModel()
        /* acoustic model to identify labels */
        Speech::ModelCombination modelCombination(
                this->select("model-combination"),
                Speech::ModelCombination::useAcousticModel,
                Am::AcousticModel::noEmissions | Am::AcousticModel::noStateTransition);
        modelCombination.load();
        acousticModel_ = modelCombination.acousticModel();
        require(acousticModel_);
        u32 nClasses = acousticModel_->nEmissions();
        this->log("number of classes of acoustic model: ") << nClasses;
    }

    PyObject* initCriterion() {
        _initCriterion();
        Py_INCREF(Py_None);
        return Py_None;
    }

    PyObject* calculateCriterion(PyObject* args, PyObject* kws) {
        static const char* kwlist[] = {
                "command",
                "posteriors",
                "orthography",
                "alignment",
                "output_error_type",
                "segment_name",
                NULL};
        const char* _cmd             = NULL;
        PyObject*   posteriorsPy     = NULL;  // borrowed
        const char* orthography      = NULL;
        PyObject*   alignmentPy      = NULL;  // borrowed
        const char* outputErrorTypeC = NULL;
        const char* segmentName      = NULL;

        if (!PyArg_ParseTupleAndKeywords(
                    args, kws, "sO|sOss:callback", (char**)kwlist,
                    &_cmd, &posteriorsPy, &orthography, &alignmentPy, &outputErrorTypeC, &segmentName))
            return NULL;

        enum {
            ET_None,
            ET_ErrorSignal,
            ET_ErrorSignalBeforeSoftmax,
            ET_PseudoTargets
        } outputErrorType = ET_None;
        if (!outputErrorTypeC || strcmp(outputErrorTypeC, "none") == 0)
            outputErrorType = ET_None;
        else if (strcmp(outputErrorTypeC, "error-signal") == 0)
            outputErrorType = ET_ErrorSignal;
        else if (strcmp(outputErrorTypeC, "error-signal-before-softmax") == 0)
            outputErrorType = ET_ErrorSignalBeforeSoftmax;
        else if (strcmp(outputErrorTypeC, "pseudo-targets") == 0)
            outputErrorType = ET_PseudoTargets;
        else {
            PyErr_Format(PyExc_ValueError, "PythonControl callback(): calculate_criterion: unknown output_error_type '%s'", outputErrorTypeC);
            return NULL;
        }

        Math::CudaMatrix<f32> posteriors;
        if (!Python::numpy2nnMatrix(getPythonCriticalErrorFunc(), posteriorsPy, posteriors))
            return NULL;
        posteriors.initComputation(true);

        Math::CudaVector<u32> alignment;
        if (alignmentPy) {
            if (orthography) {
                PyErr_Format(PyExc_ValueError, "PythonControl callback(): calculate_criterion: you should provide either an alignment, or the orthography, but not both");
                return NULL;
            }

            if (!Python::numpy2nnVector(getPythonCriticalErrorFunc(), alignmentPy, alignment))
                return NULL;
            alignment.initComputation(true);
        }

        _initCriterion();

        Bliss::Corpus        dummyCorpus;
        Bliss::Recording     dummyRecording(&dummyCorpus);
        Bliss::SpeechSegment speechSegment(&dummyRecording);  // must be in scope until end when used
        // Note that segmentName is the full segment name, so setName() is not perfectly correct.
        // We would have to split it by "/" and set the corpus-name and recording-name.
        if (segmentName)
            speechSegment.setName(segmentName);
        if (orthography)
            speechSegment.setOrth(orthography);

        if (orthography)
            criterion_->inputSpeechSegment(speechSegment, posteriors);
        else if (alignmentPy)
            criterion_->inputAlignment(alignment, posteriors);
        else
            criterion_->input(posteriors);

        if (criterion_->discardCurrentInput())
            return Py_BuildValue("(OO)", Py_None, Py_None);

        f32 resLoss = 0;
        criterion_->getObjectiveFunction(resLoss);

        Python::ObjRef resOutputError;
        switch (outputErrorType) {
            case ET_None:
                resOutputError.copyRef(Py_None);
                break;
            case ET_ErrorSignal: {
                Math::CudaMatrix<f32> errorSignal(posteriors.nRows(), posteriors.nColumns());
                errorSignal.initComputation(false);
                errorSignal.setToZero();
                criterion_->getErrorSignal(errorSignal);
                errorSignal.finishComputation(true);
                if (!Python::nnMatrix2numpy(getPythonCriticalErrorFunc(), resOutputError.obj, errorSignal))
                    return NULL;
                break;
            }
            case ET_ErrorSignalBeforeSoftmax: {
                Math::CudaMatrix<f32> errorSignal(posteriors.nRows(), posteriors.nColumns());
                errorSignal.initComputation(false);
                errorSignal.setToZero();
                SoftmaxLayer<f32> dummyLayer(getConfiguration());
                criterion_->getErrorSignal_naturalPairing(errorSignal, dummyLayer);
                errorSignal.finishComputation(true);
                if (!Python::nnMatrix2numpy(getPythonCriticalErrorFunc(), resOutputError.obj, errorSignal))
                    return NULL;
                break;
            }
            case ET_PseudoTargets: {
                Math::CudaMatrix<f32>* targets = criterion_->getPseudoTargets();
                if (targets) {
                    targets->finishComputation(true);
                    if (!Python::nnMatrix2numpy(getPythonCriticalErrorFunc(), resOutputError.obj, *targets))
                        return NULL;
                    targets->initComputation(false);
                }
                else
                    resOutputError.copyRef(Py_None);
                break;
            }
        }

        PyObject* res = Py_BuildValue("(fO)", resLoss, resOutputError.obj);
        return res;
    }

    PyObject* getCtcAlignment(PyObject* args, PyObject* kws) {
        static const char* kwlist[] = {
                "command",
                "log_posteriors",
                "orthography",
                "soft",
                "min_prob_gt",
                "gamma",
                NULL};
        const char* _cmd            = NULL;
        PyObject*   logPosteriorsPy = NULL;  // borrowed
        const char* orthography     = NULL;
        int         soft            = 1;
        float       min_prob_gt     = 0.0;
        float       gamma           = 1.0;

        if (!PyArg_ParseTupleAndKeywords(
                    args, kws, "sOs|iff:callback", (char**)kwlist,
                    &_cmd, &logPosteriorsPy, &orthography, &soft, &min_prob_gt, &gamma))
            return NULL;

        _initCriterion();
        CtcCriterion<f32>* ctc = dynamic_cast<CtcCriterion<f32>*>(criterion_.get());
        if (!ctc) {
            PyErr_Format(PyExc_ValueError, "PythonControl get_ctc_alignment(): we expect the CTC criterion but got type '%i'",
                         criterion_->getType());
            return NULL;
        }

        Math::CudaMatrix<f32> logPosteriors;
        if (!Python::numpy2nnMatrix(getPythonCriticalErrorFunc(), logPosteriorsPy, logPosteriors))
            return NULL;
        logPosteriors.initComputation(true);

        Speech::Alignment alignment;
        if (!ctc->getAlignment(alignment, logPosteriors, orthography, min_prob_gt, gamma)) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        AlignmentToPython alignmentToPython;
        alignmentToPython.parent_            = this;
        alignmentToPython.criticalErrorFunc_ = getPythonCriticalErrorFunc();
        alignmentToPython.acousticModel_     = ctc->getAcousticModel();
        Python::ObjRef alignmentPy;
        if (soft)
            alignmentToPython.extractSoftAlignment(alignment, alignmentPy);
        else
            alignmentToPython.extractViterbiAlignment(alignment, alignmentPy);
        if (!alignmentPy) {
            PyErr_Format(PyExc_ValueError, "PythonControl get_ctc_alignment(): error while converting. maybe it's a soft alignment?");
            return NULL;
        }
        return alignmentPy.release();
    }

    PyObject* exportAllophoneStateFsaByOrthography(PyObject* args, PyObject* kws) {
        static const char* kwlist[] = {
                "command",
                "orthography",
                NULL};
        const char* _cmd        = NULL;
        const char* orthography = NULL;

        if (!PyArg_ParseTupleAndKeywords(args, kws, "ss:callback", (char**)kwlist, &_cmd, &orthography))
            return NULL;

        _initAllophoneStateFsaExporter();
        AllophoneStateFsaExporter::ExportedAutomaton automaton = allophoneStateFsaExporter_->exportFsaForOrthography(std::string(orthography));

        PyObject* edges   = NULL;
        PyObject* weights = NULL;
        Python::stdVec2numpy(getPythonCriticalErrorFunc(), edges, automaton.edges);
        Python::stdVec2numpy(getPythonCriticalErrorFunc(), weights, automaton.weights);
        PyObject* result = Py_BuildValue("(IIOO)", automaton.num_states, automaton.num_edges, edges, weights);
        Py_XDECREF(edges);
        Py_XDECREF(weights);
        return result;
    }

    PyObject* exportAllophoneStateFsaBySegName(PyObject* args, PyObject* kws) {
        static const char* kwlist[] = {
                "command",
                "segment_name",
                NULL};
        const char* _cmd         = NULL;
        const char* segment_name = NULL;

        if (!PyArg_ParseTupleAndKeywords(args, kws, "ss:callback", (char**)kwlist, &_cmd, &segment_name))
            return NULL;

        _initSegmentToOrthMap();
        _initAllophoneStateFsaExporter();

        auto iter = segmentToOrthMap_->find(std::string(segment_name));
        if (iter == segmentToOrthMap_->end()) {
            PyErr_Format(PyExc_KeyError, "PythonControl export_allophone_state_fsa_by_segment_name: unknown segment name '%s'",
                         segment_name);
            return NULL;
        }
        AllophoneStateFsaExporter::ExportedAutomaton automaton = allophoneStateFsaExporter_->exportFsaForOrthography(iter->second);

        PyObject* edges   = NULL;
        PyObject* weights = NULL;
        Python::stdVec2numpy(getPythonCriticalErrorFunc(), edges, automaton.edges);
        Python::stdVec2numpy(getPythonCriticalErrorFunc(), weights, automaton.weights);
        PyObject* result = Py_BuildValue("(IIOO)", automaton.num_states, automaton.num_edges, edges, weights);
        Py_XDECREF(edges);
        Py_XDECREF(weights);
        return result;
    }

    PyObject* getOrthographyBySegmentName(PyObject* args, PyObject* kws) {
        static const char* kwlist[] = {
                "command",
                "segment_name",
                NULL};
        const char* _cmd         = NULL;
        const char* segment_name = NULL;

        if (!PyArg_ParseTupleAndKeywords(args, kws, "ss:callback", (char**)kwlist, &_cmd, &segment_name))
            return NULL;

        _initSegmentToOrthMap();

        auto iter = segmentToOrthMap_->find(std::string(segment_name));
        if (iter == segmentToOrthMap_->end()) {
            PyErr_Format(PyExc_KeyError, "PythonControl get_orthography_by_segment_name: unknown segment name '%s'",
                         segment_name);
            return NULL;
        }

        return PyUnicode_FromString(iter->second.c_str());
    }

    bool _readAlignmentFromCacheArchive(const std::shared_ptr<Core::Archive>&                a,
                                        const std::string&                                   segmentName,
                                        Flow::DataPtr<Flow::DataAdaptor<Speech::Alignment>>& alignmentRef) {
        // Be sure that the necessary Flow datatypes are registered.
        Flow::Module::instance();
        Speech::Module::instance();

        Core::ArchiveReader reader(*a, segmentName);
        if (!reader.isOpen()) {
            PyErr_Format(PyExc_ValueError, "PythonControl: cannot read entry (segment-name) '%s' in archive",
                         segmentName.c_str());
            return false;
        }

        Core::BinaryInputStream b(reader);
        std::string             datatypeName;
        if (!(b >> datatypeName)) {
            PyErr_Format(PyExc_ValueError, "PythonControl: cannot read datatype name for entry (segment-name) '%s'. not a Flow cache?",
                         segmentName.c_str());
            return false;
        }

        const Flow::Datatype* datatype = Flow::Registry::instance().getDatatype(datatypeName);
        if (!datatype) {
            PyErr_Format(PyExc_ValueError, "PythonControl: unknown datatype '%s' for entry (segment-name) '%s'",
                         datatypeName.c_str(), segmentName.c_str());
            return false;
        }

        if (Flow::DataAdaptor<Speech::Alignment>::type() != datatype) {
            PyErr_Format(PyExc_ValueError, "PythonControl: expected datatype '%s' but got '%s' for entry (segment-name) '%s'",
                         Flow::DataAdaptor<Speech::Alignment>::type()->name().c_str(),
                         datatypeName.c_str(),
                         segmentName.c_str());
            return false;
        }

        std::vector<Flow::DataPtr<Flow::Data>> dataVec;
        if (!datatype->readGatheredData(b, dataVec)) {
            PyErr_Format(PyExc_ValueError, "PythonControl: error while reading Flow cache datatype '%s'", datatype->name().c_str());
            return false;
        }

        if (dataVec.size() != 1) {
            PyErr_Format(PyExc_ValueError, "PythonControl: expected to get a single instance of '%s' but got %zu",
                         Flow::DataAdaptor<Speech::Alignment>::type()->name().c_str(),
                         dataVec.size());
            return false;
        }

        alignmentRef = Flow::DataPtr<Flow::DataAdaptor<Speech::Alignment>>(dataVec[0]);
        return true;
    }

    PyObject* getAlignmentFromCache(PyObject* args, PyObject* kws) {
        static const char* kwlist[] = {
                "command",
                "cache_filename",
                "segment_name",
                NULL};
        const char* _cmd                        = NULL;
        const char* cache_filename_c            = NULL;
        const char* segment_name_c              = NULL;
        int         silence_allophone_state_idx = -1;

        if (!PyArg_ParseTupleAndKeywords(args, kws, "sss|i:callback", (char**)kwlist,
                                         &_cmd, &cache_filename_c, &segment_name_c, &silence_allophone_state_idx))
            return NULL;

        std::shared_ptr<Core::Archive> a = getCacheArchive(cache_filename_c);
        if (!a) {
            PyErr_Format(PyExc_ValueError, "PythonControl: cannot open cache archive '%s'",
                         cache_filename_c);
            return NULL;
        }

        Flow::DataPtr<Flow::DataAdaptor<Speech::Alignment>> alignmentRef;
        if (!_readAlignmentFromCacheArchive(a, segment_name_c, alignmentRef))
            return NULL;

        require(alignmentRef);
        const Speech::Alignment& alignment = alignmentRef->data();

        if (alignment.labelType() != Speech::Alignment::allophoneStateIds) {
            PyErr_Format(PyExc_ValueError, "PythonControl: alignment label type is not allophone-state-id");
            return NULL;
        }

        Python::ObjRef pyAlignment;
        {
            std::vector<u32> alignmentVec;
            size_t           time = 0;
            for (auto iter = alignment.begin(); iter != alignment.end(); ++iter) {
                const Speech::AlignmentItem& item = *iter;
                if (item.time < time) {
                    PyErr_Format(PyExc_ValueError, "PythonControl: Viterbi alignment: expected time frame %zu, got %u."
                                                   " (maybe Baum-Welch alignment?)",
                                 time, item.time);
                    return NULL;
                }
                if (item.time > time) {
                    PyErr_Format(PyExc_ValueError, "PythonControl: Viterbi alignment: skipped time frame %zu, got %u",
                                 time, item.time);
                    return NULL;
                }
                require_eq(item.time, time);
                alignmentVec.push_back(item.emission);
                ++time;
            }
            Python::stdVec2numpy(getPythonCriticalErrorFunc(), pyAlignment.obj, alignmentVec);
        }
        return pyAlignment.release();
    }

    PyObject* analyzeAlignmentFromCache(PyObject* args, PyObject* kws) {
        static const char* kwlist[] = {
                "command",
                "cache_filename",
                "segment_name",
                "silence_allophone_state_idx",
                NULL};
        const char* _cmd                        = NULL;
        const char* cache_filename_c            = NULL;
        const char* segment_name_c              = NULL;
        int         silence_allophone_state_idx = -1;

        if (!PyArg_ParseTupleAndKeywords(args, kws, "ss|si:callback", (char**)kwlist,
                                         &_cmd, &cache_filename_c, &segment_name_c, &silence_allophone_state_idx))
            return NULL;

        Am::AllophoneStateIndex silenceAllophoneStateIdx = 0;
        if (silence_allophone_state_idx >= 0)
            silenceAllophoneStateIdx = silence_allophone_state_idx;
        else {
            _initAcousticModel();
            silenceAllophoneStateIdx = acousticModel_->silenceAllophoneStateIndex();
        }

        std::shared_ptr<Core::Archive> a = getCacheArchive(cache_filename_c);
        if (!a) {
            PyErr_Format(PyExc_ValueError, "PythonControl: cannot open cache archive '%s'",
                         cache_filename_c);
            return NULL;
        }

        size_t nTime       = 0;
        size_t nSilForward = 0, nSilLoop = 0;
        size_t nPhonForward = 0, nPhonLoop = 0;

        std::string                   segmentName;
        Core::Archive::const_iterator archive_iter;
        if (!segment_name_c)
            archive_iter = a->files();
        while (true) {
            if (!segment_name_c) {
                // Skip over *.attribs entries in the cache archive.
                while (archive_iter) {
                    const std::string& entryName = archive_iter.name();
                    if (entryName.size() >= strlen(".attribs") && entryName.substr(entryName.size() - strlen(".attribs")) == ".attribs")
                        ++archive_iter;
                    else
                        break;
                }
                if (!archive_iter)
                    break;
                segmentName = archive_iter.name();
            }
            else
                segmentName = segment_name_c;

            Flow::DataPtr<Flow::DataAdaptor<Speech::Alignment>> alignmentRef;
            if (!_readAlignmentFromCacheArchive(a, segmentName, alignmentRef))
                return NULL;

            require(alignmentRef);
            const Speech::Alignment& alignment = alignmentRef->data();

            if (alignment.labelType() != Speech::Alignment::allophoneStateIds) {
                PyErr_Format(PyExc_ValueError, "PythonControl: alignment label type is not allophone-state-id");
                return NULL;
            }

            size_t                  time                  = 0;
            Am::AllophoneStateIndex lastAllophoneStateIdx = silenceAllophoneStateIdx;
            for (auto iter = alignment.begin(); iter != alignment.end(); ++iter) {
                const Speech::AlignmentItem& item = *iter;
                if (item.time < time) {
                    PyErr_Format(PyExc_ValueError, "PythonControl: Viterbi alignment: expected time frame %zu, got %u."
                                                   " (maybe Baum-Welch alignment?)",
                                 time, item.time);
                    return NULL;
                }
                if (item.time > time) {
                    PyErr_Format(PyExc_ValueError, "PythonControl: Viterbi alignment: skipped time frame %zu, got %u",
                                 time, item.time);
                    return NULL;
                }
                require_eq(item.time, time);
                Am::AllophoneStateIndex allophoneStateIdx = item.emission;
                if (time > 0) {
                    if (lastAllophoneStateIdx == silenceAllophoneStateIdx) {
                        if (allophoneStateIdx != silenceAllophoneStateIdx)
                            nSilForward++;
                        else
                            nSilLoop++;
                    }
                    else {
                        if (allophoneStateIdx != lastAllophoneStateIdx)
                            nPhonForward++;
                        else
                            nPhonLoop++;
                    }
                }

                ++time;
                lastAllophoneStateIdx = allophoneStateIdx;
            }
            nTime += time;

            if (!segment_name_c)
                ++archive_iter;
            else
                break;
        }

        Python::ObjRef res;
        res.takeOver(PyDict_New());
        if (!res)
            return NULL;
        Python::dict_SetItemString(res.obj, "length", nTime);
        Python::dict_SetItemString(res.obj, "nSilForward", nSilForward);
        Python::dict_SetItemString(res.obj, "nSilLoop", nSilLoop);
        Python::dict_SetItemString(res.obj, "nPhonForward", nPhonForward);
        Python::dict_SetItemString(res.obj, "nPhonLoop", nPhonLoop);
        return res.release();
    }

    PyObject* callback(PyObject* args, PyObject* kws) {
        Py_ssize_t nargs = PyTuple_Size(args);
        if (nargs < 1) {
            PyErr_SetString(PyExc_TypeError, "PythonControl callback(): requires at least one arg. try callback('help')");
            return NULL;
        }

        PyObject* cmd = PyTuple_GetItem(args, 0);  // borrowed
        if (PyUnicode_KIND(cmd) != PyUnicode_1BYTE_KIND){
            PyErr_SetString(PyExc_TypeError, "PythonControl callback(): first arg is not a 1BYTE unicode string");
            return NULL;
        }
        const char* cmd_cs = (const char*)PyUnicode_1BYTE_DATA(cmd);
        if (!cmd_cs) {
            PyErr_SetString(PyExc_TypeError, "PythonControl callback(): first arg must be a string");
            return NULL;
        }
        std::string cmd_s(cmd_cs);

        if (cmd_s == "version")
            return version();
        if (cmd_s == "version_number")
            return versionNumber();
        if (cmd_s == "help")
            return help();
        if (cmd_s == "init_criterion")
            return initCriterion();
        if (cmd_s == "calculate_criterion")
            return calculateCriterion(args, kws);
        if (cmd_s == "get_ctc_alignment")
            return getCtcAlignment(args, kws);
        if (cmd_s == "export_allophone_state_fsa_by_orthography")
            return exportAllophoneStateFsaByOrthography(args, kws);
        if (cmd_s == "export_allophone_state_fsa_by_segment_name")
            return exportAllophoneStateFsaBySegName(args, kws);
        if (cmd_s == "get_orthography_by_segment_name")
            return getOrthographyBySegmentName(args, kws);
        if (cmd_s == "get_alignment_from_cache")
            return getAlignmentFromCache(args, kws);
        if (cmd_s == "analyze_alignment_from_cache")
            return analyzeAlignmentFromCache(args, kws);

        PyErr_Format(PyExc_ValueError, "PythonControl callback(): unknown command '%s'", cmd_cs);
        return NULL;
    }
};

static PyObject* callback(PyObject* self, PyObject* args, PyObject* kws) {
    PythonControl::Internal* internal = (PythonControl::Internal*)PyCapsule_GetPointer(self, capsule_internal_name);
    if (!internal)
        return NULL;
    return internal->callback(args, kws);
}

PythonControl::PythonControl(const Core::Configuration& config, const std::string& sprintUnit, bool isOptional)
        : Core::Component(config),
          sprintUnit_(sprintUnit),
          pyObject_(NULL) {
    if (isOptional) {
        if (!paramPythonControlEnabled(config)) {
            this->log("PythonControl(%s) is disabled", sprintUnit.c_str());
            return;
        }
        this->log("PythonControl(%s) is enabled", sprintUnit.c_str());
    }

    pythonInitializer_.init();

    // Get us the CPython GIL. However, when we return here,
    // it will get released and other Python threads can run.
    Python::ScopedGIL gil;

    std::string pyModPath(paramPyModPath(config));
    if (!pyModPath.empty())
        Python::addSysPath(pyModPath);

    std::string pyModName(paramPyModName(config));
    if (pyModName.empty()) {
        pythonCriticalError("PythonControl(%s): need Python module name (pymod-name)", sprintUnit.c_str());
        return;
    }

    PyObject* pyMod = PyImport_ImportModule(pyModName.c_str());
    if (!pyMod) {
        pythonCriticalError(
                "PythonControl(%s): cannot import module '%s'",
                sprintUnit.c_str(), pyModName.c_str());
        Python::dumpModulesEnv();
        return;
    }

    internal_.reset(new Internal(config));

    std::string pyConfigStr(paramPyModConfig(config));
    pyObject_ = Python::PyCallKw(
            pyMod, "init", "{s:s,s:s,s:O,s:O,s:l,s:s}",
            "name", "Sprint.PythonControl",
            "sprint_unit", sprintUnit_.c_str(),
            "reference", internal_->capsule_,
            "callback", internal_->callback_,
            "version_number", (long)versionNumber,
            "config", pyConfigStr.c_str());
    Py_CLEAR(pyMod);
    if (!pyObject_) {
        pythonCriticalError("PythonControl(%s): init() failed", sprintUnit_.c_str());
        return;
    }

    if (pyObject_ == Py_None) {
        Py_CLEAR(pyObject_);
        pythonCriticalError("PythonControl(%s): init() returned None", sprintUnit_.c_str());
        return;
    }
}

PythonControl::~PythonControl() {
    internal_.reset();
    if (pyObject_) {
        require(Py_IsInitialized());  // should not happen. only via pythonInitializer_.
        Python::ScopedGIL gil;
        Py_CLEAR(pyObject_);
    }
    pythonInitializer_.uninit();  // safe to call in any case
}

void PythonControl::exit() {
    if (!pyObject_)
        return;
    require(Py_IsInitialized());  // should not happen. only via pythonInitializer_.

    Python::ScopedGIL gil;

    PyObject* res = Python::PyCallKw(pyObject_, "exit", "{}");
    if (!res) {
        pythonCriticalError("PythonControl(%s): exit() failed", sprintUnit_.c_str());
        return;
    }

    Py_CLEAR(res);
}

static const Core::ParameterBool paramExtractAlignments(
        "extract-alignments",
        "extract alignments for PythonControl",
        false);

static const Core::ParameterBool paramSoftAlignments(
        "soft-alignments",
        "soft alignments / Baum-Welch alignments",
        false);

static const Core::ParameterString paramAlignmentPortName(
        "alignment-port-name",
        "name of the main data source port",
        "alignments");

struct NoneFeatureExtractor : public Speech::CorpusProcessor {
    typedef Speech::CorpusProcessor Precursor;
    Core::Ref<Speech::DataSource>   dataSource() const {
        return Core::Ref<Speech::DataSource>();
    }
    NoneFeatureExtractor(const Core::Configuration& c)
            : Core::Component(c), Precursor(c) {}
};

// See Speech::AligningFeatureExtractor.
template<typename BaseClass = Speech::FeatureExtractor, bool extractFeatures = true>
class PythonControlCorpusProcessor : public BaseClass {
public:
    typedef BaseClass                  Precursor;
    PythonControl&                     control_;
    bool                               firstSegment_;
    Core::Ref<const Am::AcousticModel> acousticModel_;
    std::shared_ptr<ClassLabelWrapper> classLabelWrapper_;
    Math::FastMatrix<f32>              features_;  // dim * time
    bool                               extractAlignments_;
    bool                               softAlignments_;
    Flow::PortId                       alignmentPortId_;
    size_t                             nTotalFrames_;
    AlignmentToPython                  alignmentToPython_;

    PythonControlCorpusProcessor(const Core::Configuration& c, PythonControl& control)
            : Core::Component(c),
              Precursor(c),
              control_(control),
              firstSegment_(true),
              extractAlignments_(paramExtractAlignments(c)),
              softAlignments_(paramSoftAlignments(c)),
              alignmentPortId_(Flow::IllegalPortId),
              nTotalFrames_(0) {
        if (extractAlignments_) {
            require(extractFeatures);
            const std::string alignmentPortName(paramAlignmentPortName(c));
            alignmentPortId_ = this->dataSource()->getOutput(alignmentPortName);
            if (alignmentPortId_ == Flow::IllegalPortId)
                this->criticalError("Flow network does not have an output named \"%s\"", alignmentPortName.c_str());
            requireInitAcousticModel();
            alignmentToPython_.parent_            = this;
            alignmentToPython_.criticalErrorFunc_ = control_.getPythonCriticalErrorFunc();
            alignmentToPython_.acousticModel_     = acousticModel_;
            alignmentToPython_.classLabelWrapper_ = classLabelWrapper_;
            alignmentToPython_.features_          = &features_;
        }
    }

    virtual ~PythonControlCorpusProcessor() {
        this->log("PythonControl: skipped frames: %zu, total frames: %zu", alignmentToPython_.nSkippedAlignmentFrames_, nTotalFrames_);
    }

    void requireInitAcousticModel() {
        if (acousticModel_)
            return;
        // e.g. see BufferedAlignedFeatureProcessor<T>::initAcousticModel()
        /* acoustic model to identify labels */
        Speech::ModelCombination modelCombination(this->select("model-combination"),
                                                  Speech::ModelCombination::useAcousticModel,
                                                  Am::AcousticModel::noEmissions | Am::AcousticModel::noStateTransition);
        modelCombination.load();
        acousticModel_ = modelCombination.acousticModel();
        require(acousticModel_);
        u32 nClasses = acousticModel_->nEmissions();
        this->log("number of classes of acoustic model: ") << nClasses;

        classLabelWrapper_.reset(new ClassLabelWrapper(this->select("class-labels"), nClasses));
        this->log("number of classes to accumulate: ") << classLabelWrapper_->nClassesToAccumulate();
        require_gt(classLabelWrapper_->nClassesToAccumulate(), 0);
    }

    void _extractFeatures() {
        require(extractFeatures);
        features_.resizeColsAndKeepContent(0);
        Core::Ref<Speech::Feature> feature;
        bool                       firstFeature = true;
        while (this->dataSource()->getData(feature)) {
            require_gt(feature->nStreams(), 0);
            const Speech::Feature::Vector& featureVector = *feature->mainStream();
            if (firstFeature) {
                if (features_.nRows() != featureVector.size())
                    features_.resize(featureVector.size(), 0);
                firstFeature = false;
            }
            features_.resizeColsAndKeepContent(features_.nColumns() + 1);
            features_.copy(featureVector, 0, features_.nColumns() - 1);
        }
        nTotalFrames_ += features_.nColumns();
    }

    void _extractAlignment(Python::ObjRef& pyAlignment, Python::ObjRef& pySoftAlignment) {
        require(extractFeatures);
        Flow::DataPtr<Flow::DataAdaptor<Speech::Alignment>> alignmentRef;
        if (this->dataSource()->getData(alignmentPortId_, alignmentRef)) {
            const Speech::Alignment& alignment = alignmentRef->data();
            if (softAlignments_)
                alignmentToPython_.extractSoftAlignment(alignment, pySoftAlignment);
            else
                alignmentToPython_.extractViterbiAlignment(alignment, pyAlignment);
        }
        else
            this->error("Failed to extract alignment.");
    }

    virtual void processSegment(Bliss::Segment* s) {
        // We don't call Precursor::processSegment() because we do the feature iteration here ourself.
        Python::ScopedGIL gil;

        Bliss::SpeechSegment* ss           = dynamic_cast<Bliss::SpeechSegment*>(s);
        const Bliss::Speaker* speaker      = ss ? ss->speaker() : NULL;
        const char*           speaker_name = speaker ? speaker->name().c_str() : NULL;
        Python::ObjRef        pyOrth;
        if (ss) {
            pyOrth.takeOver(PyBytes_FromStringAndSize(ss->orth().data(), ss->orth().size()));
        }

        if (extractFeatures)
            _extractFeatures();

        Python::ObjRef pyAlignment;
        Python::ObjRef pySoftAlignment;
        if (extractAlignments_)
            _extractAlignment(pyAlignment, pySoftAlignment);

        Python::ObjRef pyFeatures;
        if (extractFeatures) {
            Python::fastMatrix2numpy(control_.getPythonCriticalErrorFunc(), pyFeatures.obj, features_);
        }

        if (firstSegment_) {
            long inputDim = features_.nRows();
            if (!extractFeatures)
                inputDim = -1;
            long outputDim = -1;
            if (extractAlignments_) {
                require(classLabelWrapper_);
                outputDim = classLabelWrapper_->nClassesToAccumulate();
            }
            control_.run_custom(
                    "init_processing", "{s:l,s:l}",
                    "input_dim", inputDim,
                    "output_dim", outputDim);
            firstSegment_ = false;
        }

        control_.run_custom(
                "process_segment", "{s:s,s:O,s:s,s:O,s:O,s:O}",
                "name", s->fullName().c_str(),
                "orthography", pyOrth.obj ? pyOrth.obj : Py_None,
                "speaker_name", speaker_name,
                "features", pyFeatures.obj ? pyFeatures.obj : Py_None,
                "alignment", pyAlignment.obj ? pyAlignment.obj : Py_None,
                "soft_alignment", pySoftAlignment.obj ? pySoftAlignment.obj : Py_None);
    }
};

template<typename CorpusProcessor>
static void iterateCorpus(PythonControl& control) {
    // See NnTrainer::visitCorpus() as an example.
    CorpusProcessor corpusProcessor(control.getConfiguration(), control);

    Speech::CorpusVisitor corpusVisitor(corpusProcessor.select("corpus"));
    corpusProcessor.signOn(corpusVisitor);

    Bliss::CorpusDescription corpusDescription(corpusProcessor.select("corpus"));
    corpusDescription.accept(&corpusVisitor);
}

static const Core::ParameterBool paramExtractFeatures(
        "extract-features",
        "extract features for PythonControl",
        true);

void PythonControl::run_iterate_corpus() {
    if ((paramExtractFeatures(getConfiguration())))
        iterateCorpus<PythonControlCorpusProcessor<Speech::FeatureExtractor, true>>(*this);
    else
        iterateCorpus<PythonControlCorpusProcessor<NoneFeatureExtractor, false>>(*this);
}

void PythonControl::run_control_loop() {
    if (!pyObject_)
        return;
    Python::ScopedGIL gil;

    PyObject* res = Python::PyCallKw(
            pyObject_, "run_control_loop", "{s:O}",
            "callback", internal_->callback_);
    if (!res) {
        pythonCriticalError("PythonControl(%s): run_control_loop() failed", sprintUnit_.c_str());
        return;
    }

    Py_CLEAR(res);
}

// https://docs.python.org/2/c-api/arg.html
PyObject* PythonControl::run_custom_with_result(const char* method, const char* kwArgsFormat, ...) const {
    if (!pyObject_)
        return NULL;
    PyObject* meth   = NULL;
    PyObject* args   = NULL;
    PyObject* kwArgs = NULL;
    PyObject* res    = NULL;

    meth = PyObject_GetAttrString(pyObject_, method);
    if (!meth)
        goto final;

    args = PyTuple_New(0);
    if (!args)
        goto final;

    va_list vargs;
    va_start(vargs, kwArgsFormat);
    kwArgs = Py_VaBuildValue(kwArgsFormat, vargs);
    va_end(vargs);
    if (!kwArgs)
        goto final;

    res = PyObject_Call(meth, args, kwArgs);

final:
    Py_XDECREF(meth);
    Py_XDECREF(args);
    Py_XDECREF(kwArgs);
    if (!res) {
        pythonCriticalError("PythonControl(%s): run_custom(%s) failed", sprintUnit_.c_str(), method);
        return NULL;
    }
    return res;
}

// https://docs.python.org/2/c-api/arg.html
void PythonControl::run_custom(const char* method, const char* kwArgsFormat, ...) const {
    if (!pyObject_)
        return;
    Python::ScopedGIL gil;
    PyObject*         meth   = NULL;
    PyObject*         args   = NULL;
    PyObject*         kwArgs = NULL;
    PyObject*         res    = NULL;

    meth = PyObject_GetAttrString(pyObject_, method);
    if (!meth)
        goto final;

    args = PyTuple_New(0);
    if (!args)
        goto final;

    va_list vargs;
    va_start(vargs, kwArgsFormat);
    kwArgs = Py_VaBuildValue(kwArgsFormat, vargs);
    va_end(vargs);
    if (!kwArgs)
        goto final;

    res = PyObject_Call(meth, args, kwArgs);

final:
    Py_XDECREF(meth);
    Py_XDECREF(args);
    Py_XDECREF(kwArgs);
    if (!res) {
        pythonCriticalError("PythonControl(%s): run_custom(%s) failed", sprintUnit_.c_str(), method);
        return;
    }
    Py_CLEAR(res);
}

// Specialized over Core::Component::criticialError():
// Handles recent Python exceptions (prints them).
// Note that Py_Finalize() is not called here but registered via
// std::atexit(). See constructor code+comment.
Core::Component::Message PythonControl::pythonCriticalError(const char* msg, ...) const {
    Python::handlePythonError();

    va_list ap;
    va_start(ap, msg);
    Core::Component::Message msgHelper = Core::Component::vCriticalError(msg, ap);
    va_end(ap);
    return msgHelper;
}

Python::CriticalErrorFunc PythonControl::getPythonCriticalErrorFunc() const {
    return [this]() {
        return this->pythonCriticalError("PythonControl(%s): ", this->sprintUnit_.c_str());
    };
}

}  // namespace Nn
