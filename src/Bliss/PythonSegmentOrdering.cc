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
#include "PythonSegmentOrdering.hh"
#include <Core/Component.hh>
#include <Core/Parameter.hh>
#include <Speech/CorpusVisitor.hh>
#include <Speech/DataSource.hh>
#include <Speech/Module.hh>
#include <Python.h>

/**
 * Python interface:
 *
 * def getSegmentList(corpusName, segmentList, segmentsInfo, config)
 *
 * corpusName is the name of the root corpus.
 * segmentList is a list of strings with the segment-names of the corpus.
 * segmentsInfo is a dict segmentName -> info, where info is a dict with info such as
 *   nframes (only if use-data-source = true, but that's the default).
 * getSegmentList() can be any generator. It should yield segment-name strings.
 *
 * Dummy example:
 *
 * def getSegmentList(corpusName, segmentList, segmentsInfo, config): return segmentList
 *
 * Usually, there is only one root corpus.
 * However, the function still might get called multiple times for the same root corpus
 * because e.g. for ProgressIndicationVisitorAdaptor, we count the total amount of segments.
 *
 * Note that there are usually other CorpusVisitors behind the SegmentOrderingVisitor
 * which might further filter out the segments, such as the SegmentPartitionVisitorAdaptor.
 **/

namespace Bliss {

static const Core::ParameterBool paramWithSegmentInfo(
        "python-segment-order-with-segment-info",
        "Whether to provide additional information for each segment.",
        false);

static const Core::ParameterBool paramUseDataSource(
        "use-data-source",
        "Whether to use data-source to extract segment info such as length.",
        false);

static const Core::ParameterBool paramPythonSegmentOrderAllowCopy(
        "python-segment-order-allow-copy",
        "Whether to allow a copy. This is used just for counting the segments.",
        false);

PythonSegmentOrderingVisitor::PythonSegmentOrderingVisitor(const std::string& pyModPath, const std::string& pyModName, const std::string& pyConfig, Core::Component& owner) {
    pythonInitializer.init();
    pyConfig_  = pyConfig;
    allowCopy_ = paramPythonSegmentOrderAllowCopy(owner.getConfiguration());
    withInfo_  = paramWithSegmentInfo(owner.getConfiguration());

    {
        Python::ScopedGIL gil;

        if (!pyModPath.empty())
            Python::addSysPath(pyModPath);

        pyMod_.takeOver(PyImport_ImportModule(pyModName.c_str()));
        if (!pyMod_) {
            Python::criticalError(
                    "python-segment-order: cannot import module '%s'",
                    pyModName.c_str());
            return;
        }
    }

    if (paramUseDataSource(owner.getConfiguration())) {
        if (!withInfo_)
            owner.error("python-segment-order: python-segment-order-with-segment-info must be enabled for use-data-source");
        // See Speech::DataExtractor.
        dataSource_ = Core::ref(Speech::Module::instance().createDataSource(owner.select("feature-extraction"), /*loadFromFile*/ true));
        require(dataSource_);
        dataSource_->respondToDelayedErrors();
        dataSource_->setProgressIndication(false);
    }
}

SegmentOrderingVisitor* PythonSegmentOrderingVisitor::copy() {
    // This is usually called via CorpusDescription::totalSegmentCount().
    if (allowCopy_)
        return new PythonSegmentOrderingVisitor(*this);
    return NULL;
}

size_t PythonSegmentOrderingVisitor::getSegmentNumFramesViaDataSourceSlow() {
    // We need to go through all the data to get the frames count.
    while (dataSource_->getData())
        ;

    const std::vector<size_t>& nFramess   = dataSource_->nFrames();
    Flow::PortId               mainPortId = dataSource_->mainPortId();
    if (mainPortId < 0 || (size_t)mainPortId >= nFramess.size()) {
        dataSource_->error("invalid main port %i, have ports # %zd", mainPortId, nFramess.size());
        return 0;
    }

    return nFramess[mainPortId];
}

size_t PythonSegmentOrderingVisitor::getSegmentNumFramesViaDataSource() {
    Flow::PortId mainPortId = dataSource_->mainPortId();
    ssize_t      n          = dataSource_->getRemainingDataLen(mainPortId);
    if (n < 0) {
        static bool didWarning = false;
        if (!didWarning) {
            dataSource_->warning("Cannot get segment len in a fast way. We use the slow method instead.");
            didWarning = true;
        }
        return getSegmentNumFramesViaDataSourceSlow();
    }
    return n;
}

size_t PythonSegmentOrderingVisitor::getSegmentNumFrames(Segment* segment) {
    verify(dataSource_);

    // We need to set the Corpus parameters on the data source.
    // Normally, the CorpusVisitor would do this.
    // First, clear previous parameters, then set the current ones.
    Speech::clearSegmentParametersOnDataSource(dataSource_, segment);
    Speech::setSegmentParametersOnDataSource(dataSource_, segment);

    // See Speech::DataExtractor.
    dataSource_->initialize(segment);

    size_t nFrames = getSegmentNumFramesViaDataSource();

    dataSource_->finalize();
    return nFrames;
}

PyObject* PythonSegmentOrderingVisitor::getSegmentsInfo() {
    // We expect to have the Python GIL.

    // dict[segmentName,info].
    Python::ObjRef segmentsDict;
    segmentsDict.takeOver(PyDict_New());

    for (const std::string& segmentName : segmentList_) {
        Segment* segment = getSegmentByName(segmentName);
        if (!segment) {
            Core::Application::us()->error("segment '%s' not found", segmentName.c_str());
            continue;
        }

        // dict[attrib,value] for segment.
        Python::ObjRef infoDict;
        infoDict.takeOver(PyDict_New());

        // So far, only the length.
        if (dataSource_) {
            size_t nFrames = getSegmentNumFrames(segment);
            Python::dict_SetItemString(infoDict, "nframes", nFrames);
        }

        Python::dict_SetItemString(infoDict, "time", segment->end() - segment->start());

        PyDict_SetItemString(segmentsDict, segmentName.c_str(), infoDict);
        infoDict.clear();
    }

    if (PyErr_Occurred())
        Python::criticalError("error collecting segments info");

    return segmentsDict.release();
}

void PythonSegmentOrderingVisitor::leaveCorpus(Bliss::Corpus* corpus) {
    curCorpus_.pop_back();
    if (!curCorpus_.empty()) {
        // not the root corpus
        return;
    }
    // corpus is the root corpus. We don't have our own copy of this one.

    prepareSegmentLoop();

    {
        CustomCorpusGuide corpusGuide(this, /* root */ corpus);
        Python::ScopedGIL gil;

        Python::ObjRef pySegmentList;
        Python::ObjRef iterator, item;

        // Now, pass the current segmentList_ to Python.
        // We expect to get some iterable segment-name-list returned.
        pySegmentList.takeOver(Python::PyCallKw(
                pyMod_,
                "getSegmentList", "{s:s,s:N,s:N,s:s}",
                "corpusName", corpus->name().c_str(),
                "segmentList", Python::newList(segmentList_),
                "segmentsInfo", withInfo_ ? getSegmentsInfo() : Python::incremented(Py_None),
                "config", pyConfig_.c_str()));
        if (!pySegmentList)
            Python::criticalError("python-segment-order: getSegmentList() failed");

        iterator.takeOver(PyObject_GetIter(pySegmentList));
        if (!iterator)
            Python::criticalError("python-segment-order: getSegmentList() did not return an iterable object");

        while (item.takeOver(PyIter_Next(iterator))) {
            std::string segmentName;
            if (!Python::pyObjToStr(item, segmentName)) {
                Python::criticalError("python-segment-order: segment-name is not a string but a %s",
                                      item->ob_type->tp_name);
                break;
            }

            // Pass on the segment to the rest of Sprint.
            // Release the Python GIL meanwhile.
            Python::ScopedAllowThreads allowThreads;
            corpusGuide.showSegmentByName(segmentName);
        }

        if (PyErr_Occurred())
            Python::criticalError("python-segment-order: failed to get next segment-name");

        iterator.clear();
    }

    finishSegmentLoop();
}

}  // namespace Bliss
