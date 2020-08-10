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
#ifndef _BLISS_PYTHONSEGMENTORDERING_HH
#define _BLISS_PYTHONSEGMENTORDERING_HH

#include <Core/ReferenceCounting.hh>
#include <Python/Init.hh>
#include <Python/Utilities.hh>
#include <Python.h>

#include "SegmentOrdering.hh"

namespace Core {
class Component;
}

namespace Speech {
class DataSource;
}

namespace Bliss {

class PythonSegmentOrderingVisitor : public SegmentOrderingVisitor {
    Python::Initializer           pythonInitializer;
    Python::ObjRef                pyMod_;
    std::string                   pyConfig_;
    bool                          allowCopy_;
    bool                          withInfo_;
    Core::Ref<Speech::DataSource> dataSource_;  // used to get seq len

    size_t    getSegmentNumFramesViaDataSourceSlow();
    size_t    getSegmentNumFramesViaDataSource();
    size_t    getSegmentNumFrames(Segment* segment);
    PyObject* getSegmentsInfo();

public:
    PythonSegmentOrderingVisitor(const std::string& pyModPath, const std::string& pyModName, const std::string& pyConfig, Core::Component& owner);
    virtual SegmentOrderingVisitor* copy();

    virtual void leaveCorpus(Bliss::Corpus* corpus);
};

}  // namespace Bliss

#endif  // PYTHONSEGMENTORDERING_HH
