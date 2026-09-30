/* Copyright 2025 RWTH Aachen University. All rights reserved.
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

#include "FeatureExtractor.hh"
#include <Speech/Module.hh>

PYBIND11_DECLARE_HOLDER_TYPE(T, Core::Ref<T>, true);

class PublicFeatureExtractor : public Speech::FeatureExtractor {
public:
    using Speech::FeatureExtractor::processFeature;
    using Speech::FeatureExtractor::setFeatureDescription;
};

// Trampoline class that forwards virtual C++ callbacks to overrides on Python
// subclasses of FeatureExtractor. PYBIND11_OVERRIDE_NAME maps the C++
// camelCase method names to the snake_case names exposed by the Python API.
class PyFeatureExtractor : public Speech::FeatureExtractor {
public:
    PyFeatureExtractor(const Core::Configuration& c, bool loadFromFile = true)
            : Core::Component(c), Speech::FeatureExtractor(c, loadFromFile) {}
    void setFeatureDescription(const Mm::FeatureDescription& description) override {
        PYBIND11_OVERRIDE_NAME(void, Speech::FeatureExtractor, "set_feature_description", setFeatureDescription, description);
    }
    void processFeature(Core::Ref<const Speech::Feature> feature) override {
        PYBIND11_OVERRIDE_NAME(void, Speech::FeatureExtractor, "process_feature", processFeature, feature);
    }
    void processSegment(Bliss::Segment* segment) override {
        PYBIND11_OVERRIDE_NAME(void, Speech::FeatureExtractor, "process_segment", processSegment, segment);
    }
};

void bindFeatureExtractor(py::module_& m) {
    py::class_<Speech::DataExtractor, Speech::CorpusProcessor>(m, "DataExtractor")
            .def(py::init<const Core::Configuration&, bool>())
            .def("sign_on", &Speech::DataExtractor::signOn, py::keep_alive<2, 1>())
            .def("enter_corpus", &Speech::DataExtractor::enterCorpus)
            .def("leave_corpus", &Speech::DataExtractor::leaveCorpus)
            .def("enter_recording", &Speech::DataExtractor::enterRecording)
            .def("enter_segment", &Speech::DataExtractor::enterSegment)
            .def("leave_segment", &Speech::DataExtractor::leaveSegment)
            .def("process_segment", &Speech::DataExtractor::processSegment);

    py::class_<Speech::FeatureExtractor, Speech::DataExtractor, PyFeatureExtractor>(m, "FeatureExtractor")
            .def(py::init<const Core::Configuration&, bool>())
            .def("process_segment", &PublicFeatureExtractor::processSegment)
            .def("process_feature", &PublicFeatureExtractor::processFeature)
            .def("set_feature_description", &PublicFeatureExtractor::setFeatureDescription);
}
