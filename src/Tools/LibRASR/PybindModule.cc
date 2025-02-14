#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Python/AllophoneStateFsaBuilder.hh>
#include <Python/CallbackRegistry.hh>
#include <Python/Configuration.hh>
#include <Python/Search.hh>

#include "LibRASR.hh"

namespace py = pybind11;

PYBIND11_MODULE(librasr, m) {
    static DummyApplication app;

    m.doc() = "RASR python module";

    py::class_<Core::Configuration> baseConfigClass(m, "_BaseConfig");

    py::class_<PyConfiguration> pyRasrConfig(m, "Configuration", baseConfigClass);
    pyRasrConfig.def(py::init<>());
    pyRasrConfig.def("set_from_file",
                     (bool(Core::Configuration::*)(const std::string&)) & Core::Configuration::setFromFile);

    py::class_<AllophoneStateFsaBuilder> pyFsaBuilder(m, "AllophoneStateFsaBuilder");
    pyFsaBuilder.def(py::init<const Core::Configuration&>());
    pyFsaBuilder.def("build_by_orthography",
                     &AllophoneStateFsaBuilder::buildByOrthography);
    pyFsaBuilder.def("build_by_segment_name",
                     &AllophoneStateFsaBuilder::buildBySegmentName);

    m.def("register_callback", [](std::string const& name, py::function callback) { CallbackRegistry::instance().registerCallback(name, callback); }, "Register a Python callback function under a given name.");

    py::class_<TracebackItem> pyTracebackItem(m, "TracebackItem");
    pyTracebackItem.def_readwrite("lemma", &TracebackItem::lemma);
    pyTracebackItem.def_readwrite("am_score", &TracebackItem::amScore);
    pyTracebackItem.def_readwrite("lm_score", &TracebackItem::lmScore);
    pyTracebackItem.def_readwrite("start_time", &TracebackItem::startTime);
    pyTracebackItem.def_readwrite("end_time", &TracebackItem::endTime);
    pyTracebackItem.def("__repr__", [](TracebackItem const& t) {
        return "<TracebackItem(lemma='" + t.lemma + "', am_score=" + std::to_string(t.amScore) + ", lm_score=" + std::to_string(t.lmScore) + ", start_time=" + std::to_string(t.startTime) + ", end_time=" + std::to_string(t.endTime) + ")>";
    });

    py::class_<SearchAlgorithm> pySearchAlgorithm(m, "SearchAlgorithm");
    pySearchAlgorithm.def(py::init<const Core::Configuration&>(), py::arg("config"));
    pySearchAlgorithm.def("reset", &SearchAlgorithm::reset, "Call before starting a new recognition. Cleans up existing data structures from the previous run.");
    pySearchAlgorithm.def("enter_segment", &SearchAlgorithm::enterSegment, "Call at the beginning of a new segment.");
    pySearchAlgorithm.def("finish_segment", &SearchAlgorithm::finishSegment, "Call after all features of the current segment have been passed");
    pySearchAlgorithm.def("add_feature", &SearchAlgorithm::addFeature, py::arg("feature_vector"), "Pass a single feature as a numpy array of shape [F].");
    pySearchAlgorithm.def("add_features", &SearchAlgorithm::addFeatures, py::arg("feature_array"), "Pass multiple features as a numpy array of shape [T, F].");
    pySearchAlgorithm.def("get_current_best_transcription", &SearchAlgorithm::getCurrentBestTranscription, "Get the best transcription given all features that have been passed thus far.");
    pySearchAlgorithm.def("recognize_segment", &SearchAlgorithm::recognizeSegment, py::arg("features"), "Convenience function to start a segment, pass all the features as a numpy array of shape [T, F], finish the segment, and return the recognition result.");
    pySearchAlgorithm.def("get_current_best_traceback", &SearchAlgorithm::getCurrentBestTraceback);
}
