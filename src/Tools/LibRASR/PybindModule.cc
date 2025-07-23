#include <string>
#include <pybind11/pybind11.h>

#include <Python/AllophoneStateFsaBuilder.hh>
#include <Python/Configuration.hh>

#include "LibRASR.hh"
#include "Search.hh"

namespace py = pybind11;

PYBIND11_MODULE(librasr, m) {
    static DummyApplication app;

    m.doc() = "RASR python module";

    py::class_<Core::Configuration> baseConfigClass(m, "_BaseConfig");

    py::class_<PyConfiguration> pyRasrConfig(m, "Configuration", baseConfigClass);
    pyRasrConfig.def(py::init<>());
    pyRasrConfig.def("set_from_file", static_cast<bool (Core::Configuration::*)(const std::string&)>(&Core::Configuration::setFromFile));

    py::class_<AllophoneStateFsaBuilder> pyFsaBuilder(m, "AllophoneStateFsaBuilder");
    pyFsaBuilder.def(py::init<const Core::Configuration&>());
    pyFsaBuilder.def("build_by_orthography", &AllophoneStateFsaBuilder::buildByOrthography);
    pyFsaBuilder.def("build_by_segment_name", &AllophoneStateFsaBuilder::buildBySegmentName);

    bindSearchAlgorithm(m);
}
