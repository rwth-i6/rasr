#include <string>
#include <pybind11/pybind11.h>

#include <Python/AllophoneStateFsaBuilder.hh>
#include <Python/Configuration.hh>

#include "LabelScorer.hh"
#include "Lexicon.hh"
#include "LibRASR.hh"
#include "Search.hh"

namespace py = pybind11;

PYBIND11_MODULE(librasr, m) {
    static DummyApplication app;

    m.doc() = "RASR python module";

    // TODO: Overhaul Configuration pybinds to make Configurations better to interact with from python-side.
    py::class_<Core::Configuration> baseConfigClass(m, "_BaseConfig");
    baseConfigClass.def(
            "__getitem__",
            [](Core::Configuration const& self, std::string const& key) {
                std::string result;
                if (self.get(key, result)) {
                    return result;
                }
                else {
                    std::cerr << "WARNING: Tried to get config value for key '" << key << "' but it was not configured. Return empty string.\n";
                    return std::string();
                }
            },
            py::arg("key"),
            "Retrieve the configured value of a specific parameter key as an unprocessed string.");

    py::class_<PyConfiguration> pyRasrConfig(m, "Configuration", baseConfigClass);
    pyRasrConfig.def(py::init<>())
            .def(py::init<PyConfiguration const&>())
            .def(py::init<PyConfiguration const&, std::string const&>())
            .def("enable_logging", &Core::Configuration::enableLogging)
            .def("set_from_file", static_cast<bool (Core::Configuration::*)(const std::string&)>(&Core::Configuration::setFromFile))
            .def("get_selection", &Core::Configuration::getSelection)
            .def("get_name", &Core::Configuration::getName)
            .def("set_selection", &Core::Configuration::setSelection)
            .def("set", &PyConfiguration::set)
            .def("resolve", &Core::Configuration::resolve)
            .def("__getitem__", [](Core::Configuration const& self, std::string const& parameter) {
                std::string value;
                if (self.get(parameter, value)) {
                    return std::optional<std::string>(value);
                }
                else {
                    return std::optional<std::string>();
                }
            });

    py::class_<AllophoneStateFsaBuilder> pyFsaBuilder(m, "AllophoneStateFsaBuilder");
    pyFsaBuilder.def(py::init<const Core::Configuration&>());
    pyFsaBuilder.def("get_orthography_by_segment_name", &AllophoneStateFsaBuilder::getOrthographyBySegmentName);
    pyFsaBuilder.def("build_by_orthography", &AllophoneStateFsaBuilder::buildByOrthography);
    pyFsaBuilder.def("build_by_segment_name", &AllophoneStateFsaBuilder::buildBySegmentName);

    bindLabelScorer(m);
    bindSearchAlgorithm(m);
    bindLexicon(m);
}
