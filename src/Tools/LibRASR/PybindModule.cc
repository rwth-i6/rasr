#include <string>
#include <pybind11/pybind11.h>

#include <Python/AllophoneStateFsaBuilder.hh>
#include <Python/Configuration.hh>

#include "LabelScorer.hh"
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
    pyRasrConfig.def(py::init<>());
    pyRasrConfig.def("set_from_file", static_cast<bool (Core::Configuration::*)(const std::string&)>(&Core::Configuration::setFromFile));

    py::class_<AllophoneStateFsaBuilder> pyFsaBuilder(m, "AllophoneStateFsaBuilder");
    pyFsaBuilder.def(py::init<const Core::Configuration&>());
    pyFsaBuilder.def("build_by_orthography", &AllophoneStateFsaBuilder::buildByOrthography);
    pyFsaBuilder.def("build_by_segment_name", &AllophoneStateFsaBuilder::buildBySegmentName);

    bind_label_scorer(m);
    bind_search_algorithm(m);
}
