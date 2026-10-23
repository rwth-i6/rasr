#include <string>
#include <pybind11/pybind11.h>

#include <Python/AllophoneStateFsaBuilder.hh>

#include "Configuration.hh"
#include "LabelScorer.hh"
#include "Lexicon.hh"
#include "LibRASR.hh"
#include "Search.hh"

namespace py = pybind11;

PYBIND11_MODULE(librasr, m) {
    static DummyApplication app;

    m.doc() = "RASR python module";

    // Registered first, as many functions take a configuration
    bindConfiguration(m);

    m.def(
            "init_logging",
            [](Core::Configuration const& config) {
                app.initLogging(config);
            },
            py::arg("config"),
            "Initialize RASR logging from config. Call directly after importing librasr, before creating other RASR objects.");

    py::class_<AllophoneStateFsaBuilder> pyFsaBuilder(m, "AllophoneStateFsaBuilder");
    pyFsaBuilder.def(py::init<const Core::Configuration&>(), py::arg("config"));
    pyFsaBuilder.def("get_orthography_by_segment_name", &AllophoneStateFsaBuilder::getOrthographyBySegmentName, py::arg("segment_name"));
    pyFsaBuilder.def("build_by_orthography", &AllophoneStateFsaBuilder::buildByOrthography, py::arg("orthography"));
    pyFsaBuilder.def("build_by_segment_name", &AllophoneStateFsaBuilder::buildBySegmentName, py::arg("segment_name"));

    bindLabelScorer(m);
    bindLexicon(m);
    bindSearchAlgorithm(m);
}
