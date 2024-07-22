#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/iostream.h>

#include <Am/Module.hh>
#include <Audio/Module.hh>
#include <Bliss/CorpusDescription.hh>
#include <Bliss/CorpusParser.hh>
#include <Bliss/Lexicon.hh>
#include <Core/Application.hh>
#include <Flf/Module.hh>
#include <Flow/Module.hh>
#include <Lm/Module.hh>
#include <Math/Module.hh>
#include <Mm/Module.hh>
#include <Modules.hh>
#ifdef MODULE_NN
#include <Nn/Module.hh>
#endif
#ifdef MODULE_ONNX
#include <Onnx/Module.hh>
#endif
#include <Signal/Module.hh>
#include <Speech/Module.hh>
#ifdef MODULE_TENSORFLOW
#include <Tensorflow/Module.hh>
#endif

#include <Python/AllophoneStateFsaBuilder.hh>
#include <Python/Configuration.hh>

#include "LibRASR.hh"

namespace py = pybind11;

PYBIND11_MODULE(librasr, m) {
    static DummyApplication app;

    m.doc() = "RASR python module";

    py::class_<Core::Configuration> baseConfigClass(m, "_BaseConfig");

    py::class_<PyConfiguration> pyRasrConfig(m, "Configuration", baseConfigClass);
    pyRasrConfig.def(py::init<>());
    pyRasrConfig.def("set_from_file",
                     (bool (Core::Configuration::*)(const std::string&)) &Core::Configuration::setFromFile);

    py::class_<AllophoneStateFsaBuilder> pyFsaBuilder(m, "AllophoneStateFsaBuilder");
    pyFsaBuilder.def(py::init<const Core::Configuration&>());
    pyFsaBuilder.def("build_by_orthography",
                     &AllophoneStateFsaBuilder::buildByOrthography);
    pyFsaBuilder.def("build_by_segment_name",
                     &AllophoneStateFsaBuilder::buildBySegmentName);

    py::class_<Bliss::Symbol> symbol(m, "Symbol");
    symbol
    .def(py::init<>())
    .def(py::init<const Bliss::Symbol&>())
    .def("length", &Bliss::Symbol::length)
    .def("__eq__", &Bliss::Symbol::operator==)
    .def("__ne__", &Bliss::Symbol::operator!=)
    .def("__bool__", [](const Bliss::Symbol &self) { return !self.operator!(); })
    .def("to_bool", &Bliss::Symbol::operator bool)
    .def("to_string", &Bliss::Symbol::operator Bliss::Symbol::String)
    .def("str", &Bliss::Symbol::str)
    .def_static("cast", &Bliss::Symbol::cast);

    py::class_<Bliss::Symbol::Hash>(symbol, "Hash")
    .def("__call__", &Bliss::Symbol::Hash::operator());

    py::class_<Bliss::Symbol::Equality>(symbol, "Equality")
    .def("__call__", &Bliss::Symbol::Equality::operator());

    py::class_<Bliss::OrthographicFormList>(m, "OrthographicFormList")
    .def(py::init<const Bliss::Symbol*, const Bliss::Symbol*>())
    .def(py::init<>())
    .def(py::init<const Bliss::SymbolSequence<Bliss::OrthographicForm>&>())
    .def("valid", &Bliss::SymbolSequence<Bliss::OrthographicForm>::valid)
    .def("size", &Bliss::SymbolSequence<Bliss::OrthographicForm>::size)
    .def("length", &Bliss::SymbolSequence<Bliss::OrthographicForm>::length)
    .def("is_epsilon", &Bliss::SymbolSequence<Bliss::OrthographicForm>::isEpsilon)
    .def("front", &Bliss::SymbolSequence<Bliss::OrthographicForm>::front, py::return_value_policy::reference_internal)
    .def("begin", &Bliss::SymbolSequence<Bliss::OrthographicForm>::begin, py::return_value_policy::reference_internal)
    .def("end", &Bliss::SymbolSequence<Bliss::OrthographicForm>::end, py::return_value_policy::reference_internal)
    .def("__getitem__", &Bliss::OrthographicFormList::operator[]);

    py::class_<Bliss::SyntacticTokenSequence>(m, "SyntacticTokenSequence")
    .def(py::init<>())
    .def(py::init<const Bliss::SymbolSequence<const Bliss::SyntacticToken*>&>())
    .def("valid", &Bliss::SymbolSequence<const Bliss::SyntacticToken*>::valid)
    .def("size", &Bliss::SymbolSequence<const Bliss::SyntacticToken*>::size)
    .def("length", &Bliss::SymbolSequence<const Bliss::SyntacticToken*>::length)
    .def("is_epsilon", &Bliss::SymbolSequence<const Bliss::SyntacticToken*>::isEpsilon)
    .def("front", &Bliss::SymbolSequence<const Bliss::SyntacticToken*>::front, py::return_value_policy::reference_internal)
    .def("__getitem__", &Bliss::SymbolSequence<const Bliss::SyntacticToken*>::operator[]);

    py::class_<Bliss::Token>(m, "Token")
    .def("symbol", &Bliss::Token::symbol)
    .def("id", &Bliss::Token::id)
    .def_readonly_static("invalid_id", &Bliss::Token::invalidId);

}
