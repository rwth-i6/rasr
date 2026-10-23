#ifndef _PYTHON_ALLOPHONESTATEFSABUILDER_HH
#define _PYTHON_ALLOPHONESTATEFSABUILDER_HH

#pragma push_macro("ensure")  // Macro duplication in numpy.h
#undef ensure
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/typing.h>
#pragma pop_macro("ensure")

#include <Core/Application.hh>
#include <Core/Archive.hh>
#include <Core/Configuration.hh>
#include <Nn/AllophoneStateFsaExporter.hh>

namespace py = pybind11;

class AllophoneStateFsaBuilder : public Core::Component {
public:
    // Number of states, number of edges, edges and weights of the automaton
    typedef py::typing::Tuple<py::int_, py::int_, py::array_t<u32>, py::array_t<f32>> Fsa;

    AllophoneStateFsaBuilder(const Core::Configuration& c);
    ~AllophoneStateFsaBuilder() = default;

    std::string getOrthographyBySegmentName(const std::string& segmentName);

    Fsa buildBySegmentName(const std::string& segmentName);

    Fsa buildByOrthography(const std::string& orthography);

private:
    std::shared_ptr<Nn::AllophoneStateFsaExporter>    allophoneStateFsaExporter_;
    std::shared_ptr<Core::StringHashMap<std::string>> segmentToOrthMap_;
};

#endif  // _PYTHON_ALLOPHONESTATEFSABUILDER_HH
