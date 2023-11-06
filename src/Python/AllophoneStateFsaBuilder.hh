#ifndef _PYTHON_ALLOPHONESTATEFSABUILDER_HH
#define _PYTHON_ALLOPHONESTATEFSABUILDER_HH

#include <pybind11/pybind11.h>

#include <Core/Application.hh>
#include <Core/Archive.hh>
#include <Core/Configuration.hh>
#include <Nn/AllophoneStateFsaExporter.hh>

namespace py = pybind11;

class AllophoneStateFsaBuilder : public Core::Component {
public:
    AllophoneStateFsaBuilder(const Core::Configuration& c);
    ~AllophoneStateFsaBuilder() = default;

    py::tuple buildBySegmentName(const std::string& segmentName);

    py::tuple buildByOrthography(const std::string& orthography);

private:
    std::shared_ptr<Nn::AllophoneStateFsaExporter>    allophoneStateFsaExporter_;
    std::shared_ptr<Core::StringHashMap<std::string>> segmentToOrthMap_;

};

#endif // _PYTHON_ALLOPHONESTATEFSABUILDER_HH