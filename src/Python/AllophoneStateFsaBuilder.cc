#include "AllophoneStateFsaBuilder.hh"

#include <memory>
#include <vector>

#undef ensure  // macro duplication in pybind11/numpy.h
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Bliss/CorpusDescription.hh>
#include <Core/Application.hh>
#include <Core/Archive.hh>
#include <Core/Component.hh>
#include <Nn/AllophoneStateFsaExporter.hh>

namespace py = pybind11;

struct BuildSegmentToOrthMapVisitor : public Bliss::CorpusVisitor {
    BuildSegmentToOrthMapVisitor()
            : Bliss::CorpusVisitor(), map_(new Core::StringHashMap<std::string>()) {}

    virtual void visitSpeechSegment(Bliss::SpeechSegment* s) {
        (*map_)[s->fullName()] = s->orth();
    }

    std::shared_ptr<Core::StringHashMap<std::string>> map_;
};

static std::shared_ptr<Core::StringHashMap<std::string>> build_segment_to_orth_map(Core::Configuration const& config) {
    Bliss::CorpusDescription     corpus(config);
    BuildSegmentToOrthMapVisitor visitor;
    corpus.accept(&visitor);
    return visitor.map_;
}

AllophoneStateFsaBuilder::AllophoneStateFsaBuilder(const Core::Configuration& c)
        : Core::Component(c),
          allophoneStateFsaExporter_(nullptr),
          segmentToOrthMap_(nullptr) {
    // init exporter and orth map
    allophoneStateFsaExporter_ = std::make_shared<Nn::AllophoneStateFsaExporter>(select("alignment-fsa-exporter"));
    segmentToOrthMap_          = build_segment_to_orth_map(select("corpus"));
}

std::string AllophoneStateFsaBuilder::getOrthographyBySegmentName(const std::string& segmentName) {
    auto iter = segmentToOrthMap_->find(segmentName);
    if (iter == segmentToOrthMap_->end()) {
        throw std::invalid_argument("Could not find segment with name " + segmentName);
    }
    return iter->second;
}

py::tuple AllophoneStateFsaBuilder::buildBySegmentName(const std::string& segmentName) {
    return buildByOrthography(getOrthography(segmentName));
}

py::tuple AllophoneStateFsaBuilder::buildByOrthography(const std::string& orthography) {
    Nn::AllophoneStateFsaExporter::ExportedAutomaton automaton = allophoneStateFsaExporter_->exportFsaForOrthography(orthography);
    return py::make_tuple(
            automaton.num_states,
            automaton.num_edges,
            py::array_t<u32>(automaton.edges.size(), automaton.edges.data()),
            py::array_t<f32>(automaton.weights.size(), automaton.weights.data()));
}
