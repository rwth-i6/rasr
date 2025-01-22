#include "Search.hh"
#include <Flf/LatticeHandler.hh>
#include <Flf/Module.hh>
#include <Flow/Data.hh>
#include <Fsa/tBest.hh>
#include <Nn/Types.hh>

#include <Search/Module.hh>
#include <Speech/ModelCombination.hh>

namespace py = pybind11;

SearchAlgorithm::SearchAlgorithm(const Core::Configuration& c)
        : Core::Component(c),
          searchAlgorithm_(Search::Module::instance().createSearchAlgorithm(select("search-algorithm"))),
          modelCombination_(config) {
    modelCombination_.build(searchAlgorithm_->modelCombinationNeeded(), searchAlgorithm_->acousticModelNeeded());
    searchAlgorithm_->setModelCombination(modelCombination_);
}

void SearchAlgorithm::reset() {
    searchAlgorithm_->reset();
}

void SearchAlgorithm::enterSegment() {
    searchAlgorithm_->enterSegment();
}

void SearchAlgorithm::finishSegment() {
    searchAlgorithm_->finishSegment();
}

void SearchAlgorithm::logStatistics() {
    searchAlgorithm_->logStatistics();
}

void SearchAlgorithm::resetStatistics() {
    searchAlgorithm_->resetStatistics();
}

void SearchAlgorithm::addFeature(py::array_t<f32> const& feature) {
    size_t F = 0ul;
    if (feature.ndim() == 2) {
        if (feature.shape(0) != 1) {
            error() << "Received feature tensor with non-trivial batch dimension " << feature.shape(0) << "; should be 1";
        }
        F = feature.shape(1);
    }
    else if (feature.ndim() != 1) {
        error() << "Received feature vector of invalid dim " << feature.ndim() << "; should be 1";
        F = feature.shape(0);
    }

    // `dataPtr` is a shared_ptr wrapper around `input`.
    // Since we don't actually own the underlying data, it has a custom deleter that does nothing
    auto dataPtr = std::shared_ptr<const f32[]>(feature.data(), [](const f32*) {});
    searchAlgorithm_->addFeature(dataPtr, F);
}

void SearchAlgorithm::addFeatures(py::array_t<f32> const& features) {
    size_t T = 0ul;
    size_t F = 0ul;
    if (features.ndim() == 3) {
        if (features.shape(0) != 1) {
            error() << "Received feature tensor with non-trivial batch dimension " << features.shape(0) << "; should be 1";
        }
        T = features.shape(1);
        F = features.shape(2);
    }
    else if (features.ndim() != 2) {
        error() << "Received feature tensor of invalid dim " << features.ndim() << "; should be 2 or 3";
        T = features.shape(0);
        F = features.shape(1);
    }

    // `dataPtr` is a shared_ptr wrapper around `input`.
    // Since we don't actually own the underlying data, it has a custom deleter that does nothing
    auto dataPtr = std::shared_ptr<const f32[]>(features.data(), [](const f32*) {});
    searchAlgorithm_->addFeatures(dataPtr, T, F);
}

std::string SearchAlgorithm::getCurrentBestTranscription() {
    decodeMore();

    auto traceback = searchAlgorithm_->getCurrentBestTraceback();

    std::stringstream ss;

    for (auto it = traceback->begin(); it != traceback->end(); ++it) {
        if (it->lemma) {
            ss << it->lemma->symbol() << " ";
        }
    }

    return ss.str();
}

bool SearchAlgorithm::decodeMore() {
    return searchAlgorithm_->decodeMore();
}

std::string SearchAlgorithm::recognizeSegment(py::array_t<f32> const& features) {
    reset();
    resetStatistics();
    enterSegment();
    addFeatures(features);
    finishSegment();
    decodeMore();
    auto result = getCurrentBestTranscription();
    logStatistics();
    return result;
}
