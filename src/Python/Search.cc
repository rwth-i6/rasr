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
          modelCombination_(new Speech::ModelCombination(config, searchAlgorithm_->requiredModelCombination(), searchAlgorithm_->requiredAcousticModel())) {
    searchAlgorithm_->setModelCombination(*modelCombination_);
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
    searchAlgorithm_->putFeature(dataPtr, F);
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
    searchAlgorithm_->putFeatures(dataPtr, T, F);
}

std::string SearchAlgorithm::getCurrentBestTranscription() {
    decodeManySteps();

    auto traceback = searchAlgorithm_->getCurrentBestTraceback();

    std::stringstream ss;

    for (auto it = traceback->begin(); it != traceback->end(); ++it) {
        if (it->pronunciation) {
            ss << it->pronunciation->lemma()->symbol() << " ";
        }
    }

    return ss.str();
}

Traceback SearchAlgorithm::getCurrentBestTraceback() {
    decodeManySteps();

    auto                       traceback = searchAlgorithm_->getCurrentBestTraceback();
    std::vector<TracebackItem> result;
    result.reserve(traceback->size());

    u32 prevTime = 0;

    for (auto it = traceback->begin(); it != traceback->end(); ++it) {
        result.push_back({
                it->pronunciation->lemma()->symbol(),
                it->score.acoustic,
                it->score.lm,
                prevTime,
                it->time,
        });
        prevTime = it->time;
    }
    return result;
}

bool SearchAlgorithm::decodeManySteps() {
    return searchAlgorithm_->decodeManySteps();
}

std::string SearchAlgorithm::recognizeSegment(py::array_t<f32> const& features) {
    reset();
    enterSegment();
    addFeatures(features);
    finishSegment();
    decodeManySteps();
    auto result = getCurrentBestTranscription();
    return result;
}

Nn::LabelScorer& SearchAlgorithm::getLabelScorer() const {
    return *searchAlgorithm_->getLabelScorer();
}
