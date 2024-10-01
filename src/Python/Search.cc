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

void SearchAlgorithm::addFeature(py::array_t<double> feature) {
    // Transform features of shape [1, F] to [F]
    if (feature.ndim() == 2) {
        if (feature.shape(0) != 1) {
            error() << "Received feature tensor with non-trivial batch dimension " << feature.shape(0) << "; should be 1";
        }
        feature = feature.reshape({feature.shape(1)});
    }

    if (feature.ndim() != 1) {
        error() << "Received feature vector of invalid dim " << feature.ndim() << "; should be 1";
    }

    // Read-only view of the array (without bounds checking)
    auto buffer = feature.unchecked<1>();

    // Shape of the array: [F]
    auto F = buffer.shape(1);

    std::vector<f32> currentFeature;
    currentFeature.reserve(F);
    for (ssize_t f = 0ul; f < F; ++f) {
        currentFeature.push_back(buffer(f));
    }
    addFeatureInternal(currentFeature);
}

void SearchAlgorithm::addFeatures(py::array_t<double> features) {
    // Transform features of shape [1, T, F] to [T, F]
    if (features.ndim() == 3) {
        if (features.shape(0) != 1) {
            error() << "Received feature tensor with non-trivial batch dimension " << features.shape(0) << "; should be 1";
        }
        features = features.reshape({features.shape(1), features.shape(2)});
    }

    if (features.ndim() != 2) {
        error() << "Received feature tensor of invalid dim " << features.ndim() << "; should be 2 or 3";
    }

    // Read-only view of the array (without bounds checking)
    auto buffer = features.unchecked<2>();

    // Shape of the array: [T, F]
    auto T = buffer.shape(0);
    auto F = buffer.shape(1);

    // Iterate over time axis and slice off each individual feature vector
    for (ssize_t t = 0ul; t < T; ++t) {
        std::vector<f32> currentFeature;
        currentFeature.reserve(F);
        for (ssize_t f = 0ul; f < F; ++f) {
            currentFeature.push_back(buffer(t, f));
        }
        addFeatureInternal(currentFeature);
    }
}

void SearchAlgorithm::addFeatureInternal(const std::vector<float>& feature) {
    // Abuse feature index as timestamp
    searchAlgorithm_->addFeature(Flow::dataPtr(new Nn::FeatureVector(feature, currentFeatureIdx_, currentFeatureIdx_ + 1)));
    ++currentFeatureIdx_;
}

std::string SearchAlgorithm::getCurrentBestTranscription() {
    decodeMore();

    auto traceback = searchAlgorithm_->getCurrentBestTraceback();

    std::stringstream ss;

    for (auto it = traceback->begin(); it != traceback->end(); ++it) {
        ss << it->lemma->symbol() << " ";
    }

    return ss.str();
}

bool SearchAlgorithm::decodeMore() {
    return searchAlgorithm_->decodeMore();
}

std::string SearchAlgorithm::recognizeSegment(py::array_t<double> features) {
    enterSegment();
    addFeatures(features);
    finishSegment();
    decodeMore();
    auto result = getCurrentBestTranscription();
    reset();
    return result;
}
