#ifndef _PYTHON_SEARCH_HH
#define _PYTHON_SEARCH_HH

#include <Search/SearchV2.hh>

#undef ensure  // macro duplication in pybind11/numpy.h
#include <Core/Application.hh>
#include <Core/Archive.hh>
#include <Core/Configuration.hh>
#include <Flf/FlfCore/Lattice.hh>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class SearchAlgorithm : public Core::Component {
public:
    SearchAlgorithm(const Core::Configuration& c);
    virtual ~SearchAlgorithm() {
        delete searchAlgorithm_;
    };

    // Call before starting a new recognition. Clean up existing data structures
    // from the previous run.
    void reset();

    // Call at the beginning of a new segment.
    // A segment can be one recording segment in a corpus for offline recognition
    // or a chunk of audio for online recognition.
    void enterSegment();

    // Call after all features of the current segment have been passed
    void finishSegment();

    // Write search statistics in log file
    void logStatistics();

    // Reset search statistics
    void resetStatistics();

    // Pass a feature tensor of shape [F] or [1, F]
    void addFeature(py::array_t<f32>);

    // Pass a tensor of features of shape [T, F] or [1, T, F]
    void addFeatures(py::array_t<f32>);

    // Return the current best result. May contain unstable results.
    std::string getCurrentBestTranscription();

    // Convenience function to recognize a full segment given all the features as a tensor of shape [T, F]
    // Returns the recognition result
    std::string recognizeSegment(py::array_t<f32>);

private:
    // Decode as much as possible given the currently available features. Return bool indicates whether any steps could be made.
    bool decodeMore();

    Flf::ConstLatticeRef buildLattice(Core::Ref<const Search::LatticeAdaptor>);

    Search::SearchAlgorithmV2* searchAlgorithm_;
    Speech::ModelCombination   modelCombination_;

    size_t currentFeatureIdx_;
};

#endif  // _PYTHON_SEARCH_HH
