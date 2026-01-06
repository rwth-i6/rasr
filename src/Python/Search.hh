/** Copyright 2025 RWTH Aachen University. All rights reserved.
 *
 *  Licensed under the RWTH ASR License (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef _PYTHON_SEARCH_HH
#define _PYTHON_SEARCH_HH

#include <Flf/LatticeHandler.hh>
#include <Flf/Lexicon.hh>
#include <Search/SearchV2.hh>

#pragma push_macro("ensure")  // Macro duplication in numpy.h
#undef ensure
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#pragma pop_macro("ensure")

namespace py = pybind11;

struct TracebackItem {
    std::string        lemma;
    f32                amScore;
    f32                lmScore;
    std::optional<f32> confidenceScore;
    u32                startTime;
    u32                endTime;
};

typedef std::vector<TracebackItem> Traceback;

class SearchAlgorithm : public Core::Component {
public:
    static const Core::ParameterBool paramConfidenceScores;

    SearchAlgorithm(const Core::Configuration& c);

    // Call before starting a new recognition. Clean up existing data structures
    // from the previous run.
    void reset();

    // Call at the beginning of a new segment.
    void enterSegment();

    // Call after all features of the current segment have been passed
    void finishSegment();

    // Pass a feature array of shape [F] or [1, F]
    void putFeature(py::array_t<f32> const& feature);

    // Pass an array of features of shape [T, F] or [1, T, F]
    void putFeatures(py::array_t<f32> const& features);

    // Return the current best result. May contain unstable results.
    Traceback getCurrentBestTraceback();

    // Return the current stable result.
    Traceback getCurrentStableTraceback();

    // Return the current best n-best list. May contain unstable results.
    std::vector<Traceback> getCurrentNBestList(size_t nBestSize);

    // Convenience function to recognize a full segment given all the features as a tensor of shape [T, F]
    // Returns the recognition result
    Traceback recognizeSegment(py::array_t<f32> const& features);

    // Convenience function to recognize a full segment given all the features as a tensor of shape [T, F]
    // Returns a n-best list of recognition results
    std::vector<Traceback> recognizeSegmentNBest(py::array_t<f32> const& features, size_t nBestSize);

private:
    Traceback getTracebackWithoutConfidence();
    Traceback getTracebackWithConfidence();

    bool addConfidenceScores_;

    std::unique_ptr<Flf::LatticeHandler>       latticeHandler_;
    std::unique_ptr<Search::SearchAlgorithmV2> searchAlgorithm_;
    Flf::LexiconRef                            lexicon_;
    Speech::ModelCombination                   modelCombination_;
};

#endif  // _PYTHON_SEARCH_HH
