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

#include <functional>

#include <Flf/LatticeHandler.hh>
#include <Flf/Lexicon.hh>
#include <Search/SearchV2.hh>
#include <Search/SearchV3.hh>

#pragma push_macro("ensure")  // Macro duplication in numpy.h
#undef ensure
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#pragma pop_macro("ensure")

namespace py = pybind11;

struct TracebackItem {
    std::string lemma;
    f32         amScore;
    f32         lmScore;
    u32         startTime;
    u32         endTime;
};

typedef std::vector<TracebackItem> Traceback;

class SearchAlgorithm : public Core::Component {
public:
    SearchAlgorithm(const Core::Configuration& c);

    // Closes a segment that is still open so the log stays well-formed.
    ~SearchAlgorithm();

    // Return the model combination used by the search.
    Speech::ModelCombination& modelCombination();

    // Call at the beginning of a new segment.
    // Opens a `segment` element in the log that groups everything logged until `finishSegment`.
    // `name` only identifies the segment in the log and may be empty.
    void enterSegment(std::string const& name = "");

    // Call after all features of the current segment have been passed
    void finishSegment();

    // Pass a feature array of shape [F] or [1, F]
    void putFeature(py::array_t<f32> const& feature);

    // Pass an array of features of shape [T, F] or [1, T, F]
    void putFeatures(py::array_t<f32> const& features);

    // Return the current best result. May contain unstable results.
    Traceback getCurrentBestTraceback();

    // Return the current stable result, i.e. common prefix of all current search hypotheses.
    Traceback getCommonPrefix();

    // Return the current best n-best list. May contain unstable results.
    std::vector<Traceback> getCurrentNBestList(size_t nBestSize);

    // Convenience function to recognize a full segment given all the features as a tensor of shape [T, F]
    // Returns the recognition result
    Traceback recognizeSegment(py::array_t<f32> const& features, std::string const& name = "");

    // Convenience function to recognize a full segment given all the features as a tensor of shape [T, F]
    // Returns a n-best list of recognition results
    std::vector<Traceback> recognizeSegmentNBest(py::array_t<f32> const& features, size_t nBestSize, std::string const& name = "");

private:
    std::unique_ptr<Flf::LatticeHandler>       latticeHandler_;
    std::unique_ptr<Search::SearchAlgorithmV2> searchAlgorithm_;
    Flf::LexiconRef                            lexicon_;
    Speech::ModelCombination                   modelCombination_;

    bool   segmentOpen_  = false;  // Whether a `segment` log element is currently open
    size_t segmentIndex_ = 0ul;    // Running index of the current segment, used to identify it in the log

    // Close the currently open `segment` log element, if there is one
    void closeSegmentLog();
};

/*
 * Recognition of several segments at the same time with a `Search::SearchAlgorithmV3`.
 *
 * Segments are identified by handles. The low-level functions mirror the `SearchAlgorithmV3` interface:
 * `add_segment` -> `put_features` -> `signal_no_more_features` and `decode_step`/`decode_many_steps` which advance all
 * active segments together. Results of finished segments can be retrieved via `get_current_best_traceback`.
 * Unlike `SearchAlgorithm`, retrieving results does not decode any further.
 *
 * The convenience functions `recognize_segments(_n_best)` recognize a list of segments with continuous batching:
 * up to `max_active_segments` segments are decoded at the same time and whenever a segment is finished, the next one
 * is added.
 */
class BatchedSearchAlgorithm : public Core::Component {
public:
    typedef Search::SearchAlgorithmV3::SegmentHandle SegmentHandle;

    BatchedSearchAlgorithm(const Core::Configuration& c);

    // Return the model combination used by the search.
    Speech::ModelCombination& modelCombination();

    // Maximum number of segments that can be decoded at the same time
    size_t maxActiveSegments() const;

    // Number of segments that have been added and are not finished yet
    size_t numActiveSegments() const;

    // Add a new segment and return its handle
    SegmentHandle addSegment(std::string const& name);

    // Call after all features of the segment have been passed
    void signalNoMoreFeatures(SegmentHandle segment);

    // Pass a feature array of shape [F] or [1, F] of the segment
    void putFeature(SegmentHandle segment, py::array_t<f32> const& feature);

    // Pass an array of features of shape [T, F] or [1, T, F] of the segment
    void putFeatures(SegmentHandle segment, py::array_t<f32> const& features);

    // Decode one step for the active segments (with pipelining only for one group). Returns whether anything changed.
    bool decodeStep();

    // Decode as far as possible for all active segments. Returns the number of `decodeStep` calls that made progress.
    unsigned decodeManySteps();

    // Whether the search of the segment is finished
    bool isFinished(SegmentHandle segment) const;

    // Release all data of the segment
    void removeSegment(SegmentHandle segment);

    // Handles of all segments that have not been removed yet
    std::vector<SegmentHandle> segments() const;

    // Return the current best result of the segment. May contain unstable results if the segment is not finished.
    Traceback getCurrentBestTraceback(SegmentHandle segment);

    // Return the current stable result of the segment, i.e. common prefix of all current search hypotheses.
    Traceback getCommonPrefix(SegmentHandle segment);

    // Return the current best n-best list of the segment. May contain unstable results if the segment is not finished.
    std::vector<Traceback> getCurrentNBestList(SegmentHandle segment, size_t nBestSize);

    // Convenience function to recognize several segments given their features as tensors of shape [T, F] each.
    // Decodes up to `maxActiveSegments()` segments at the same time. `names` (only used for logging) is either empty
    // or has one entry per segment.
    // Returns the recognition results in the order of the inputs.
    std::vector<Traceback> recognizeSegments(std::vector<py::array_t<f32>> const& features, std::vector<std::string> const& names = {});

    // Same as `recognizeSegments` but returns a n-best list of recognition results for each segment.
    std::vector<std::vector<Traceback>> recognizeSegmentsNBest(std::vector<py::array_t<f32>> const& features, size_t nBestSize, std::vector<std::string> const& names = {});

private:
    std::unique_ptr<Flf::LatticeHandler>       latticeHandler_;
    std::unique_ptr<Search::SearchAlgorithmV3> searchAlgorithm_;
    Flf::LexiconRef                            lexicon_;
    Speech::ModelCombination                   modelCombination_;

    // Recognize the segments with continuous batching and call `storeResult` for each segment once it is finished
    void recognizeSegments(std::vector<py::array_t<f32>> const& features, std::vector<std::string> const& names, std::function<void(size_t, SegmentHandle)> const& storeResult);
};

#endif  // _PYTHON_SEARCH_HH
