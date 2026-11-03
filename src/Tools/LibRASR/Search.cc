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

#include "Search.hh"

#include <sstream>

#include <Lm/ScaledLanguageModel.hh>
#include <Nn/LabelScorer/ScaledLabelScorer.hh>
#include <Python/Search.hh>
#include <Speech/ModelCombination.hh>

// Return the lanuage model
Lm::ScaledLanguageModel* getLanguageModel(Speech::ModelCombination& modelCombination) {
    return modelCombination.languageModel().get();
}

// Return the label scorer
Nn::ScaledLabelScorer* getLabelScorer(Speech::ModelCombination& modelCombination, size_t index = 0) {
    return dynamic_cast<Nn::ScaledLabelScorer*>(modelCombination.labelScorer(index).get());
}

void bindSearchAlgorithm(py::module_& module) {
    /*
     * ========================
     * === Traceback ==========
     * ========================
     */
    py::class_<TracebackItem> pyTracebackItem(
            module,
            "TracebackItem",
            "Represents attributes of a single traceback item.");
    pyTracebackItem.def_readwrite("lemma", &TracebackItem::lemma);
    pyTracebackItem.def_readwrite("am_score", &TracebackItem::amScore);
    pyTracebackItem.def_readwrite("lm_score", &TracebackItem::lmScore);
    pyTracebackItem.def_readwrite("start_time", &TracebackItem::startTime);
    pyTracebackItem.def_readwrite("end_time", &TracebackItem::endTime);

    pyTracebackItem.def(
            "__repr__",
            [](TracebackItem const& t) {
                std::stringstream ss;
                ss << "<TracebackItem(";
                ss << "lemma='" << t.lemma << "'";
                ss << ", am_score=" << t.amScore;
                ss << ", lm_score=" << t.lmScore;
                ss << ", start_time=" << t.startTime;
                ss << ", end_time=" << t.endTime;
                ss << ")>";
                return ss.str();
            });

    pyTracebackItem.def(
            "__str__",
            [](TracebackItem const& t) {
                return t.lemma;
            });

    /*
     * =========================
     * === Model Combination ===
     * =========================
     */

    py::class_<Speech::ModelCombination> pyModelCombination(
            module,
            "ModelCombination",
            "Combination of lexicon, acoustic model, label scorer and language model.");

    pyModelCombination.def(
            "label_scorer",
            &getLabelScorer,
            py::arg("index") = 0ul,
            py::return_value_policy::reference_internal,
            "Return the label scorer.");

    pyModelCombination.def(
            "language_model",
            &getLanguageModel,
            py::return_value_policy::reference_internal,
            "Return the language model.");

    py::class_<Lm::ScaledLanguageModel> pyScaledLanguageModel(
            module,
            "ScaledLanguageModel",
            "Language model with configurable scale.");

    pyScaledLanguageModel.def(
            "set_scale",
            [](Lm::ScaledLanguageModel& lm, Mc::Scale scale) {
                lm.setOwnScale(scale / lm.parentScale());
            },
            py::arg("scale"),
            "Set the effective LM scale, overriding the value from the config.");

    pyScaledLanguageModel.def(
            "scale",
            &Lm::ScaledLanguageModel::scale,
            "Return the current effective LM scale.");

    /*
     * ========================
     * === Search Algorithm ===
     * ========================
     */
    py::class_<SearchAlgorithm> pySearchAlgorithm(
            module,
            "SearchAlgorithm",
            "Class that can perform recognition using RASR.\n\n"
            "The search algorithm is configured with a RASR config object.\n"
            "It works by calling `enter_segment()`, passing segment features\n"
            "via `put_feature` or `put_features` and finally calling `finish_segment()`.\n"
            "Intermediate and final results can be retrieved via `get_current_best_traceback()`.\n"
            "There is also a convenience function `recognize_segment` that performs all\n"
            "these steps in one go given an array of segment features.");

    pySearchAlgorithm.def(
            py::init<const Core::Configuration&>(),
            py::arg("config"),
            "Initialize search algorithm using a RASR config.");

    pySearchAlgorithm.def(
            "enter_segment",
            &SearchAlgorithm::enterSegment,
            py::arg("name") = "",
            "Call at the beginning of a new segment.\n\n"
            "Opens a `segment` element in the log so that all messages until\n"
            "`finish_segment()` are grouped under it. The optional `name` is only\n"
            "used to identify the segment in the log.");

    pySearchAlgorithm.def(
            "finish_segment",
            &SearchAlgorithm::finishSegment,
            "Call after all features of the current segment have been passed");

    pySearchAlgorithm.def(
            "put_feature",
            &SearchAlgorithm::putFeature,
            py::arg("feature_vector"),
            "Pass a single feature as a numpy array of shape [F] or [1, F].");

    pySearchAlgorithm.def(
            "put_features",
            &SearchAlgorithm::putFeatures,
            py::arg("feature_array"),
            "Pass multiple features as a numpy array of shape [T, F] or [1, T, F].");

    pySearchAlgorithm.def(
            "get_current_best_traceback",
            &SearchAlgorithm::getCurrentBestTraceback,
            "Get the best traceback given all features that have been passed thus far.");

    pySearchAlgorithm.def(
            "get_common_prefix",
            &SearchAlgorithm::getCommonPrefix,
            "Get the stable traceback given all features that have been passed thus far.");

    pySearchAlgorithm.def(
            "get_current_n_best_list",
            &SearchAlgorithm::getCurrentNBestList,
            py::arg("n"),
            "Get a n-best list of tracebacks given all features that have been passed thus far.");

    pySearchAlgorithm.def(
            "recognize_segment",
            &SearchAlgorithm::recognizeSegment,
            py::arg("features"),
            py::arg("name") = "",
            "Convenience function to start a segment, pass all the features as a numpy array of shape [T, F] or [1, T, F], finish the segment, and return the recognition result.\n\n"
            "The optional `name` is only used to identify the segment in the log, as in\n"
            "`enter_segment()`.");

    pySearchAlgorithm.def(
            "recognize_segment_n_best",
            &SearchAlgorithm::recognizeSegmentNBest,
            py::arg("features"),
            py::arg("n"),
            py::arg("name") = "",
            "Convenience function to start a segment, pass all the features as a numpy array of shape [T, F] or [1, T, F], finish the segment, and return a n-best list of results.\n\n"
            "The optional `name` is only used to identify the segment in the log, as in\n"
            "`enter_segment()`.");

    pySearchAlgorithm.def(
            "model_combination",
            &SearchAlgorithm::modelCombination,
            py::return_value_policy::reference_internal,
            "Return the model combination used by the search.");

    /*
     * ================================
     * === Batched Search Algorithm ===
     * ================================
     */
    py::class_<BatchedSearchAlgorithm> pyBatchedSearchAlgorithm(
            module,
            "BatchedSearchAlgorithm",
            "Class that can perform recognition of several segments at the same time using RASR.\n\n"
            "The search algorithm is configured with a RASR config object (`search-algorithm.type` selects a multi-segment search).\n"
            "All active segments are decoded in lockstep so that the label scorer computations of all segments are batched.\n"
            "Segments are identified by handles: `add_segment()` returns a handle, features are passed via `put_features(handle, ...)`\n"
            "and `signal_no_more_features(handle)` signals that all features of the segment have been passed. `decode_step()` and\n"
            "`decode_many_steps()` advance the active segments. Once `is_finished(handle)` is true, the result can be retrieved via\n"
            "`get_current_best_traceback(handle)` and the segment should be released via `remove_segment(handle)`.\n"
            "New segments can be added at any time, also while other segments are being decoded (continuous batching).\n"
            "The convenience function `recognize_segments` performs all these steps for a list of feature arrays.");

    pyBatchedSearchAlgorithm.def(
            py::init<const Core::Configuration&>(),
            py::arg("config"),
            "Initialize batched search algorithm using a RASR config.");

    pyBatchedSearchAlgorithm.def(
            "max_active_segments",
            &BatchedSearchAlgorithm::maxActiveSegments,
            "Maximum number of segments that can be decoded at the same time.");

    pyBatchedSearchAlgorithm.def(
            "num_active_segments",
            &BatchedSearchAlgorithm::numActiveSegments,
            "Number of segments that have been added and are not finished yet.");

    pyBatchedSearchAlgorithm.def(
            "add_segment",
            &BatchedSearchAlgorithm::addSegment,
            py::arg("name") = "",
            "Add a new segment and return its handle. The optional name is used for logging.");

    pyBatchedSearchAlgorithm.def(
            "signal_no_more_features",
            &BatchedSearchAlgorithm::signalNoMoreFeatures,
            py::arg("segment"),
            "Call after all features of the segment have been passed. Does not decode the segment yet.");

    pyBatchedSearchAlgorithm.def(
            "put_feature",
            &BatchedSearchAlgorithm::putFeature,
            py::arg("segment"),
            py::arg("feature_vector"),
            "Pass a single feature of the segment as a numpy array of shape [F] or [1, F].");

    pyBatchedSearchAlgorithm.def(
            "put_features",
            &BatchedSearchAlgorithm::putFeatures,
            py::arg("segment"),
            py::arg("feature_array"),
            "Pass multiple features of the segment as a numpy array of shape [T, F] or [1, T, F].");

    pyBatchedSearchAlgorithm.def(
            "decode_step",
            &BatchedSearchAlgorithm::decodeStep,
            "Decode one step for the active segments (with pipelining only for one group). Returns whether anything changed.");

    pyBatchedSearchAlgorithm.def(
            "decode_many_steps",
            &BatchedSearchAlgorithm::decodeManySteps,
            "Decode as far as possible for all active segments given their features. Returns the number of steps.");

    pyBatchedSearchAlgorithm.def(
            "is_finished",
            &BatchedSearchAlgorithm::isFinished,
            py::arg("segment"),
            "Whether the search of the segment is finished.");

    pyBatchedSearchAlgorithm.def(
            "remove_segment",
            &BatchedSearchAlgorithm::removeSegment,
            py::arg("segment"),
            "Release all data of the segment. The handle becomes invalid afterwards.");

    pyBatchedSearchAlgorithm.def(
            "segments",
            &BatchedSearchAlgorithm::segments,
            "Handles of all segments that have been added and not removed yet.");

    pyBatchedSearchAlgorithm.def(
            "get_current_best_traceback",
            &BatchedSearchAlgorithm::getCurrentBestTraceback,
            py::arg("segment"),
            "Get the best traceback of the segment. Doesn't decode any further.");

    pyBatchedSearchAlgorithm.def(
            "get_common_prefix",
            &BatchedSearchAlgorithm::getCommonPrefix,
            py::arg("segment"),
            "Get the stable traceback of the segment. Doesn't decode any further.");

    pyBatchedSearchAlgorithm.def(
            "get_current_n_best_list",
            &BatchedSearchAlgorithm::getCurrentNBestList,
            py::arg("segment"),
            py::arg("n"),
            "Get a n-best list of tracebacks of the segment. Doesn't decode any further.");

    pyBatchedSearchAlgorithm.def(
            "recognize_segments",
            [](BatchedSearchAlgorithm&              self,
               std::vector<py::array_t<f32>> const& features,
               std::vector<std::string> const&      names) {
                return self.recognizeSegments(features, names);
            },
            py::arg("features"),
            py::arg("names") = std::vector<std::string>(),
            "Convenience function to recognize a list of segments given their features as numpy arrays of shape [T, F] or [1, T, F].\n"
            "Decodes up to `max_active_segments()` segments at the same time (configured in the search) with continuous\n"
            "batching and returns the recognition results in the order of the inputs.\n"
            "Optionally, `names` (only used for logging) can be given with one entry per segment.");

    pyBatchedSearchAlgorithm.def(
            "recognize_segments_n_best",
            [](BatchedSearchAlgorithm&              self,
               std::vector<py::array_t<f32>> const& features,
               size_t                               nBestSize,
               std::vector<std::string> const&      names) {
                return self.recognizeSegmentsNBest(features, nBestSize, names);
            },
            py::arg("features"),
            py::arg("n"),
            py::arg("names") = std::vector<std::string>(),
            "Same as `recognize_segments` but returns a n-best list of recognition results for each segment.");

    pyBatchedSearchAlgorithm.def(
            "model_combination",
            &BatchedSearchAlgorithm::modelCombination,
            py::return_value_policy::reference_internal,
            "Return the model combination used by the search.");
}
