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

#include <Python/Search.hh>

void bind_search_algorithm(py::module_& module) {
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
            "Before recognizing the next segment, `reset` should be called.\n"
            "There is also a convenience function `recognize_segment` that performs all\n"
            "these steps in one go given an array of segment features.");

    pySearchAlgorithm.def(
            py::init<const Core::Configuration&>(),
            py::arg("config"),
            "Initialize search algorithm using a RASR config.");

    pySearchAlgorithm.def(
            "reset",
            &SearchAlgorithm::reset,
            "Call before starting a new recognition. Cleans up existing data structures from the previous run.");

    pySearchAlgorithm.def(
            "enter_segment",
            &SearchAlgorithm::enterSegment,
            "Call at the beginning of a new segment.");

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
            "recognize_segment",
            &SearchAlgorithm::recognizeSegment,
            py::arg("features"),
            "Convenience function to reset the search algorithm, start a segment, pass all the features as a numpy array of shape [T, F] or [1, T, F], finish the segment, and return the recognition result.");
}
