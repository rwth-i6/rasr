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

#include "Align.hh"

#include <Python/Align.hh>

void bindAligner(py::module_& module) {
    /*
     * ========================
     * === Aligner ============
     * ========================
     */
    py::class_<Aligner> pyAligner(
            module,
            "Aligner",
            "Class that can compute forced alignments using RASR.\n\n"
            "The aligner is configured with a RASR config object.\n"
            "Internally, the aligner just performs recognition using a\n"
            "search algorithm v2 while incorporating a cheating LM\n"
            "to restrict the search space to only the correct transcription.\n"
            "Thus, a config for a SearchAlgorithm is also applicable\n"
            "to configure the aligner.");

    pyAligner.def(
            py::init<const Core::Configuration&>(),
            py::arg("config"),
            "Initialize aligner using a RASR config.");

    pyAligner.def(
            "align_segment",
            &Aligner::alignSegment,
            py::arg("features"),
            py::arg("orth"),
            "Compute forced-alignment of a segment described by a feature numpy array of shape [T, F] and a transcription string.");
}
