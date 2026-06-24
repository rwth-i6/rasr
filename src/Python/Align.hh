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

#ifndef _PYTHON_ALIGN_HH
#define _PYTHON_ALIGN_HH

#include <Search/SearchV2.hh>
#include "Search.hh"

#pragma push_macro("ensure")  // Macro duplication in numpy.h
#undef ensure
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#pragma pop_macro("ensure")

namespace py = pybind11;

class Aligner : public Core::Component {
public:
    Aligner(const Core::Configuration& c);

    // Align a speech segment given all the features as a tensor of shape [T, F] and the transcription string
    Traceback alignSegment(py::array_t<f32> const& features, std::string const& orth);

private:
    std::unique_ptr<Search::SearchAlgorithmV2> searchAlgorithm_;

    // Pass an array of features of shape [T, F] or [1, T, F]
    void putFeatures(py::array_t<f32> const& features);

    // Return the current best result.
    Traceback getBestTraceback();
};

// Compute forced alignment of targets using an array of scores of shape [T, V] or [1, T, V] with CTC label topology
std::vector<s32> ctcAlignment(py::array_t<f32> const& scores, py::array_t<s32> const& targets, s32 blankId);

// Compute forced alignment of targets using an array of scores of shape [T, V] or [1, T, V] with RNA label topology
std::vector<s32> rnaAlignment(py::array_t<f32> const& scores, py::array_t<s32> const& targets, s32 blankId);

#endif  // _PYTHON_ALIGN_HH
