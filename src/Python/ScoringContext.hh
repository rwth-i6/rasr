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

#ifndef PYTHON_SCORING_CONTEXT_HH
#define PYTHON_SCORING_CONTEXT_HH

#include <pybind11/pybind11.h>

#include <Nn/LabelScorer/ScoringContext.hh>

namespace py = pybind11;

namespace Python {

/*
 * Scoring context containing some arbitrary (hashable) python object
 */
struct PythonScoringContext : public Nn::ScoringContext {
    py::object object;

    PythonScoringContext()
            : object(py::none()) {}

    PythonScoringContext(py::object&& object)
            : object(object) {}

    bool   isEqual(Nn::ScoringContextRef const& other) const;
    size_t hash() const;
};

typedef Core::Ref<const PythonScoringContext> PythonScoringContextRef;

}  // namespace Python

#endif  // PYTHON_SCORING_CONTEXT_HH
