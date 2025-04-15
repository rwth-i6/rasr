/** Copyright 2024 RWTH Aachen University. All rights reserved.
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
#include "ScoringContext.hh"
#include <Core/MurmurHash.hh>
#include <pybind11/cast.h>
#include <pybind11/pytypes.h>

namespace Nn {

typedef Mm::EmissionIndex LabelIndex;

// Auxiliary function to merge multiple hashes into one via the boost way
// See https://www.boost.org/doc/libs/1_43_0/doc/html/hash/reference.html#boost.hash_combine
size_t combineHashes(size_t hash1, size_t hash2) {
    if (hash1 == 0ul) {
        return hash2;
    }
    if (hash2 == 0ul) {
        return hash1;
    }
    return hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
}

/*
 * =============================
 * ====== ScoringContext =======
 * =============================
 */
bool ScoringContext::isEqual(ScoringContextRef const& other) const {
    return true;
}

size_t ScoringContext::hash() const {
    return 0ul;
}

/*
 * =============================
 * === StepScoringContext ======
 * =============================
 */
size_t StepScoringContext::hash() const {
    return currentStep;
}

bool StepScoringContext::isEqual(ScoringContextRef const& other) const {
    return currentStep == dynamic_cast<const StepScoringContext*>(other.get())->currentStep;
}

/*
 * =============================
 * === LabelSeqScoringContext ==
 * =============================
 */
size_t LabelSeqScoringContext::hash() const {
    return Core::MurmurHash3_x64_64(reinterpret_cast<void const*>(labelSeq.data()), labelSeq.size() * sizeof(LabelIndex), 0x78b174eb);
}

bool LabelSeqScoringContext::isEqual(ScoringContextRef const& other) const {
    auto* otherPtr = dynamic_cast<const LabelSeqScoringContext*>(other.get());
    if (labelSeq.size() != otherPtr->labelSeq.size()) {
        return false;
    }

    for (auto it_l = labelSeq.begin(), it_r = otherPtr->labelSeq.begin(); it_l != labelSeq.end(); ++it_l, ++it_r) {
        if (*it_l != *it_r) {
            return false;
        }
    }

    return true;
}

/*
 * =============================
 * === SeqStepScoringContext ===
 * =============================
 */
size_t SeqStepScoringContext::hash() const {
    return combineHashes(currentStep, Core::MurmurHash3_x64_64(reinterpret_cast<void const*>(labelSeq.data()), labelSeq.size() * sizeof(LabelIndex), 0x78b174eb));
}

bool SeqStepScoringContext::isEqual(ScoringContextRef const& other) const {
    auto* otherPtr = dynamic_cast<const SeqStepScoringContext*>(other.get());
    if (currentStep != otherPtr->currentStep) {
        return false;
    }

    if (labelSeq.size() != otherPtr->labelSeq.size()) {
        return false;
    }

    for (auto it_l = labelSeq.begin(), it_r = otherPtr->labelSeq.begin(); it_l != labelSeq.end(); ++it_l, ++it_r) {
        if (*it_l != *it_r) {
            return false;
        }
    }

    return true;
}

#ifdef MODULE_ONNX
/*
 * =============================
 * = HiddenStateScoringContext =
 * =============================
 */
size_t HiddenStateScoringContext::hash() const {
    return Core::MurmurHash3_x64_64(reinterpret_cast<void const*>(labelSeq.data()), labelSeq.size() * sizeof(LabelIndex), 0x78b174eb);
}

bool HiddenStateScoringContext::isEqual(ScoringContextRef const& other) const {
    auto* otherPtr = dynamic_cast<const HiddenStateScoringContext*>(other.get());
    if (labelSeq.size() != otherPtr->labelSeq.size()) {
        return false;
    }

    for (auto it_l = labelSeq.begin(), it_r = otherPtr->labelSeq.begin(); it_l != labelSeq.end(); ++it_l, ++it_r) {
        if (*it_l != *it_r) {
            return false;
        }
    }

    return true;
}
#endif  // MODULE_ONNX

/*
 * =============================
 * == CTCPrefixScoringContext ==
 * =============================
 */

size_t CTCPrefixScoringContext::hash() const {
    return combineHashes(Core::MurmurHash3_x64_64(reinterpret_cast<void const*>(prefixScores.data()), prefixScores.size() * sizeof(PrefixScore), 0x78b174eb), lastLabel);
}

bool CTCPrefixScoringContext::isEqual(ScoringContextRef const& other) const {
    auto* otherPtr = dynamic_cast<const CTCPrefixScoringContext*>(other.get());

    if (lastLabel != otherPtr->lastLabel) {
        return false;
    }

    if (prefixScores.size() != otherPtr->prefixScores.size()) {
        return false;
    }

    for (auto it_l = prefixScores.begin(), it_r = otherPtr->prefixScores.begin(); it_l != prefixScores.end(); ++it_l, ++it_r) {
        if (it_l->blankEndingScore != it_r->blankEndingScore or it_l->nonBlankEndingScore != it_r->nonBlankEndingScore) {
            return false;
        }
    }

    return true;
}

/*
 * =============================
 * === CombineScoringContext ===
 * =============================
 */
size_t CombineScoringContext::hash() const {
    size_t value = 0ul;
    for (auto const& scoringContext : scoringContexts) {
        value = combineHashes(value, scoringContext->hash());
    }
    return value;
}

bool CombineScoringContext::isEqual(ScoringContextRef const& other) const {
    auto* otherPtr = dynamic_cast<const CombineScoringContext*>(other.get());

    if (otherPtr == nullptr or scoringContexts.size() != otherPtr->scoringContexts.size()) {
        return false;
    }

    for (auto it_l = scoringContexts.begin(), it_r = otherPtr->scoringContexts.begin(); it_l != scoringContexts.end(); ++it_l, ++it_r) {
        if (!(*it_l)->isEqual(*it_r)) {
            return false;
        }
    }

    return true;
}

/*
 * =============================
 * === PythonScoringContext ====
 * =============================
 */
size_t PythonScoringContext::hash() const {
    return py::hash(py::cast<py::handle>(object));
}

bool PythonScoringContext::isEqual(ScoringContextRef const& other) const {
    auto* otherPtr = dynamic_cast<const PythonScoringContext*>(other.get());
    if (step != otherPtr->step) {
        return false;
    }

    return object.equal(py::cast<py::handle>(otherPtr->object));
}

}  // namespace Nn
