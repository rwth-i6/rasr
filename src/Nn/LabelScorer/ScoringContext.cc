/** Copyright 2020 RWTH Aachen University. All rights reserved.
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

namespace Nn {

typedef Mm::EmissionIndex LabelIndex;

// Auxiliary function to merge multiple hashes into one via the boost way
static size_t combineHashes(size_t hash1, size_t hash2) {
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
 * === ScoringContext ============
 * =============================
 */
size_t ScoringContextHash::operator()(ScoringContextRef history) const {
    return 0ul;
}

bool ScoringContextEq::operator()(ScoringContextRef lhs, ScoringContextRef rhs) const {
    return true;
}

/*
 * =============================
 * === StepScoringContext ========
 * =============================
 */
size_t StepScoringContextHash::operator()(StepScoringContextRef history) const {
    return history->currentStep;
}

bool StepScoringContextEq::operator()(StepScoringContextRef lhs, StepScoringContextRef rhs) const {
    return lhs->currentStep == rhs->currentStep;
}

/*
 * =============================
 * === LabelSeqScoringContext ==
 * =============================
 */
size_t LabelSeqScoringContextHash::operator()(LabelSeqScoringContextRef history) const {
    return Core::MurmurHash3_x64_64(reinterpret_cast<void const*>(history->labelSeq.data()), history->labelSeq.size() * sizeof(LabelIndex), 0x78b174eb);
}

bool LabelSeqScoringContextEq::operator()(LabelSeqScoringContextRef lhs, LabelSeqScoringContextRef rhs) const {
    if (lhs == rhs) {
        return true;
    }

    if (lhs->labelSeq.size() != rhs->labelSeq.size()) {
        return false;
    }

    for (auto it_l = lhs->labelSeq.begin(), it_r = rhs->labelSeq.begin(); it_l != lhs->labelSeq.end(); ++it_l, ++it_r) {
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
size_t SeqStepScoringContextHash::operator()(SeqStepScoringContextRef history) const {
    return combineHashes(history->currentStep, Core::MurmurHash3_x64_64(reinterpret_cast<void const*>(history->labelSeq.data()), history->labelSeq.size() * sizeof(LabelIndex), 0x78b174eb));
}

bool SeqStepScoringContextEq::operator()(SeqStepScoringContextRef lhs, SeqStepScoringContextRef rhs) const {
    if (lhs == rhs) {
        return true;
    }

    if (lhs->currentStep != rhs->currentStep) {
        return false;
    }

    if (lhs->labelSeq.size() != rhs->labelSeq.size()) {
        return false;
    }

    for (auto it_l = lhs->labelSeq.begin(), it_r = rhs->labelSeq.begin(); it_l != lhs->labelSeq.end(); ++it_l, ++it_r) {
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
size_t HiddenStateScoringContextHash::operator()(HiddenStateScoringContextRef history) const {
    return Core::MurmurHash3_x64_64(reinterpret_cast<void const*>(history->labelSeq.data()), history->labelSeq.size() * sizeof(LabelIndex), 0x78b174eb);
}

bool HiddenStateScoringContextEq::operator()(HiddenStateScoringContextRef lhs, HiddenStateScoringContextRef rhs) const {
    if (lhs == rhs) {
        return true;
    }

    if (lhs->labelSeq.size() != rhs->labelSeq.size()) {
        return false;
    }

    for (auto it_l = lhs->labelSeq.begin(), it_r = rhs->labelSeq.begin(); it_l != lhs->labelSeq.end(); ++it_l, ++it_r) {
        if (*it_l != *it_r) {
            return false;
        }
    }

    return true;
}
#endif  // MODULE_ONNX

}  // namespace Nn
