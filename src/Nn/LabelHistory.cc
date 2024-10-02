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
#include "LabelHistory.hh"
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
 * === LabelHistory ============
 * =============================
 */
size_t LabelHistoryHash::operator()(const LabelHistory* history) const {
    return 0ul;
}

bool LabelHistoryEq::operator()(const LabelHistory* lhs, const LabelHistory* rhs) const {
    return true;
}

/*
 * =============================
 * === StepLabelHistory ========
 * =============================
 */
size_t StepLabelHistoryHash::operator()(const StepLabelHistory* history) const {
    return history->currentStep;
}

bool StepLabelHistoryEq::operator()(const StepLabelHistory* lhs, const StepLabelHistory* rhs) const {
    return lhs->currentStep == rhs->currentStep;
}

/*
 * =============================
 * === SeqLabelHistory =========
 * =============================
 */
size_t SeqLabelHistoryHash::operator()(const SeqLabelHistory* history) const {
    return Core::MurmurHash3_x64_64(reinterpret_cast<void const*>(history->labelSeq.data()), history->labelSeq.size() * sizeof(LabelIndex), 0x78b174eb);
}

bool SeqLabelHistoryEq::operator()(const SeqLabelHistory* lhs, const SeqLabelHistory* rhs) const {
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
 * === SeqLabelHistory ========
 * =============================
 */
size_t SeqStepLabelHistoryHash::operator()(const SeqStepLabelHistory* history) const {
    return combineHashes(history->currentStep, Core::MurmurHash3_x64_64(reinterpret_cast<void const*>(history->labelSeq.data()), history->labelSeq.size() * sizeof(LabelIndex), 0x78b174eb));
}

bool SeqStepLabelHistoryEq::operator()(const SeqStepLabelHistory* lhs, const SeqStepLabelHistory* rhs) const {
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

}  // namespace Nn
