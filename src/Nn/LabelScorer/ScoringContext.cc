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
 * === ScoringContext ==========
 * =============================
 */
bool ScoringContext::isEqual(const ScoringContextRef& other) const {
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

bool StepScoringContext::isEqual(const ScoringContextRef& other) const {
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

bool LabelSeqScoringContext::isEqual(const ScoringContextRef& other) const {
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

bool SeqStepScoringContext::isEqual(const ScoringContextRef& other) const {
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

bool HiddenStateScoringContext::isEqual(const ScoringContextRef& other) const {
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
 * === CombineScoringContext ===
 * =============================
 */
size_t CombineScoringContext::hash() const {
    size_t value = 0ul;
    for (const auto& scoringContext : scoringContexts) {
        value = combineHashes(value, scoringContext->hash());
    }
    return value;
}

bool CombineScoringContext::isEqual(const ScoringContextRef& other) const {
    auto* otherPtr = dynamic_cast<const CombineScoringContext*>(other.get());

    if (scoringContexts.size() != otherPtr->scoringContexts.size()) {
        return false;
    }

    for (auto it_l = scoringContexts.begin(), it_r = otherPtr->scoringContexts.begin(); it_l != scoringContexts.end(); ++it_l, ++it_r) {
        if (!(*it_l)->isEqual(*it_r)) {
            return false;
        }
    }

    return true;
}

}  // namespace Nn
