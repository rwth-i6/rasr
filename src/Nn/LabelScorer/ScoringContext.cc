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

namespace Nn {

typedef Mm::EmissionIndex LabelIndex;

/*
 * =============================
 * ====== ScoringContext =======
 * =============================
 */
size_t ScoringContext::hash() const {
    return 0ul;
}

bool ScoringContext::isEqual(ScoringContextRef const& other) const {
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
        value = Core::combineHashes(value, scoringContext->hash());
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
 * ==== StepScoringContext =====
 * =============================
 */
size_t StepScoringContext::hash() const {
    return currentStep;
}

bool StepScoringContext::isEqual(ScoringContextRef const& other) const {
    StepScoringContext const* o = dynamic_cast<StepScoringContext const*>(other.get());
    return o != nullptr and currentStep == o->currentStep;
}

}  // namespace Nn
