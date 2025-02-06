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
 * === ScoringContext ==========
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
        value = combineHashes(value, scoringContext->hash());
    }
    return value;
}

bool CombineScoringContext::isEqual(ScoringContextRef const& other) const {
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
