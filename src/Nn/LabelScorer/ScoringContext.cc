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

namespace Nn {

typedef Mm::EmissionIndex LabelIndex;

size_t labelSeqHash(std::vector<LabelIndex> const& labelSeq) {
    return Core::MurmurHash3_x64_64(reinterpret_cast<void const*>(labelSeq.data()), labelSeq.size() * sizeof(LabelIndex), 0x78b174eb);
}

bool labelSeqEqual(std::vector<LabelIndex> const& lhs, std::vector<LabelIndex> const& rhs) {
    return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

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
