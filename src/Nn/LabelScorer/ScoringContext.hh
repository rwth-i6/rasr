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

#ifndef SCORING_CONTEXT_HH
#define SCORING_CONTEXT_HH

#include <Core/Hash.hh>
#include <Core/ReferenceCounting.hh>
#include <Mm/Types.hh>
#include <Nn/AbstractStateManager.hh>

#include "Types.hh"

namespace Nn {

size_t labelSeqHash(std::vector<LabelIndex> const& labelSeq);
bool   labelSeqEqual(std::vector<LabelIndex> const& lhs, std::vector<LabelIndex> const& rhs);

/*
 * Empty scoring context base class
 *
 * Each scoring context belongs to a segment. Label scorers that support several segments at the same time set the segment
 * when they create a context and propagate it to extended contexts. Single-segment usage always uses the default segment.
 * `isEqual` and `hash` of the concrete contexts only describe the context itself, the segment is taken into account
 * by `ScoringContextHash` and `ScoringContextEq` so that contexts of different segments are never treated as equal.
 */
struct ScoringContext : public Core::ReferenceCounted {
    SegmentId segment = defaultSegmentId;

    virtual ~ScoringContext() = default;

    virtual bool   isEqual(Core::Ref<ScoringContext const> const& other) const;
    virtual size_t hash() const;
};

typedef Core::Ref<ScoringContext const> ScoringContextRef;

struct ScoringContextHash {
    size_t operator()(ScoringContextRef const& scoringContext) const {
        if (scoringContext->segment == defaultSegmentId) {
            return scoringContext->hash();
        }
        return Core::combineHashes(scoringContext->hash(), scoringContext->segment);
    }
};

struct ScoringContextEq {
    bool operator()(ScoringContextRef const& lhs, ScoringContextRef const& rhs) const {
        return lhs->segment == rhs->segment and lhs->isEqual(rhs);
    }
};

/*
 * Scoring context that only describes the current decoding step
 */
struct StepScoringContext : public ScoringContext {
    Speech::TimeframeIndex currentStep;

    StepScoringContext()
            : currentStep(0u) {}

    StepScoringContext(Speech::TimeframeIndex step)
            : currentStep(step) {}

    bool   isEqual(ScoringContextRef const& other) const;
    size_t hash() const;
};

typedef Core::Ref<StepScoringContext const> StepScoringContextRef;

}  // namespace Nn

#endif  // SCORING_CONTEXT_HH
