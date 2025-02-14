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

#include <Core/ReferenceCounting.hh>
#include <Mm/Types.hh>
#include <Speech/Types.hh>

namespace Nn {

typedef Mm::EmissionIndex LabelIndex;

/*
 * Empty scoring context base class
 */
struct ScoringContext : public Core::ReferenceCounted {
    virtual ~ScoringContext() = default;

    virtual bool   isEqual(Core::Ref<const ScoringContext> const& other) const;
    virtual size_t hash() const;
};

typedef Core::Ref<const ScoringContext> ScoringContextRef;

struct ScoringContextHash {
    size_t operator()(ScoringContextRef const& scoringContext) const {
        return scoringContext->hash();
    }
};

struct ScoringContextEq {
    bool operator()(ScoringContextRef const& lhs, ScoringContextRef const& rhs) const {
        return lhs->isEqual(rhs);
    }
};

/*
 * Combines multiple scoring contexts at once
 */
struct CombineScoringContext : public ScoringContext {
    std::vector<ScoringContextRef> scoringContexts;

    CombineScoringContext()
            : scoringContexts() {}

    CombineScoringContext(std::vector<ScoringContextRef>&& scoringContexts)
            : scoringContexts(scoringContexts) {}

    bool   isEqual(ScoringContextRef const& other) const;
    size_t hash() const;
};

typedef Core::Ref<const CombineScoringContext> CombineScoringContextRef;

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

typedef Core::Ref<const StepScoringContext> StepScoringContextRef;

}  // namespace Nn

#endif  // SCORING_CONTEXT_HH
