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
#include <Nn/AbstractStateManager.hh>

#include "Types.hh"

namespace Nn {

size_t labelSeqHash(std::vector<LabelIndex> const& labelSeq);
bool   labelSeqEqual(std::vector<LabelIndex> const& lhs, std::vector<LabelIndex> const& rhs);

/*
 * Empty scoring context base class
 */
struct ScoringContext : public Core::ReferenceCounted {
    virtual ~ScoringContext() = default;

    virtual size_t hash() const;
    virtual bool   isEqual(Core::Ref<ScoringContext const> const& other) const;
};

typedef Core::Ref<ScoringContext const> ScoringContextRef;

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
 * Scoring context that only describes the current decoding step
 */
struct StepScoringContext : public ScoringContext {
    Speech::TimeframeIndex currentStep;

    StepScoringContext()
            : currentStep(0u) {}

    StepScoringContext(Speech::TimeframeIndex step)
            : currentStep(step) {}

    size_t hash() const;
    bool   isEqual(ScoringContextRef const& other) const;
};

typedef Core::Ref<StepScoringContext const> StepScoringContextRef;

/*
 * Scoring context that describes the complete sequence of previously observed labels as well as the current decoding step
 */
struct SeqStepScoringContext : public ScoringContext {
    std::vector<LabelIndex> labelSeq;
    Speech::TimeframeIndex  currentStep;

    SeqStepScoringContext()
            : labelSeq(), currentStep(0ul) {}
    SeqStepScoringContext(std::vector<LabelIndex> const& seq, Speech::TimeframeIndex step)
            : labelSeq(seq), currentStep(step) {}
    SeqStepScoringContext(std::vector<LabelIndex>&& seq, Speech::TimeframeIndex step)
            : labelSeq(std::move(seq)), currentStep(step) {}

    size_t hash() const override;
    bool   isEqual(ScoringContextRef const& other) const override;
};

typedef Core::Ref<SeqStepScoringContext const> SeqStepScoringContextRef;

}  // namespace Nn

#endif  // SCORING_CONTEXT_HH
