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
#include <Onnx/Value.hh>
#include "Types.hh"

namespace Nn {

/*
 * Empty scoring context base class
 */
struct ScoringContext : public Core::ReferenceCounted {
    virtual ~ScoringContext() = default;

    virtual bool   isEqual(Core::Ref<ScoringContext const> const& other) const;
    virtual size_t hash() const;
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

typedef Core::Ref<CombineScoringContext const> CombineScoringContextRef;

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

/*
 * Scoring context that describes a sequence of previously observed labels as well as the current decoding step
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

    bool   isEqual(ScoringContextRef const& other) const;
    size_t hash() const;
};

typedef Core::Ref<SeqStepScoringContext const> SeqStepScoringContextRef;

/*
 * Hidden state represented by a dictionary of named ONNX values
 */
struct OnnxHiddenState : public Core::ReferenceCounted {
    std::unordered_map<std::string, Onnx::Value> stateValueMap;

    OnnxHiddenState()
            : stateValueMap() {}

    OnnxHiddenState(std::vector<std::string>&& names, std::vector<Onnx::Value>&& values) {
        verify(names.size() == values.size());
        stateValueMap.reserve(names.size());
        for (size_t i = 0ul; i < names.size(); ++i) {
            stateValueMap.emplace(std::move(names[i]), std::move(values[i]));
        }
    }
};

typedef Core::Ref<OnnxHiddenState const> OnnxHiddenStateRef;

/*
 * Scoring context consisting of a hidden state.
 * Assumes that two hidden states are equal if and only if they were created
 * from the same label history.
 */
struct OnnxHiddenStateScoringContext : public ScoringContext {
    std::vector<LabelIndex>    labelSeq;  // Used for hashing
    mutable OnnxHiddenStateRef hiddenState;
    mutable bool               requiresFinalize;

    OnnxHiddenStateScoringContext()
            : labelSeq(), hiddenState(), requiresFinalize(false) {}

    OnnxHiddenStateScoringContext(std::vector<LabelIndex> const& labelSeq, OnnxHiddenStateRef state, bool requiresFinalize)
            : labelSeq(labelSeq), hiddenState(state), requiresFinalize(requiresFinalize) {}

    bool   isEqual(ScoringContextRef const& other) const;
    size_t hash() const;
};

typedef Core::Ref<OnnxHiddenStateScoringContext const> OnnxHiddenStateScoringContextRef;

/*
 * Scoring context for computation of CTC prefix scores.
 * Contains time-wise and overall score for prefix as well as a cache for the score of
 * extended prefixes.
 * Hash and equality operators are based on the label sequence.
 */
struct CtcPrefixScoringContext : public ScoringContext {
    struct PrefixScore {
        Score blankEndingScore    = std::numeric_limits<Score>::infinity();
        Score nonBlankEndingScore = std::numeric_limits<Score>::infinity();

        Score totalScore() const {
            return Math::scoreSum(blankEndingScore, nonBlankEndingScore);
        }
    };

    std::vector<LabelIndex>                           labelSeq;
    mutable std::shared_ptr<std::vector<PrefixScore>> timePrefixScores;  // Represents neg-log-probabilities of emitting `labelSeq` ending in blank or nonblank up to time t for each t = 0, ..., T
    mutable Score                                     prefixScore;       // -log P(prefix, ...)
    mutable std::unordered_map<LabelIndex, Score>     extScores;         // Cache for -log P(prefix + token, ...) to avoid repeated computation
    mutable bool                                      requiresFinalize;

    CtcPrefixScoringContext()
            : labelSeq(), timePrefixScores(), prefixScore(0.0), extScores(), requiresFinalize(true) {}

    CtcPrefixScoringContext(std::vector<LabelIndex> const& seq, std::shared_ptr<std::vector<PrefixScore>> const& timePrefixScores, Score prefixScore, bool requiresFinalize)
            : labelSeq(seq), timePrefixScores(timePrefixScores), prefixScore(prefixScore), extScores(), requiresFinalize(requiresFinalize) {}

    bool   isEqual(ScoringContextRef const& other) const;
    size_t hash() const;
};

typedef Core::Ref<const CtcPrefixScoringContext> CtcPrefixScoringContextRef;

}  // namespace Nn

#endif  // SCORING_CONTEXT_HH
