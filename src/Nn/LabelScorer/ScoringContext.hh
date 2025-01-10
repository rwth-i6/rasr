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

#ifndef SCORING_CONTEXT_HH
#define SCORING_CONTEXT_HH

#include <Am/ClassicStateModel.hh>
#include <Core/ReferenceCounting.hh>
#include <Math/FastVector.hh>
#include <Mm/Types.hh>
#include <Speech/Types.hh>
#ifdef MODULE_ONNX
#include "Onnx/Value.hh"
#endif

namespace Nn {

typedef Mm::EmissionIndex LabelIndex;

/*
 * Empty scoring context base class
 */
struct ScoringContext : public Core::ReferenceCounted {
    virtual ~ScoringContext() = default;

    virtual bool   isEqual(const Core::Ref<const ScoringContext>& other) const;
    virtual size_t hash() const;
};

typedef Core::Ref<const ScoringContext> ScoringContextRef;

struct ScoringContextHash {
    size_t operator()(const ScoringContextRef& history) const {
        return history->hash();
    }
};

struct ScoringContextEq {
    bool operator()(const ScoringContextRef& lhs, const ScoringContextRef& rhs) const {
        return lhs->isEqual(rhs);
    }
};

/*
 * Scoring context that only describes the current decoding step
 */

struct StepScoringContext : public ScoringContext {
    Speech::TimeframeIndex currentStep;

    StepScoringContext()
            : currentStep(0ul) {}

    StepScoringContext(Speech::TimeframeIndex step)
            : currentStep(step) {}

    bool   isEqual(const ScoringContextRef& other) const;
    size_t hash() const;
};

typedef Core::Ref<const StepScoringContext> StepScoringContextRef;

/*
 * Scoring context that describes a sequence of previously observed labels
 */
struct LabelSeqScoringContext : public ScoringContext {
    std::vector<LabelIndex> labelSeq;

    LabelSeqScoringContext()
            : labelSeq() {}
    LabelSeqScoringContext(const std::vector<LabelIndex>& seq)
            : labelSeq(seq) {}

    bool   isEqual(const ScoringContextRef& other) const;
    size_t hash() const;
};

typedef Core::Ref<const LabelSeqScoringContext> LabelSeqScoringContextRef;

/*
 * Scoring context that describes a sequence of previously observed labels as well as the current decoding step
 */
struct SeqStepScoringContext : public ScoringContext {
    std::vector<LabelIndex> labelSeq;
    Speech::TimeframeIndex  currentStep;

    SeqStepScoringContext()
            : labelSeq(), currentStep(0ul) {}
    SeqStepScoringContext(const std::vector<LabelIndex>& seq, Speech::TimeframeIndex step)
            : labelSeq(seq), currentStep(step) {}

    bool   isEqual(const ScoringContextRef& other) const;
    size_t hash() const;
};

typedef Core::Ref<const SeqStepScoringContext> SeqStepScoringContextRef;

#ifdef MODULE_ONNX

/*
 * Hidden state containing a dictionary of named ONNX values
 */
struct HiddenState : public Core::ReferenceCounted {
    std::unordered_map<std::string, Onnx::Value> stateValueMap;

    HiddenState()
            : stateValueMap() {}

    HiddenState(std::vector<std::string>&& names, std::vector<Onnx::Value>&& values) {
        verify(names.size() == values.size());
        stateValueMap.reserve(names.size());
        for (size_t i = 0ul; i < names.size(); ++i) {
            stateValueMap.emplace(std::move(names[i]), std::move(values[i]));
        }
    }
};

typedef Core::Ref<HiddenState> HiddenStateRef;

/*
 * Scoring context that uses hidden state values
 * Assumes that the hidden state is uniquely determined by the label history
 */
struct HiddenStateScoringContext : public ScoringContext {
    std::vector<LabelIndex> labelSeq;  // Used for hashing
    Core::Ref<HiddenState>  hiddenState;

    HiddenStateScoringContext()
            : labelSeq(), hiddenState() {}

    HiddenStateScoringContext(const std::vector<LabelIndex>& labelSeq, HiddenStateRef state)
            : labelSeq(labelSeq), hiddenState(state) {}

    bool   isEqual(const ScoringContextRef& other) const;
    size_t hash() const;
};

typedef Core::Ref<const HiddenStateScoringContext> HiddenStateScoringContextRef;

#endif  // MODULE_ONNX

/*
 * Combines multiple scoring contexts at once
 */
struct CombineScoringContext : public ScoringContext {
    std::vector<ScoringContextRef> scoringContexts;

    CombineScoringContext()
            : scoringContexts() {}

    CombineScoringContext(std::vector<ScoringContextRef>&& scoringContexts)
            : scoringContexts(scoringContexts) {}

    bool   isEqual(const ScoringContextRef& other) const;
    size_t hash() const;
};

typedef Core::Ref<const CombineScoringContext> CombineScoringContextRef;

}  // namespace Nn

#endif  // SCORING_CONTEXT_HH
