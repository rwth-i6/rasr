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

#include <Am/ClassicStateModel.hh>
#include <Core/ReferenceCounting.hh>
#include <Core/Types.hh>
#include <Math/FastVector.hh>
#include <Mm/Types.hh>
#include <Onnx/Value.hh>
#include <Speech/Types.hh>
#include <limits>
#ifdef MODULE_PYTHON
#include <pybind11/pytypes.h>
namespace py = pybind11;
#endif
#include <Search/Types.hh>

namespace Nn {

typedef Mm::EmissionIndex   LabelIndex;
static constexpr LabelIndex invalidLabelIndex = Core::Type<LabelIndex>::max;

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
 * Scoring context that only describes the current decoding step
 */
struct StepScoringContext : public ScoringContext {
    Speech::TimeframeIndex currentStep;

    StepScoringContext()
            : currentStep(0ul) {}

    StepScoringContext(Speech::TimeframeIndex step)
            : currentStep(step) {}

    bool   isEqual(ScoringContextRef const& other) const;
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
    LabelSeqScoringContext(std::vector<LabelIndex> const& seq)
            : labelSeq(seq) {}

    bool   isEqual(ScoringContextRef const& other) const;
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
    SeqStepScoringContext(std::vector<LabelIndex> const& seq, Speech::TimeframeIndex step)
            : labelSeq(seq), currentStep(step) {}

    bool   isEqual(ScoringContextRef const& other) const;
    size_t hash() const;
};

typedef Core::Ref<const SeqStepScoringContext> SeqStepScoringContextRef;

struct CTCPrefixScoringContext : public ScoringContext {
    struct PrefixScore {
        Search::Score blankEndingScore    = std::numeric_limits<Search::Score>::infinity();
        Search::Score nonBlankEndingScore = std::numeric_limits<Search::Score>::infinity();

        Search::Score totalScore() const {
            return Math::scoreSum(blankEndingScore, nonBlankEndingScore);
        }
    };

    std::vector<LabelIndex>                               labelSeq;
    mutable std::shared_ptr<std::vector<PrefixScore>>     timePrefixScores;  // Represents probabilities of emitting `labelSeq` ending in blank or nonblank up to time t for each t = 0, ..., T
    mutable Search::Score                                 prefixScore;       // -log P(prefix, ...)
    mutable std::unordered_map<LabelIndex, Search::Score> extScores;         // -log P(prefix + token, ...)
    mutable bool                                          requiresFinalize;

    CTCPrefixScoringContext()
            : labelSeq(), timePrefixScores(), prefixScore(0.0), extScores(), requiresFinalize(true) {}

    CTCPrefixScoringContext(std::vector<LabelIndex> const& seq, std::shared_ptr<std::vector<PrefixScore>> const& timePrefixScores, Search::Score prefixScore, bool requiresFinalize)
            : labelSeq(seq), timePrefixScores(timePrefixScores), prefixScore(prefixScore), extScores(), requiresFinalize(requiresFinalize) {}

    bool   isEqual(ScoringContextRef const& other) const;
    size_t hash() const;
};

typedef Core::Ref<const CTCPrefixScoringContext> CTCPrefixScoringContextRef;

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

#ifdef MODULE_PYTHON

/*
 * Scoring context containing some arbitrary python object
 */
struct PythonScoringContext : public ScoringContext {
    py::object object;
    size_t     step;

    PythonScoringContext()
            : object(py::none()), step(0ul) {}

    PythonScoringContext(py::object&& object, size_t step)
            : object(object), step(step) {}

    bool   isEqual(ScoringContextRef const& other) const;
    size_t hash() const;
};

typedef Core::Ref<const PythonScoringContext> PythonScoringContextRef;

#endif

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

typedef Core::Ref<const OnnxHiddenState> OnnxHiddenStateRef;

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

typedef Core::Ref<const OnnxHiddenStateScoringContext> OnnxHiddenStateScoringContextRef;

/*
 * Scoring context consisting of a hidden state and a step.
 * Assumes that two hidden states are equal if and only if they were created
 * from the same label history.
 */
struct StepOnnxHiddenStateScoringContext : public ScoringContext {
    Speech::TimeframeIndex     currentStep;
    std::vector<LabelIndex>    labelSeq;  // Used for hashing
    mutable OnnxHiddenStateRef hiddenState;
    mutable bool               requiresFinalize;

    StepOnnxHiddenStateScoringContext()
            : currentStep(0u), labelSeq(), hiddenState(), requiresFinalize(false) {}

    StepOnnxHiddenStateScoringContext(Speech::TimeframeIndex step, std::vector<LabelIndex> const& labelSeq, OnnxHiddenStateRef state)
            : currentStep(step), labelSeq(labelSeq), hiddenState(state), requiresFinalize(false) {}

    bool   isEqual(ScoringContextRef const& other) const;
    size_t hash() const;
};

typedef Core::Ref<const StepOnnxHiddenStateScoringContext> StepOnnxHiddenStateScoringContextRef;

}  // namespace Nn

#endif  // SCORING_CONTEXT_HH
