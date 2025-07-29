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
#include <Search/Types.hh>
#include <Speech/Types.hh>
#ifdef MODULE_ONNX
#include <Onnx/Value.hh>
#endif
#ifdef MODULE_PYTHON
#include <pybind11/pytypes.h>
namespace py = pybind11;
#endif

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

    HiddenStateScoringContext(std::vector<LabelIndex> const& labelSeq, HiddenStateRef state)
            : labelSeq(labelSeq), hiddenState(state) {}

    bool   isEqual(ScoringContextRef const& other) const;
    size_t hash() const;
};

typedef Core::Ref<const HiddenStateScoringContext> HiddenStateScoringContextRef;

#endif  // MODULE_ONNX

struct CTCPrefixScoringContext : public ScoringContext {
    struct PrefixScore {
        Search::Score blankEndingScore;
        Search::Score nonBlankEndingScore;
    };

    std::vector<PrefixScore> prefixScores;
    LabelIndex               lastLabel;

    CTCPrefixScoringContext()
            : prefixScores(), lastLabel(Core::Type<LabelIndex>::max) {}

    CTCPrefixScoringContext(std::vector<PrefixScore>&& prefixScores, LabelIndex lastLabel)
            : prefixScores(std::move(prefixScores)), lastLabel(lastLabel) {}

    bool   isEqual(ScoringContextRef const& other) const;
    size_t hash() const;
};

typedef Core::Ref<const CTCPrefixScoringContext> PrefixScoringContextRef;

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

}  // namespace Nn

#endif  // SCORING_CONTEXT_HH
