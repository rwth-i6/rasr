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

#ifndef LABEL_HISTORY_HH
#define LABEL_HISTORY_HH

#include <Am/ClassicStateModel.hh>
#include <Core/ReferenceCounting.hh>
#include <Mm/Types.hh>
#include <Speech/Types.hh>

namespace Nn {

typedef Mm::EmissionIndex LabelIndex;

/*
 * Empty label history base class
 */
struct LabelHistory : public Core::ReferenceCounted {
    virtual ~LabelHistory() = default;
};

typedef Core::Ref<const LabelHistory> LabelHistoryRef;

struct LabelHistoryHash {
    size_t operator()(LabelHistoryRef history) const;
};

struct LabelHistoryEq {
    bool operator()(LabelHistoryRef lhs, LabelHistoryRef rhs) const;
};

/*
 * Label history that only describes the current decoding step
 */

struct StepLabelHistory : public LabelHistory {
    Speech::TimeframeIndex currentStep;

    StepLabelHistory()
            : currentStep(0ul) {}

    StepLabelHistory(Speech::TimeframeIndex step)
            : currentStep(step) {}
};

typedef Core::Ref<const StepLabelHistory> StepLabelHistoryRef;

struct StepLabelHistoryHash {
    size_t operator()(StepLabelHistoryRef history) const;
};

struct StepLabelHistoryEq {
    bool operator()(StepLabelHistoryRef lhs, StepLabelHistoryRef rhs) const;
};

/*
 * Label history that describes a sequence of previously observed labels
 */
struct SeqLabelHistory : public LabelHistory {
    std::vector<LabelIndex> labelSeq;

    SeqLabelHistory()
            : labelSeq() {}
    SeqLabelHistory(const std::vector<LabelIndex>& seq)
            : labelSeq(seq) {}
};

typedef Core::Ref<const SeqLabelHistory> SeqLabelHistoryRef;

struct SeqLabelHistoryHash {
    size_t operator()(SeqLabelHistoryRef history) const;
};

struct SeqLabelHistoryEq {
    bool operator()(SeqLabelHistoryRef lhs, SeqLabelHistoryRef rhs) const;
};

/*
 * Label history that describes a sequence of previously observed labels as well as the current decoding step
 */
struct SeqStepLabelHistory : public LabelHistory {
    std::vector<LabelIndex> labelSeq;
    Speech::TimeframeIndex  currentStep;

    SeqStepLabelHistory()
            : labelSeq(), currentStep(0ul) {}
    SeqStepLabelHistory(const std::vector<LabelIndex>& seq, Speech::TimeframeIndex step)
            : labelSeq(seq), currentStep(step) {}
};

typedef Core::Ref<const SeqStepLabelHistory> SeqStepLabelHistoryRef;

struct SeqStepLabelHistoryHash {
    size_t operator()(SeqStepLabelHistoryRef history) const;
};

struct SeqStepLabelHistoryEq {
    bool operator()(SeqStepLabelHistoryRef lhs, SeqStepLabelHistoryRef rhs) const;
};

}  // namespace Nn

#endif  // LABEL_HISTORY_HH
