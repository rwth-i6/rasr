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

#include "Core/ReferenceCounting.hh"
#include "Search/Types.hh"

namespace Nn {

typedef Search::Index LabelIndex;

struct LabelHistory : public Core::ReferenceCounted {
    virtual ~LabelHistory() = default;
};

struct StepLabelHistory : public LabelHistory {
    size_t currentStep = 0ul;
};

struct SeqLabelHistory : public LabelHistory {
    std::vector<LabelIndex> labelSeq;
};

struct SeqStepHistory : public LabelHistory {
    std::vector<LabelIndex> labelSeq;
    size_t                  currentStep = 0ul;
};

}  // namespace Nn

#endif  // LABEL_HISTORY_HH
