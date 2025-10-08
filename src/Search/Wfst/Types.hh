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
#ifndef _SEARCH_WFST_TYPES_HH
#define _SEARCH_WFST_TYPES_HH

namespace Search {
namespace Wfst {

enum OutputType {
    OutputLemmaPronunciation,
    OutputLemma,
    OutputSyntacticToken
};

enum LookAheadFlags {
    NoLookAheadFlag    = 0,
    LabelLookAheadFlag = 1,
    PushWeightsFlag    = 2,
    PushLabelsFlag     = 4,
    ArcLookAheadFlag   = 8
};

enum LookAheadType {
    NoLookAhead    = NoLookAheadFlag,
    LabelLookAhead = LabelLookAheadFlag,
    PushWeights    = LabelLookAheadFlag | PushWeightsFlag,
    PushLabels     = LabelLookAheadFlag | PushWeightsFlag | PushLabelsFlag,
    PushLabelsOnly = LabelLookAheadFlag | PushLabelsFlag,
    ArcLookAhead   = ArcLookAheadFlag
};

}  // namespace Wfst
}  // namespace Search

#endif  // _SEARCH_WFST_TYPES_HH
