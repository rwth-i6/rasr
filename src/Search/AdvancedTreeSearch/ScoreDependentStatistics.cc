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
#include "ScoreDependentStatistics.hh"

using namespace Search;

void ScoreDependentVectorStatistic::addValue(Score relativeStartScore, u32 offset, f32 value) {
    if (relativeStartScore < 0)
        relativeStartScore = 0;

    if (relativeStartScore > maxRelativeScore_)
        relativeStartScore = maxRelativeScore_;

    u32 index = (u32)((relativeStartScore / maxRelativeScore_) * granularity_);
    if (index == granularity_)
        --index;

    if (effort_[index].size() <= offset)
        effort_[index].resize(offset + 1, std::make_pair(0u, 0u));

    ++effort_[index][offset].first;
    effort_[index][offset].second += value;
}
