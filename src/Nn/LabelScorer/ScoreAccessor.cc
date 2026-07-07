/** Copyright 2026 RWTH Aachen University. All rights reserved.
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

#include "ScoreAccessor.hh"

#include <numeric>
#include "Types.hh"

namespace Nn {

/*
 * =============================
 * ==== VectorScoreAccessor ====
 * =============================
 */

VectorScoreAccessor::VectorScoreAccessor(std::shared_ptr<std::vector<Score>> scores, TimeframeIndex time)
        : scores_(scores),
          time_(time) {}

Score VectorScoreAccessor::getScore(TransitionType transitionType, LabelIndex labelIndex) const {
    return scores_->at(labelIndex);
}

TimeframeIndex VectorScoreAccessor::getTime() const {
    return time_;
}

/*
 * =============================
 * === DataViewScoreAccessor ===
 * =============================
 */

DataViewScoreAccessor::DataViewScoreAccessor(DataView const& dataView, TimeframeIndex time)
        : dataView_(dataView),
          time_(time) {}

Score DataViewScoreAccessor::getScore(TransitionType transitionType, LabelIndex labelIndex) const {
    return dataView_[labelIndex];
}

TimeframeIndex DataViewScoreAccessor::getTime() const {
    return time_;
}

}  // namespace Nn
