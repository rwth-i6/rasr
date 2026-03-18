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
 * ====== ScoreAccessor ========
 * =============================
 */

Score ScoreAccessor::getScore(TransitionType transitionType, LabelIndex labelIndex) const {
    return 0.0;
};

TimeframeIndex ScoreAccessor::getTime() const {
    return 0;
};

/*
 * =============================
 * === ScaledScoreAccessor =====
 * =============================
 */
ScaledScoreAccessor::ScaledScoreAccessor(ScoreAccessorRef base, Score scale)
        : base_(base),
          scale_(scale) {}

Score ScaledScoreAccessor::getScore(TransitionType transitionType, LabelIndex labelIndex) const {
    return base_->getScore(transitionType, labelIndex) * scale_;
}

TimeframeIndex ScaledScoreAccessor::getTime() const {
    return base_->getTime();
}

/*
 * =============================
 * === CombinedScoreAccessor ===
 * =============================
 */
CombinedScoreAccessor::CombinedScoreAccessor()
        : subAccessors_() {}

void CombinedScoreAccessor::addSubAccessor(ScoreAccessorRef subAccessor) {
    subAccessors_.push_back(subAccessor);
}

Score CombinedScoreAccessor::getScore(TransitionType transitionType, LabelIndex labelIndex) const {
    return std::accumulate(subAccessors_.begin(), subAccessors_.end(), 0.0, [transitionType, labelIndex](Score acc, ScoreAccessorRef subAccessor) {
        return acc + subAccessor->getScore(transitionType, labelIndex);
    });
}

TimeframeIndex CombinedScoreAccessor::getTime() const {
    return std::accumulate(subAccessors_.begin(), subAccessors_.end(), 0, [](TimeframeIndex max, ScoreAccessorRef subAccessor) {
        return std::max(max, subAccessor->getTime());
    });
}

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

/*
 * =============================
 * == TransitionScoreAccessor ==
 * =============================
 */

FixedTransitionScoreAccessor::FixedTransitionScoreAccessor()
        : transitionScores_() {
    for (auto const& [stringIdentifier, enumValue] : TransitionTypeArray) {
        setScore(enumValue, 0.0);
    }
}

void FixedTransitionScoreAccessor::setScore(TransitionType transitionType, Score score) {
    transitionScores_[static_cast<size_t>(transitionType)] = score;
}

Score FixedTransitionScoreAccessor::getScore(TransitionType transitionType, LabelIndex labelIndex) const {
    return transitionScores_.at(transitionType);
}

TimeframeIndex FixedTransitionScoreAccessor::getTime() const {
    return 0;
}

}  // namespace Nn
