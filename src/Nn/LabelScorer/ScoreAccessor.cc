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

#include "Types.hh"

namespace Nn {

/*
 * =======================
 * ==== ScoreAccessor ====
 * =======================
 */

Score ScoreAccessor::getScore(TransitionType transitionType, LabelIndex labelIndex) const {
    return 0.0;
}

std::optional<DenseScoreSpan> ScoreAccessor::getDenseScores() const {
    return std::nullopt;
}

TimeframeIndex ScoreAccessor::getTime() const {
    return 0;
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
    if (labelIndex == invalidLabelIndex) {
        return 0.0;
    }
    return scores_->at(labelIndex);
}

std::optional<DenseScoreSpan> VectorScoreAccessor::getDenseScores() const {
    return DenseScoreSpan(DenseScoreTerm{.scores = std::span<Score const>(*scores_)});
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
    if (labelIndex == invalidLabelIndex) {
        return 0.0;
    }
    return dataView_[labelIndex];
}

std::optional<DenseScoreSpan> DataViewScoreAccessor::getDenseScores() const {
    return DenseScoreSpan(DenseScoreTerm{.scores = std::span<Score const>(dataView_.data(), dataView_.size())});
}

TimeframeIndex DataViewScoreAccessor::getTime() const {
    return time_;
}

}  // namespace Nn
