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

Score ScoreAccessor::getScoreForLabel(LabelIndex labelIndex) const {
    return 0.0;
};

Score ScoreAccessor::getScoreForTransition(TransitionType transitionType) const {
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

Score ScaledScoreAccessor::getScoreForLabel(LabelIndex labelIndex) const {
    return base_->getScoreForLabel(labelIndex) * scale_;
}

Score ScaledScoreAccessor::getScoreForTransition(TransitionType transitionType) const {
    return base_->getScoreForTransition(transitionType) * scale_;
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

Score CombinedScoreAccessor::getScoreForLabel(LabelIndex labelIndex) const {
    return std::accumulate(subAccessors_.begin(), subAccessors_.end(), 0.0, [labelIndex](Score acc, ScoreAccessorRef subAccessor) {
        return acc + subAccessor->getScoreForLabel(labelIndex);
    });
}

Score CombinedScoreAccessor::getScoreForTransition(TransitionType transitionType) const {
    return std::accumulate(subAccessors_.begin(), subAccessors_.end(), 0.0, [transitionType](Score acc, ScoreAccessorRef subAccessor) {
        return acc + subAccessor->getScoreForTransition(transitionType);
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

Score VectorScoreAccessor::getScoreForLabel(LabelIndex labelIndex) const {
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

Score DataViewScoreAccessor::getScoreForLabel(LabelIndex labelIndex) const {
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
    transitionScores_.emplace(transitionType, score);
}

Score FixedTransitionScoreAccessor::getScoreForTransition(TransitionType transitionType) const {
    return transitionScores_.at(transitionType);
}

TimeframeIndex FixedTransitionScoreAccessor::getTime() const {
    return 0;
}

/*
 * =============================
 * == CTCPrefixScoreAccessor ===
 * =============================
 */

CTCPrefixScoreAccessor::CTCPrefixScoreAccessor(CTCPrefixScoringContextRef const& scoringContext, std::shared_ptr<Math::FastMatrix<Score>> const& ctcScores) : ctcScores_(ctcScores), scoringContext_(scoringContext) {
}

Score CTCPrefixScoreAccessor::getScoreForLabel(LabelIndex labelIndex) const {
    auto it = scoringContext_->extScores.find(labelIndex);
    if (it != scoringContext_->extScores.end()) {
        return it->second = scoringContext_->prefixScore;
    }

    Score totalScore;

    if (scoringContext_->labelSeq.empty()) {
        totalScore = ctcScores_->at(labelIndex, 0);
    }
    else {
        totalScore = std::numeric_limits<Score>::infinity();
    }
    for (size_t t = 1ul; t < ctcScores_->nColumns(); ++t) {
        // Prefix can always end by time t-1 with blank label
        Score timestepScore;
        if (not scoringContext_->labelSeq.empty() and labelIndex == scoringContext_->labelSeq.back()) {
            // If prefix ends in the same token as `nextToken` there must a blank between, thus the prefix can only end in blank at t-1
            timestepScore = scoringContext_->timePrefixScores->at(t - 1).blankEndingScore;
        }
        else {
            // If prefix ends in a different token as `nextToken` can it end in both blank or non-blank at t-1
            timestepScore = scoringContext_->timePrefixScores->at(t - 1).totalScore();
        }

        timestepScore += ctcScores_->at(labelIndex, t);

        totalScore = std::min(totalScore, timestepScore);
    }
    scoringContext_->extScores.emplace(labelIndex, totalScore);

    return totalScore - scoringContext_->prefixScore;
}

Score CTCPrefixScoreAccessor::getScoreForTransition(TransitionType transitionType) const {
    if (transitionType == SENTENCE_END) {
        return scoringContext_->timePrefixScores->back().totalScore() - scoringContext_->prefixScore;
    }
    return 0;
}

TimeframeIndex CTCPrefixScoreAccessor::getTime() const {
    return scoringContext_->labelSeq.size() + 1;
}

/*
 * ==============================
 * = FeatureScorerScoreAccessor =
 * ==============================
 */
FeatureScorerScoreAccessor::FeatureScorerScoreAccessor(Mm::FeatureScorer::Scorer const& scorer, TimeframeIndex time) : scorer_(scorer), time_(time) {}

Score FeatureScorerScoreAccessor::getScoreForLabel(LabelIndex labelIndex) const {
    return scorer_->score(labelIndex);
}

TimeframeIndex FeatureScorerScoreAccessor::getTime() const {
    return time_;
}

}  // namespace Nn
