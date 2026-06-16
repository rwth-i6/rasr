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

#ifndef SCORE_ACCESSOR_HH
#define SCORE_ACCESSOR_HH

#include <span>

#include <Core/ReferenceCounting.hh>
#include "DataView.hh"
#include "TransitionTypes.hh"
#include "Types.hh"

namespace Nn {

struct DenseScoreTerm {
    std::span<Score const> scores;
    Score                  scale = 1.0;
};

/*
 * Dense score view over one or more contiguous vocabulary score spans.
 */
struct DenseScoreSpan {
    std::vector<DenseScoreTerm> terms;

    DenseScoreSpan(std::vector<DenseScoreTerm>&& terms)
            : terms(std::move(terms)) {}
    DenseScoreSpan(DenseScoreTerm&& term)
            : terms{std::move(term)} {};

    size_t size() const {
        return terms.empty() ? 0ul : terms.front().scores.size();
    }

    Score operator[](size_t idx) const {
        if (idx == Nn::invalidLabelIndex) {
            return 0.0;
        }
        // Fast path without loop for common 1-element case
        if (terms.size() == 1ul) {
            return terms.front().scores[idx] * terms.front().scale;
        }

        Score result = 0.0;
        for (auto const& term : terms) {
            result += term.scores[idx] * term.scale;
        }
        return result;
    }
};

/*
 * Abstract base class for score accessor interface
 */
class ScoreAccessor : public Core::ReferenceCounted {
public:
    virtual Score getScore(TransitionType transitionType, LabelIndex labelIndex = invalidLabelIndex) const;

    // Optional contiguous score view over vocabulary for accessors that support it
    virtual std::optional<DenseScoreSpan> getDenseScores() const;

    virtual TimeframeIndex getTime() const;

    virtual ~ScoreAccessor() = default;
};

typedef Core::Ref<ScoreAccessor> ScoreAccessorRef;

/*
 * Score accessor that contains a vector of scores for each label
 */
class VectorScoreAccessor : public ScoreAccessor {
public:
    VectorScoreAccessor(std::shared_ptr<std::vector<Score>> scores, TimeframeIndex time);

    Score                         getScore(TransitionType transitionType, LabelIndex labelIndex = invalidLabelIndex) const override;
    std::optional<DenseScoreSpan> getDenseScores() const override;
    TimeframeIndex                getTime() const override;

private:
    std::shared_ptr<std::vector<Score>> scores_;
    TimeframeIndex                      time_;
};

/*
 * Score accessor that contains a DataView of scores
 */
class DataViewScoreAccessor : public ScoreAccessor {
public:
    DataViewScoreAccessor(DataView const& dataView, TimeframeIndex time);

    Score                         getScore(TransitionType transitionType, LabelIndex labelIndex = invalidLabelIndex) const override;
    std::optional<DenseScoreSpan> getDenseScores() const override;
    TimeframeIndex                getTime() const override;

private:
    DataView       dataView_;
    TimeframeIndex time_;
};

}  // namespace Nn

#endif  // SCORE_ACCESSOR_HH
