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

#include <Core/ReferenceCounting.hh>
#include "DataView.hh"
#include "TransitionTypes.hh"
#include "Types.hh"

namespace Nn {

/*
 * Abstract base class for score accessor interface
 */
class ScoreAccessor : public Core::ReferenceCounted {
public:
    virtual Score          getScore(TransitionType transitionType, LabelIndex labelIndex = invalidLabelIndex) const;
    virtual TimeframeIndex getTime() const;

    virtual ~ScoreAccessor() = default;
};

typedef Core::Ref<ScoreAccessor> ScoreAccessorRef;

inline Score ScoreAccessor::getScore(TransitionType transitionType, LabelIndex labelIndex) const {
    return 0.0;
};

inline TimeframeIndex ScoreAccessor::getTime() const {
    return 0;
};

/*
 * Score accessor that contains a vector of scores for each label
 */
class VectorScoreAccessor : public ScoreAccessor {
public:
    VectorScoreAccessor(std::shared_ptr<std::vector<Score>> scores, TimeframeIndex time);

    Score          getScore(TransitionType transitionType, LabelIndex labelIndex = invalidLabelIndex) const override;
    TimeframeIndex getTime() const override;

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

    Score          getScore(TransitionType transitionType, LabelIndex labelIndex = invalidLabelIndex) const override;
    TimeframeIndex getTime() const override;

private:
    DataView       dataView_;
    TimeframeIndex time_;
};

}  // namespace Nn

#endif  // SCORE_ACCESSOR_HH
