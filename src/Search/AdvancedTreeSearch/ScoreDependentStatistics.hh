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
#ifndef SEARCH_SCOREDEPENDENTSTATISTICS_HH
#define SEARCH_SCOREDEPENDENTSTATISTICS_HH

#include <Core/Assertions.hh>
#include <Search/Types.hh>
#include <string>
#include <vector>

namespace Search {
class ScoreDependentStatistic {
public:
    f32 getValue(Score relativeStartScore) {
        if (relativeStartScore < 0)
            relativeStartScore = 0;

        if (relativeStartScore > maxRelativeScore_)
            return 0;

        u32 index = (u32)((relativeStartScore / maxRelativeScore_) * granularity_);
        if (index == granularity_)
            --index;

        verify(granularity_ < 1000);

        verify(index < effort_.size());

        if (effort_[index].first)
            return effort_[index].second / effort_[index].first;
        else
            return 0;
    }

    void addValue(Score relativeStartScore, f32 value) {
        if (relativeStartScore < 0) {
            relativeStartScore = 0;
        }

        if (relativeStartScore > maxRelativeScore_) {
            relativeStartScore = maxRelativeScore_;
        }

        u32 index = (u32)((relativeStartScore / maxRelativeScore_) * granularity_);
        if (index == granularity_)
            --index;

        verify(granularity_ < 1000);

        verify(index < effort_.size());

        ++effort_[index].first;
        effort_[index].second += value;
    }

    std::vector<u32> histogram() const {
        std::vector<u32> ret;
        for (u32 a = 0; a < effort_.size(); ++a)
            ret.push_back(effort_[a].first);
        return ret;
    }

    std::vector<f32> efforts() const {
        std::vector<f32> ret;
        for (u32 a = 0; a < effort_.size(); ++a) {
            f32 effort = 0;
            if (effort_[a].first) {
                effort = effort_[a].second / f32(effort_[a].first);
            }
            ret.push_back(effort);
        }
        return ret;
    }

    int granularity() const {
        return granularity_;
    }

    Score maxRelativeScore() const {
        return maxRelativeScore_;
    }

    void initialize(u32 granularity, Score maxRelativeScore) {
        granularity_      = granularity;
        maxRelativeScore_ = maxRelativeScore;

        effort_.clear();
        effort_.resize(granularity_, std::make_pair(0u, 0u));
    }

    std::string print() const {
        std::ostringstream ret;
        ret << "{";
        bool first = true;
        for (u32 a = 0; a < effort_.size(); ++a) {
            if (!first) {
                ret << ", ";
            }
            else {
                first = false;
            }
            Score score  = (f32(a) / granularity_) * maxRelativeScore_;
            f32   effort = 0;
            if (effort_[a].first) {
                effort = effort_[a].second / f32(effort_[a].first);
            }
            ret << score << ":" << effort;
        }
        ret << "}";
        return ret.str();
    }

    std::string printHistogram() const {
        std::ostringstream ret;
        ret << "{";
        bool first = true;
        for (u32 a = 0; a < effort_.size(); ++a) {
            if (!first) {
                ret << ", ";
            }
            else {
                first = false;
            }
            Score score = (f32(a) / granularity_) * maxRelativeScore_;

            ret << score << ":" << effort_[a].first;
        }
        ret << "}";
        return ret.str();
    }

private:
    u32                              granularity_;
    Score                            maxRelativeScore_;
    std::vector<std::pair<u32, f32>> effort_;
};

class ScoreDependentVectorStatistic {
public:
    void addValue(Score relativeStartScore, u32 offset, f32 value);

    void initialize(u32 granularity, Score maxRelativeScore) {
        granularity_      = granularity;
        maxRelativeScore_ = maxRelativeScore;

        effort_.clear();
        effort_.resize(granularity_);
    }

    std::string print() {
        std::ostringstream ret;
        ret << "{";
        bool first = true;
        for (u32 a = 0; a < effort_.size(); ++a) {
            if (!first) {
                ret << ", ";
            }
            else {
                first = false;
            }

            Score score = (f32(a) / granularity_) * maxRelativeScore_;
            ret << score << ": {";
            bool first2 = true;

            for (u32 b = 0; b < effort_[a].size(); ++b) {
                if (!first2) {
                    ret << ", ";
                }
                else {
                    first2 = false;
                }
                f32 effort = 0;
                if (effort_[a][b].first) {
                    effort = effort_[a][b].second / f32(effort_[a][b].first);
                }
                ret << b << ":" << effort;
            }
            ret << "} ";
        }
        ret << "}";
        return ret.str();
    }

    const std::vector<std::vector<std::pair<u32, f32>>>& data() {
        return effort_;
    }

private:
    u32                                           granularity_;
    Score                                         maxRelativeScore_;
    std::vector<std::vector<std::pair<u32, f32>>> effort_;
};
}  // namespace Search

#endif  // SEARCH_SCOREDEPENDENTSTATISTICS_HH
