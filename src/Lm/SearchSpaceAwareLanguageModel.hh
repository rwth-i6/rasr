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
#ifndef _LM_SEARCH_SPACE_AWARE_LANGUAGE_MODEL_HH
#define _LM_SEARCH_SPACE_AWARE_LANGUAGE_MODEL_HH

#include <Search/Types.hh>

#include "LanguageModel.hh"

namespace Lm {
struct SearchSpaceInformation {
    SearchSpaceInformation()
            : minLabelDistance(std::numeric_limits<unsigned>::max()),
              bestScore(std::numeric_limits<Score>::max()),
              bestScoreOffset(std::numeric_limits<Score>::max()),
              numStates(0u) {
    }
    ~SearchSpaceInformation() {}

    unsigned minLabelDistance;
    Score    bestScore;
    Score    bestScoreOffset;
    unsigned numStates;
};

class SearchSpaceAwareLanguageModel {
public:
    virtual void startFrame(Search::TimeframeIndex time) const                          = 0;
    virtual void setInfo(History const& hist, SearchSpaceInformation const& info) const = 0;
};
}  // namespace Lm

#endif  // _LM_SEARCH_SPACE_AWARE_LANGUAGE_MODEL_HH
