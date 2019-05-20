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
