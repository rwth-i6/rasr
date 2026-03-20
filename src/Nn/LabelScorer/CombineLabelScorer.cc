/** Copyright 2025 RWTH Aachen University. All rights reserved.
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

#include "CombineLabelScorer.hh"
#include <Nn/Module.hh>
#include "Types.hh"

namespace {

using namespace Nn;

/*
 * Score accessor that contains a list of sub-accessors and adds up the scores they return
 */
class CombinedScoreAccessor : public ScoreAccessor {
public:
    CombinedScoreAccessor()
        : subAccessors_() {}

    void addSubAccessor(ScoreAccessorRef subAccessor) {
        subAccessors_.push_back(subAccessor);
    }

    // Sum of scores from sub-scorers
    Score getScore(TransitionType transitionType, LabelIndex labelIndex = invalidLabelIndex) const override {
        return std::accumulate(subAccessors_.begin(), subAccessors_.end(), 0.0, [transitionType, labelIndex](Score acc, ScoreAccessorRef subAccessor) {
            return acc + subAccessor->getScore(transitionType, labelIndex);
        });
    }

    // Max of timeframes from sub-scorers
    TimeframeIndex getTime() const override {
        return std::accumulate(subAccessors_.begin(), subAccessors_.end(), 0, [](TimeframeIndex max, ScoreAccessorRef subAccessor) {
            return std::max(max, subAccessor->getTime());
        });
    }

private:
    std::vector<ScoreAccessorRef> subAccessors_;
};

}  // namespace

namespace Nn {

Core::ParameterInt CombineLabelScorer::paramNumLabelScorers(
        "num-scorers", "Number of label scorers to combine", 1, 1);

CombineLabelScorer::CombineLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::ALL) {
    size_t numLabelScorers = paramNumLabelScorers(config);
    for (size_t i = 0ul; i < numLabelScorers; ++i) {
        Core::Configuration subConfig = select(std::string("scorer-") + std::to_string(i + 1));
        scorers_.push_back(Nn::Module::instance().labelScorerFactory().createLabelScorer(subConfig));
        enabledTransitions_.enableIntersection(scorers_.back()->enabledTransitions());
    }
}

void CombineLabelScorer::reset() {
    for (auto& scorer : scorers_) {
        scorer->reset();
    }
}

void CombineLabelScorer::signalNoMoreFeatures() {
    for (auto& scorer : scorers_) {
        scorer->signalNoMoreFeatures();
    }
}

ScoringContextRef CombineLabelScorer::getInitialScoringContext() {
    std::vector<ScoringContextRef> scoringContexts;
    scoringContexts.reserve(scorers_.size());

    for (auto const& scorer : scorers_) {
        scoringContexts.push_back(scorer->getInitialScoringContext());
    }
    return Core::ref(new CombineScoringContext(std::move(scoringContexts)));
}

ScoringContextRef CombineLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) {
    auto combineContext = dynamic_cast<CombineScoringContext const*>(scoringContext.get());

    std::vector<ScoringContextRef> extScoringContexts;
    extScoringContexts.reserve(scorers_.size());

    auto scorerIt  = scorers_.begin();
    auto contextIt = combineContext->scoringContexts.begin();

    for (; scorerIt != scorers_.end(); ++scorerIt, ++contextIt) {
        extScoringContexts.push_back((*scorerIt)->extendedScoringContext(*contextIt, nextToken, transitionType));
    }
    return Core::ref(new CombineScoringContext(std::move(extScoringContexts)));
}

void CombineLabelScorer::cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) {
    std::vector<CombineScoringContext const*> combineContexts;
    combineContexts.reserve(activeContexts.internalSize());
    for (auto const& activeContext : activeContexts.internalData()) {
        combineContexts.push_back(dynamic_cast<CombineScoringContext const*>(activeContext.get()));
    }

    for (size_t scorerIdx = 0ul; scorerIdx < scorers_.size(); ++scorerIdx) {
        auto const&                              scorer = scorers_[scorerIdx];
        Core::CollapsedVector<ScoringContextRef> subScoringContexts;
        for (auto const& combineContext : combineContexts) {
            subScoringContexts.push_back(combineContext->scoringContexts[scorerIdx]);
        }

        scorer->cleanupCaches(subScoringContexts);
    }
}

void CombineLabelScorer::addInput(DataView const& input) {
    for (auto& scorer : scorers_) {
        scorer->addInput(input);
    }
}

void CombineLabelScorer::addInputs(DataView const& input, size_t nTimesteps) {
    for (auto& scorer : scorers_) {
        scorer->addInputs(input, nTimesteps);
    }
}

std::optional<ScoreAccessorRef> CombineLabelScorer::getScoreAccessor(ScoringContextRef scoringContext) {
    auto combineContext = dynamic_cast<CombineScoringContext const*>(scoringContext.get());

    auto                          combinedAccessor = Core::ref(new CombinedScoreAccessor());
    std::vector<ScoreAccessorRef> subAccessors;
    subAccessors.reserve(scorers_.size());

    auto scorerIt  = scorers_.begin();
    auto contextIt = combineContext->scoringContexts.begin();
    for (; scorerIt != scorers_.end(); ++scorerIt, ++contextIt) {
        auto subAccessor = (*scorerIt)->getScoreAccessor(*contextIt);
        // If any of the sub-scorers can't score, the overall result is also None
        if (not subAccessor) {
            return {};
        }

        combinedAccessor->addSubAccessor(*subAccessor);
    }

    return combinedAccessor;
}

std::vector<std::optional<ScoreAccessorRef>> CombineLabelScorer::getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts) {
    // Collect CombineScoringContexts
    std::vector<CombineScoringContext const*> combineContexts;
    combineContexts.reserve(scoringContexts.size());
    for (auto const& scoringContext : scoringContexts) {
        combineContexts.push_back(dynamic_cast<CombineScoringContext const*>(scoringContext.get()));
    }

    std::vector<std::optional<ScoreAccessorRef>> combinedAccessors;
    combinedAccessors.reserve(scoringContexts.size());
    for (size_t i = 0ul; i < scoringContexts.size(); ++i) {
        combinedAccessors.push_back(Core::ref(new CombinedScoreAccessor()));
    }

    // Count how many score accessors are already set to std::nullopt in order to
    // break early when everything is null.
    size_t numNullAccessors = 0ul;

    for (size_t scorerIdx = 0ul; scorerIdx < scorers_.size(); ++scorerIdx) {
        // Extract contexts for current scorer from
        std::vector<ScoringContextRef> subScorerContexts;
        subScorerContexts.reserve(combineContexts.size());
        for (auto const& combineContext : combineContexts) {
            subScorerContexts.push_back(combineContext->scoringContexts[scorerIdx]);
        }

        // Get score accessors for sub-scorer and merge them into the existing combinedAccessors
        auto subAccessors = scorers_[scorerIdx]->getScoreAccessors(subScorerContexts);
        for (size_t i = 0ul; i < combinedAccessors.size(); ++i) {
            if (not combinedAccessors[i]) {
                // Null accessors continue to be null for all further scorers
                continue;
            }
            if (subAccessors[i]) {
                dynamic_cast<CombinedScoreAccessor*>(combinedAccessors[i]->get())->addSubAccessor(*subAccessors[i]);
            }
            else {
                combinedAccessors[i] = std::nullopt;
                ++numNullAccessors;

                // Break early if all elements are null
                if (numNullAccessors == combinedAccessors.size()) {
                    return combinedAccessors;
                }
            }
        }
    }

    return combinedAccessors;
}

}  // namespace Nn
