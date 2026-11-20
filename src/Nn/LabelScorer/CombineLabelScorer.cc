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

#include <algorithm>
#include <mutex>

#include <Math/Utilities.hh>
#include <Nn/Module.hh>
#include "Types.hh"

namespace {

using namespace Nn;

/*
 * Combines multiple scoring contexts at once
 */
struct CombineScoringContext : public ScoringContext {
    std::vector<ScoringContextRef> scoringContexts;

    CombineScoringContext()
            : scoringContexts() {}

    CombineScoringContext(std::vector<ScoringContextRef>&& scoringContexts)
            : scoringContexts(scoringContexts) {}

    bool isEqual(ScoringContextRef const& other) const {
        auto* otherPtr = dynamic_cast<CombineScoringContext const*>(other.get());

        if (otherPtr == nullptr) {
            return false;
        }

        return std::equal(
                scoringContexts.begin(),
                scoringContexts.end(),
                otherPtr->scoringContexts.begin(),
                otherPtr->scoringContexts.end(),
                [](auto const& sc1, auto const& sc2) {
                    return sc1->isEqual(sc2);
                });
    }

    size_t hash() const {
        size_t value = 0ul;
        for (auto const& scoringContext : scoringContexts) {
            value = Core::combineHashes(value, scoringContext->hash());
        }
        return value;
    }
};

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

    std::optional<DenseScoreSpan> getDenseScores() const override {
        if (subAccessors_.empty()) {
            return std::nullopt;
        }

        std::vector<DenseScoreTerm> denseScoreTerms;
        size_t                      denseSize = 0ul;
        for (auto const& subAccessor : subAccessors_) {
            auto denseScores = subAccessor->getDenseScores();
            if (not denseScores) {
                return std::nullopt;
            }
            if (denseScoreTerms.empty()) {
                denseSize = denseScores->size();
            }
            else if (denseScores->size() != denseSize) {
                return std::nullopt;
            }
            denseScoreTerms.insert(denseScoreTerms.end(), denseScores->terms.begin(), denseScores->terms.end());
        }

        return DenseScoreSpan(std::move(denseScoreTerms));
    }

    // Sparse if at least one sub-accessor is sparse. The candidates are the labels of the shortest sparse list that
    // all other sparse sub-accessors also list, with the summed scores of all sub-accessors (non-finite sums dropped).
    std::optional<SparseScoreSpan> getSparseScores() const override {
        std::call_once(sparseScoresComputed_, [this]() { computeSparseScores(); });
        if (not hasSparseScores_) {
            return std::nullopt;
        }
        return SparseScoreSpan{.labels = sparseLabels_, .scores = sparseScoreValues_};
    }

    // Max of timeframes from sub-scorers
    TimeframeIndex getTime() const override {
        return std::accumulate(subAccessors_.begin(), subAccessors_.end(), 0, [](TimeframeIndex max, ScoreAccessorRef subAccessor) {
            return std::max(max, subAccessor->getTime());
        });
    }

private:
    void computeSparseScores() const {
        std::vector<std::optional<SparseScoreSpan>> sparseSpans(subAccessors_.size());
        std::vector<std::optional<DenseScoreSpan>>  denseSpans(subAccessors_.size());
        std::optional<size_t>                       shortestIdx;
        for (size_t i = 0ul; i < subAccessors_.size(); ++i) {
            sparseSpans[i] = subAccessors_[i]->getSparseScores();
            if (sparseSpans[i]) {
                if (not shortestIdx or sparseSpans[i]->size() < sparseSpans[*shortestIdx]->size()) {
                    shortestIdx = i;
                }
                continue;
            }
            denseSpans[i] = subAccessors_[i]->getDenseScores();
            if (not denseSpans[i]) {
                return;  // scores of this sub-accessor only via `getScore`
            }
        }
        if (not shortestIdx) {
            return;
        }
        hasSparseScores_ = true;

        auto const& shortest = *sparseSpans[*shortestIdx];
        sparseLabels_.reserve(shortest.size());
        sparseScoreValues_.reserve(shortest.size());
        for (size_t labelIdx = 0ul; labelIdx < shortest.size(); ++labelIdx) {
            LabelIndex label  = shortest.labels[labelIdx];
            Score      score  = shortest[labelIdx];
            bool       listed = true;
            for (size_t i = 0ul; i < subAccessors_.size() and listed; ++i) {
                if (i == *shortestIdx) {
                    continue;
                }
                if (sparseSpans[i]) {
                    auto const& labels = sparseSpans[i]->labels;
                    auto        it     = std::find(labels.begin(), labels.end(), label);
                    listed             = it != labels.end();
                    if (listed) {
                        score += (*sparseSpans[i])[it - labels.begin()];
                    }
                }
                else {
                    listed = label < denseSpans[i]->size();
                    if (listed) {
                        score += (*denseSpans[i])[label];
                    }
                }
            }
            if (not listed or Math::isinf(score) or score >= Core::Type<Score>::max) {
                continue;
            }
            sparseLabels_.push_back(label);
            sparseScoreValues_.push_back(score);
        }
    }

    std::vector<ScoreAccessorRef> subAccessors_;

    mutable std::once_flag          sparseScoresComputed_;
    mutable bool                    hasSparseScores_ = false;
    mutable std::vector<LabelIndex> sparseLabels_;
    mutable std::vector<Score>      sparseScoreValues_;
};

}  // namespace

namespace Nn {

Core::ParameterInt CombineLabelScorer::paramNumLabelScorers(
        "num-scorers", "Number of label scorers to combine", 1, 1);

CombineLabelScorer::CombineLabelScorer(Core::Configuration const& config, ModelCache& modelCache)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::ALL) {
    size_t numLabelScorers = paramNumLabelScorers(config);
    for (size_t i = 0ul; i < numLabelScorers; ++i) {
        Core::Configuration subConfig = select(std::string("scorer-") + std::to_string(i + 1));
        scorers_.push_back(Nn::Module::instance().labelScorerFactory().createLabelScorer(subConfig, modelCache));
        enabledTransitions_.enableIntersection(scorers_.back()->enabledTransitions());
    }
}

Core::Ref<ScaledLabelScorer> CombineLabelScorer::getSubScorer(size_t index) const {
    require(index < scorers_.size());
    return scorers_[index];
}

void CombineLabelScorer::logStatistics() const {
    for (auto const& scorer : scorers_) {
        scorer->logStatistics();
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

void CombineLabelScorer::reset() {
    for (auto& scorer : scorers_) {
        scorer->reset();
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
