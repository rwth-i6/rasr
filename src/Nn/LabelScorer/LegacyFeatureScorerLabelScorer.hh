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

#ifndef LEGACY_FEATURE_SCORER_LABEL_SCORER_HH
#define LEGACY_FEATURE_SCORER_LABEL_SCORER_HH

#include <Mm/FeatureScorer.hh>
#include "LabelScorer.hh"

namespace Nn {

/*
 * Wrapper around legacy Mm::FeatureScorer.
 * Inputs are treated as features for the FeatureScorer.
 * After adding features, whenever possible (depending on FeatureScorer buffering)
 * directly prepare ContextScorers based on them and cache these.
 * Upon receiving the feature stream end signal, all available ContextScorers are flushed.
 */
class LegacyFeatureScorerLabelScorer : public LabelScorer {
    using Precursor = LabelScorer;

public:
    LegacyFeatureScorerLabelScorer(const Core::Configuration& config);

    // Reset internal feature scorer and clear cache of context scorers
    void reset() override;

    // Add feature to internal feature scorer. Afterwards prepare and cache context scorer if possible.
    void addInput(SharedDataHolder const& input, size_t featureSize) override;
    void addInput(std::vector<f32> const& input) override;

    // Flush and cache all remaining context scorers
    void signalNoMoreFeatures() override;

    // Initial context just contains step 0.
    ScoringContextRef getInitialScoringContext() override;

    // Scoring context with step incremented by 1.
    ScoringContextRef extendedScoringContext(LabelScorer::Request const& request) override;

    // Use cached context scorer at given step to score the next token.
    std::optional<LabelScorer::ScoreWithTime> computeScoreWithTime(LabelScorer::Request const& request) override;

private:
    Core::Ref<Mm::FeatureScorer>           featureScorer_;
    std::vector<Mm::FeatureScorer::Scorer> scoreCache_;
};

}  // namespace Nn

#endif  // LEGACY_FEATURE_SCORER_LABEL_SCORER_HH
