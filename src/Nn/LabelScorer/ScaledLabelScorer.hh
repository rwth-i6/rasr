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

#ifndef SCALED_LABEL_SCORER_HH
#define SCALED_LABEL_SCORER_HH

#include <Core/Configuration.hh>
#include <cstddef>

#include "LabelScorer.hh"

namespace Nn {

/*
 * Wraps a sub label scorer and scales all the scores by a given factor
 */
class ScaledLabelScorer : public LabelScorer {
public:
    static const Core::ParameterFloat paramScale;

    ScaledLabelScorer(Core::Configuration const& config, Core::Ref<LabelScorer> const& scorer);

    // Return the wrapped label scorer
    Core::Ref<LabelScorer> labelScorer() const;

    // Return the current scale
    Score scale() const;

    // Set the scale, overriding the value from the config
    void setScale(Score scale);

    // Forwarded to the wrapped scorer; scaling itself accumulates no statistics
    void logStatistics() const override;

    // Forward signal to sub-scorer
    Core::Status signalNoMoreFeatures() override;

    // Initial ScoringContext from sub-scorer
    Core::Status getInitialScoringContext(ScoringContextRef& initialScoringContext) override;

    // Extended ScoringContext from sub-scorer
    Core::Status extendedScoringContext(ScoringContextRef context, LabelIndex nextToken, TransitionType transitionType, ScoringContextRef& extendedContext) override;

    // Cleanup sub-scorer
    void cleanupCaches(Core::CollapsedVector<ScoringContextRef> const& activeContexts) override;

    // Add input to sub-scorer
    Core::Status addInput(DataView const& input) override;

    // Add inputs to sub-scorer
    virtual Core::Status addInputs(DataView const& input, size_t nTimesteps) override;

    // Reset sub-scorer
    void reset() override;

    // Score accessor wrapper that scales the scores
    Core::Status getScoreAccessor(ScoringContextRef scoringContext, std::optional<ScoreAccessorRef>& scoreAccessor) override;

    // Score accessor wrapper that scales the scores
    Core::Status getScoreAccessors(std::vector<ScoringContextRef> const& scoringContexts, std::vector<std::optional<ScoreAccessorRef>>& scoreAccessors) override;

private:
    Core::Ref<LabelScorer> scorer_;
    Score                  scale_;
};

}  // namespace Nn

#endif  // SCALED_LABEL_SCORER_HH
