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

#include "CtcPrefixLabelScorer.hh"

#include <Nn/Module.hh>
#include <cmath>
#include <limits>

namespace Nn {

namespace {

// Compute score difference while accounting for inf values
Score scoreDelta(Score extScore, Score prefixScore) {
    if (std::isinf(extScore) or std::isinf(prefixScore)) {
        return Core::Type<Score>::max;
    }
    return extScore - prefixScore;
}

// Compute score of extended prefix consisting of a base prefix from `scoringContext_` plus `labelIndex` afterwards.
// This is decomposed into the sum of scores of the extended prefix occurring with `labelIndex` first observed
// at time t, which can be computed as the score of the base prefix occurring up to time t-1 plus an emission
// of `labelIndex` at time t.
Score extendedPrefixScore(
        CtcPrefixScoringContextRef const&               scoringContext,
        LabelIndex                                      labelIndex,
        std::shared_ptr<Math::FastMatrix<Score>> const& ctcScores) {
    verify(scoringContext->timePrefixScores);

    // Degenerate case such as empty segment
    if (ctcScores->nColumns() == 0 or scoringContext->timePrefixScores->empty()) {
        return std::numeric_limits<Score>::infinity();
    }

    // Check whether result is already cached
    auto it = scoringContext->extScores.find(labelIndex);
    if (it != scoringContext->extScores.end()) {
        return it->second;
    }

    // Compute totalScore by summing over all timesteps
    // t = 0
    Score totalScore;
    if (scoringContext->labelSeq.empty()) {
        totalScore = ctcScores->at(labelIndex, 0);
    }
    else {
        totalScore = std::numeric_limits<Score>::infinity();
    }

    // t > 0
    for (size_t t = 1ul; t < ctcScores->nColumns(); ++t) {
        Score timestepScore;
        if (not scoringContext->labelSeq.empty() and labelIndex == scoringContext->labelSeq.back()) {
            // If prefix ends in the same token as `labelIndex` there must a blank between, thus the prefix can only end in blank at t-1
            timestepScore = scoringContext->timePrefixScores->at(t - 1).blankEndingScore;
        }
        else {
            // If prefix ends in a different token as `labelIndex` it can end in both blank or non-blank at t-1
            timestepScore = scoringContext->timePrefixScores->at(t - 1).totalScore();
        }
        timestepScore += ctcScores->at(labelIndex, t);
        totalScore = Math::scoreSum(totalScore, timestepScore);
    }
    scoringContext->extScores.emplace(labelIndex, totalScore);  // Cache result to avoid repeated computation

    return totalScore;
}

}  // namespace

/*
 * =============================
 * == CtcPrefixScoreAccessor ===
 * =============================
 */

CtcPrefixScoreAccessor::CtcPrefixScoreAccessor(CtcPrefixScoringContextRef const& scoringContext, std::shared_ptr<Math::FastMatrix<Score>> const& ctcScores)
        : scoringContext_(scoringContext),
          ctcScores_(ctcScores) {
}

Score CtcPrefixScoreAccessor::getScore(TransitionType transitionType, LabelIndex labelIndex) const {
    // Since this function should return the score-delta for the new label, subtract the total score of the base prefix
    // before returning.
    verify(scoringContext_->prefixScore);
    verify(scoringContext_->timePrefixScores);

    // Degenerate case such as empty segment
    if (ctcScores_->nColumns() == 0 or scoringContext_->timePrefixScores->empty()) {
        return Core::Type<Score>::max;
    }

    if (transitionType == SENTENCE_END) {
        // Score of the exact prefix at the last timestep
        return scoreDelta(scoringContext_->timePrefixScores->back().totalScore(), *scoringContext_->prefixScore);
    }

    // Score of the extended prefix with any arbitrary suffix
    return scoreDelta(extendedPrefixScore(scoringContext_, labelIndex, ctcScores_), *scoringContext_->prefixScore);
}

TimeframeIndex CtcPrefixScoreAccessor::getTime() const {
    return scoringContext_->labelSeq.size();
}

/*
 * =============================
 * === CtcPrefixLabelScorer ====
 * =============================
 */

const Core::ParameterInt CtcPrefixLabelScorer::paramBlankIndex("blank-label-index", "Index of blank symbol in vocabulary.");
const Core::ParameterInt CtcPrefixLabelScorer::paramVocabSize("vocab-size", "Number of labels in CTC scorer vocabulary.");

CtcPrefixLabelScorer::CtcPrefixLabelScorer(Core::Configuration const& config, ModelCache& modelCache)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::LM),
          blankIndex_(paramBlankIndex(config)),
          vocabSize_(paramVocabSize(config)),
          ctcScorer_(Module::instance().labelScorerFactory().createLabelScorer(select("ctc-scorer"), modelCache)),
          expectMoreFeatures_(true),
          ctcScores_(std::make_shared<Math::FastMatrix<Score>>(vocabSize_, 0)) {
}

void CtcPrefixLabelScorer::reset() {
    ctcScorer_->reset();
    expectMoreFeatures_ = true;
}

void CtcPrefixLabelScorer::signalNoMoreFeatures() {
    ctcScorer_->signalNoMoreFeatures();
    expectMoreFeatures_ = false;
    setupCTCScores();
}

void CtcPrefixLabelScorer::addInput(DataView const& input) {
    ctcScorer_->addInput(input);
}

void CtcPrefixLabelScorer::addInputs(DataView const& inputs, size_t nTimesteps) {
    ctcScorer_->addInputs(inputs, nTimesteps);
}

ScoringContextRef CtcPrefixLabelScorer::getInitialScoringContext() {
    return Core::ref(new CtcPrefixScoringContext());
}

ScoringContextRef CtcPrefixLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) {
    auto context = Core::ref(dynamic_cast<CtcPrefixScoringContext const*>(scoringContext.get()));

    std::vector<LabelIndex> newLabelSeq(context->labelSeq);
    newLabelSeq.push_back(nextToken);

    return Core::ref(new CtcPrefixScoringContext(std::move(newLabelSeq), scoringContext));
}

std::optional<ScoreAccessorRef> CtcPrefixLabelScorer::getScoreAccessor(ScoringContextRef scoringContext) {
    if (expectMoreFeatures_) {
        return {};
    }

    auto context = Core::ref(dynamic_cast<const CtcPrefixScoringContext*>(scoringContext.get()));
    finalizeScoringContext(context);

    return Core::ref(new CtcPrefixScoreAccessor(context, ctcScores_));
}

void CtcPrefixLabelScorer::setupCTCScores() {
    auto ctcScorerContext = ctcScorer_->getInitialScoringContext();

    // Collect score accessors for all timesteps
    std::vector<ScoreAccessorRef> scoreAccessors;
    while (true) {
        auto scoreAccessor = ctcScorer_->getScoreAccessor(ctcScorerContext);
        if (not scoreAccessor) {
            break;
        }
        scoreAccessors.push_back(*scoreAccessor);

        // Transition type and next token assumed to not influence the scoring context
        ctcScorerContext = ctcScorer_->extendedScoringContext(ctcScorerContext, invalidLabelIndex, LABEL_TO_BLANK);
    }

    // Write score values into matrix
    ctcScores_->resize(vocabSize_, scoreAccessors.size());
    for (size_t t = 0ul; t < scoreAccessors.size(); ++t) {
        for (size_t v = 0ul; v < vocabSize_; ++v) {
            // Transition type can be anything as we assume that the score is independent of it
            ctcScores_->at(v, t) = scoreAccessors[t]->getScore(Nn::TransitionType::LABEL_TO_BLANK, v);
        }
    }
}

void CtcPrefixLabelScorer::finalizeScoringContext(CtcPrefixScoringContextRef const& scoringContext) const {
    if (not scoringContext->requiresFinalize) {
        return;
    }

    if (scoringContext->labelSeq.empty()) {
        scoringContext->prefixScore = 0.0;

        // `timePrefixScores` stores one entry per real CTC frame; there is no separate virtual predecessor entry.
        // For the empty prefix, the score after frame t is the cumulative blank score over frames 0, ..., t.
        // The non-blank ending score stays at infinity because the empty prefix cannot end in a non-blank label.

        auto timePrefixScores = std::make_shared<std::vector<CtcPrefixScoringContext::PrefixScore>>();
        timePrefixScores->resize(ctcScores_->nColumns());

        Score cumulativeBlankScore = 0.0;
        for (size_t t = 0ul; t < ctcScores_->nColumns(); ++t) {
            cumulativeBlankScore += ctcScores_->at(blankIndex_, t);
            timePrefixScores->at(t).blankEndingScore = cumulativeBlankScore;
        }

        scoringContext->timePrefixScores = timePrefixScores;
    }
    else {
        // Set `prefixScore` via parent. Parent may also need to be finalized before (resulting in recursion).
        auto parent = Core::ref(dynamic_cast<const CtcPrefixScoringContext*>(scoringContext->parent.get()));
        verify(parent);
        finalizeScoringContext(parent);
        scoringContext->prefixScore = extendedPrefixScore(parent, scoringContext->labelSeq.back(), ctcScores_);

        // We are given stored scores for the parent prefix [..., a] after each real CTC frame t = 0, ..., T - 1,
        // and compute stored scores for the extended prefix [..., a, b] over the same frame indices.
        // Frame 0 is handled explicitly below. For frames t >= 1, the recurrence is:
        // PrefixScore_t([..., a, b], blank) = LogSumExp(
        //                                          PrefixScore_{t-1}([..., a, b], blank) + CTCScore_t(blank),
        //                                          PrefixScore_{t-1}([..., a, b], nonblank) + CTCScore_t(blank)
        //                                     )
        // and
        // PrefixScore_t([..., a, b], nonblank) = LogSumExp(
        //                                            PrefixScore_{t-1}([..., a], blank) + CTCScore_t(b),
        //                                            PrefixScore_{t-1}([..., a, b], nonblank) + CTCScore_t(b),
        //                                           [PrefixScore_{t-1}([..., a], nonblank) + CTCScore_t(b) only if a != b]
        //                                        )
        auto const& timePrefixScores    = parent->timePrefixScores;
        auto        extTimePrefixScores = std::make_shared<std::vector<CtcPrefixScoringContext::PrefixScore>>();
        auto        nextToken           = scoringContext->labelSeq.back();
        verify(timePrefixScores);

        extTimePrefixScores->resize(timePrefixScores->size());
        if (not extTimePrefixScores->empty() and scoringContext->labelSeq.size() == 1) {
            extTimePrefixScores->at(0).nonBlankEndingScore = ctcScores_->at(nextToken, 0);
        }

        for (size_t t = 1ul; t < ctcScores_->nColumns(); ++t) {
            Score& blankEndingScore    = extTimePrefixScores->at(t).blankEndingScore;
            Score& nonBlankEndingScore = extTimePrefixScores->at(t).nonBlankEndingScore;

            blankEndingScore    = extTimePrefixScores->at(t - 1).totalScore() + ctcScores_->at(blankIndex_, t);
            nonBlankEndingScore = extTimePrefixScores->at(t - 1).nonBlankEndingScore;  // Label loop
            if (scoringContext->labelSeq.size() >= 2 and nextToken == scoringContext->labelSeq[scoringContext->labelSeq.size() - 2]) {
                // If the last label is equal to the one before it, there must be a blank in between, i.e., the prefix must not end at non-blank at t-1
                nonBlankEndingScore = Math::scoreSum(nonBlankEndingScore, timePrefixScores->at(t - 1).blankEndingScore);  // Blank-to-label
            }
            else {
                // If the last two labels are different, the prefix can end in both blank or non-blank
                nonBlankEndingScore = Math::scoreSum(nonBlankEndingScore, timePrefixScores->at(t - 1).totalScore());  // Blank-to-label or label-to-label
            }
            nonBlankEndingScore += ctcScores_->at(nextToken, t);
        }

        scoringContext->timePrefixScores = extTimePrefixScores;
    }

    scoringContext->requiresFinalize = false;
}

}  // namespace Nn
