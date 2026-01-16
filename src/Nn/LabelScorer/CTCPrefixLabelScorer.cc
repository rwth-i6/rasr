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

#include "CTCPrefixLabelScorer.hh"
#include "ScoringContext.hh"

#include <Core/Assertions.hh>
#include <Core/Parameter.hh>
#include <Core/ReferenceCounting.hh>
#include <Flow/Timestamp.hh>
#include <Math/FastMatrix.hh>
#include <Math/Utilities.hh>
#include <Mm/Module.hh>
#include <Nn/Module.hh>
#include <Speech/Types.hh>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace Nn {

/*
 * =============================
 * === CTCPrefixLabelScorer ====
 * =============================
 */

const Core::ParameterInt CTCPrefixLabelScorer::paramBlankIndex("blank-label-index", "Index of blank symbol in vocabulary.");
const Core::ParameterInt CTCPrefixLabelScorer::paramVocabSize("vocab-size", "Number of labels in CTC scorer vocabulary.");

CTCPrefixLabelScorer::CTCPrefixLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config, TransitionPresetType::LM),
          blankIndex_(paramBlankIndex(config)),
          vocabSize_(paramVocabSize(config)),
          ctcScorer_(Module::instance().labelScorerFactory().createLabelScorer(select("ctc-scorer"))),
          expectMoreFeatures_(true) {
}

void CTCPrefixLabelScorer::reset() {
    ctcScorer_->reset();
    expectMoreFeatures_ = true;
}

void CTCPrefixLabelScorer::signalNoMoreFeatures() {
    ctcScorer_->signalNoMoreFeatures();
    expectMoreFeatures_ = false;
    setupCTCScores();
}

void CTCPrefixLabelScorer::addInput(DataView const& input) {
    ctcScorer_->addInput(input);
}

void CTCPrefixLabelScorer::addInputs(DataView const& inputs, size_t nTimesteps) {
    ctcScorer_->addInputs(inputs, nTimesteps);
}

ScoringContextRef CTCPrefixLabelScorer::getInitialScoringContext() {
    return Core::ref(new CTCPrefixScoringContext());
}

ScoringContextRef CTCPrefixLabelScorer::extendedScoringContextInternal(LabelScorer::Request const& request) {
    auto context = Core::ref(dynamic_cast<CTCPrefixScoringContext const*>(request.context.get()));

    std::vector<LabelIndex> newLabelSeq(context->labelSeq);
    newLabelSeq.push_back(request.nextToken);
    // Before scoring context extension, the score has already been computed in `computeScoreWithTimeInternal`
    auto extScore = context->extScores.at(request.nextToken);

    return Core::ref(new CTCPrefixScoringContext(std::move(newLabelSeq), context->timePrefixScores, extScore, true));
}

std::optional<LabelScorer::ScoreWithTime> CTCPrefixLabelScorer::computeScoreWithTimeInternal(LabelScorer::Request const& request) {
    if (expectMoreFeatures_) {
        return {};
    }

    auto context = Core::ref(dynamic_cast<const CTCPrefixScoringContext*>(request.context.get()));
    finalizeScoringContext(context);

    Score totalScore;
    if (request.transitionType == SENTENCE_END) {
        // For EOS, the exact prefix must have been emitted until time T, so we only need
        // to consider the last element of the prefixScores vector which corresponds to t=T.
        totalScore = context->timePrefixScores->back().totalScore();
    }
    else {
        if (context->labelSeq.empty()) {
            totalScore = ctcScores_.at(request.nextToken, 0);
        }
        else {
            totalScore = std::numeric_limits<Score>::infinity();
        }
        for (size_t t = 1ul; t < ctcScores_.nColumns(); ++t) {
            // Prefix can always end by time t-1 with blank label
            Score timestepScore;
            if (not context->labelSeq.empty() and request.nextToken == context->labelSeq.back()) {
                // If prefix ends in the same token as `nextToken` there must a blank between, thus the prefix can only end in blank at t-1
                timestepScore = context->timePrefixScores->at(t - 1).blankEndingScore;
            }
            else {
                // If prefix ends in a different token as `nextToken` can it end in both blank or non-blank at t-1
                timestepScore = context->timePrefixScores->at(t - 1).totalScore();
            }

            timestepScore += ctcScores_.at(request.nextToken, t);

            totalScore = Math::scoreSum(totalScore, timestepScore);
        }
    }
    context->extScores.emplace(request.nextToken, totalScore);

    return ScoreWithTime{totalScore - context->prefixScore, static_cast<Speech::TimeframeIndex>(context->labelSeq.size() + 1)};
}

void CTCPrefixLabelScorer::setupCTCScores() {
    ctcScores_.resize(vocabSize_, 0);
    auto ctcScorerContext = ctcScorer_->getInitialScoringContext();
    while (true) {
        // Check if scores for next timestep are available
        if (not ctcScorer_->computeScoreWithTime({.context = ctcScorerContext, .nextToken = 0, .transitionType = LABEL_TO_BLANK})) {
            break;
        }

        // Add column for next timestep to matrix and insert score values into it
        ctcScores_.resizeColsAndKeepContent(ctcScores_.nColumns() + 1);
        for (LabelIndex v = 0ul; v < vocabSize_; ++v) {
            // Transition type can be anything as we assume that the score is independent of it
            ctcScores_.at(v, ctcScores_.nColumns() - 1) = ctcScorer_->computeScoreWithTime({.context = ctcScorerContext, .nextToken = v, .transitionType = LABEL_TO_BLANK})->score;
        }
        // Transition type and next token assumed to not influence the scoring context
        ctcScorerContext = ctcScorer_->extendedScoringContext({.context = ctcScorerContext, .nextToken = invalidLabelIndex, .transitionType = LABEL_TO_BLANK});
    }
}

void CTCPrefixLabelScorer::finalizeScoringContext(CTCPrefixScoringContextRef const& scoringContext) const {
    if (not scoringContext->requiresFinalize) {
        return;
    }

    if (scoringContext->labelSeq.empty()) {
        // In the beginning the prefix is empty, which can only be achieved by emitting pure blanks.
        // So PrefixScore_0([], blank) = 0 and PrefixScore_t([], blank) = sum_{t'=1}^t -log p_{t'}(<blank>) for t >= 1
        // PrefixScore_t([], nonblank) = -inf for t >= 0

        auto prefixScores = std::make_shared<std::vector<CTCPrefixScoringContext::PrefixScore>>();
        prefixScores->resize(ctcScores_.nColumns());

        Score cumulativeBlankScore = 0.0;
        for (size_t t = 0ul; t < ctcScores_.nColumns(); ++t) {
            cumulativeBlankScore += ctcScores_.at(blankIndex_, t);
            prefixScores->at(t).blankEndingScore = cumulativeBlankScore;
        }

        scoringContext->timePrefixScores = prefixScores;
    }
    else {
        // We are given PrefixScore_t([..., a], blank) and PrefixScore_t([..., a], nonblank) for t >= 0
        // as well as CTCScore_t(v) for t >= 1 for any blank or non-blank label v.
        // We want PrefixScore_t([..., a, b], blank) and PrefixScore_t([..., a, b], nonblank).
        // To do this we use the following recursive equations:
        // PrefixScore_0([..., a, b], blank) = PrefixScore_0([..., a, b], nonblank) = -inf
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
        // for t >= 1
        auto const& prefixScores    = scoringContext->timePrefixScores;
        auto        extPrefixScores = std::make_shared<std::vector<CTCPrefixScoringContext::PrefixScore>>();
        auto        nextToken       = scoringContext->labelSeq.back();

        extPrefixScores->resize(prefixScores->size());
        if (scoringContext->labelSeq.size() == 1) {
            extPrefixScores->at(0).nonBlankEndingScore = ctcScores_.at(nextToken, 0);
        }

        for (size_t t = 1ul; t < ctcScores_.nColumns(); ++t) {
            Score& blankEndingScore    = extPrefixScores->at(t).blankEndingScore;
            Score& nonBlankEndingScore = extPrefixScores->at(t).nonBlankEndingScore;

            blankEndingScore    = extPrefixScores->at(t - 1).totalScore() + ctcScores_.at(blankIndex_, t);
            nonBlankEndingScore = extPrefixScores->at(t - 1).nonBlankEndingScore;  // Label loop
            if (scoringContext->labelSeq.size() >= 2 and nextToken == scoringContext->labelSeq[scoringContext->labelSeq.size() - 2]) {
                // If the last label is equal to the one before it, there must be a blank in between, i.e., the prefix must not end at non-blank at t-1
                nonBlankEndingScore = Math::scoreSum(nonBlankEndingScore, prefixScores->at(t - 1).blankEndingScore);  // Blank-to-label
            }
            else {
                // If the last two labels are different, the prefix can end in both blank or non-blank
                nonBlankEndingScore = Math::scoreSum(nonBlankEndingScore, prefixScores->at(t - 1).totalScore());  // Blank-to-label or label-to-label
            }
            nonBlankEndingScore += ctcScores_.at(nextToken, t);
        }

        scoringContext->timePrefixScores = extPrefixScores;
    }

    scoringContext->requiresFinalize = false;
}

}  // namespace Nn
