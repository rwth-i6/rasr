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
          expectMoreFeatures_(true),
          ctcScores_(std::make_shared<Math::FastMatrix<Score>>(vocabSize_, 0)) {
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

ScoringContextRef CTCPrefixLabelScorer::extendedScoringContext(ScoringContextRef scoringContext, LabelIndex nextToken, TransitionType transitionType) {
    auto context = Core::ref(dynamic_cast<CTCPrefixScoringContext const*>(scoringContext.get()));

    std::vector<LabelIndex> newLabelSeq(context->labelSeq);
    newLabelSeq.push_back(nextToken);
    // Before scoring context extension, the score has already been computed in `computeScoreWithTimeInternal`
    auto extScore = context->extScores.at(nextToken);

    return Core::ref(new CTCPrefixScoringContext(std::move(newLabelSeq), context->timePrefixScores, extScore, true));
}

std::optional<ScoreAccessorRef> CTCPrefixLabelScorer::getScoreAccessor(ScoringContextRef scoringContext) {
    if (expectMoreFeatures_) {
        return {};
    }

    auto context = Core::ref(dynamic_cast<const CTCPrefixScoringContext*>(scoringContext.get()));
    finalizeScoringContext(context);

    return Core::ref(new CTCPrefixScoreAccessor(context, ctcScores_));
}

void CTCPrefixLabelScorer::setupCTCScores() {
    ctcScores_->resize(vocabSize_, 0);
    auto ctcScorerContext = ctcScorer_->getInitialScoringContext();
    while (true) {
        auto scoreAccessor = ctcScorer_->getScoreAccessor(ctcScorerContext);

        if (not scoreAccessor) {
            break;
        }

        // Add column for next timestep to matrix and insert score values into it
        ctcScores_->resizeColsAndKeepContent(ctcScores_->nColumns() + 1);
        for (LabelIndex v = 0ul; v < vocabSize_; ++v) {
            // Transition type can be anything as we assume that the score is independent of it
            ctcScores_->at(v, ctcScores_->nColumns() - 1) = (*scoreAccessor)->getScoreForLabel(v);
        }
        // Transition type and next token assumed to not influence the scoring context
        ctcScorerContext = ctcScorer_->extendedScoringContext(ctcScorerContext, invalidLabelIndex, LABEL_TO_BLANK);
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
        prefixScores->resize(ctcScores_->nColumns());

        Score cumulativeBlankScore = 0.0;
        for (size_t t = 0ul; t < ctcScores_->nColumns(); ++t) {
            cumulativeBlankScore += ctcScores_->at(blankIndex_, t);
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
            extPrefixScores->at(0).nonBlankEndingScore = ctcScores_->at(nextToken, 0);
        }

        for (size_t t = 1ul; t < ctcScores_->nColumns(); ++t) {
            Score& blankEndingScore    = extPrefixScores->at(t).blankEndingScore;
            Score& nonBlankEndingScore = extPrefixScores->at(t).nonBlankEndingScore;

            blankEndingScore    = extPrefixScores->at(t - 1).totalScore() + ctcScores_->at(blankIndex_, t);
            nonBlankEndingScore = extPrefixScores->at(t - 1).nonBlankEndingScore;  // Label loop
            if (scoringContext->labelSeq.size() >= 2 and nextToken == scoringContext->labelSeq[scoringContext->labelSeq.size() - 2]) {
                // If the last label is equal to the one before it, there must be a blank in between, i.e., the prefix must not end at non-blank at t-1
                nonBlankEndingScore = std::min(nonBlankEndingScore, prefixScores->at(t - 1).blankEndingScore);  // Blank-to-label
            }
            else {
                // If the last two labels are different, the prefix can end in both blank or non-blank
                nonBlankEndingScore = std::min(nonBlankEndingScore, prefixScores->at(t - 1).totalScore());  // Blank-to-label or label-to-label
            }
            nonBlankEndingScore += ctcScores_->at(nextToken, t);
        }

        scoringContext->timePrefixScores = extPrefixScores;
    }

    scoringContext->requiresFinalize = false;
}

}  // namespace Nn
