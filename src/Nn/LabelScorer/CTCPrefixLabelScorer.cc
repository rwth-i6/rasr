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
#include "Nn/LabelScorer/LabelScorer.hh"
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

const Core::ParameterInt CTCPrefixLabelScorer::paramBlankIndex("blank-label-index", "Index of blank symbol in vocabulary");

CTCPrefixLabelScorer::CTCPrefixLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          blankIndex_(paramBlankIndex(config)),
          ctcScorer_(Module::instance().createLabelScorer(select("ctc-scorer"))),
          expectMoreFeatures_(true) {
    log() << "Create CTCPrefixLabelScorer";
}

void CTCPrefixLabelScorer::reset() {
    ctcScorer_->reset();
    expectMoreFeatures_ = true;
}

void CTCPrefixLabelScorer::signalNoMoreFeatures() {
    ctcScorer_->signalNoMoreFeatures();
    expectMoreFeatures_ = false;
}

void CTCPrefixLabelScorer::addInput(SharedDataHolder const& input, size_t featureSize) {
    ctcScorer_->addInput(input, featureSize);
}

void CTCPrefixLabelScorer::addInputs(SharedDataHolder const& inputs, size_t timeSize, size_t featureSize) {
    ctcScorer_->addInputs(inputs, timeSize, featureSize);
}

ScoringContextRef CTCPrefixLabelScorer::getInitialScoringContext() {
    // Empty ref as sentinel value since the proper initial scoring context can only be computed
    // after the features have been passed
    return {};
}

PrefixScoringContextRef CTCPrefixLabelScorer::getProperInitialScoringContext() {
    // In the beginning the prefix is empty, so it can only be achieved by emitting pure blanks.
    // So PrefixScore_0([], blank) = 0 and PrefixScore_t([], blank) = sum_{t'=1}^t -log p_{t'}(<blank>) for t >= 1
    // PrefixScore_t([], nonblank) = -inf for t >= 0
    ctcScores_.resize(blankIndex_ + 1, 0);  // TODO: HACK! blankIndex_ is not necessarily the last index in the alphabet
    auto ctcScorerContext = ctcScorer_->getInitialScoringContext();
    while (true) {
        if (not ctcScorer_->computeScoreWithTime({ctcScorerContext, 0ul, BLANK_LOOP})) {
            break;
        }
        ctcScores_.resizeColsAndKeepContent(ctcScores_.nColumns() + 1);
        for (LabelIndex v = 0ul; v <= blankIndex_; ++v) {  // TODO: See above
            ctcScores_.at(v, ctcScores_.nColumns() - 1) = ctcScorer_->computeScoreWithTime({ctcScorerContext, v, BLANK_LOOP})->score;
        }
        ctcScorerContext = ctcScorer_->extendedScoringContext({ctcScorerContext, blankIndex_, BLANK_LOOP});
    }

    std::vector<CTCPrefixScoringContext::PrefixScore> prefixScores;
    prefixScores.reserve(ctcScores_.nColumns() + 1);

    Score cumulativeBlankScore = 0.0;
    prefixScores.push_back({cumulativeBlankScore, std::numeric_limits<Score>::infinity()});

    for (size_t t = 0ul; t < ctcScores_.nColumns(); ++t) {
        cumulativeBlankScore += ctcScores_.at(blankIndex_, t);
        prefixScores.push_back({cumulativeBlankScore, std::numeric_limits<Score>::infinity()});
    }
    prefixScores.push_back({cumulativeBlankScore, std::numeric_limits<Score>::infinity()});

    return Core::ref(new CTCPrefixScoringContext(std::move(prefixScores), Core::Type<LabelIndex>::max));  // `lastLabel` is set to invalid index
}

ScoringContextRef CTCPrefixLabelScorer::extendedScoringContext(LabelScorer::Request const& request) {
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
    PrefixScoringContextRef context;
    if (request.context) {
        context = Core::ref(dynamic_cast<const CTCPrefixScoringContext*>(request.context.get()));
    }
    else {
        context = getProperInitialScoringContext();
    }

    if (request.transitionType == BLANK_LOOP or request.transitionType == LABEL_TO_BLANK) {
        return context;
    }

    const auto&                                       prefixScores = context->prefixScores;
    std::vector<CTCPrefixScoringContext::PrefixScore> extPrefixScores;
    extPrefixScores.reserve(prefixScores.size());
    extPrefixScores.push_back({std::numeric_limits<Score>::infinity(), std::numeric_limits<Score>::infinity()});

    for (size_t t = 0ul; t < ctcScores_.nColumns(); ++t) {
        auto nonBlankScore = ctcScores_.at(request.nextToken, t);
        auto blankScore    = ctcScores_.at(blankIndex_, t);

        Score blankEndingScore = Math::scoreSum(
                extPrefixScores[t].blankEndingScore + blankScore,      // Blank-loop
                extPrefixScores[t].nonBlankEndingScore + blankScore);  // Label-to-blank
        Score nonBlankEndingScore = Math::scoreSum(
                prefixScores[t].blankEndingScore + nonBlankScore,         // Blank-to-label
                extPrefixScores[t].nonBlankEndingScore + nonBlankScore);  // Continue label-loop
        if (request.nextToken != context->lastLabel) {
            nonBlankEndingScore = Math::scoreSum(
                    nonBlankEndingScore,
                    prefixScores[t].nonBlankEndingScore + nonBlankScore);  // Label-to-label
        }
        extPrefixScores.push_back({blankEndingScore, nonBlankEndingScore});
    }

    return Core::ref(new CTCPrefixScoringContext(std::move(extPrefixScores), request.nextToken));
}

std::optional<LabelScorer::ScoreWithTime> CTCPrefixLabelScorer::computeScoreWithTime(LabelScorer::Request const& request) {
    if (expectMoreFeatures_) {
        return {};
    }

    if (request.transitionType == BLANK_LOOP or request.transitionType == LABEL_TO_BLANK) {
        return ScoreWithTime{0.0, 0};
    }

    PrefixScoringContextRef context;
    if (request.context) {
        context = Core::ref(dynamic_cast<const CTCPrefixScoringContext*>(request.context.get()));
    }
    else {
        context = getProperInitialScoringContext();
    }

    Score totalScore = std::numeric_limits<Score>::infinity();

    for (size_t t = 0ul; t < ctcScores_.nColumns(); ++t) {
        auto ctcScore = ctcScores_.at(request.nextToken, t);

        // Different to previous label: Prefix can end both in blank or in previous label
        totalScore = Math::scoreSum(totalScore, context->prefixScores[t].blankEndingScore + ctcScore);

        if (request.nextToken != context->lastLabel) {
            // Prefix can end in non-blank only if the non-blank label is different to the new one, otherwise it's considered a loop
            totalScore = Math::scoreSum(totalScore, context->prefixScores[t].nonBlankEndingScore + ctcScore);
        }
    }

    return ScoreWithTime{totalScore, static_cast<Speech::TimeframeIndex>(0ul)};
}

}  // namespace Nn
