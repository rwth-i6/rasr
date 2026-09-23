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

#include "LexiconfreeRNNTTimesyncBeamSearch.hh"

#include <algorithm>
#include <strings.h>

#include <Core/CollapsedVector.hh>
#include <Core/XmlStream.hh>
#include <Lattice/LatticeAdaptor.hh>
#include <Math/Utilities.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/Traceback.hh>
#include <Search/TracebackHelper.hh>

namespace Search {

namespace {

enum RecombinationMode {
    RecombinationModeOff,
    RecombinationModeOn,
};

}  // namespace

/*
 * =======================
 * === LabelHypothesis ===
 * =======================
 */

LexiconfreeRNNTTimesyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContexts(),
          currentToken(Nn::invalidLabelIndex),
          length(1),
          score(0.0),
          scaledScore(0.0),
          outputTokens(),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))),
          reachedSentenceEnd(false) {}

LexiconfreeRNNTTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        LexiconfreeRNNTTimesyncBeamSearch::LabelHypothesis const&    base,
        LexiconfreeRNNTTimesyncBeamSearch::ExtensionCandidate const& extension,
        std::vector<Nn::ScoringContextRef> const&                    newScoringContexts,
        float                                                        lengthNormScale)
        : scoringContexts(newScoringContexts),
          currentToken(extension.nextToken),
          length(base.length),
          score(extension.score),
          scaledScore(score / std::pow(length, lengthNormScale)),
          outputTokens(base.outputTokens),
          trace(),
          reachedSentenceEnd(base.reachedSentenceEnd or extension.transitionType == Nn::SENTENCE_END) {
    // In an inner hyp (a non-blank label was predicted):
    // increment length, update the scaled score and
    // append new label to the vector of predicted labels
    switch (extension.transitionType) {
        case Nn::INITIAL_LABEL:
        case Nn::LABEL_TO_LABEL:
        case Nn::BLANK_TO_LABEL:
            length += 1;
            scaledScore = score / std::pow(length, lengthNormScale);
            outputTokens.push_back(currentToken);
            break;
        default:
            break;
    }

    Core::Ref<LatticeTrace> predecessor;
    switch (extension.transitionType) {
        case Nn::TransitionType::LABEL_LOOP:
        case Nn::TransitionType::BLANK_LOOP:
            predecessor = base.trace->predecessor;
            break;
        default:
            predecessor = base.trace;
            break;
    }
    trace = Core::ref(new LatticeTrace(
            predecessor,
            extension.pron,
            extension.timeframe,
            {score, 0},
            {}));
}

std::string LexiconfreeRNNTTimesyncBeamSearch::LabelHypothesis::toString() const {
    std::stringstream ss;
    ss << "Score: " << score << ", traceback: ";

    auto traceback = trace->performTraceback();

    for (auto& item : *traceback) {
        if (item.pronunciation and item.pronunciation->lemma()) {
            ss << item.pronunciation->lemma()->symbol() << " ";
        }
    }
    return ss.str();
}

/*
 * =====================================
 * === LexiconfreeRNNTTimesyncBeamSearch ===
 * =====================================
 */

const Core::ParameterIntVector LexiconfreeRNNTTimesyncBeamSearch::paramMaxBeamSizes(
        "max-beam-size",
        "Maximum number of elements in the search beam. Pruning is applied after each intermediate label scorer.",
        "",
        1);

const Core::ParameterFloatVector LexiconfreeRNNTTimesyncBeamSearch::paramScoreThresholds(
        "score-threshold",
        "Prune any hypotheses with a score that is at least this much worse than the best hypothesis. Pruning is applied after each intermediate label scorer.",
        "",
        0,
        Core::Type<Score>::max);

const Core::ParameterFloat LexiconfreeRNNTTimesyncBeamSearch::paramLengthNormScale(
        "length-norm-scale",
        "Exponent of length for the hypothesis length normalization. Scaled scores are computed as score / length^length_norm_scale.",
        0.0);

const Core::ParameterInt LexiconfreeRNNTTimesyncBeamSearch::paramMaxLabelsPerFrame(
        "max-labels-per-timeframe",
        "Maximum number of non-blank label predictions per hypothesis in one timestep.",
        10, 0);

const Core::ParameterInt LexiconfreeRNNTTimesyncBeamSearch::paramBlankLabelIndex(
        "blank-label-index",
        "Index of the blank label in the lexicon. Can also be inferred from lexicon if it has a lemma with `special='blank'`. If not set, the search will not use blank.",
        Nn::invalidLabelIndex);

const Core::ParameterInt LexiconfreeRNNTTimesyncBeamSearch::paramSentenceEndLabelIndex(
        "sentence-end-label-index",
        "Index of the sentence end label in the lexicon. Can also be inferred from lexicon if it has a lemma with `special='sentence-end'` or `special='sentence-boundary'`. If not set, the search will not use sentence end.",
        Nn::invalidLabelIndex);

const Core::ParameterBool LexiconfreeRNNTTimesyncBeamSearch::paramSentenceEndFallBack(
        "sentence-end-fall-back",
        "Allow for fallback solution if no active word-end hypothesis exists at the end of a segment.",
        true);

const Core::ParameterBool LexiconfreeRNNTTimesyncBeamSearch::paramCollapseRepeatedLabels(
        "collapse-repeated-labels",
        "Collapse repeated emission of the same label into one output. If false, every emission is treated like a new output.",
        false);

const Core::ParameterBool LexiconfreeRNNTTimesyncBeamSearch::paramLogStepwiseStatistics(
        "log-stepwise-statistics",
        "Log statistics about the beam at every search step.",
        false);

const Core::ParameterInt LexiconfreeRNNTTimesyncBeamSearch::paramCacheCleanupInterval(
        "cache-cleanup-interval",
        "Interval of search steps after which buffered inputs that are not needed anymore get cleaned up.",
        10,
        1);

const Core::ParameterInt LexiconfreeRNNTTimesyncBeamSearch::paramMaximumStableDelay(
        "maximum-stable-delay",
        "Introduce a cutoff point at `current-time` - `delay`. Every hypothesis that disagrees with the current best anywhere before the cutoff gets pruned."
        "This way words in the traceback become stable after at most `delay` frames.",
        Core::Type<int>::max,
        0);

const Core::ParameterInt LexiconfreeRNNTTimesyncBeamSearch::paramMaximumStableDelayPruningInterval(
        "maximum-stable-delay-pruning-interval",
        "Interval of search steps after which the maximum-stable-delay-pruning gets applied.",
        10,
        1);

const Core::Choice LexiconfreeRNNTTimesyncBeamSearch::choiceRecombinationMode(
        "off", RecombinationModeOff,
        "on", RecombinationModeOn,
        Core::Choice::endMark());

const Core::ParameterChoice LexiconfreeRNNTTimesyncBeamSearch::paramRecombinationMode(
        "recombination-mode",
        &choiceRecombinationMode,
        "Whether hypotheses with identical recombination state should be recombined.",
        RecombinationModeOn);

LexiconfreeRNNTTimesyncBeamSearch::LexiconfreeRNNTTimesyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          lengthNormScale_(paramLengthNormScale(config)),
          maxLabelsPerFrame_(paramMaxLabelsPerFrame(config)),
          blankLabelIndex_(paramBlankLabelIndex(config)),
          sentenceEndLemma_(),
          sentenceEndLabelIndex_(paramSentenceEndLabelIndex(config)),
          sentenceEndFallback_(paramSentenceEndFallBack(config)),
          collapseRepeatedLabels_(paramCollapseRepeatedLabels(config)),
          cacheCleanupInterval_(paramCacheCleanupInterval(config)),
          maximumStableDelay_(paramMaximumStableDelay(config)),
          maximumStableDelayPruningInterval_(paramMaximumStableDelayPruningInterval(config)),
          recombinationEnabled_(paramRecombinationMode(config) == RecombinationModeOn),
          logStepwiseStatistics_(paramLogStepwiseStatistics(config)),
          debugChannel_(config, "debug"),
          labelScorers_(),
          beam_(),
          innerHyps_(),
          outerHyps_(),
          hypIndexToContextIndexMap_(),
          extensions_(),
          newBeam_(),
          scoringContexts_(),
          tempHypotheses_(),
          initializationTime_(),
          featureProcessingTime_(),
          scoringTime_(),
          contextExtensionTime_(),
          numActiveHyps_("num-active-hyps"),
          numOuterHyps_("num-outer-hyps"),
          numInnerHyps_("num-inner-hyps"),
          numInnerAndOuterHyps_("num-inner-and-outer-hyps"),
          currentSearchStep_(0ul),
          finishedSegment_(false) {
    auto maxBeamSizes = paramMaxBeamSizes(config);
    maxBeamSizes_.insert(maxBeamSizes_.begin(), maxBeamSizes.begin(), maxBeamSizes.end());

    auto scoreThresholds = paramScoreThresholds(config);
    scoreThresholds_.insert(scoreThresholds_.begin(), scoreThresholds.begin(), scoreThresholds.end());
    // Fill up with default value
    for (size_t i = scoreThresholds_.size(); i < maxBeamSizes_.size(); ++i) {
        scoreThresholds_.push_back(Core::Type<Score>::max);
    }

    for (size_t i = 0ul; i < scoreThresholds_.size(); ++i) {
        useScorePruning_.push_back(scoreThresholds_[i] != Core::Type<Score>::max);
    }

    if (blankLabelIndex_ != Nn::invalidLabelIndex) {
        log() << "Use blank label with index " << blankLabelIndex_;
    }

    useSentenceEnd_ = sentenceEndLabelIndex_ != Nn::invalidLabelIndex;
    if (useSentenceEnd_) {
        log() << "Use sentence end label with index " << sentenceEndLabelIndex_;
    }
}

Speech::ModelCombination::Mode LexiconfreeRNNTTimesyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon;
}

bool LexiconfreeRNNTTimesyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_      = modelCombination.lexicon();
    labelScorers_ = modelCombination.labelScorers();

    if (labelScorers_.size() > maxBeamSizes_.size()) {
        error() << "Number of label scorers (" << labelScorers_.size() << ") exceeds number of configured max beam sizes (" << maxBeamSizes_.size() << ")";
    }
    if (labelScorers_.size() < maxBeamSizes_.size()) {
        warning() << "Number of label scorers (" << labelScorers_.size() << ") is less than number of configured max beam sizes (" << maxBeamSizes_.size() << ")";
    }

    auto blankLemma = lexicon_->specialLemma("blank");
    if (blankLemma) {
        if (blankLabelIndex_ == Nn::invalidLabelIndex) {
            blankLabelIndex_ = blankLemma->id();
            log() << "Use blank index " << blankLabelIndex_ << " inferred from lexicon";
        }
        else if (blankLabelIndex_ != static_cast<Nn::LabelIndex>(blankLemma->id())) {
            warning() << "Blank lemma exists in lexicon with id " << blankLemma->id() << " but is overwritten by config parameter with value " << blankLabelIndex_;
        }
    }
    if (blankLabelIndex_ == Nn::invalidLabelIndex) {
        error() << "Blank label index is not defined and cannot be inferred from the lexicon";
    }

    sentenceEndLemma_ = lexicon_->specialLemma("sentence-end");
    if (!sentenceEndLemma_) {
        sentenceEndLemma_ = lexicon_->specialLemma("sentence-boundary");
    }
    if (sentenceEndLemma_) {
        if (sentenceEndLabelIndex_ == Nn::invalidLabelIndex) {
            sentenceEndLabelIndex_ = sentenceEndLemma_->id();
            useSentenceEnd_        = true;
            log() << "Use sentence-end index " << sentenceEndLabelIndex_ << " inferred from lexicon";
        }
        else if (sentenceEndLabelIndex_ != static_cast<Nn::LabelIndex>(sentenceEndLemma_->id())) {
            warning() << "SentenceEnd lemma exists in lexicon with id " << sentenceEndLemma_->id() << " but is overwritten by config parameter with value " << sentenceEndLabelIndex_;
        }
    }

    return true;
}

void LexiconfreeRNNTTimesyncBeamSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    resetStatistics();

    initializationTime_.start();

    for (auto& labelScorer : labelScorers_) {
        labelScorer->reset();
    }

    // Reset beam to a single empty hypothesis
    beam_.clear();
    beam_.push_back(LabelHypothesis());
    beam_.front().scoringContexts.clear();
    for (auto& labelScorer : labelScorers_) {
        beam_.front().scoringContexts.push_back(labelScorer->getInitialScoringContext());
    }

    currentSearchStep_ = 0ul;
    finishedSegment_   = false;

    initializationTime_.stop();
}

void LexiconfreeRNNTTimesyncBeamSearch::finishSegment() {
    featureProcessingTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->signalNoMoreFeatures();
    }
    featureProcessingTime_.stop();
    decodeManySteps();
    finalizeHypotheses();
    logStatistics();
    finishedSegment_ = true;
}

void LexiconfreeRNNTTimesyncBeamSearch::putFeature(Nn::DataView const& feature) {
    featureProcessingTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInput(feature);
    }
    featureProcessingTime_.stop();
}

void LexiconfreeRNNTTimesyncBeamSearch::putFeatures(Nn::DataView const& features, size_t nTimesteps) {
    featureProcessingTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInputs(features, nTimesteps);
    }
    featureProcessingTime_.stop();
}

Core::Ref<const Traceback> LexiconfreeRNNTTimesyncBeamSearch::getCurrentBestTraceback() const {
    return getBestHypothesis().trace->performTraceback();
}

Core::Ref<const LatticeAdaptor> LexiconfreeRNNTTimesyncBeamSearch::getCurrentBestWordLattice() const {
    auto&        bestHypothesis = getBestHypothesis();
    LatticeTrace endTrace(bestHypothesis.trace, 0, bestHypothesis.trace->time + 1, bestHypothesis.trace->score, {});

    for (auto const& hyp : beam_) {
        // The best hypothesis is already represented in endTrace
        if (&hyp == &bestHypothesis) {
            continue;
        }
        auto siblingTrace = Core::ref(new LatticeTrace(hyp.trace, 0, hyp.trace->time, hyp.trace->score, {}));
        endTrace.appendSiblingToChain(siblingTrace);
    }

    return endTrace.buildWordLattice(lexicon_);
}

Core::Ref<const LatticeTrace> LexiconfreeRNNTTimesyncBeamSearch::getCurrentBestLatticeTrace() const {
    return getBestHypothesis().trace;
}

Core::Ref<const LatticeTrace> LexiconfreeRNNTTimesyncBeamSearch::getCommonPrefix() const {
    std::vector<Core::Ref<LatticeTrace>> traces(beam_.size());
    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        traces[hypIndex] = beam_[hypIndex].trace;
    }

    RootTraceSearcher searcher(traces);
    if (not searcher.rootTrace()) {
        warning("Common prefix of all traces is a sentinel value");
    }

    return Core::Ref<const LatticeTrace>(searcher.rootTrace());
}

bool LexiconfreeRNNTTimesyncBeamSearch::decodeStep() {
    if (finishedSegment_) {
        return false;
    }

    if (logStepwiseStatistics_) {
        clog() << Core::XmlOpen("search-step-stats");
        clog() << Core::XmlFull("timestep", currentSearchStep_);
    }

    // Assume the output labels are stored as lexicon lemma orth and ordered consistently with NN output index
    auto lemmas = lexicon_->lemmas();

    // Start timestep with beam of previous timestep
    innerHyps_ = beam_;

    size_t symbolStep = 0;

    // Start inner loop of this timestep
    while (true) {
        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("symbolstep", symbolStep);
        }

        // Early stopping if no inner hyps left
        // This happens if all inner hyps were worse than the worst outer hyp
        if (innerHyps_.empty()) {
            break;
        }

        /*
         * Get score accessors of the first label scorer for all inner hyps. These get reused below for
         * both the blank and the non-blank extension candidates, since both start from the same inner
         * hyps and the same first label scorer
         */
        scoringContexts_.clear();
        for (auto const& hyp : innerHyps_) {
            scoringContexts_.push_back(hyp.scoringContexts.front());
        }

        scoringTime_.start();
        auto scoreAccessors = labelScorers_.front()->getScoreAccessors(scoringContexts_);
        scoringTime_.stop();

        // Check if any scoring context could be scored
        bool anyScored = std::any_of(scoreAccessors.begin(), scoreAccessors.end(),
                                     [](auto const& a) { return a.has_value(); });
        if (not anyScored) {
            return false;
        }

        /*
         * Extend inner hyps with the blank label, so they become outer hyps
         */
        extensions_.clear();
        for (size_t hypIndex = 0ul; hypIndex < innerHyps_.size(); ++hypIndex) {
            auto& hyp = innerHyps_[hypIndex];

            auto const& scoreAccessor = scoreAccessors[hypIndex];
            if (not scoreAccessor) {
                continue;
            }

            auto  transitionType = inferTransitionType(hyp.currentToken, blankLabelIndex_);
            Score extScore       = hyp.score;
            if (labelScorers_.front()->scoresTransition(transitionType)) {
                extScore += (*scoreAccessor)->getScore(transitionType, blankLabelIndex_);
            }

            extensions_.push_back(
                    {blankLabelIndex_,
                     nullptr,
                     extScore,
                     (*scoreAccessor)->getTime(),
                     transitionType,
                     hypIndex});
        }

        // Unlike the non-blank extensions below, the blank extensions are not pruned after the first label scorer
        // because there is only one blank candidate per inner hyp, so an intermediate cut isn't needed for performance here
        scoreWithRemainingLabelScorers(extensions_, innerHyps_);

        // Create new label hypotheses from extension candidates
        newBeam_.clear();
        for (auto const& extension : extensions_) {
            auto const& baseHyp = innerHyps_[extension.baseHypIndex];
            newBeam_.push_back({baseHyp, extension, extendedScoringContexts(baseHyp, extension), lengthNormScale_});
        }

        // Add these new outer hyps to the set of all outer hyps of this timestep
        outerHyps_.insert(outerHyps_.end(), newBeam_.begin(), newBeam_.end());

        recombination(outerHyps_);

        numOuterHyps_ += outerHyps_.size();
        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("outer-hyps", outerHyps_.size());
        }

        // Finish this step if the maximum number of output symbols has been reached
        if (symbolStep >= maxLabelsPerFrame_) {
            break;
        }

        /*
         * Extend inner hyps with non-blank labels
         */
        extensions_.clear();
        for (size_t hypIndex = 0ul; hypIndex < innerHyps_.size(); ++hypIndex) {
            auto& hyp = innerHyps_[hypIndex];

            auto const& scoreAccessor = scoreAccessors[hypIndex];
            if (not scoreAccessor) {
                continue;
            }

            auto denseScores = (*scoreAccessor)->getDenseScores();
            auto scoreTime   = (*scoreAccessor)->getTime();

            // Iterate over possible successors (all lemmas)
            for (auto lemmaIt = lemmas.first; lemmaIt != lemmas.second; ++lemmaIt) {
                const Bliss::Lemma* lemma(*lemmaIt);
                Nn::LabelIndex      tokenIdx = lemma->id();

                // Blank is not allowed as an extension for the inner hyps
                if (tokenIdx == blankLabelIndex_) {
                    continue;
                }

                auto  transitionType = inferTransitionType(hyp.currentToken, tokenIdx);
                Score extScore       = hyp.score;
                if (labelScorers_.front()->scoresTransition(transitionType)) {
                    extScore += (denseScores and tokenIdx < denseScores->size())
                                        ? (*denseScores)[tokenIdx]
                                        : (*scoreAccessor)->getScore(transitionType, tokenIdx);
                }

                extensions_.push_back(
                        {tokenIdx,
                         lemma->pronunciations().first,
                         extScore,
                         scoreTime,
                         transitionType,
                         hypIndex});
            }
        }

        // Score/beam-size-prune extension candidates after the first label scorer
        {
            size_t maxBeamSize = extensions_.size();
            if (labelScorers_.size() > 1ul) {
                maxBeamSize = maxBeamSizes_.front();
            }
            scorePruning(extensions_, scoreThresholds_.front(), maxBeamSize);
        }
        scoreWithRemainingLabelScorers(extensions_, innerHyps_);

        // Create new label hypotheses from extension candidates
        newBeam_.clear();
        for (auto const& extension : extensions_) {
            auto const& baseHyp = innerHyps_[extension.baseHypIndex];
            newBeam_.push_back({baseHyp, extension, extendedScoringContexts(baseHyp, extension), lengthNormScale_});
        }

        // Prune new inner hyps down to maxBeamSize based on the raw score
        beamSizePruning(newBeam_);

        // If there are already more than maxBeamSize outer hyps,
        // remove all inner hyps with a score that is lower than the worst score of the max-beam-size best outer hyps
        Score outerHypsThreshold = std::numeric_limits<Score>::infinity();
        if (outerHyps_.size() >= maxBeamSizes_.back()) {
            auto kth = outerHyps_.begin() + (maxBeamSizes_.back() - 1);
            std::nth_element(outerHyps_.begin(), kth, outerHyps_.end(),
                             [](auto const& a, auto const& b) { return a.score < b.score; });
            outerHypsThreshold = kth->score;

            innerHyps_.clear();
            for (auto const& hyp : newBeam_) {
                if (hyp.score < outerHypsThreshold) {
                    innerHyps_.push_back(hyp);
                }
            }
        }
        else {
            innerHyps_.swap(newBeam_);
        }

        numInnerHyps_ += innerHyps_.size();
        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("inner-hyps", innerHyps_.size());
        }

        numInnerAndOuterHyps_ += (innerHyps_.size() + outerHyps_.size());
        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("inner-and-outer-hyps", innerHyps_.size() + outerHyps_.size());
        }

        ++symbolStep;

    }  // end of inner loop

    // Prune all hyps of this timestep at the end of this timestep based on the length-normalized score
    if (useScorePruning_.back()) {
        scorePruningLengthnormalized(outerHyps_);
    }
    beamSizePruningLengthnormalized(outerHyps_);

    // The leftover outer hyps of this timestep will be the inner hyps to start with in the next timestep
    beam_ = outerHyps_;
    outerHyps_.clear();

    numActiveHyps_ += beam_.size();

    /*
     * Clean up label scorer caches.
     */
    if (++currentSearchStep_ % cacheCleanupInterval_ == 0) {
        for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
            Core::CollapsedVector<Nn::ScoringContextRef> activeContexts;
            for (auto const& hyp : beam_) {
                activeContexts.push_back(hyp.scoringContexts[scorerIdx]);
            }
            labelScorers_[scorerIdx]->cleanupCaches(activeContexts);
        }
    }

    /*
     * Perform maximum-stable-delay-pruning.
     */
    if (currentSearchStep_ % maximumStableDelayPruningInterval_ == 0) {
        maximumStableDelayPruning();
        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("num-hyps-after-maximum-stable-delay-pruning", beam_.size());
        }
    }

    /*
     * Log statistics about the new beam after this step.
     */
    if (debugChannel_.isOpen()) {
        std::stringstream ss;
        for (size_t hypIdx = 0ul; hypIdx < beam_.size(); ++hypIdx) {
            ss << "Hypothesis " << hypIdx + 1ul << ":  " << beam_[hypIdx].toString() << "\n";
        }
        ss << "\n";
        debugChannel_ << ss.str();
    }

    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("active-hyps", beam_.size());
        clog() << Core::XmlFull("best-hyp-score", getBestHypothesis().score);
        clog() << Core::XmlFull("worst-hyp-score", getWorstHypothesis().score);
        clog() << Core::XmlClose("search-step-stats");
    }

    return true;
}

LexiconfreeRNNTTimesyncBeamSearch::LabelHypothesis const& LexiconfreeRNNTTimesyncBeamSearch::getBestHypothesis() const {
    verify(not beam_.empty());

    return *std::min_element(beam_.begin(), beam_.end());
}

LexiconfreeRNNTTimesyncBeamSearch::LabelHypothesis const& LexiconfreeRNNTTimesyncBeamSearch::getWorstHypothesis() const {
    verify(not beam_.empty());

    return *std::max_element(beam_.begin(), beam_.end());
}

void LexiconfreeRNNTTimesyncBeamSearch::resetStatistics() {
    initializationTime_.reset();
    featureProcessingTime_.reset();
    scoringTime_.reset();
    contextExtensionTime_.reset();
    numActiveHyps_.clear();
    numOuterHyps_.clear();
    numInnerHyps_.clear();
    numInnerAndOuterHyps_.clear();
}

void LexiconfreeRNNTTimesyncBeamSearch::logStatistics() const {
    clog() << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    clog() << Core::XmlOpen("initialization-time") << initializationTime_.elapsedMilliseconds() << Core::XmlClose("initialization-time");
    clog() << Core::XmlOpen("feature-processing-time") << featureProcessingTime_.elapsedMilliseconds() << Core::XmlClose("feature-processing-time");
    clog() << Core::XmlOpen("scoring-time") << scoringTime_.elapsedMilliseconds() << Core::XmlClose("scoring-time");
    clog() << Core::XmlOpen("context-extension-time") << contextExtensionTime_.elapsedMilliseconds() << Core::XmlClose("context-extension-time");
    clog() << Core::XmlClose("timing-statistics");
    numActiveHyps_.write(clog());
    numOuterHyps_.write(clog());
    numInnerHyps_.write(clog());
    numInnerAndOuterHyps_.write(clog());
}

Nn::TransitionType LexiconfreeRNNTTimesyncBeamSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const {
    bool prevIsBlank       = prevLabel == blankLabelIndex_;
    bool nextIsBlank       = nextLabel == blankLabelIndex_;
    bool nextIsSentenceEnd = (useSentenceEnd_ and nextLabel == sentenceEndLabelIndex_);

    if (prevLabel == Nn::invalidLabelIndex) {
        if (nextIsBlank) {
            return Nn::TransitionType::INITIAL_BLANK;
        }
        else if (nextIsSentenceEnd) {
            return Nn::TransitionType::SENTENCE_END;
        }
        else {
            return Nn::TransitionType::INITIAL_LABEL;
        }
    }

    if (prevIsBlank) {
        if (nextIsBlank) {
            return Nn::TransitionType::BLANK_LOOP;
        }
        else if (nextIsSentenceEnd) {
            return Nn::TransitionType::SENTENCE_END;
        }
        else {
            return Nn::TransitionType::BLANK_TO_LABEL;
        }
    }
    else {
        if (nextIsBlank) {
            return Nn::TransitionType::LABEL_TO_BLANK;
        }
        else if (collapseRepeatedLabels_ and prevLabel == nextLabel) {
            return Nn::TransitionType::LABEL_LOOP;
        }
        else if (nextIsSentenceEnd) {
            return Nn::TransitionType::SENTENCE_END;
        }
        else {
            return Nn::TransitionType::LABEL_TO_LABEL;
        }
    }
}

void LexiconfreeRNNTTimesyncBeamSearch::scoreWithRemainingLabelScorers(
        std::vector<ExtensionCandidate>&    extensions,
        std::vector<LabelHypothesis> const& baseHyps) {
    for (size_t scorerIdx = 1ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
        auto const& labelScorer = labelScorers_[scorerIdx];

        // Collect the scoring contexts of this scorer needed for the surviving extensions
        scoringContexts_.clear();
        hypIndexToContextIndexMap_.assign(baseHyps.size(), -1);
        for (auto const& ext : extensions) {
            if (hypIndexToContextIndexMap_[ext.baseHypIndex] == -1) {
                hypIndexToContextIndexMap_[ext.baseHypIndex] = scoringContexts_.size();
                scoringContexts_.push_back(baseHyps[ext.baseHypIndex].scoringContexts[scorerIdx]);
            }
        }

        scoringTime_.start();
        auto scoreAccessors = labelScorer->getScoreAccessors(scoringContexts_);
        scoringTime_.stop();

        std::vector<std::optional<Nn::DenseScoreSpan>> denseScoreSpans(scoreAccessors.size(), std::nullopt);
        std::vector<Nn::TimeframeIndex>                scoreTimes(scoreAccessors.size(), 0);
        for (size_t accessorIdx = 0ul; accessorIdx < scoreAccessors.size(); ++accessorIdx) {
            if (scoreAccessors[accessorIdx]) {
                denseScoreSpans[accessorIdx] = (*scoreAccessors[accessorIdx])->getDenseScores();
                scoreTimes[accessorIdx]      = (*scoreAccessors[accessorIdx])->getTime();
            }
        }

        for (auto& ext : extensions) {
            if (not labelScorer->scoresTransition(ext.transitionType)) {
                continue;
            }

            auto        contextIdx    = hypIndexToContextIndexMap_[ext.baseHypIndex];
            auto const& scoreAccessor = scoreAccessors[contextIdx];
            if (not scoreAccessor) {
                // Extension is not scorable so set the score to max in order to prune it later
                ext.score = Core::Type<Score>::max;
                continue;
            }

            auto const& denseScores = denseScoreSpans[contextIdx];
            ext.score += (denseScores and ext.nextToken < denseScores->size())
                                 ? (*denseScores)[ext.nextToken]
                                 : (*scoreAccessor)->getScore(ext.transitionType, ext.nextToken);
            ext.timeframe = std::max(ext.timeframe, scoreTimes[contextIdx]);
        }

        size_t maxBeamSize = extensions.size();
        if (scorerIdx < labelScorers_.size() - 1) {
            maxBeamSize = maxBeamSizes_[scorerIdx];
        }
        scorePruning(extensions, scoreThresholds_[scorerIdx], maxBeamSize);
    }
}

std::vector<Nn::ScoringContextRef> LexiconfreeRNNTTimesyncBeamSearch::extendedScoringContexts(
        LabelHypothesis const&    baseHyp,
        ExtensionCandidate const& extension) {
    contextExtensionTime_.start();
    std::vector<Nn::ScoringContextRef> newScoringContexts;
    newScoringContexts.reserve(labelScorers_.size());
    for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
        newScoringContexts.push_back(labelScorers_[scorerIdx]->extendedScoringContext(
                baseHyp.scoringContexts[scorerIdx],
                extension.nextToken,
                extension.transitionType));
    }
    contextExtensionTime_.stop();
    return newScoringContexts;
}

void LexiconfreeRNNTTimesyncBeamSearch::beamSizePruning(std::vector<LabelHypothesis>& hypotheses) const {
    if (hypotheses.size() <= maxBeamSizes_.back()) {
        return;
    }

    // Reorder the hypotheses by associated score value such that the first `beamSize_` elements are the best
    std::nth_element(hypotheses.begin(), hypotheses.begin() + maxBeamSizes_.back(), hypotheses.end(),
                     [](auto const& a, auto const& b) { return a.score < b.score; });
    hypotheses.resize(maxBeamSizes_.back());  // Get rid of excessive elements
}

void LexiconfreeRNNTTimesyncBeamSearch::beamSizePruningLengthnormalized(std::vector<LabelHypothesis>& hypotheses) const {
    if (hypotheses.size() <= maxBeamSizes_.back()) {
        return;
    }

    // Reorder the hypotheses by associated scaledScore value such that the first `beamSize_` elements are the best
    std::nth_element(hypotheses.begin(), hypotheses.begin() + maxBeamSizes_.back(), hypotheses.end());
    hypotheses.resize(maxBeamSizes_.back());  // Get rid of excessive elements
}

void LexiconfreeRNNTTimesyncBeamSearch::scorePruning(std::vector<ExtensionCandidate>& extensions, Score relativeThreshold, size_t maxBeamSize) const {
    // Remove extensions that could not be scored by some label scorer
    extensions.erase(
            std::remove_if(
                    extensions.begin(),
                    extensions.end(),
                    [](auto const& ext) { return Math::isinf(ext.score) or ext.score >= Core::Type<Score>::max; }),
            extensions.end());

    if (extensions.empty()) {
        return;
    }

    // Prune by relative score threshold
    if (relativeThreshold != Core::Type<Score>::max) {
        auto bestScore = std::min_element(
                                 extensions.begin(),
                                 extensions.end(),
                                 [](auto const& a, auto const& b) { return a.score < b.score; })
                                 ->score;
        auto pruningThreshold = bestScore + relativeThreshold;

        extensions.erase(
                std::remove_if(
                        extensions.begin(),
                        extensions.end(),
                        [=](auto const& ext) { return ext.score > pruningThreshold; }),
                extensions.end());
    }

    // Prune by max beam size
    if (extensions.size() > maxBeamSize) {
        std::nth_element(extensions.begin(), extensions.begin() + maxBeamSize, extensions.end(),
                         [](auto const& a, auto const& b) { return a.score < b.score; });
        extensions.resize(maxBeamSize);
    }
}

void LexiconfreeRNNTTimesyncBeamSearch::scorePruningLengthnormalized(std::vector<LabelHypothesis>& hypotheses) const {
    if (hypotheses.empty()) {
        return;
    }

    // Compute the pruning threshold
    auto bestHyp = *std::min_element(
            hypotheses.begin(),
            hypotheses.end());
    auto pruningThreshold = (bestHyp.score + scoreThresholds_.back()) / std::pow(bestHyp.length, lengthNormScale_);

    // Remove elements with scaledScore > pruningThreshold
    hypotheses.erase(
            std::remove_if(
                    hypotheses.begin(),
                    hypotheses.end(),
                    [=](auto const& hyp) { return hyp.scaledScore > pruningThreshold; }),
            hypotheses.end());
}

void LexiconfreeRNNTTimesyncBeamSearch::recombination(std::vector<LexiconfreeRNNTTimesyncBeamSearch::LabelHypothesis>& hypotheses) {
    if (not recombinationEnabled_) {
        return;
    }

    // Represents a unique combination of currentToken, scoringContexts and the previous (non-blank) output tokens
    struct RecombinationContext {
        Nn::LabelIndex                     currentToken;
        std::vector<Nn::ScoringContextRef> scoringContexts;
        std::vector<int>                   outputTokens;

        RecombinationContext(LabelHypothesis const& hyp)
                : currentToken(hyp.currentToken), scoringContexts(hyp.scoringContexts), outputTokens(hyp.outputTokens) {}

        bool operator==(RecombinationContext const& other) const {
            if (currentToken != other.currentToken or outputTokens != other.outputTokens) {
                return false;
            }
            if (scoringContexts.size() != other.scoringContexts.size()) {
                return false;
            }
            for (size_t i = 0ul; i < scoringContexts.size(); ++i) {
                if (not Nn::ScoringContextEq{}(scoringContexts[i], other.scoringContexts[i])) {
                    return false;
                }
            }
            return true;
        }
    };
    struct RecombinationContextHash {
        size_t operator()(RecombinationContext const& context) const {
            size_t h1 = context.currentToken;
            size_t h2 = 0;
            for (auto const& scoringContext : context.scoringContexts) {
                h2 = Core::combineHashes(h2, Nn::ScoringContextHash{}(scoringContext));
            }
            size_t h3 = 0;
            for (size_t i = 0; i < context.outputTokens.size(); ++i) {
                h3 = Core::combineHashes(h3, std::hash<uint32_t>()(context.outputTokens[i]));
            }
            return Core::combineHashes(Core::combineHashes(h1, h2), h3);
        }
    };

    tempHypotheses_.clear();
    // Reserve capacity because future reallocations would break the raw pointer we are storing later
    tempHypotheses_.reserve(hypotheses.size());
    // Map each unique ScoringContext in newHypotheses to its hypothesis
    std::unordered_map<RecombinationContext, LabelHypothesis*, RecombinationContextHash> seenScoringContexts;

    for (auto& hyp : hypotheses) {
        auto [it, inserted] = seenScoringContexts.try_emplace({hyp}, nullptr);

        if (inserted) {
            // First time seeing this context -> keep this hyp as representative
            tempHypotheses_.push_back(std::move(hyp));
            it->second = &tempHypotheses_.back();
        }
        else {
            verify(not hyp.trace->sibling);

            auto* existingHyp = it->second;

            // Merge scores in probability space (log-sum-exp of the two path scores)
            // numerically stable form: min(a, b) - log1p(exp(-|a - b|)), which is <= min(a, b)
            Score mergedScore = std::min(existingHyp->score, hyp.score) - std::log1p(std::exp(-std::fabs(existingHyp->score - hyp.score)));

            if (hyp.score < existingHyp->score) {
                // New hyp is better -> keep it as representative and add existing one as sibling
                hyp.trace->sibling = existingHyp->trace;
                *existingHyp       = std::move(hyp);  // Overwrite in-place
            }
            else {
                // New hyp is worse -> add it as sibling to the existing representative
                hyp.trace->sibling          = existingHyp->trace->sibling;
                existingHyp->trace->sibling = hyp.trace;
            }

            // Recompute scaled score from the merged score
            existingHyp->score       = mergedScore;
            const auto len           = std::max<std::size_t>(1, existingHyp->length);
            existingHyp->scaledScore = existingHyp->score / std::pow(static_cast<double>(len), static_cast<double>(lengthNormScale_));
        }
    }

    hypotheses.swap(tempHypotheses_);
}

void LexiconfreeRNNTTimesyncBeamSearch::maximumStableDelayPruning() {
    if (currentSearchStep_ + 1 <= maximumStableDelay_) {
        return;
    }

    auto cutoff = currentSearchStep_ + 1 - maximumStableDelay_;

    // Find trace of current best hypothesis that has a recent word-end within the limit
    Score                   bestScore = Core::Type<Score>::max;
    Core::Ref<LatticeTrace> root;

    for (auto const& hyp : beam_) {
        if (hyp.score < bestScore and hyp.trace->time >= cutoff) {
            bestScore = hyp.score;
            root      = hyp.trace;
        }
    }

    // No Hypothesis with a recent word-end was found so just take the overall best as fallback
    if (not root) {
        root = getBestHypothesis().trace;
        warning() << "Most recent label in best hypothesis is before cutoff point for maximum-stable-delay-pruning so the limit will be surpassed";
    }

    // Determine the right predecessor of best trace for pruning. `root->time` should be after the cutoff and `root->predecessor->time` before the cutoff
    Core::Ref<LatticeTrace> preRoot = root->predecessor;

    while (preRoot and preRoot->time >= cutoff) {
        root    = preRoot;
        preRoot = preRoot->predecessor;
    }

    // Perform pruning on root
    tempHypotheses_.clear();
    for (auto const& hyp : beam_) {
        auto curr = hyp.trace;
        while (curr and curr != root and curr->time > root->time) {
            curr = curr->predecessor;
        }
        if (curr == root) {
            tempHypotheses_.push_back(hyp);
        }
    }
    beam_.swap(tempHypotheses_);
}

void LexiconfreeRNNTTimesyncBeamSearch::finalizeHypotheses() {
    if (not useSentenceEnd_) {
        return;
    }

    newBeam_.clear();
    for (auto const& hyp : beam_) {
        if (hyp.reachedSentenceEnd) {
            newBeam_.push_back(hyp);
        }
    }

    if (newBeam_.empty()) {  // There was no valid final hypothesis in the beam
        warning("No hypothesis has produced sentence-end by the end of the segment.");
        if (sentenceEndFallback_) {
            log() << "Use sentence-end fallback";
            // Keep `beam_` as it is
        }
        else {
            newBeam_.push_back(LabelHypothesis());
            newBeam_.front().trace->time          = beam_.front().trace->time;  // Retrieve the timeframe from any hyp in the old beam
            newBeam_.front().trace->pronunciation = nullptr;
            newBeam_.front().trace->predecessor   = Core::ref(new LatticeTrace(0, {0, 0}, {}));
            newBeam_.front().reachedSentenceEnd   = true;
            beam_.swap(newBeam_);
        }
    }
    else {
        newBeam_.swap(beam_);
    }
}

}  // namespace Search
