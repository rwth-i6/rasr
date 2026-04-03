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
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/Traceback.hh>
#include <Search/TracebackHelper.hh>

namespace Search {

/*
 * =======================
 * === LabelHypothesis ===
 * =======================
 */

LexiconfreeRNNTTimesyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContext(),
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
        Nn::ScoringContextRef const&                                 newScoringContext,
        float                                                        lengthNormScale)
        : scoringContext(newScoringContext),
          currentToken(extension.nextToken),
          length(base.length),
          score(extension.score),
          scaledScore(score / std::pow(length, lengthNormScale)),
          outputTokens(base.outputTokens),
          trace(),
          reachedSentenceEnd(base.reachedSentenceEnd or extension.transitionType == Nn::LabelScorer::SENTENCE_END) {
    // In an inner hyp (a non-blank label was predicted):
    // increment length, update the scaled score and
    // append new label to the vector of predicted labels
    switch (extension.transitionType) {
        case Nn::LabelScorer::INITIAL_LABEL:
        case Nn::LabelScorer::LABEL_TO_LABEL:
        case Nn::LabelScorer::BLANK_TO_LABEL:
            length += 1;
            scaledScore = score / std::pow(length, lengthNormScale);
            outputTokens.push_back(currentToken);
            break;
        default:
            break;
    }

    Core::Ref<LatticeTrace> predecessor;
    switch (extension.transitionType) {
        case Nn::LabelScorer::TransitionType::LABEL_LOOP:
        case Nn::LabelScorer::TransitionType::BLANK_LOOP:
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

const Core::ParameterInt LexiconfreeRNNTTimesyncBeamSearch::paramMaxBeamSize(
        "max-beam-size",
        "Maximum number of elements in the search beam.",
        1, 1);

const Core::ParameterFloat LexiconfreeRNNTTimesyncBeamSearch::paramScoreThreshold(
        "score-threshold",
        "Prune any hypotheses with a score that is at least this much worse than the best hypothesis. If not set, no score pruning will be done.",
        Core::Type<Score>::max, 0);

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

const Core::ParameterBool LexiconfreeRNNTTimesyncBeamSearch::paramAllowBlankAfterSentenceEnd(
        "allow-blank-after-sentence-end",
        "blanks can still be produced after the sentence-end has been reached",
        true);

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

const Core::ParameterBool LexiconfreeRNNTTimesyncBeamSearch::paramCacheCleanupInterval(
        "cache-cleanup-interval",
        "Interval of search steps after which buffered inputs that are not needed anymore get cleaned up.",
        10);

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

LexiconfreeRNNTTimesyncBeamSearch::LexiconfreeRNNTTimesyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          maxBeamSize_(paramMaxBeamSize(config)),
          scoreThreshold_(paramScoreThreshold(config)),
          lengthNormScale_(paramLengthNormScale(config)),
          maxLabelsPerFrame_(paramMaxLabelsPerFrame(config)),
          blankLabelIndex_(paramBlankLabelIndex(config)),
          allowBlankAfterSentenceEnd_(paramAllowBlankAfterSentenceEnd(config)),
          sentenceEndLemma_(),
          sentenceEndLabelIndex_(paramSentenceEndLabelIndex(config)),
          sentenceEndFallback_(paramSentenceEndFallBack(config)),
          collapseRepeatedLabels_(paramCollapseRepeatedLabels(config)),
          cacheCleanupInterval_(paramCacheCleanupInterval(config)),
          maximumStableDelay_(paramMaximumStableDelay(config)),
          maximumStableDelayPruningInterval_(paramMaximumStableDelayPruningInterval(config)),
          logStepwiseStatistics_(paramLogStepwiseStatistics(config)),
          debugChannel_(config, "debug"),
          labelScorer_(),
          beam_(),
          innerHyps_(),
          outerHyps_(),
          extensions_(),
          newBeam_(),
          requests_(),
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
    beam_.reserve(maxBeamSize_);
    if (blankLabelIndex_ != Nn::invalidLabelIndex) {
        log() << "Use blank label with index " << blankLabelIndex_;
    }

    useSentenceEnd_ = sentenceEndLabelIndex_ != Nn::invalidLabelIndex;
    if (useSentenceEnd_) {
        log() << "Use sentence end label with index " << sentenceEndLabelIndex_;
    }

    useScorePruning_ = scoreThreshold_ != Core::Type<Score>::max;
}

Speech::ModelCombination::Mode LexiconfreeRNNTTimesyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon;
}

bool LexiconfreeRNNTTimesyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_     = modelCombination.lexicon();
    labelScorer_ = modelCombination.labelScorer();

    extensions_.reserve(maxBeamSize_ * lexicon_->nLemmas());
    requests_.reserve(extensions_.size());
    newBeam_.reserve(extensions_.size());

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

    reset();
    return true;
}

void LexiconfreeRNNTTimesyncBeamSearch::reset() {
    initializationTime_.start();

    labelScorer_->reset();

    // Reset beam to a single empty hypothesis
    beam_.clear();
    beam_.push_back(LabelHypothesis());
    beam_.front().scoringContext = labelScorer_->getInitialScoringContext();

    currentSearchStep_ = 0ul;
    finishedSegment_   = false;

    initializationTime_.stop();
}

void LexiconfreeRNNTTimesyncBeamSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    initializationTime_.start();
    labelScorer_->reset();
    resetStatistics();
    initializationTime_.stop();
    currentSearchStep_ = 0ul;
    finishedSegment_   = false;
}

void LexiconfreeRNNTTimesyncBeamSearch::finishSegment() {
    featureProcessingTime_.start();
    labelScorer_->signalNoMoreFeatures();
    featureProcessingTime_.stop();
    decodeManySteps();
    finalizeHypotheses();
    logStatistics();
    finishedSegment_ = true;
}

void LexiconfreeRNNTTimesyncBeamSearch::putFeature(Nn::DataView const& feature) {
    featureProcessingTime_.start();
    labelScorer_->addInput(feature);
    featureProcessingTime_.stop();
}

void LexiconfreeRNNTTimesyncBeamSearch::putFeatures(Nn::DataView const& features, size_t nTimesteps) {
    featureProcessingTime_.start();
    labelScorer_->addInputs(features, nTimesteps);
    featureProcessingTime_.stop();
}

Core::Ref<const Traceback> LexiconfreeRNNTTimesyncBeamSearch::getCurrentBestTraceback() const {
    return getBestHypothesis().trace->performTraceback();
}

Core::Ref<const LatticeAdaptor> LexiconfreeRNNTTimesyncBeamSearch::getCurrentBestWordLattice() const {
    auto&        bestHypothesis = getBestHypothesis();
    LatticeTrace endTrace(bestHypothesis.trace, 0, bestHypothesis.trace->time + 1, bestHypothesis.trace->score, {});

    for (size_t hypIdx = 1ul; hypIdx < beam_.size(); ++hypIdx) {
        auto& hyp          = beam_[hypIdx];
        auto  siblingTrace = Core::ref(new LatticeTrace(hyp.trace, 0, hyp.trace->time, hyp.trace->score, {}));
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

        extensions_.clear();
        requests_.clear();

        /*
         * Extend inner hyps with the blank label, so they become outer hyps
         */
        for (size_t hypIndex = 0ul; hypIndex < innerHyps_.size(); ++hypIndex) {
            auto& hyp = innerHyps_[hypIndex];

            auto transitionType = inferTransitionType(hyp.currentToken, blankLabelIndex_);

            extensions_.push_back(
                    {blankLabelIndex_,
                     nullptr,
                     hyp.score,
                     0,
                     transitionType,
                     hypIndex});
            requests_.push_back({innerHyps_[hypIndex].scoringContext, blankLabelIndex_, transitionType});
        }

        /*
         * Perform scoring of all the requests with the label scorer.
         */
        scoringTime_.start();
        auto result = labelScorer_->computeScoresWithTimes(requests_);
        scoringTime_.stop();

        if (not result) {
            // LabelScorer could not compute scores -> no search step can be made.
            return false;
        }

        for (size_t extensionIdx = 0ul; extensionIdx < extensions_.size(); ++extensionIdx) {
            extensions_[extensionIdx].score += result->scores[extensionIdx];
            extensions_[extensionIdx].timeframe = result->timeframes[extensionIdx];
        }

        // Create new label hypotheses from extension candidates
        newBeam_.clear();
        for (auto const& extension : extensions_) {
            auto const& baseHyp = innerHyps_[extension.baseHypIndex];

            contextExtensionTime_.start();
            auto newScoringContext = labelScorer_->extendedScoringContext(
                    {baseHyp.scoringContext,
                     extension.nextToken,
                     extension.transitionType});
            contextExtensionTime_.stop();

            newBeam_.push_back({baseHyp, extension, newScoringContext, lengthNormScale_});
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

        extensions_.clear();
        requests_.clear();

        /*
         * Extend inner hyps with non-blank labels
         */
        for (size_t hypIndex = 0ul; hypIndex < innerHyps_.size(); ++hypIndex) {
            auto& hyp = innerHyps_[hypIndex];

            // Iterate over possible successors (all lemmas)
            for (auto lemmaIt = lemmas.first; lemmaIt != lemmas.second; ++lemmaIt) {
                const Bliss::Lemma* lemma(*lemmaIt);
                Nn::LabelIndex      tokenIdx = lemma->id();

                // Blank is not allowed as an extension for the inner hyps
                if (tokenIdx == blankLabelIndex_) {
                    continue;
                }

                auto transitionType = inferTransitionType(hyp.currentToken, tokenIdx);

                extensions_.push_back(
                        {tokenIdx,
                         lemma->pronunciations().first,
                         hyp.score,
                         0,
                         transitionType,
                         hypIndex});
                requests_.push_back({innerHyps_[hypIndex].scoringContext, tokenIdx, transitionType});
            }
        }

        /*
         * Perform scoring of all the requests with the label scorer.
         */
        scoringTime_.start();
        auto resultInner = labelScorer_->computeScoresWithTimes(requests_);
        scoringTime_.stop();

        if (not resultInner) {
            // LabelScorer could not compute scores -> no search step can be made.
            return false;
        }

        for (size_t extensionIdx = 0ul; extensionIdx < extensions_.size(); ++extensionIdx) {
            extensions_[extensionIdx].score += resultInner->scores[extensionIdx];
            extensions_[extensionIdx].timeframe = resultInner->timeframes[extensionIdx];
        }

        // Score-prune extension candidates
        if (useScorePruning_) {
            scorePruning(extensions_);
        }

        // Create new label hypotheses from extension candidates
        newBeam_.clear();
        for (auto const& extension : extensions_) {
            auto const& baseHyp = innerHyps_[extension.baseHypIndex];

            contextExtensionTime_.start();
            auto newScoringContext = labelScorer_->extendedScoringContext(
                    {baseHyp.scoringContext,
                     extension.nextToken,
                     extension.transitionType});
            contextExtensionTime_.stop();

            newBeam_.push_back({baseHyp, extension, newScoringContext, lengthNormScale_});
        }

        // Prune new inner hyps down to maxBeamSize based on the raw score
        beamSizePruning(newBeam_);

        // If there are already more than maxBeamSize outer hyps,
        // remove all inner hyps with a score that is lower than the worst score of the max-beam-size best outer hyps
        Score outerHypsThreshold = std::numeric_limits<Score>::infinity();
        if (outerHyps_.size() >= maxBeamSize_) {
            auto kth = outerHyps_.begin() + (maxBeamSize_ - 1);
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
    if (useScorePruning_) {
        scorePruningLengthnormalized(outerHyps_);
    }
    beamSizePruningLengthnormalized(outerHyps_);

    // The leftover outer hyps of this timestep will be the inner hyps to start with in the next timestep
    beam_ = outerHyps_;
    outerHyps_.clear();

    numActiveHyps_ += beam_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("active-hyps", beam_.size());
        clog() << Core::XmlClose("search-step-stats");
    }

    /*
     * Clean up label scorer caches.
     */
    if (++currentSearchStep_ % cacheCleanupInterval_ == 0) {
        Core::CollapsedVector<Nn::ScoringContextRef> activeContexts;
        for (auto const& hyp : beam_) {
            activeContexts.push_back(hyp.scoringContext);
        }
        labelScorer_->cleanupCaches(activeContexts);
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

Nn::LabelScorer::TransitionType LexiconfreeRNNTTimesyncBeamSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const {
    bool prevIsBlank       = prevLabel == blankLabelIndex_;
    bool nextIsBlank       = nextLabel == blankLabelIndex_;
    bool nextIsSentenceEnd = (useSentenceEnd_ and nextLabel == sentenceEndLabelIndex_);

    if (prevLabel == Nn::invalidLabelIndex) {
        if (nextIsBlank) {
            return Nn::LabelScorer::TransitionType::INITIAL_BLANK;
        }
        else if (nextIsSentenceEnd) {
            return Nn::LabelScorer::TransitionType::SENTENCE_END;
        }
        else {
            return Nn::LabelScorer::TransitionType::INITIAL_LABEL;
        }
    }

    if (prevIsBlank) {
        if (nextIsBlank) {
            return Nn::LabelScorer::TransitionType::BLANK_LOOP;
        }
        else if (nextIsSentenceEnd) {
            return Nn::LabelScorer::TransitionType::SENTENCE_END;
        }
        else {
            return Nn::LabelScorer::TransitionType::BLANK_TO_LABEL;
        }
    }
    else {
        if (nextIsBlank) {
            return Nn::LabelScorer::TransitionType::LABEL_TO_BLANK;
        }
        else if (collapseRepeatedLabels_ and prevLabel == nextLabel) {
            return Nn::LabelScorer::TransitionType::LABEL_LOOP;
        }
        else if (nextIsSentenceEnd) {
            return Nn::LabelScorer::TransitionType::SENTENCE_END;
        }
        else {
            return Nn::LabelScorer::TransitionType::LABEL_TO_LABEL;
        }
    }
}

void LexiconfreeRNNTTimesyncBeamSearch::beamSizePruning(std::vector<LabelHypothesis>& hypotheses) const {
    if (hypotheses.size() <= maxBeamSize_) {
        return;
    }

    // Reorder the hypotheses by associated score value such that the first `beamSize_` elements are the best
    std::nth_element(hypotheses.begin(), hypotheses.begin() + maxBeamSize_, hypotheses.end(),
                     [](auto const& a, auto const& b) { return a.score < b.score; });
    hypotheses.resize(maxBeamSize_);  // Get rid of excessive elements
}

void LexiconfreeRNNTTimesyncBeamSearch::beamSizePruningLengthnormalized(std::vector<LabelHypothesis>& hypotheses) const {
    if (hypotheses.size() <= maxBeamSize_) {
        return;
    }

    // Reorder the hypotheses by associated scaledScore value such that the first `beamSize_` elements are the best
    std::nth_element(hypotheses.begin(), hypotheses.begin() + maxBeamSize_, hypotheses.end());
    hypotheses.resize(maxBeamSize_);  // Get rid of excessive elements
}

void LexiconfreeRNNTTimesyncBeamSearch::scorePruning(std::vector<ExtensionCandidate>& extensions) const {
    if (extensions.empty()) {
        return;
    }

    // Compute the pruning threshold
    auto bestHyp = *std::min_element(
            extensions.begin(),
            extensions.end(),
            [](auto const& a, auto const& b) {
                return a.score < b.score;
            });

    auto pruningThreshold = bestHyp.score + scoreThreshold_;

    // Remove elements with score > pruningThreshold
    extensions.erase(
            std::remove_if(
                    extensions.begin(),
                    extensions.end(),
                    [=](auto const& hyp) { return hyp.score > pruningThreshold; }),
            extensions.end());
}

void LexiconfreeRNNTTimesyncBeamSearch::scorePruningLengthnormalized(std::vector<LabelHypothesis>& hypotheses) const {
    if (hypotheses.empty()) {
        return;
    }

    // Compute the pruning threshold
    auto bestHyp = *std::min_element(
            hypotheses.begin(),
            hypotheses.end());
    auto pruningThreshold = (bestHyp.score + scoreThreshold_) / std::pow(bestHyp.length, lengthNormScale_);

    // Remove elements with scaledScore > pruningThreshold
    hypotheses.erase(
            std::remove_if(
                    hypotheses.begin(),
                    hypotheses.end(),
                    [=](auto const& hyp) { return hyp.scaledScore > pruningThreshold; }),
            hypotheses.end());
}

void LexiconfreeRNNTTimesyncBeamSearch::recombination(std::vector<LexiconfreeRNNTTimesyncBeamSearch::LabelHypothesis>& hypotheses) {
    // Represents a unique combination of currentToken, scoringContext and the previous (non-blank) output tokens
    struct RecombinationContext {
        Nn::LabelIndex        currentToken;
        Nn::ScoringContextRef scoringContext;
        std::vector<int>      outputTokens;

        RecombinationContext(LabelHypothesis const& hyp)
                : currentToken(hyp.currentToken), scoringContext(hyp.scoringContext), outputTokens(hyp.outputTokens) {}

        bool operator==(RecombinationContext const& other) const {
            return currentToken == other.currentToken and Nn::ScoringContextEq{}(scoringContext, other.scoringContext) and outputTokens == other.outputTokens;
        }
    };
    struct RecombinationContextHash {
        size_t operator()(RecombinationContext const& context) const {
            size_t h1 = context.currentToken;
            size_t h2 = Nn::ScoringContextHash{}(context.scoringContext);
            size_t h3 = 0;
            for (size_t i = 0; i < context.outputTokens.size(); ++i) {
                Core::combineHashes(h3, std::hash<uint32_t>()(context.outputTokens[i]));
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

            auto* existingHyp           = it->second;
            hyp.trace->sibling          = existingHyp->trace->sibling;
            existingHyp->trace->sibling = hyp.trace;

            // Add this hyp's score to existing hyp's score
            existingHyp->score += hyp.score;

            // Recompute scaled score from the merged score
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
    auto&                   bestHyp   = beam_.front();
    Score                   bestScore = Core::Type<Score>::max;
    Core::Ref<LatticeTrace> root;

    for (auto const& hyp : beam_) {
        if (hyp.score < bestScore and hyp.trace->time >= cutoff) {
            bestScore = hyp.score;
            bestHyp   = hyp;
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
