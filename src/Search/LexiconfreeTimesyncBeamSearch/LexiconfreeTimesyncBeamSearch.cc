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

#include "LexiconfreeTimesyncBeamSearch.hh"

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

LexiconfreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContexts(),
          currentToken(Nn::invalidLabelIndex),
          score(0.0),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))),
          reachedSentenceEnd(false) {}

LexiconfreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        LexiconfreeTimesyncBeamSearch::LabelHypothesis const&    base,
        LexiconfreeTimesyncBeamSearch::ExtensionCandidate const& extension,
        std::vector<Nn::ScoringContextRef> const&                newScoringContexts)
        : scoringContexts(newScoringContexts),
          currentToken(extension.nextToken),
          score(extension.score),
          trace(),
          reachedSentenceEnd(base.reachedSentenceEnd or extension.transitionType == Nn::LabelScorer::SENTENCE_END) {
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
            extension.timeframe + 1,
            {score, 0},
            {}));
}

std::string LexiconfreeTimesyncBeamSearch::LabelHypothesis::toString() const {
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
 * === LexiconfreeTimesyncBeamSearch ===
 * =====================================
 */

const Core::ParameterIntVector LexiconfreeTimesyncBeamSearch::paramMaxBeamSizes(
        "max-beam-size",
        "Maximum number of elements in the search beam. Pruning is applied after each intermediate label scorer.",
        "",
        1);

const Core::ParameterFloatVector LexiconfreeTimesyncBeamSearch::paramScoreThresholds(
        "score-threshold",
        "Prune any hypotheses with a score that is at least this much worse than the best hypothesis. Pruning is applied after each intermediate label scorer.",
        "",
        0,
        Core::Type<Score>::max);

const Core::ParameterInt LexiconfreeTimesyncBeamSearch::paramBlankLabelIndex(
        "blank-label-index",
        "Index of the blank label in the lexicon. Can also be inferred from lexicon if it has a lemma with `special='blank'`. If not set, the search will not use blank.",
        Nn::invalidLabelIndex);

const Core::ParameterInt LexiconfreeTimesyncBeamSearch::paramSentenceEndLabelIndex(
        "sentence-end-label-index",
        "Index of the sentence end label in the lexicon. Can also be inferred from lexicon if it has a lemma with `special='sentence-end'` or `special='sentence-boundary'`. If not set, the search will not use sentence end.",
        Nn::invalidLabelIndex);

const Core::ParameterBool LexiconfreeTimesyncBeamSearch::paramAllowBlankAfterSentenceEnd(
        "allow-blank-after-sentence-end",
        "blanks can still be produced after the sentence-end has been reached",
        true);

const Core::ParameterBool LexiconfreeTimesyncBeamSearch::paramSentenceEndFallBack(
        "sentence-end-fall-back",
        "Allow for fallback solution if no active word-end hypothesis exists at the end of a segment.",
        true);

const Core::ParameterBool LexiconfreeTimesyncBeamSearch::paramCollapseRepeatedLabels(
        "collapse-repeated-labels",
        "Collapse repeated emission of the same label into one output. If false, every emission is treated like a new output.",
        false);

const Core::ParameterBool LexiconfreeTimesyncBeamSearch::paramLogStepwiseStatistics(
        "log-stepwise-statistics",
        "Log statistics about the beam at every search step.",
        false);

const Core::ParameterBool LexiconfreeTimesyncBeamSearch::paramCacheCleanupInterval(
        "cache-cleanup-interval",
        "Interval of search steps after which buffered inputs that are not needed anymore get cleaned up.",
        10);

const Core::ParameterInt LexiconfreeTimesyncBeamSearch::paramMaximumStableDelay(
        "maximum-stable-delay",
        "Introduce a cutoff point at `current-time` - `delay`. Every hypothesis that disagrees with the current best anywhere before the cutoff gets pruned."
        "This way words in the traceback become stable after at most `delay` frames.",
        Core::Type<int>::max,
        0);

const Core::ParameterInt LexiconfreeTimesyncBeamSearch::paramMaximumStableDelayPruningInterval(
        "maximum-stable-delay-pruning-interval",
        "Interval of search steps after which the maximum-stable-delay-pruning gets applied.",
        10,
        1);

LexiconfreeTimesyncBeamSearch::LexiconfreeTimesyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          blankLabelIndex_(paramBlankLabelIndex(config)),
          allowBlankAfterSentenceEnd_(paramAllowBlankAfterSentenceEnd(config)),
          sentenceEndLemma_(),
          sentenceEndLabelIndex_(paramSentenceEndLabelIndex(config)),
          sentenceEndFallback_(paramSentenceEndFallBack(config)),
          collapseRepeatedLabels_(paramCollapseRepeatedLabels(config)),
          logStepwiseStatistics_(paramLogStepwiseStatistics(config)),
          cacheCleanupInterval_(paramCacheCleanupInterval(config)),
          debugChannel_(config, "debug"),
          labelScorers_(),
          beam_(),
          extensions_(),
          newBeam_(),
          requests_(),
          tempHypotheses_(),
          initializationTime_(),
          featureProcessingTime_(),
          scoringTime_(),
          contextExtensionTime_(),
          numHypsAfterRecombination_("num-hyps-after-recombination"),
          numActiveHyps_("num-active-hyps"),
          currentSearchStep_(0ul),
          finishedSegment_(false),
          maximumStableDelay_(paramMaximumStableDelay(config)),
          maximumStableDelayPruningInterval_(paramMaximumStableDelayPruningInterval(config)) {
    auto maxBeamSizes = paramMaxBeamSizes(config);
    maxBeamSizes_.insert(maxBeamSizes_.begin(), maxBeamSizes.begin(), maxBeamSizes.end());

    auto scoreThresholds = paramScoreThresholds(config);
    scoreThresholds_.insert(scoreThresholds_.begin(), scoreThresholds.begin(), scoreThresholds.end());
    // Fill up with default value
    for (size_t i = scoreThresholds_.size(); i < maxBeamSizes_.size(); ++i) {
        scoreThresholds_.push_back(Core::Type<Score>::max);
    }

    useBlank_ = blankLabelIndex_ != Nn::invalidLabelIndex;
    if (useBlank_) {
        log() << "Use blank label with index " << blankLabelIndex_;
    }

    for (size_t i = 0; i < scoreThresholds_.size(); ++i) {
        useScorePruning_.push_back(scoreThresholds_[i] != Core::Type<Score>::max);
    }

    for (size_t i = 1ul; i <= scoreThresholds_.size(); ++i) {
        numHypsAfterScorePruning_.push_back({"num-hyps-after-score-pruning-" + std::to_string(i)});
    }
    for (size_t i = 1ul; i <= maxBeamSizes_.size(); ++i) {
        numHypsAfterBeamPruning_.push_back({"num-hyps-after-beam-pruning-" + std::to_string(i)});
    }
    useSentenceEnd_ = sentenceEndLabelIndex_ != Nn::invalidLabelIndex;
    if (useSentenceEnd_) {
        log() << "Use sentence end label with index " << sentenceEndLabelIndex_;
    }
}

Speech::ModelCombination::Mode LexiconfreeTimesyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon;
}

bool LexiconfreeTimesyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
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
            useBlank_        = true;
            log() << "Use blank index " << blankLabelIndex_ << " inferred from lexicon";
        }
        else if (blankLabelIndex_ != static_cast<Nn::LabelIndex>(blankLemma->id())) {
            warning() << "Blank lemma exists in lexicon with id " << blankLemma->id() << " but is overwritten by config parameter with value " << blankLabelIndex_;
        }
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

void LexiconfreeTimesyncBeamSearch::reset() {
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

void LexiconfreeTimesyncBeamSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    initializationTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->reset();
    }
    resetStatistics();
    initializationTime_.stop();
    currentSearchStep_ = 0ul;
    finishedSegment_   = false;
}

void LexiconfreeTimesyncBeamSearch::finishSegment() {
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

void LexiconfreeTimesyncBeamSearch::putFeature(Nn::DataView const& feature) {
    featureProcessingTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInput(feature);
    }
    featureProcessingTime_.stop();
}

void LexiconfreeTimesyncBeamSearch::putFeatures(Nn::DataView const& features, size_t nTimesteps) {
    featureProcessingTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInputs(features, nTimesteps);
    }
    featureProcessingTime_.stop();
}

Core::Ref<const Traceback> LexiconfreeTimesyncBeamSearch::getCurrentBestTraceback() const {
    return getBestHypothesis().trace->performTraceback();
}

Core::Ref<const LatticeAdaptor> LexiconfreeTimesyncBeamSearch::getCurrentBestWordLattice() const {
    auto&        bestHypothesis = getBestHypothesis();
    LatticeTrace endTrace(bestHypothesis.trace, 0, bestHypothesis.trace->time + 1, bestHypothesis.trace->score, {});

    for (size_t hypIdx = 1ul; hypIdx < beam_.size(); ++hypIdx) {
        auto& hyp          = beam_[hypIdx];
        auto  siblingTrace = Core::ref(new LatticeTrace(hyp.trace, 0, hyp.trace->time, hyp.trace->score, {}));
        endTrace.appendSiblingToChain(siblingTrace);
    }

    return endTrace.buildWordLattice(lexicon_);
}

Core::Ref<const LatticeTrace> LexiconfreeTimesyncBeamSearch::getCurrentBestLatticeTrace() const {
    return getBestHypothesis().trace;
}

Core::Ref<const LatticeTrace> LexiconfreeTimesyncBeamSearch::getCommonPrefix() const {
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

bool LexiconfreeTimesyncBeamSearch::decodeStep() {
    if (finishedSegment_) {
        return false;
    }

    // Assume the output labels are stored as lexicon lemma orth and ordered consistently with NN output index
    auto lemmas = lexicon_->lemmas();

    /*
     * Collect all possible extensions for all hypotheses in the beam.
     * Each extension candidate makes up a request.
     */
    extensions_.clear();
    extensions_.reserve(beam_.size() * lexicon_->nLemmas());
    requests_.reserve(extensions_.size());

    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        auto& hyp = beam_[hypIndex];

        // Iterate over possible successors (all lemmas)
        for (auto lemmaIt = lemmas.first; lemmaIt != lemmas.second; ++lemmaIt) {
            const Bliss::Lemma* lemma(*lemmaIt);
            Nn::LabelIndex      tokenIdx = lemma->id();

            // After first sentence-end token only allow looping that sentence-end or blanks afterwards
            if (hyp.reachedSentenceEnd and
                not(
                        (collapseRepeatedLabels_ and hyp.currentToken == sentenceEndLabelIndex_ and tokenIdx == sentenceEndLabelIndex_)  // sentence-end-loop
                        or (allowBlankAfterSentenceEnd_ and tokenIdx == blankLabelIndex_))) {                                            // blank
                continue;
            }

            auto transitionType = inferTransitionType(hyp.currentToken, tokenIdx);

            extensions_.push_back(
                    {.nextToken      = tokenIdx,
                     .pron           = lemma->pronunciations().first,
                     .score          = hyp.score,
                     .timeframe      = hyp.trace->time,
                     .transitionType = transitionType,
                     .baseHypIndex   = hypIndex});
        }
    }

    if (logStepwiseStatistics_) {
        clog() << Core::XmlOpen("search-step-stats");
    }

    for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
        requests_.clear();

        /*
         * Create scoring requests for the current label scorer.
         */
        for (auto const& ext : extensions_) {
            requests_.push_back({beam_[ext.baseHypIndex].scoringContexts[scorerIdx], ext.nextToken, ext.transitionType});
        }

        /*
         * Perform scoring of all the requests with the label scorer.
         */
        scoringTime_.start();
        auto result = labelScorers_[scorerIdx]->computeScoresWithTimes(requests_);
        scoringTime_.stop();

        if (not result) {
            // LabelScorer could not compute scores -> no search step can be made.
            if (logStepwiseStatistics_) {
                clog() << Core::XmlClose("search-step-stats");
            }
            return false;
        }

        for (size_t extensionIdx = 0ul; extensionIdx < extensions_.size(); ++extensionIdx) {
            auto& ext = extensions_[extensionIdx];
            ext.score += result->scores[extensionIdx];
            ext.timeframe = std::max(ext.timeframe, result->timeframes[extensionIdx]);
        }

        /*
         * Prune set of possible extensions by max beam size and possibly also by score.
         */

        if (useScorePruning_[scorerIdx]) {
            scorePruning(extensions_, scoreThresholds_[scorerIdx]);

            numHypsAfterScorePruning_[scorerIdx] += extensions_.size();

            if (logStepwiseStatistics_) {
                clog() << Core::XmlFull("num-hyps-after-score-pruning-" + std::to_string(scorerIdx + 1), extensions_.size());
            }
        }

        if (scorerIdx < labelScorers_.size() - 1) {
            beamSizePruning(extensions_, maxBeamSizes_[scorerIdx]);
            numHypsAfterBeamPruning_[scorerIdx] += extensions_.size();

            if (logStepwiseStatistics_) {
                clog() << Core::XmlFull("num-hyps-after-beam-pruning-" + std::to_string(scorerIdx + 1), extensions_.size());
            }
        }
    }

    // Create new beam from surviving extensions.
    newBeam_.clear();
    for (auto const& extension : extensions_) {
        auto const& baseHyp = beam_[extension.baseHypIndex];

        contextExtensionTime_.start();
        std::vector<Nn::ScoringContextRef> newScoringContexts;
        for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
            newScoringContexts.push_back(labelScorers_[scorerIdx]->extendedScoringContext(
                    {baseHyp.scoringContexts[scorerIdx],
                     extension.nextToken,
                     extension.transitionType}));
        }
        contextExtensionTime_.stop();

        newBeam_.push_back({baseHyp, extension, newScoringContexts});
    }

    // For all hypotheses with the same scoring context keep only the best since they will all develop in the same way.
    recombination(newBeam_);
    numHypsAfterRecombination_ += newBeam_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-recombination", newBeam_.size());
    }

    beamSizePruning(newBeam_, maxBeamSizes_[labelScorers_.size() - 1]);
    numHypsAfterBeamPruning_[labelScorers_.size() - 1] += newBeam_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-beam-pruning-" + std::to_string(labelScorers_.size()), newBeam_.size());
    }

    beam_.swap(newBeam_);

    numActiveHyps_ += beam_.size();

    ++currentSearchStep_;

    /*
     * Clean up label scorer caches.
     */
    if (currentSearchStep_ % cacheCleanupInterval_ == 0) {
        for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
            Core::CollapsedVector<Nn::ScoringContextRef> activeContexts;
            for (auto const& hyp : newBeam_) {
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

LexiconfreeTimesyncBeamSearch::LabelHypothesis const& LexiconfreeTimesyncBeamSearch::getBestHypothesis() const {
    verify(not beam_.empty());

    return *std::min_element(beam_.begin(), beam_.end());
}

LexiconfreeTimesyncBeamSearch::LabelHypothesis const& LexiconfreeTimesyncBeamSearch::getWorstHypothesis() const {
    verify(not beam_.empty());

    return *std::max_element(beam_.begin(), beam_.end());
}

void LexiconfreeTimesyncBeamSearch::resetStatistics() {
    initializationTime_.reset();
    featureProcessingTime_.reset();
    scoringTime_.reset();
    contextExtensionTime_.reset();
    for (auto& stat : numHypsAfterScorePruning_) {
        stat.clear();
    }
    for (auto& stat : numHypsAfterBeamPruning_) {
        stat.clear();
    }
    numHypsAfterRecombination_.clear();
    numActiveHyps_.clear();
}

void LexiconfreeTimesyncBeamSearch::logStatistics() const {
    clog() << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    clog() << Core::XmlOpen("initialization-time") << initializationTime_.elapsedMilliseconds() << Core::XmlClose("initialization-time");
    clog() << Core::XmlOpen("feature-processing-time") << featureProcessingTime_.elapsedMilliseconds() << Core::XmlClose("feature-processing-time");
    clog() << Core::XmlOpen("scoring-time") << scoringTime_.elapsedMilliseconds() << Core::XmlClose("scoring-time");
    clog() << Core::XmlOpen("context-extension-time") << contextExtensionTime_.elapsedMilliseconds() << Core::XmlClose("context-extension-time");
    clog() << Core::XmlClose("timing-statistics");
    for (auto const& stat : numHypsAfterScorePruning_) {
        stat.write(clog());
    }
    for (auto const& stat : numHypsAfterBeamPruning_) {
        stat.write(clog());
    }
    numHypsAfterRecombination_.write(clog());
    numActiveHyps_.write(clog());
}

Nn::LabelScorer::TransitionType LexiconfreeTimesyncBeamSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const {
    bool prevIsBlank       = (useBlank_ and prevLabel == blankLabelIndex_);
    bool nextIsBlank       = (useBlank_ and nextLabel == blankLabelIndex_);
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

template<typename Element>
void LexiconfreeTimesyncBeamSearch::beamSizePruning(std::vector<Element>& hypotheses, size_t maxSize) const {
    if (hypotheses.size() <= maxSize) {
        return;
    }

    // Reorder the hypotheses by associated score value such that the first `beamSize_` elements are the best
    std::nth_element(hypotheses.begin(), hypotheses.begin() + maxSize, hypotheses.end());
    hypotheses.resize(maxSize);  // Get rid of excessive elements
}

template void LexiconfreeTimesyncBeamSearch::beamSizePruning<LexiconfreeTimesyncBeamSearch::ExtensionCandidate>(std::vector<LexiconfreeTimesyncBeamSearch::ExtensionCandidate>&, size_t) const;
template void LexiconfreeTimesyncBeamSearch::beamSizePruning<LexiconfreeTimesyncBeamSearch::LabelHypothesis>(std::vector<LexiconfreeTimesyncBeamSearch::LabelHypothesis>&, size_t) const;

void LexiconfreeTimesyncBeamSearch::scorePruning(std::vector<LexiconfreeTimesyncBeamSearch::ExtensionCandidate>& extensions, Score threshold) const {
    if (extensions.empty()) {
        return;
    }

    // Compute the pruning threshold
    auto bestScore        = std::min_element(extensions.begin(), extensions.end())->score;
    auto pruningThreshold = bestScore + threshold;

    // Remove elements with score > pruningThreshold
    extensions.erase(
            std::remove_if(
                    extensions.begin(),
                    extensions.end(),
                    [=](auto const& ext) { return ext.score > pruningThreshold; }),
            extensions.end());
}

void LexiconfreeTimesyncBeamSearch::recombination(std::vector<LexiconfreeTimesyncBeamSearch::LabelHypothesis>& hypotheses) {
    // Represents a unique combination of currentToken and scoringContext
    struct RecombinationContext {
        Nn::LabelIndex                     currentToken;
        std::vector<Nn::ScoringContextRef> scoringContexts;

        RecombinationContext(LabelHypothesis const& hyp)
                : currentToken(hyp.currentToken), scoringContexts(hyp.scoringContexts) {}

        bool operator==(RecombinationContext const& other) const {
            if (currentToken != other.currentToken) {
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
            size_t hash = context.currentToken;
            for (auto const& scoringContext : context.scoringContexts) {
                hash = Core::combineHashes(hash, Nn::ScoringContextHash{}(scoringContext));
            }
            return hash;
        }
    };

    tempHypotheses_.clear();
    // Reserve capacity because future reallocations would break the raw pointer we are storing later
    tempHypotheses_.reserve(hypotheses.size());
    // Map each unique ScoringContext in newHypotheses to its hypothesis
    std::unordered_map<RecombinationContext, LabelHypothesis*, RecombinationContextHash> seenScoringContexts;
    for (auto const& hyp : hypotheses) {
        // Use try_emplace to check if the scoring context already exists and create a new entry if not at the same time
        auto [it, inserted] = seenScoringContexts.try_emplace({hyp}, nullptr);

        if (inserted) {
            // First time seeing this scoring context so move it over to `newHypotheses`
            tempHypotheses_.push_back(std::move(hyp));
            it->second = &tempHypotheses_.back();
        }
        else {
            auto* existingHyp = it->second;
            if (hyp.score < existingHyp->score) {
                // New hyp is better -> replace in `newHypotheses` and add existing one as sibling
                hyp.trace->sibling = existingHyp->trace;
                *existingHyp       = std::move(hyp);  // Overwrite in-place
            }
            else {
                // New hyp is worse -> add to existing one as sibling
                hyp.trace->sibling          = existingHyp->trace->sibling;
                existingHyp->trace->sibling = hyp.trace;
            }
        }
    }

    hypotheses.swap(tempHypotheses_);
}

void LexiconfreeTimesyncBeamSearch::maximumStableDelayPruning() {
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
        warning() << "Most recent word in best hypothesis is before cutoff point for maximum-stable-delay-pruning so the limit will be surpassed";
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

void LexiconfreeTimesyncBeamSearch::finalizeHypotheses() {
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
