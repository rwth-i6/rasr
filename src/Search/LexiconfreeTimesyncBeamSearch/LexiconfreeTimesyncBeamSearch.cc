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

namespace Search {

/*
 * =======================
 * === LabelHypothesis ===
 * =======================
 */

LexiconfreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContext(),
          currentPron(nullptr),
          currentToken(Nn::invalidLabelIndex),
          timeframe(0),
          score(0.0),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))) {}

LexiconfreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        LexiconfreeTimesyncBeamSearch::LabelHypothesis const&    base,
        LexiconfreeTimesyncBeamSearch::ExtensionCandidate const& extension,
        Nn::ScoringContextRef const&                             newScoringContext)
        : scoringContext(newScoringContext),
          currentPron(base.currentPron),
          currentToken(extension.nextToken),
          timeframe(base.timeframe),
          score(extension.score),
          trace(base.trace),
          recentTransitionType(extension.transitionType) {
    switch (extension.transitionType) {
        case Nn::LabelScorer::TransitionType::BLANK_TO_LABEL:
        case Nn::LabelScorer::TransitionType::LABEL_TO_BLANK:
        case Nn::LabelScorer::TransitionType::LABEL_TO_LABEL:
        case Nn::LabelScorer::TransitionType::SENTENCE_END:
            commitTrace();
            break;
        default:
            break;
    }

    currentPron = extension.pron;
    timeframe   = extension.timeframe;
}

void LexiconfreeTimesyncBeamSearch::LabelHypothesis::commitTrace() {
    trace = Core::ref(new LatticeTrace(
            trace,
            currentPron,
            timeframe + 1,
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

const Core::ParameterInt LexiconfreeTimesyncBeamSearch::paramMaxBeamSize(
        "max-beam-size",
        "Maximum number of elements in the search beam.",
        1, 1);

const Core::ParameterFloat LexiconfreeTimesyncBeamSearch::paramScoreThreshold(
        "score-threshold",
        "Prune any hypotheses with a score that is at least this much worse than the best hypothesis. If not set, no score pruning will be done.",
        Core::Type<Score>::max, 0);

const Core::ParameterFloat LexiconfreeTimesyncBeamSearch::paramIntermediateScoreThreshold(
        "intermediate-score-threshold",
        "Prune any intermediate hypotheses of sub-scorers with a score that is at least this much worse than the best hypothesis. If not set, no intermediate score pruning will be done.",
        Core::Type<Score>::max, 0);

const Core::ParameterInt LexiconfreeTimesyncBeamSearch::paramIntermediateMaxBeamSize(
        "intermediate-max-beam-size",
        "",
        Core::Type<int>::max, 0);

const Core::ParameterInt LexiconfreeTimesyncBeamSearch::paramBlankLabelIndex(
        "blank-label-index",
        "Index of the blank label in the lexicon. Can also be inferred from lexicon if it has a lemma with `special='blank'`. If not set, the search will not use blank.",
        Nn::invalidLabelIndex);

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
        "Introduce a cutoff point at `current-time` - `delay`. Every hypothesis that disagrees with the current best anywhere before the cutoff gets pruned. This way words in the traceback become stable after at most `delay` frames."
        "maximum number of frames before results are required to be stable",
        Core::Type<int>::max, 0);

LexiconfreeTimesyncBeamSearch::LexiconfreeTimesyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          maxBeamSize_(paramMaxBeamSize(config)),
          intermediateMaxBeamSize_(paramIntermediateMaxBeamSize(config)),
          scoreThreshold_(paramScoreThreshold(config)),
          intermediateScoreThreshold_(paramIntermediateScoreThreshold(config)),
          blankLabelIndex_(paramBlankLabelIndex(config)),
          collapseRepeatedLabels_(paramCollapseRepeatedLabels(config)),
          logStepwiseStatistics_(paramLogStepwiseStatistics(config)),
          cacheCleanupInterval_(paramCacheCleanupInterval(config)),
          debugChannel_(config, "debug"),
          labelScorer_(),
          beam_(),
          extensions_(),
          newBeam_(),
          requests_(),
          tempHypotheses_(),
          initializationTime_(),
          featureProcessingTime_(),
          scoringTime_(),
          contextExtensionTime_(),
          numHypsAfterScorePruning_("num-hyps-after-score-pruning"),
          numHypsAfterRecombination_("num-hyps-after-recombination"),
          numHypsAfterBeamPruning_("num-hyps-after-beam-pruning"),
          numActiveHyps_("num-active-hyps"),
          currentSearchStep_(0ul),
          finishedSegment_(false),
          stableTraceTracker_(),
          canUpdateStablePrefix_(false),
          maximumStableDelay_(paramMaximumStableDelay(config)) {
    beam_.reserve(maxBeamSize_);
    newBeam_.reserve(maxBeamSize_);
    tempHypotheses_.reserve(maxBeamSize_);
    useBlank_ = blankLabelIndex_ != Nn::invalidLabelIndex;
    if (useBlank_) {
        log() << "Use blank label with index " << blankLabelIndex_;
    }
    useScorePruning_             = scoreThreshold_ != Core::Type<Score>::max;
    useIntermediateScorePruning_ = intermediateScoreThreshold_ != Core::Type<Score>::max;
    useIntermediateBeamPruning_  = intermediateMaxBeamSize_ != Core::Type<int>::max;
}

Speech::ModelCombination::Mode LexiconfreeTimesyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon;
}

bool LexiconfreeTimesyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_     = modelCombination.lexicon();
    labelScorer_ = modelCombination.labelScorer();

    extensions_.reserve(maxBeamSize_ * lexicon_->nLemmas());
    requests_.reserve(extensions_.size());

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

    reset();
    return true;
}

void LexiconfreeTimesyncBeamSearch::reset() {
    initializationTime_.start();

    labelScorer_->reset();

    // Reset beam to a single empty hypothesis
    beam_.clear();
    beam_.push_back(LabelHypothesis());
    beam_.front().scoringContext = labelScorer_->getInitialScoringContext();

    stableTraceTracker_.setTrace(beam_.front().trace);
    canUpdateStablePrefix_ = false;

    currentSearchStep_ = 0ul;
    finishedSegment_   = false;

    initializationTime_.stop();
}

void LexiconfreeTimesyncBeamSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    initializationTime_.start();
    labelScorer_->reset();
    resetStatistics();
    initializationTime_.stop();
    currentSearchStep_ = 0ul;
    finishedSegment_   = false;
}

void LexiconfreeTimesyncBeamSearch::finishSegment() {
    featureProcessingTime_.start();
    labelScorer_->signalNoMoreFeatures();
    featureProcessingTime_.stop();
    decodeManySteps();
    for (auto& hyp : beam_) {
        hyp.commitTrace();
    }
    logStatistics();
    finishedSegment_ = true;
}

void LexiconfreeTimesyncBeamSearch::putFeature(Nn::DataView const& feature) {
    featureProcessingTime_.start();
    labelScorer_->addInput(feature);
    featureProcessingTime_.stop();
}

void LexiconfreeTimesyncBeamSearch::putFeatures(Nn::DataView const& features, size_t nTimesteps) {
    featureProcessingTime_.start();
    labelScorer_->addInputs(features, nTimesteps);
    featureProcessingTime_.stop();
}

Core::Ref<const Traceback> LexiconfreeTimesyncBeamSearch::getCurrentBestTraceback() const {
    return getBestHypothesis().trace->performTraceback();
}

Core::Ref<const Traceback> LexiconfreeTimesyncBeamSearch::getCurrentStableTraceback() {
    if (canUpdateStablePrefix_) {
        maximumStableDelayPruning();

        std::vector<Core::Ref<LatticeTrace const>> traces;
        traces.reserve(beam_.size());
        for (auto const& hyp : beam_) {
            traces.push_back(hyp.trace);
        }
        stableTraceTracker_.advanceStablePrefix(traces);
        canUpdateStablePrefix_ = false;
    }

    return stableTraceTracker_.getStablePrefixTrace()->performTraceback();
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

bool LexiconfreeTimesyncBeamSearch::decodeStep() {
    if (finishedSegment_) {
        return false;
    }

    // Assume the output labels are stored as lexicon lemma orth and ordered consistently with NN output index
    auto lemmas = lexicon_->lemmas();

    /*
     * Collect all possible extensions for all hypotheses in the beam.
     * Also Create scoring requests for the label scorer.
     * Each extension candidate makes up a request.
     */
    extensions_.clear();

    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        auto& hyp = beam_[hypIndex];

        // Iterate over possible successors (all lemmas)
        for (auto lemmaIt = lemmas.first; lemmaIt != lemmas.second; ++lemmaIt) {
            const Bliss::Lemma* lemma(*lemmaIt);
            Nn::LabelIndex      tokenIdx = lemma->id();

            auto transitionType = inferTransitionType(hyp.currentToken, tokenIdx);

            extensions_.push_back(
                    {tokenIdx,
                     lemma->pronunciations().first,
                     hyp.score,
                     0,
                     transitionType,
                     hypIndex});
        }
    }

    if (logStepwiseStatistics_) {
        clog() << Core::XmlOpen("search-step-stats");
    }

    for (size_t subScorerIdx = 0ul; subScorerIdx < labelScorer_->numSubScorers(); ++subScorerIdx) {
        requests_.clear();
        for (auto const& extension : extensions_) {
            requests_.push_back({beam_[extension.baseHypIndex].scoringContext, extension.nextToken, extension.transitionType});
        }

        /*
         * Perform scoring of all the requests with the label scorer.
         */
        scoringTime_.start();
        auto result = labelScorer_->computeScoresWithTimes(requests_, subScorerIdx);
        scoringTime_.stop();

        if (not result) {
            // LabelScorer could not compute scores -> no search step can be made.
            if (logStepwiseStatistics_) {
                clog() << Core::XmlClose("search-step-stats");
            }
            return false;
        }

        for (size_t extensionIdx = 0ul; extensionIdx < extensions_.size(); ++extensionIdx) {
            extensions_[extensionIdx].score += result->scores[extensionIdx];
            extensions_[extensionIdx].timeframe = std::max(extensions_[extensionIdx].timeframe, result->timeframes[extensionIdx]);
        }

        /*
         * Prune set of possible extensions by max beam size and possibly also by score.
         */
        if (subScorerIdx + 1 < labelScorer_->numSubScorers()) {
            if (useIntermediateScorePruning_) {
                scorePruning(extensions_, intermediateScoreThreshold_);

                if (logStepwiseStatistics_) {
                    clog() << Core::XmlFull("num-hyps-after-intermediate-score-pruning-" + std::to_string(subScorerIdx), extensions_.size());
                }
            }

            if (useIntermediateBeamPruning_) {
                beamSizePruning(extensions_, intermediateMaxBeamSize_);

                if (logStepwiseStatistics_) {
                    clog() << Core::XmlFull("num-hyps-after-intermediate-beam-pruning-" + std::to_string(subScorerIdx), extensions_.size());
                }
            }
        }
    }

    if (useScorePruning_) {
        scorePruning(extensions_, scoreThreshold_);

        numHypsAfterScorePruning_ += extensions_.size();

        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("num-hyps-after-score-pruning", extensions_.size());
        }
    }

    // Create new beam from surviving extensions.
    newBeam_.clear();
    for (auto const& extension : extensions_) {
        auto const& baseHyp = beam_[extension.baseHypIndex];

        auto newScoringContext = labelScorer_->extendedScoringContext(
                {baseHyp.scoringContext,
                 extension.nextToken,
                 extension.transitionType});

        newBeam_.push_back({baseHyp, extension, newScoringContext});
    }

    // For all hypotheses with the same scoring context keep only the best since they will all develop in the same way.
    recombination(newBeam_);
    numHypsAfterRecombination_ += newBeam_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-recombination", newBeam_.size());
    }

    beamSizePruning(newBeam_, maxBeamSize_);
    numHypsAfterBeamPruning_ += newBeam_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-beam-pruning", newBeam_.size());
    }

    for (auto& hyp : newBeam_) {
        auto newScoringContext = labelScorer_->finalizeScoringContext(
                {hyp.scoringContext,
                 hyp.currentToken,
                 hyp.recentTransitionType});

        hyp.scoringContext = newScoringContext;
    }

    beam_.swap(newBeam_);

    numActiveHyps_ += beam_.size();

    /*
     * Clean up label scorer caches.
     */
    if (++currentSearchStep_ % cacheCleanupInterval_ == 0) {
        Core::CollapsedVector<Nn::ScoringContextRef> activeContexts;
        for (auto const& hyp : newBeam_) {
            activeContexts.push_back(hyp.scoringContext);
        }
        labelScorer_->cleanupCaches(activeContexts);
    }

    /*
     * Log statistics about the new beam after this step.
     */

    canUpdateStablePrefix_ = true;

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
    numHypsAfterScorePruning_.clear();
    numHypsAfterRecombination_.clear();
    numHypsAfterBeamPruning_.clear();
    numActiveHyps_.clear();
}

void LexiconfreeTimesyncBeamSearch::logStatistics() const {
    clog() << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    clog() << Core::XmlOpen("initialization-time") << initializationTime_.elapsedMilliseconds() << Core::XmlClose("initialization-time");
    clog() << Core::XmlOpen("feature-processing-time") << featureProcessingTime_.elapsedMilliseconds() << Core::XmlClose("feature-processing-time");
    clog() << Core::XmlOpen("scoring-time") << scoringTime_.elapsedMilliseconds() << Core::XmlClose("scoring-time");
    clog() << Core::XmlOpen("context-extension-time") << contextExtensionTime_.elapsedMilliseconds() << Core::XmlClose("context-extension-time");
    clog() << Core::XmlClose("timing-statistics");
    numHypsAfterScorePruning_.write(clog());
    numHypsAfterRecombination_.write(clog());
    numHypsAfterBeamPruning_.write(clog());
    numActiveHyps_.write(clog());
}

Nn::LabelScorer::TransitionType LexiconfreeTimesyncBeamSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const {
    bool prevIsBlank = (useBlank_ and prevLabel == blankLabelIndex_);
    bool nextIsBlank = (useBlank_ and nextLabel == blankLabelIndex_);

    if (prevLabel == Nn::invalidLabelIndex) {
        if (nextIsBlank) {
            return Nn::LabelScorer::TransitionType::INITIAL_BLANK;
        }
        else {
            return Nn::LabelScorer::TransitionType::INITIAL_LABEL;
        }
    }

    if (prevIsBlank) {
        if (nextIsBlank) {
            return Nn::LabelScorer::TransitionType::BLANK_LOOP;
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
        else {
            return Nn::LabelScorer::TransitionType::LABEL_TO_LABEL;
        }
    }
}

template<typename Element>
void LexiconfreeTimesyncBeamSearch::beamSizePruning(std::vector<Element>& hypotheses, size_t maxBeamSize) const {
    if (hypotheses.size() <= maxBeamSize) {
        return;
    }

    // Reorder the hypotheses by associated score value such that the first `beamSize_` elements are the best
    std::nth_element(hypotheses.begin(), hypotheses.begin() + maxBeamSize, hypotheses.end());
    hypotheses.resize(maxBeamSize);  // Get rid of excessive elements
}

template void LexiconfreeTimesyncBeamSearch::beamSizePruning<LexiconfreeTimesyncBeamSearch::ExtensionCandidate>(std::vector<LexiconfreeTimesyncBeamSearch::ExtensionCandidate>&, size_t) const;
template void LexiconfreeTimesyncBeamSearch::beamSizePruning<LexiconfreeTimesyncBeamSearch::LabelHypothesis>(std::vector<LexiconfreeTimesyncBeamSearch::LabelHypothesis>&, size_t) const;

void LexiconfreeTimesyncBeamSearch::scorePruning(std::vector<LexiconfreeTimesyncBeamSearch::ExtensionCandidate>& extensions, Score scoreThreshold) const {
    if (extensions.empty()) {
        return;
    }

    // Compute the pruning threshold
    auto bestScore        = std::min_element(extensions.begin(), extensions.end())->score;
    auto pruningThreshold = bestScore + scoreThreshold;

    // Remove elements with score > pruningThreshold
    extensions.erase(
            std::remove_if(
                    extensions.begin(),
                    extensions.end(),
                    [=](auto const& ext) { return ext.score > pruningThreshold; }),
            extensions.end());
}

void LexiconfreeTimesyncBeamSearch::recombination(std::vector<LexiconfreeTimesyncBeamSearch::LabelHypothesis>& hypotheses) {
    tempHypotheses_.clear();
    tempHypotheses_.reserve(hypotheses.size());
    // Map each unique ScoringContext in newHypotheses to its hypothesis
    std::unordered_map<Nn::ScoringContextRef, LabelHypothesis*, Nn::ScoringContextHash, Nn::ScoringContextEq> seenScoringContexts;
    for (auto const& hyp : hypotheses) {
        // Use try_emplace to check if the scoring context already exists and create a new entry if not at the same time
        auto [it, inserted] = seenScoringContexts.try_emplace(hyp.scoringContext, nullptr);

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

}  // namespace Search
