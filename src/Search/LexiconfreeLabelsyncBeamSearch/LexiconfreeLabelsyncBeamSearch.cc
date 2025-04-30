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

#include "LexiconfreeLabelsyncBeamSearch.hh"

#include <algorithm>
#include <strings.h>

#include <Core/CollapsedVector.hh>
#include <Core/XmlStream.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/Traceback.hh>

namespace Search {

/*
 * =======================
 * === LabelHypothesis ===
 * =======================
 */

LexiconfreeLabelsyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContext(),
          currentToken(Core::Type<Nn::LabelIndex>::max),
          length(0),
          score(0.0),
          scaledScore(0.0),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))) {}

LexiconfreeLabelsyncBeamSearch::LabelHypothesis::LabelHypothesis(
        LexiconfreeLabelsyncBeamSearch::LabelHypothesis const&    base,
        LexiconfreeLabelsyncBeamSearch::ExtensionCandidate const& extension,
        Nn::ScoringContextRef const&                              newScoringContext,
        float                                                     lengthNormScale)
        : scoringContext(newScoringContext),
          currentToken(extension.nextToken),
          length(base.length + 1),
          score(extension.score),
          scaledScore(extension.scaledScore),
          trace(Core::ref(new LatticeTrace(
                  base.trace,
                  extension.pron,
                  extension.timeframe + 1,
                  {extension.score, 0},
                  {}))) {
}

std::string LexiconfreeLabelsyncBeamSearch::LabelHypothesis::toString() const {
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
 * === LexiconfreeLabelsyncBeamSearch ==
 * =====================================
 */

const Core::ParameterInt LexiconfreeLabelsyncBeamSearch::paramMaxBeamSize(
        "max-beam-size-active",
        "Maximum number of hypotheses in the search beam.",
        1, 1);

const Core::ParameterFloat LexiconfreeLabelsyncBeamSearch::paramScoreThreshold(
        "score-threshold-active",
        "Prune any hypotheses with a score that is at least this much worse than the best hypothesis. If not set, no score pruning will be done.",
        Core::Type<Score>::max, 0);

const Core::ParameterInt LexiconfreeLabelsyncBeamSearch::paramSentenceEndLabelIndex(
        "sentence-end-index",
        "Index of the sentence-end label in the lexicon. Can also be inferred from lexicon if it has a lemma with `special='blank'`. If not set, the search will not use blank.",
        Core::Type<int>::max);

const Core::ParameterFloat LexiconfreeLabelsyncBeamSearch::paramLengthNormScale(
        "length-norm-scale",
        "Scaling factor for the hypothesis length normalization.",
        0.0);

const Core::ParameterFloat LexiconfreeLabelsyncBeamSearch::paramMaxLabelsPerTimestep(
        "max-labels-per-timestep",
        "Maximum number of emitted labels",
        1.0);

const Core::ParameterBool LexiconfreeLabelsyncBeamSearch::paramLogStepwiseStatistics(
        "log-stepwise-statistics",
        "Log statistics about the beam at every search step.",
        false);

LexiconfreeLabelsyncBeamSearch::LexiconfreeLabelsyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          maxBeamSize_(paramMaxBeamSize(config)),
          scoreThreshold_(paramScoreThreshold(config)),
          lengthNormScale_(paramLengthNormScale(config)),
          maxLabelsPerTimestep_(paramMaxLabelsPerTimestep(config)),
          sentenceEndLabelIndex_(paramSentenceEndLabelIndex(config)),
          logStepwiseStatistics_(paramLogStepwiseStatistics(config)),
          debugChannel_(config, "debug"),
          labelScorer_(),
          beamActive_(),
          beamTerminated_(),
          extensions_(),
          newBeamActive_(),
          requests_(),
          recombinedHypotheses_(),
          initializationTime_(),
          featureProcessingTime_(),
          scoringTime_(),
          contextExtensionTime_(),
          numActiveHypsAfterScorePruning_("num-active-hyps-after-score-pruning"),
          numActiveHypsAfterBeamPruning_("num-active-hyps-after-beam-pruning"),
          numTerminatedHypsAfterScorePruning_("num-terminated-hyps-after-score-pruning"),
          numTerminatedHypsAfterBeamPruning_("num-terminated-hyps-after-beam-pruning"),
          numActiveHyps_("num-active-hyps"),
          numTerminatedHyps_("num-terminated-hyps"),
          currentSearchStep_(0ul),
          totalTimesteps_(0ul),
          finishedSegment_(false) {
    beamActive_.reserve(maxBeamSize_);
    beamTerminated_.reserve(maxBeamSize_);
    newBeamActive_.reserve(maxBeamSize_);
    recombinedHypotheses_.reserve(maxBeamSize_);

    useScorePruning_ = scoreThreshold_ != Core::Type<Score>::max;

    log() << "Use sentence-end label with index " << sentenceEndLabelIndex_;
}

Speech::ModelCombination::Mode LexiconfreeLabelsyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon;
}

bool LexiconfreeLabelsyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_     = modelCombination.lexicon();
    labelScorer_ = modelCombination.labelScorer();

    extensions_.reserve(maxBeamSize_ * lexicon_->nLemmas());
    requests_.reserve(extensions_.size());

    auto sentenceEndLemma = lexicon_->specialLemma("sentence-end");
    if (!sentenceEndLemma) {
        sentenceEndLemma = lexicon_->specialLemma("sentence-boundary");
    }
    if (sentenceEndLemma) {
        if (sentenceEndLabelIndex_ == Core::Type<int>::max) {
            sentenceEndLabelIndex_ = sentenceEndLemma->id();
            log() << "Use sentence-end index " << sentenceEndLabelIndex_ << " inferred from lexicon";
        }
        else if (sentenceEndLabelIndex_ != static_cast<Nn::LabelIndex>(sentenceEndLemma->id())) {
            warning() << "SentenceEnd lemma exists in lexicon with id " << sentenceEndLemma->id() << " but is overwritten by config parameter with value " << sentenceEndLabelIndex_;
        }
    }

    reset();
    return true;
}

void LexiconfreeLabelsyncBeamSearch::reset() {
    initializationTime_.start();

    labelScorer_->reset();

    // Reset beam to a single empty hypothesis
    beamActive_.clear();
    beamActive_.push_back(LabelHypothesis());
    beamActive_.front().scoringContext = labelScorer_->getInitialScoringContext();

    beamTerminated_.clear();

    finishedSegment_   = false;
    totalTimesteps_    = 0ul;
    currentSearchStep_ = 0ul;

    initializationTime_.stop();
}

void LexiconfreeLabelsyncBeamSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    initializationTime_.start();
    labelScorer_->reset();
    resetStatistics();
    initializationTime_.stop();
    finishedSegment_   = false;
    totalTimesteps_    = 0ul;
    currentSearchStep_ = 0ul;
}

void LexiconfreeLabelsyncBeamSearch::finishSegment() {
    featureProcessingTime_.start();
    labelScorer_->signalNoMoreFeatures();
    featureProcessingTime_.stop();
    decodeManySteps();
    logStatistics();
    finishedSegment_ = true;
}

void LexiconfreeLabelsyncBeamSearch::putFeature(Nn::DataView const& feature) {
    featureProcessingTime_.start();
    labelScorer_->addInput(feature);
    ++totalTimesteps_;
    featureProcessingTime_.stop();
}

void LexiconfreeLabelsyncBeamSearch::putFeatures(Nn::DataView const& features, size_t nTimesteps) {
    featureProcessingTime_.start();
    labelScorer_->addInputs(features, nTimesteps);
    totalTimesteps_ += nTimesteps;
    featureProcessingTime_.stop();
}

Core::Ref<const Traceback> LexiconfreeLabelsyncBeamSearch::getCurrentBestTraceback() const {
    return getBestHypothesis().trace->performTraceback();
}

Core::Ref<const LatticeAdaptor> LexiconfreeLabelsyncBeamSearch::getCurrentBestWordLattice() const {
    auto&       bestHypothesis = getBestHypothesis();
    auto const& hyps           = not beamTerminated_.empty() ? beamTerminated_ : beamActive_;

    LatticeTrace endTrace(bestHypothesis.trace, 0, bestHypothesis.trace->time + 1, bestHypothesis.trace->score, {});

    for (size_t hypIdx = 1ul; hypIdx < hyps.size(); ++hypIdx) {
        auto& hyp          = hyps[hypIdx];
        auto  siblingTrace = Core::ref(new LatticeTrace(hyp.trace, 0, hyp.trace->time, hyp.trace->score, {}));
        endTrace.appendSiblingToChain(siblingTrace);
    }

    return endTrace.buildWordLattice(lexicon_);
}

bool LexiconfreeLabelsyncBeamSearch::decodeStep() {
    if (finishedSegment_) {
        return false;
    }
    if (currentSearchStep_ * maxLabelsPerTimestep_ >= totalTimesteps_) {
        return false;
    }
    if (beamActive_.empty()) {
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
    requests_.clear();

    for (size_t hypIndex = 0ul; hypIndex < beamActive_.size(); ++hypIndex) {
        auto& hyp = beamActive_[hypIndex];

        // Iterate over possible successors (all lemmas)
        for (auto lemmaIt = lemmas.first; lemmaIt != lemmas.second; ++lemmaIt) {
            const Bliss::Lemma* lemma(*lemmaIt);
            Nn::LabelIndex      tokenIdx = lemma->id();

            auto transitionType = Nn::LabelScorer::TransitionType::LABEL_TO_LABEL;
            if (hyp.currentToken == Core::Type<Nn::LabelIndex>::max) {
                transitionType = Nn::LabelScorer::TransitionType::INITIAL_LABEL;
            }

            extensions_.push_back(
                    {tokenIdx,
                     lemma->pronunciations().first,
                     hyp.score,
                     static_cast<Score>(hyp.score / std::pow(hyp.length + 1, lengthNormScale_)),
                     0,
                     transitionType,
                     hypIndex});
            requests_.push_back({beamActive_[hypIndex].scoringContext, tokenIdx, transitionType});
        }
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

    if (logStepwiseStatistics_) {
        clog() << Core::XmlOpen("search-step-stats");
    }

    /*
     * Prune set of possible extensions by max beam size and possibly also by score.
     */

    if (useScorePruning_) {
        scorePruning(extensions_);

        numActiveHypsAfterScorePruning_ += extensions_.size();

        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("num-active-hyps-after-score-pruning", extensions_.size());
        }
    }

    beamSizePruning(extensions_);
    numActiveHypsAfterBeamPruning_ += extensions_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-active-hyps-after-beam-pruning", extensions_.size());
    }

    /*
     * Create new beam from surviving extensions.
     */
    newBeamActive_.clear();

    for (auto const& extension : extensions_) {
        auto const& baseHyp = beamActive_[extension.baseHypIndex];

        // If the next token is the sentence-end label, add it to the terminated beam
        if (extension.nextToken == sentenceEndLabelIndex_) {
            beamTerminated_.push_back({baseHyp, extension, baseHyp.scoringContext, lengthNormScale_});
        }
        else {
            auto newScoringContext = labelScorer_->extendedScoringContext(
                    {baseHyp.scoringContext,
                     extension.nextToken,
                     extension.transitionType});
            newBeamActive_.push_back({baseHyp, extension, newScoringContext, lengthNormScale_});
        }
    }

    /*
     * For all hypotheses with the same scoring context keep only the best since they will
     * all develop in the same way.
     */
    recombination(newBeamActive_);
    numActiveHyps_ += newBeamActive_.size();

    /*
     * Prune terminated hypotheses among each other
     */
    if (useScorePruning_) {
        scorePruningTerminated();

        numTerminatedHypsAfterScorePruning_ += beamTerminated_.size();

        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("num-terminated-hyps-after-score-pruning", beamTerminated_.size());
        }
    }

    beamSizePruningTerminated();
    numTerminatedHypsAfterBeamPruning_ += beamTerminated_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-terminated-hyps-after-beam-pruning", beamTerminated_.size());
    }

    /*
     * Clean up label scorer caches.
     */
    Core::CollapsedVector<Nn::ScoringContextRef> activeContexts;
    for (auto const& hyp : newBeamActive_) {
        activeContexts.push_back(hyp.scoringContext);
    }
    labelScorer_->cleanupCaches(activeContexts);

    /*
     * Log statistics about the new beam after this step.
     */
    beamActive_.swap(newBeamActive_);

    if (debugChannel_.isOpen()) {
        std::stringstream ss;
        for (size_t hypIdx = 0ul; hypIdx < beamActive_.size(); ++hypIdx) {
            ss << "Active hypothesis " << hypIdx + 1ul << ":  " << beamActive_[hypIdx].toString() << "\n";
        }
        for (size_t hypIdx = 0ul; hypIdx < beamTerminated_.size(); ++hypIdx) {
            ss << "Terminated hypothesis " << hypIdx + 1ul << ":  " << beamTerminated_[hypIdx].toString() << "\n";
        }
        ss << "\n";
        debugChannel_ << ss.str();
    }

    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("active-hyps", beamActive_.size());
        clog() << Core::XmlFull("terminated-hyps", beamTerminated_.size());
        clog() << Core::XmlFull("best-hyp-score", getBestHypothesis().score);
        clog() << Core::XmlFull("worst-hyp-score", getWorstHypothesis().score);
        clog() << Core::XmlClose("search-step-stats");
    }

    ++currentSearchStep_;
    return true;
}

LexiconfreeLabelsyncBeamSearch::LabelHypothesis const& LexiconfreeLabelsyncBeamSearch::getBestHypothesis() const {
    if (not beamTerminated_.empty()) {
        return *std::min_element(beamTerminated_.begin(), beamTerminated_.end());
    }
    else {
        return *std::min_element(beamActive_.begin(), beamActive_.end());
    }
}

LexiconfreeLabelsyncBeamSearch::LabelHypothesis const& LexiconfreeLabelsyncBeamSearch::getWorstHypothesis() const {
    if (not beamTerminated_.empty()) {
        return *std::max_element(beamTerminated_.begin(), beamTerminated_.end());
    }
    else {
        return *std::max_element(beamActive_.begin(), beamActive_.end());
    }
}

void LexiconfreeLabelsyncBeamSearch::resetStatistics() {
    initializationTime_.reset();
    featureProcessingTime_.reset();
    scoringTime_.reset();
    contextExtensionTime_.reset();
    numActiveHypsAfterScorePruning_.clear();
    numActiveHypsAfterBeamPruning_.clear();
    numTerminatedHypsAfterScorePruning_.clear();
    numTerminatedHypsAfterBeamPruning_.clear();
    numActiveHyps_.clear();
}

void LexiconfreeLabelsyncBeamSearch::logStatistics() const {
    clog() << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    clog() << Core::XmlOpen("initialization-time") << initializationTime_.elapsedMilliseconds() << Core::XmlClose("initialization-time");
    clog() << Core::XmlOpen("feature-processing-time") << featureProcessingTime_.elapsedMilliseconds() << Core::XmlClose("feature-processing-time");
    clog() << Core::XmlOpen("scoring-time") << scoringTime_.elapsedMilliseconds() << Core::XmlClose("scoring-time");
    clog() << Core::XmlOpen("context-extension-time") << contextExtensionTime_.elapsedMilliseconds() << Core::XmlClose("context-extension-time");
    clog() << Core::XmlClose("timing-statistics");
    numActiveHypsAfterScorePruning_.write(clog());
    numActiveHypsAfterBeamPruning_.write(clog());
    numTerminatedHypsAfterScorePruning_.write(clog());
    numTerminatedHypsAfterBeamPruning_.write(clog());
    numActiveHyps_.write(clog());
}

void LexiconfreeLabelsyncBeamSearch::beamSizePruning(std::vector<LexiconfreeLabelsyncBeamSearch::ExtensionCandidate>& extensions) const {
    if (extensions.size() <= maxBeamSize_) {
        return;
    }

    // Reorder the hypotheses by associated score value such that the first `beamSizeActive_` elements are the best
    std::nth_element(extensions.begin(), extensions.begin() + maxBeamSize_, extensions.end());
    extensions.resize(maxBeamSize_);  // Get rid of excessive elements
}

void LexiconfreeLabelsyncBeamSearch::beamSizePruningTerminated() {
    if (beamTerminated_.size() <= maxBeamSizeTerminated_) {
        return;
    }

    // Reorder the hypotheses by associated score value such that the first `beamSizeTerminated_` elements are the best
    std::nth_element(beamTerminated_.begin(), beamTerminated_.begin() + maxBeamSizeTerminated_, beamTerminated_.end());
    beamTerminated_.resize(maxBeamSizeTerminated_);  // Get rid of excessive elements
}

void LexiconfreeLabelsyncBeamSearch::scorePruning(std::vector<LexiconfreeLabelsyncBeamSearch::ExtensionCandidate>& extensions) const {
    if (extensions.empty()) {
        return;
    }

    // Compute the pruning threshold
    auto bestScore        = std::min_element(extensions.begin(), extensions.end())->score;
    auto pruningThreshold = bestScore + scoreThresholdActive_;

    // Remove elements with score > pruningThreshold
    extensions.erase(
            std::remove_if(
                    extensions.begin(),
                    extensions.end(),
                    [&](auto const& ext) { return ext.score > pruningThreshold; }),
            extensions.end());

    // Compare to terminated hypotheses
    if (beamTerminated_.empty()) {
        return;
    }
    auto bestHypTerminated          = *std::min_element(beamTerminated_.begin(), beamTerminated_.end());
    auto pruningThresholdTerminated = (bestHypTerminated.score + scoreThresholdActive_) / std::pow(bestHypTerminated.length, lengthNormScale_);

    // Remove elements with score > pruningThreshold
    extensions.erase(
            std::remove_if(
                    extensions.begin(),
                    extensions.end(),
                    [&](auto const& ext) { return ext.scaledScore > pruningThresholdTerminated; }),
            extensions.end());
}

void LexiconfreeLabelsyncBeamSearch::scorePruningTerminated() {
    if (beamTerminated_.empty()) {
        return;
    }

    // Compute the pruning threshold
    auto bestHyp = *std::min_element(
            beamTerminated_.begin(),
            beamTerminated_.end());

    // Remove elements with score > pruningThreshold
    auto pruningThreshold = (bestHyp.score + scoreThresholdTerminated_) / std::pow(bestHyp.length, lengthNormScale_);
    beamTerminated_.erase(
            std::remove_if(
                    beamTerminated_.begin(),
                    beamTerminated_.end(),
                    [&](auto const& hyp) { return hyp.scaledScore > pruningThreshold; }),
            beamTerminated_.end());

    // Compare to active hypotheses
    if (beamActive_.empty()) {
        return;
    }
    auto bestHypActive              = *std::min_element(beamActive_.begin(), beamActive_.end());
    auto pruningThresholdTerminated = (bestHypActive.score + scoreThresholdTerminated_) / std::pow(bestHyp.length, lengthNormScale_);

    // Remove elements with score > pruningThreshold
    extensions.erase(
            std::remove_if(
                    extensions.begin(),
                    extensions.end(),
                    [&](auto const& ext) { return ext.scaledScore > pruningThresholdTerminated; }),
            extensions.end());
}

void LexiconfreeLabelsyncBeamSearch::recombination(std::vector<LexiconfreeLabelsyncBeamSearch::LabelHypothesis>& hypotheses) {
    recombinedHypotheses_.clear();
    // Map each unique ScoringContext in newHypotheses to its hypothesis
    std::unordered_map<Nn::ScoringContextRef, LabelHypothesis*, Nn::ScoringContextHash, Nn::ScoringContextEq> seenScoringContexts;
    for (auto const& hyp : hypotheses) {
        // Use try_emplace to check if the scoring context already exists and create a new entry if not at the same time
        auto [it, inserted] = seenScoringContexts.try_emplace(hyp.scoringContext, nullptr);

        if (inserted) {
            // First time seeing this scoring context so move it over to `newHypotheses`
            recombinedHypotheses_.push_back(std::move(hyp));
            it->second = &recombinedHypotheses_.back();
        }
        else {
            verify(not hyp.trace->sibling);

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

    hypotheses.swap(recombinedHypotheses_);
}

}  // namespace Search
