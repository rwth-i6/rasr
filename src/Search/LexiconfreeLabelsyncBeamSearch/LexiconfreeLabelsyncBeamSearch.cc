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
#include <numeric>
#include <strings.h>

#include <Core/CollapsedVector.hh>
#include <Core/XmlStream.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Search/Traceback.hh>

namespace Search {

/*
 * =======================
 * === LabelHypothesis ===
 * =======================
 */

LexiconfreeLabelsyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContext(),
          currentToken(Nn::invalidLabelIndex),
          length(0),
          score(0.0),
          scaledScore(0.0),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))),
          isActive(true) {}

LexiconfreeLabelsyncBeamSearch::LabelHypothesis::LabelHypothesis(
        LexiconfreeLabelsyncBeamSearch::LabelHypothesis const&    base,
        LexiconfreeLabelsyncBeamSearch::ExtensionCandidate const& extension,
        Nn::ScoringContextRef const&                              newScoringContext,
        float                                                     lengthNormScale)
        : scoringContext(newScoringContext),
          currentToken(extension.nextToken),
          length(base.length + 1),
          score(extension.score),
          scaledScore(score / std::pow(length, lengthNormScale)),
          trace(Core::ref(new LatticeTrace(
                  base.trace,
                  extension.pron,
                  extension.timeframe + 1,
                  {extension.score, 0},
                  {}))),
          isActive(extension.transitionType != Nn::LabelScorer::TransitionType::SENTENCE_END) {
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
        "max-beam-size",
        "Maximum number of elements in the search beam.",
        1, 1);

const Core::ParameterFloat LexiconfreeLabelsyncBeamSearch::paramScoreThreshold(
        "score-threshold",
        "Prune any hypotheses with a score that is at least this much worse than the best hypothesis."
        "If length normalization is enabled, the score threshold is added to the raw score before normalization."
        "If not set, no score pruning will be done.",
        Core::Type<Score>::max, 0);

const Core::ParameterInt LexiconfreeLabelsyncBeamSearch::paramSentenceEndLabelIndex(
        "sentence-end-index",
        "Index of the sentence-end label in the lexicon."
        "Can also be inferred from lexicon if it has a lemma with `special='sentence-end'` or `special='sentence-boundary'`");

const Core::ParameterFloat LexiconfreeLabelsyncBeamSearch::paramLengthNormScale(
        "length-norm-scale",
        "Exponent of length for the hypothesis length normalization. Scaled scores are computed as score / length^length_norm_scale.",
        0.0);

const Core::ParameterFloat LexiconfreeLabelsyncBeamSearch::paramMaxLabelsPerTimestep(
        "max-labels-per-timestep",
        "Maximum number of emitted labels per input timestep counted via `addInput`/`addInputs`.",
        1.0);

const Core::ParameterBool LexiconfreeLabelsyncBeamSearch::paramLogStepwiseStatistics(
        "log-stepwise-statistics",
        "Log statistics about the beam at every search step.",
        false);

const Core::ParameterBool LexiconfreeLabelsyncBeamSearch::paramCacheCleanupInterval(
        "cache-cleanup-interval",
        "Interval of search steps after which buffered inputs that are not needed anymore get cleaned up.",
        10);

LexiconfreeLabelsyncBeamSearch::LexiconfreeLabelsyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          maxBeamSize_(paramMaxBeamSize(config)),
          scoreThreshold_(paramScoreThreshold(config)),
          lengthNormScale_(paramLengthNormScale(config)),
          maxLabelsPerTimestep_(paramMaxLabelsPerTimestep(config)),
          sentenceEndLabelIndex_(paramSentenceEndLabelIndex(config)),
          logStepwiseStatistics_(paramLogStepwiseStatistics(config)),
          cacheCleanupInterval_(paramCacheCleanupInterval(config)),
          debugChannel_(config, "debug"),
          labelScorer_(),
          beam_(),
          extensions_(),
          newBeam_(),
          requests_(),
          recombinedHypotheses_(),
          initializationTime_(),
          featureProcessingTime_(),
          scoringTime_(),
          contextExtensionTime_(),
          numTerminatedHypsAfterScorePruning_("num-termianted-hyps-after-score-pruning"),
          numTerminatedHypsAfterRecombination_("num-terminated-hyps-after-recombination"),
          numTerminatedHypsAfterBeamPruning_("num-terminated-hyps-after-beam-pruning"),
          numActiveHypsAfterScorePruning_("num-active-hyps-after-score-pruning"),
          numActiveHypsAfterRecombination_("num-active-hyps-after-recombination"),
          numActiveHypsAfterBeamPruning_("num-active-hyps-after-beam-pruning"),
          currentSearchStep_(0ul),
          totalTimesteps_(0ul),
          finishedSegment_(false) {
    useScorePruning_ = scoreThreshold_ != Core::Type<Score>::max;

    if (sentenceEndLabelIndex_ != Core::Type<s32>::max) {
        log() << "Use sentence-end label with index " << sentenceEndLabelIndex_;
    }
}

Speech::ModelCombination::Mode LexiconfreeLabelsyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon;
}

bool LexiconfreeLabelsyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_     = modelCombination.lexicon();
    labelScorer_ = modelCombination.labelScorer();

    auto sentenceEndLemma = lexicon_->specialLemma("sentence-end");
    if (!sentenceEndLemma) {
        sentenceEndLemma = lexicon_->specialLemma("sentence-boundary");
    }
    if (sentenceEndLemma) {
        if (sentenceEndLabelIndex_ == Core::Type<s32>::max) {
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
    beam_.clear();
    beam_.push_back(LabelHypothesis());
    beam_.front().scoringContext = labelScorer_->getInitialScoringContext();

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
    auto& bestHypothesis = getBestHypothesis();

    LatticeTrace endTrace(bestHypothesis.trace, 0, bestHypothesis.trace->time + 1, bestHypothesis.trace->score, {});

    for (auto const& hyp : beam_) {
        if (hyp.isActive != bestHypothesis.isActive) {
            continue;
        }
        auto siblingTrace = Core::ref(new LatticeTrace(hyp.trace, 0, hyp.trace->time, hyp.trace->score, {}));
        endTrace.appendSiblingToChain(siblingTrace);
    }

    return endTrace.buildWordLattice(lexicon_);
}

bool LexiconfreeLabelsyncBeamSearch::decodeStep() {
    if (finishedSegment_) {
        return false;
    }
    if (currentSearchStep_ >= maxLabelsPerTimestep_ * totalTimesteps_) {
        warning() << "Terminated search due to reaching max number of label outputs given input count";
        finishedSegment_ = true;
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

    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        auto& hyp = beam_[hypIndex];

        if (not hyp.isActive) {
            continue;
        }

        // Iterate over possible successors (all lemmas)
        for (auto lemmaIt = lemmas.first; lemmaIt != lemmas.second; ++lemmaIt) {
            const Bliss::Lemma* lemma(*lemmaIt);
            Nn::LabelIndex      tokenIdx = lemma->id();

            auto transitionType = Nn::LabelScorer::TransitionType::LABEL_TO_LABEL;
            if (hyp.currentToken == Core::Type<Nn::LabelIndex>::max) {
                transitionType = Nn::LabelScorer::TransitionType::INITIAL_LABEL;
            }
            if (tokenIdx == sentenceEndLabelIndex_) {
                transitionType = Nn::LabelScorer::TransitionType::SENTENCE_END;
            }

            extensions_.push_back(
                    {tokenIdx,
                     lemma->pronunciations().first,
                     hyp.score,
                     0,
                     transitionType,
                     hypIndex});
            requests_.push_back({beam_[hypIndex].scoringContext, tokenIdx, transitionType});
        }
    }

    if (requests_.empty()) {
        // All hypotheses are terminated -> no search step can be made.
        finishedSegment_ = true;
        return false;
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
     * Maybe prune set of possible extensions by score.
     */
    if (useScorePruning_) {
        scorePruningExtensions();
        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("num-extensions-after-score-pruning", extensions_.size());
        }
    }

    /*
     * Create new beam from surviving extensions.
     */
    newBeam_.clear();

    for (auto const& hyp : beam_) {
        if (not hyp.isActive) {
            newBeam_.push_back(hyp);
        }
    }

    for (auto const& extension : extensions_) {
        auto const& baseHyp = beam_[extension.baseHypIndex];

        auto newScoringContext = labelScorer_->extendedScoringContext(
                {baseHyp.scoringContext,
                 extension.nextToken,
                 extension.transitionType});
        newBeam_.push_back({baseHyp, extension, newScoringContext, lengthNormScale_});
    }

    /*
     * Jointly prune terminated and active hypotheses by score
     */
    if (useScorePruning_) {
        scorePruning();

        size_t numActive     = numActiveHyps();
        size_t numTerminated = newBeam_.size() - numActive;

        numTerminatedHypsAfterScorePruning_ += numTerminated;
        numActiveHypsAfterScorePruning_ += numActive;

        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("num-terminated-hyps-after-score-pruning", numTerminated);
            clog() << Core::XmlFull("num-active-hyps-after-score-pruning", numActive);
        }
    }

    /*
     * For all hypotheses with the same scoring context keep only the best since they will
     * all develop in the same way.
     */
    recombination();

    size_t numActive     = numActiveHyps();
    size_t numTerminated = newBeam_.size() - numActive;

    numTerminatedHypsAfterRecombination_ += numTerminated;
    numActiveHypsAfterRecombination_ += numActive;

    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-terminated-hyps-after-recombination", numTerminated);
        clog() << Core::XmlFull("num-active-hyps-after-recombination", numActive);
    }

    beamSizePruning();

    numActive     = numActiveHyps();
    numTerminated = newBeam_.size() - numActive;

    numTerminatedHypsAfterBeamPruning_ += numTerminated;
    numActiveHypsAfterBeamPruning_ += numActive;

    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-terminated-hyps-after-beam-pruning", numTerminated);
        clog() << Core::XmlFull("num-active-hyps-after-beam-pruning", numActive);
    }

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

    beam_.swap(newBeam_);

    /*
     * Log statistics about the new beam after this step.
     */

    if (debugChannel_.isOpen()) {
        std::stringstream ssActive;
        std::stringstream ssTerminated;
        for (size_t hypIdx = 0ul; hypIdx < beam_.size(); ++hypIdx) {
            auto const& hyp = beam_[hypIdx];
            if (not hyp.isActive) {
                ssTerminated << "Terminated hypothesis " << hypIdx + 1ul << ":  " << beam_[hypIdx].toString() << "\n";
            }
            else {
                ssActive << "Active hypothesis " << hypIdx + 1ul << ":  " << beam_[hypIdx].toString() << "\n";
            }
        }
        ssActive << "\n";
        ssTerminated << "\n";
        debugChannel_ << ssActive.str() << ssTerminated.str();
    }

    if (logStepwiseStatistics_) {
        auto const* bestTerminatedHyp  = getBestTerminatedHypothesis();
        auto const* worstTerminatedHyp = getWorstActiveHypothesis();
        auto const* bestActiveHyp      = getBestActiveHypothesis();
        auto const* worstActiveHyp     = getWorstActiveHypothesis();
        if (bestTerminatedHyp != nullptr) {
            clog() << Core::XmlFull("best-terminated-hyp-score", bestTerminatedHyp->score);
            clog() << Core::XmlFull("best-terminated-hyp-normalized-score", bestTerminatedHyp->scaledScore);
        }
        if (worstTerminatedHyp != nullptr) {
            clog() << Core::XmlFull("worst-terminated-hyp-score", worstTerminatedHyp->score);
            clog() << Core::XmlFull("worst-terminated-hyp-normalized-score", worstTerminatedHyp->scaledScore);
        }
        if (bestActiveHyp != nullptr) {
            clog() << Core::XmlFull("best-active-hyp-score", bestActiveHyp->score);
            clog() << Core::XmlFull("best-active-hyp-normalized-score", bestActiveHyp->scaledScore);
        }
        if (worstActiveHyp != nullptr) {
            clog() << Core::XmlFull("worst-active-hyp-score", worstActiveHyp->score);
            clog() << Core::XmlFull("worst-active-hyp-normalized-score", worstActiveHyp->scaledScore);
        }
        clog() << Core::XmlClose("search-step-stats");
    }

    return true;
}

LexiconfreeLabelsyncBeamSearch::LabelHypothesis const* LexiconfreeLabelsyncBeamSearch::getBestTerminatedHypothesis() const {
    LabelHypothesis const* best = nullptr;

    for (auto const& hyp : beam_) {
        if (not hyp.isActive) {
            if (best == nullptr or hyp < *best) {
                best = &hyp;
            }
        }
    }

    return best;
}

LexiconfreeLabelsyncBeamSearch::LabelHypothesis const* LexiconfreeLabelsyncBeamSearch::getWorstTerminatedHypothesis() const {
    LabelHypothesis const* worst = nullptr;

    for (auto const& hyp : beam_) {
        if (not hyp.isActive) {
            if (worst == nullptr or hyp > *worst) {
                worst = &hyp;
            }
        }
    }

    return worst;
}

LexiconfreeLabelsyncBeamSearch::LabelHypothesis const* LexiconfreeLabelsyncBeamSearch::getBestActiveHypothesis() const {
    LabelHypothesis const* best = nullptr;

    for (auto const& hyp : beam_) {
        if (hyp.isActive) {
            if (best == nullptr or hyp < *best) {
                best = &hyp;
            }
        }
    }

    return best;
}

LexiconfreeLabelsyncBeamSearch::LabelHypothesis const* LexiconfreeLabelsyncBeamSearch::getWorstActiveHypothesis() const {
    LabelHypothesis const* worst = nullptr;

    for (auto const& hyp : beam_) {
        if (hyp.isActive) {
            if (worst == nullptr or hyp > *worst) {
                worst = &hyp;
            }
        }
    }

    return worst;
}

LexiconfreeLabelsyncBeamSearch::LabelHypothesis const& LexiconfreeLabelsyncBeamSearch::getBestHypothesis() const {
    auto const* result = getBestTerminatedHypothesis();
    if (result != nullptr) {
        return *result;
    }
    result = getBestActiveHypothesis();
    verify(result != nullptr);
    return *result;
}

LexiconfreeLabelsyncBeamSearch::LabelHypothesis const& LexiconfreeLabelsyncBeamSearch::getWorstHypothesis() const {
    auto const* result = getWorstTerminatedHypothesis();
    if (result != nullptr) {
        return *result;
    }
    result = getWorstActiveHypothesis();
    verify(result != nullptr);
    return *result;
}

void LexiconfreeLabelsyncBeamSearch::resetStatistics() {
    initializationTime_.reset();
    featureProcessingTime_.reset();
    scoringTime_.reset();
    contextExtensionTime_.reset();
    numTerminatedHypsAfterScorePruning_.clear();
    numTerminatedHypsAfterBeamPruning_.clear();
    numActiveHypsAfterScorePruning_.clear();
    numActiveHypsAfterBeamPruning_.clear();
}

void LexiconfreeLabelsyncBeamSearch::logStatistics() const {
    clog() << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    clog() << Core::XmlOpen("initialization-time") << initializationTime_.elapsedMilliseconds() << Core::XmlClose("initialization-time");
    clog() << Core::XmlOpen("feature-processing-time") << featureProcessingTime_.elapsedMilliseconds() << Core::XmlClose("feature-processing-time");
    clog() << Core::XmlOpen("scoring-time") << scoringTime_.elapsedMilliseconds() << Core::XmlClose("scoring-time");
    clog() << Core::XmlOpen("context-extension-time") << contextExtensionTime_.elapsedMilliseconds() << Core::XmlClose("context-extension-time");
    clog() << Core::XmlClose("timing-statistics");
    numTerminatedHypsAfterScorePruning_.write(clog());
    numTerminatedHypsAfterBeamPruning_.write(clog());
    numActiveHypsAfterScorePruning_.write(clog());
    numActiveHypsAfterBeamPruning_.write(clog());
}

void LexiconfreeLabelsyncBeamSearch::beamSizePruning() {
    if (newBeam_.size() <= maxBeamSize_) {
        return;
    }

    // Reorder the hypotheses by associated score value such that the first `maxBeamSize_` elements are the best
    std::nth_element(newBeam_.begin(), newBeam_.begin() + maxBeamSize_, newBeam_.end());
    newBeam_.resize(maxBeamSize_);  // Get rid of excessive elements
}

void LexiconfreeLabelsyncBeamSearch::scorePruningExtensions() {
    if (extensions_.empty()) {
        return;
    }

    // Compute the pruning threshold
    auto bestScore        = std::min_element(extensions_.begin(), extensions_.end())->score;
    auto pruningThreshold = bestScore + scoreThreshold_;

    // Remove elements with score > pruningThreshold
    extensions_.erase(
            std::remove_if(
                    extensions_.begin(),
                    extensions_.end(),
                    [&](auto const& ext) { return ext.score > pruningThreshold; }),
            extensions_.end());
}

void LexiconfreeLabelsyncBeamSearch::scorePruning() {
    if (newBeam_.empty()) {
        return;
    }

    // Compute the pruning threshold
    auto bestHyp = *std::min_element(
            newBeam_.begin(),
            newBeam_.end());

    // Remove elements with score > pruningThreshold
    auto pruningThreshold = (bestHyp.score + scoreThreshold_) / std::pow(bestHyp.length, lengthNormScale_);
    newBeam_.erase(
            std::remove_if(
                    newBeam_.begin(),
                    newBeam_.end(),
                    [&](auto const& hyp) { return hyp.scaledScore > pruningThreshold; }),
            newBeam_.end());
}

void LexiconfreeLabelsyncBeamSearch::recombination() {
    recombinedHypotheses_.clear();

    // Map each unique ScoringContext in `newBeam_` to its hypothesis
    std::unordered_map<Nn::ScoringContextRef, LabelHypothesis*, Nn::ScoringContextHash, Nn::ScoringContextEq> seenScoringContexts;
    for (auto const& hyp : newBeam_) {
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
            if (hyp < *existingHyp) {
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

    newBeam_.swap(recombinedHypotheses_);
}

size_t LexiconfreeLabelsyncBeamSearch::numActiveHyps() const {
    return std::accumulate(
            newBeam_.begin(),
            newBeam_.end(),
            0ul,
            [](size_t acc, auto const& hyp) { return acc + static_cast<size_t>(hyp.isActive); });
}

}  // namespace Search
