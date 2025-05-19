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
        "Maximum number of hypotheses in the search beam.",
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
          beam_(),
          extensions_(),
          newBeam_(),
          requests_(),
          recombinedHypotheses_(),
          initializationTime_(),
          featureProcessingTime_(),
          scoringTime_(),
          contextExtensionTime_(),
          numHypsAfterScorePruning_("num-hyps-after-score-pruning"),
          numHypsAfterBeamPruning_("num-hyps-after-beam-pruning"),
          currentSearchStep_(0ul),
          totalTimesteps_(0ul),
          finishedSegment_(false) {
    beam_.reserve(maxBeamSize_);
    newBeam_.reserve(maxBeamSize_ * 2);  // terminated + active
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
    if (currentSearchStep_ >= maxLabelsPerTimestep_ * std::max(totalTimesteps_, 1ul)) {
        warning() << "Terminated search due to reaching max number of labels";
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
     * Prune set of possible extensions by max beam size and possibly also by score.
     */

    if (useScorePruning_) {
        scorePruningExtensions();
    }

    beamSizePruningExtensions();

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
     * For all hypotheses with the same scoring context keep only the best since they will
     * all develop in the same way.
     */
    recombination();

    /*
     * Jointly prune terminated and active hypotheses
     */
    if (useScorePruning_) {
        scorePruning();

        numHypsAfterScorePruning_ += newBeam_.size();

        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("num-hyps-after-score-pruning", newBeam_.size());
        }
    }

    beamSizePruning();
    numHypsAfterBeamPruning_ += newBeam_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-beam-pruning", newBeam_.size());
    }

    /*
     * Clean up label scorer caches.
     */
    Core::CollapsedVector<Nn::ScoringContextRef> activeContexts;
    for (auto const& hyp : newBeam_) {
        activeContexts.push_back(hyp.scoringContext);
    }
    labelScorer_->cleanupCaches(activeContexts);

    /*
     * Log statistics about the new beam after this step.
     */
    beam_.swap(newBeam_);

    if (debugChannel_.isOpen()) {
        std::stringstream ssActive;
        std::stringstream ssTerminated;
        for (size_t hypIdx = 0ul; hypIdx < beam_.size(); ++hypIdx) {
            auto const& hyp = beam_[hypIdx];
            if (hyp.isActive) {
                ssActive << "Active hypothesis " << hypIdx + 1ul << ":  " << beam_[hypIdx].toString() << "\n";
            }
            else {
                ssTerminated << "Terminated hypothesis " << hypIdx + 1ul << ":  " << beam_[hypIdx].toString() << "\n";
            }
        }
        ssActive << "\n";
        ssTerminated << "\n";
        debugChannel_ << ssActive.str() << ssTerminated.str();
    }

    if (logStepwiseStatistics_) {
        size_t numActive = std::accumulate(
                beam_.begin(),
                beam_.end(),
                0ul,
                [](size_t acc, auto const& hyp) { return acc + static_cast<size_t>(hyp.isActive); });
        auto const& bestHyp  = getBestHypothesis();
        auto const& worstHyp = getWorstHypothesis();
        clog() << Core::XmlFull("active-hyps", numActive);
        clog() << Core::XmlFull("terminated-hyps", beam_.size() - numActive);
        clog() << Core::XmlFull("best-hyp-score", bestHyp.score);
        clog() << Core::XmlFull("worst-hyp-score", worstHyp.score);
        clog() << Core::XmlFull("best-hyp-normed-score", bestHyp.scaledScore);
        clog() << Core::XmlFull("worst-hyp-normed-score", worstHyp.scaledScore);
        clog() << Core::XmlClose("search-step-stats");
    }

    ++currentSearchStep_;
    return true;
}

LexiconfreeLabelsyncBeamSearch::LabelHypothesis const& LexiconfreeLabelsyncBeamSearch::getBestHypothesis() const {
    LabelHypothesis const* bestActive     = nullptr;
    LabelHypothesis const* bestTerminated = nullptr;

    for (auto const& hyp : beam_) {
        if (hyp.isActive) {
            if (not bestActive or hyp < *bestActive) {
                bestActive = &hyp;
            }
        }
        else {
            if (not bestTerminated or hyp < *bestTerminated) {
                bestTerminated = &hyp;
            }
        }
    }

    if (bestTerminated) {
        return *bestTerminated;
    }
    else {
        return *bestActive;
    }
}

LexiconfreeLabelsyncBeamSearch::LabelHypothesis const& LexiconfreeLabelsyncBeamSearch::getWorstHypothesis() const {
    LabelHypothesis const* worstActive     = nullptr;
    LabelHypothesis const* worstTerminated = nullptr;

    for (auto const& hyp : beam_) {
        if (hyp.isActive) {
            if (not worstActive or hyp > *worstActive) {
                worstActive = &hyp;
            }
        }
        else {
            if (not worstTerminated or hyp > *worstTerminated) {
                worstTerminated = &hyp;
            }
        }
    }

    if (worstTerminated) {
        return *worstTerminated;
    }
    else {
        return *worstActive;
    }
}

void LexiconfreeLabelsyncBeamSearch::resetStatistics() {
    initializationTime_.reset();
    featureProcessingTime_.reset();
    scoringTime_.reset();
    contextExtensionTime_.reset();
    numHypsAfterScorePruning_.clear();
    numHypsAfterBeamPruning_.clear();
}

void LexiconfreeLabelsyncBeamSearch::logStatistics() const {
    clog() << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    clog() << Core::XmlOpen("initialization-time") << initializationTime_.elapsedMilliseconds() << Core::XmlClose("initialization-time");
    clog() << Core::XmlOpen("feature-processing-time") << featureProcessingTime_.elapsedMilliseconds() << Core::XmlClose("feature-processing-time");
    clog() << Core::XmlOpen("scoring-time") << scoringTime_.elapsedMilliseconds() << Core::XmlClose("scoring-time");
    clog() << Core::XmlOpen("context-extension-time") << contextExtensionTime_.elapsedMilliseconds() << Core::XmlClose("context-extension-time");
    clog() << Core::XmlClose("timing-statistics");
    numHypsAfterScorePruning_.write(clog());
    numHypsAfterBeamPruning_.write(clog());
}

void LexiconfreeLabelsyncBeamSearch::beamSizePruningExtensions() {
    if (extensions_.size() <= maxBeamSize_) {
        return;
    }

    // Reorder the hypotheses by associated score value such that the first `beamSizeActive_` elements are the best
    std::nth_element(extensions_.begin(), extensions_.begin() + maxBeamSize_, extensions_.end());
    extensions_.resize(maxBeamSize_);  // Get rid of excessive elements
}

void LexiconfreeLabelsyncBeamSearch::beamSizePruning() {
    if (newBeam_.size() <= maxBeamSize_) {
        return;
    }

    // Reorder the hypotheses by associated score value such that the first `beamSizeTerminated_` elements are the best
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
    // Map each unique ScoringContext in newHypotheses to its hypothesis
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

}  // namespace Search
