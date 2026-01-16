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
#include <cstddef>
#include <numeric>
#include <strings.h>

#include <Core/CollapsedVector.hh>
#include <Core/XmlStream.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Search/Traceback.hh>
#include <Search/TracebackHelper.hh>

namespace Search {

/*
 * =======================
 * === LabelHypothesis ===
 * =======================
 */

LexiconfreeLabelsyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContexts(),
          currentToken(Nn::invalidLabelIndex),
          length(0),
          score(0.0),
          scaledScore(0.0),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))),
          isActive(true) {}

LexiconfreeLabelsyncBeamSearch::LabelHypothesis::LabelHypothesis(
        LexiconfreeLabelsyncBeamSearch::LabelHypothesis const&    base,
        LexiconfreeLabelsyncBeamSearch::ExtensionCandidate const& extension,
        std::vector<Nn::ScoringContextRef> const&                 newScoringContexts,
        float                                                     lengthNormScale)
        : scoringContexts(newScoringContexts),
          currentToken(extension.nextToken),
          score(extension.score),
          trace(),
          isActive(extension.transitionType != Nn::LabelScorer::TransitionType::SENTENCE_END) {
    switch (extension.transitionType) {
        case Nn::LabelScorer::TransitionType::LABEL_TO_LABEL:
        case Nn::LabelScorer::TransitionType::BLANK_TO_LABEL:
        case Nn::LabelScorer::TransitionType::INITIAL_LABEL:
        case Nn::LabelScorer::TransitionType::SENTENCE_END:
            length = base.length + 1;
            break;
        default:
            length = base.length;
    }
    scaledScore = score / std::pow(length, lengthNormScale);

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

std::string LexiconfreeLabelsyncBeamSearch::LabelHypothesis::toString() const {
    std::stringstream ss;
    ss << "Score: " << score << ", length: " << length << ", scaledScore: " << scaledScore << ", traceback: ";

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

const Core::ParameterIntVector LexiconfreeLabelsyncBeamSearch::paramMaxBeamSizes(
        "max-beam-size",
        "Maximum number of elements in the search beam. Pruning is applied after each intermediate label scorer.",
        "",
        1);

const Core::ParameterFloatVector LexiconfreeLabelsyncBeamSearch::paramScoreThresholds(
        "score-threshold",
        "Prune any hypotheses with a score that is at least this much worse than the best hypothesis."
        "Pruning is applied after each intermediate label scorer."
        "If length normalization is enabled, the score threshold is added to the raw score before normalization.",
        "",
        0);

const Core::ParameterInt LexiconfreeLabelsyncBeamSearch::paramSentenceEndLabelIndex(
        "sentence-end-label-index",
        "Index of the sentence-end label in the lexicon."
        "Can also be inferred from lexicon if it has a lemma with `special='sentence-end'` or `special='sentence-boundary'`");

const Core::ParameterFloat LexiconfreeLabelsyncBeamSearch::paramLengthNormScale(
        "length-norm-scale",
        "Exponent of length for the hypothesis score length normalization. Scaled scores are computed as score / length^length_norm_scale.",
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

const Core::ParameterInt LexiconfreeLabelsyncBeamSearch::paramMaximumStableDelay(
        "maximum-stable-delay",
        "Introduce a cutoff point at `current-time` - `delay`. Every hypothesis that disagrees with the current best anywhere before the cutoff gets pruned."
        "This way words in the traceback become stable after at most `delay` frames.",
        Core::Type<int>::max,
        0);

const Core::ParameterInt LexiconfreeLabelsyncBeamSearch::paramMaximumStableDelayPruningInterval(
        "maximum-stable-delay-pruning-interval",
        "Interval of search steps after which the maximum-stable-delay-pruning gets applied.",
        10,
        1);

LexiconfreeLabelsyncBeamSearch::LexiconfreeLabelsyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          lengthNormScale_(paramLengthNormScale(config)),
          maxLabelsPerTimestep_(paramMaxLabelsPerTimestep(config)),
          sentenceEndLabelIndex_(paramSentenceEndLabelIndex(config)),
          logStepwiseStatistics_(paramLogStepwiseStatistics(config)),
          cacheCleanupInterval_(paramCacheCleanupInterval(config)),
          maximumStableDelay_(paramMaximumStableDelay(config)),
          maximumStableDelayPruningInterval_(paramMaximumStableDelayPruningInterval(config)),
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
          numTerminatedHyps_("num-terminated-hyps"),
          currentSearchStep_(0ul),
          totalTimesteps_(0ul),
          finishedSegment_(false) {
    auto maxBeamSizes = paramMaxBeamSizes(config);
    maxBeamSizes_.insert(maxBeamSizes_.begin(), maxBeamSizes.begin(), maxBeamSizes.end());

    auto scoreThresholds = paramScoreThresholds(config);
    scoreThresholds_.insert(scoreThresholds_.begin(), scoreThresholds.begin(), scoreThresholds.end());
    // Fill up with default value
    for (size_t i = scoreThresholds_.size(); i < maxBeamSizes_.size(); ++i) {
        scoreThresholds_.push_back(Core::Type<Score>::max);
    }

    for (size_t i = 1ul; i <= scoreThresholds_.size(); ++i) {
        useScorePruning_.push_back(scoreThresholds_[i] != Core::Type<Score>::max);
    }
    for (size_t i = 1ul; i <= scoreThresholds_.size(); ++i) {
        numHypsAfterScorePruning_.push_back({"num-hyps-after-score-pruning-" + std::to_string(i)});
    }
    for (size_t i = 1ul; i <= maxBeamSizes_.size(); ++i) {
        numHypsAfterBeamPruning_.push_back({"num-hyps-after-beam-pruning-" + std::to_string(i)});
    }

    if (sentenceEndLabelIndex_ != Core::Type<s32>::max) {
        log() << "Use sentence-end label with index " << sentenceEndLabelIndex_;
    }
}

Speech::ModelCombination::Mode LexiconfreeLabelsyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon;
}

bool LexiconfreeLabelsyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_      = modelCombination.lexicon();
    labelScorers_ = modelCombination.labelScorers();

    if (labelScorers_.size() > maxBeamSizes_.size()) {
        error() << "Number of label scorers (" << labelScorers_.size() << ") exceeds number of configured max beam sizes (" << maxBeamSizes_.size() << ")";
    }
    if (labelScorers_.size() < maxBeamSizes_.size()) {
        warning() << "Number of label scorers (" << labelScorers_.size() << ") is less than number of configured max beam sizes (" << maxBeamSizes_.size() << ")";
    }

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

    finishedSegment_   = false;
    totalTimesteps_    = 0ul;
    currentSearchStep_ = 0ul;

    initializationTime_.stop();
}

void LexiconfreeLabelsyncBeamSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    initializationTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->reset();
    }
    resetStatistics();
    initializationTime_.stop();
    finishedSegment_   = false;
    totalTimesteps_    = 0ul;
    currentSearchStep_ = 0ul;
}

void LexiconfreeLabelsyncBeamSearch::finishSegment() {
    featureProcessingTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->signalNoMoreFeatures();
    }
    featureProcessingTime_.stop();
    decodeManySteps();
    logStatistics();
    finishedSegment_ = true;
}

void LexiconfreeLabelsyncBeamSearch::putFeature(Nn::DataView const& feature) {
    featureProcessingTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInput(feature);
    }
    ++totalTimesteps_;
    featureProcessingTime_.stop();
}

void LexiconfreeLabelsyncBeamSearch::putFeatures(Nn::DataView const& features, size_t nTimesteps) {
    featureProcessingTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInputs(features, nTimesteps);
    }
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

Core::Ref<const LatticeTrace> LexiconfreeLabelsyncBeamSearch::getCurrentBestLatticeTrace() const {
    return getBestHypothesis().trace;
}

Core::Ref<const LatticeTrace> LexiconfreeLabelsyncBeamSearch::getCommonPrefix() const {
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
    extensions_.reserve(beam_.size() * lexicon_->nLemmas());
    requests_.reserve(extensions_.size());

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
        }
    }

    if (logStepwiseStatistics_) {
        clog() << Core::XmlOpen("search-step-stats");
    }

    for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
        requests_.clear();

        for (auto const& extension : extensions_) {
            requests_.push_back(
                    {.context        = beam_[extension.baseHypIndex].scoringContexts[scorerIdx],
                     .nextToken      = extension.nextToken,
                     .transitionType = extension.transitionType});
        }

        if (requests_.empty()) {
            // All hypotheses are terminated -> no search step can be made.
            finishedSegment_ = true;
            if (logStepwiseStatistics_) {
                clog() << Core::XmlClose("search-step-stats");
            }
            return false;
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
            extensions_[extensionIdx].score += result->scores[extensionIdx];
            extensions_[extensionIdx].timeframe = std::max(extensions_[extensionIdx].timeframe, result->timeframes[extensionIdx]);
        }

        /*
         * Prune set of possible extensions by max beam size and possibly also by score.
         */

        if (useScorePruning_[scorerIdx]) {
            scorePruningExtensions(scoreThresholds_[scorerIdx]);
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
    if (debugChannel_.isOpen()) {
        std::stringstream ssExtensions;
        for (size_t hypIdx = 0ul; hypIdx < extensions_.size(); ++hypIdx) {
            auto const& hyp = extensions_[hypIdx];
            ssExtensions << "Extension " << hypIdx + 1ul << " token " << hyp.pron->lemma()->symbol() << ", score " << hyp.score << ", transitionType " << hyp.transitionType << ", base: " << beam_[hyp.baseHypIndex].toString() << "\n";
        }
        debugChannel_ << ssExtensions.str();
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

        std::vector<Nn::ScoringContextRef> newScoringContexts;
        for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
            newScoringContexts.push_back(labelScorers_[scorerIdx]->extendedScoringContext(
                    {.context        = baseHyp.scoringContexts[scorerIdx],
                     .nextToken      = extension.nextToken,
                     .transitionType = extension.transitionType}));
        }
        newBeam_.push_back({baseHyp, extension, newScoringContexts, lengthNormScale_});
    }

    /*
     * For all hypotheses with the same scoring context keep only the best since they will
     * all develop in the same way.
     */
    recombination();

    numHypsAfterRecombination_ += newBeam_.size();

    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-recombination", newBeam_.size());
    }

    beamSizePruning(newBeam_, maxBeamSizes_[labelScorers_.size() - 1]);

    numHypsAfterBeamPruning_[labelScorers_.size() - 1] += newBeam_.size();

    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-beam-pruning", newBeam_.size());
    }

    auto numActive = numActiveHyps();
    numActiveHyps_ += numActive;
    numTerminatedHyps_ += newBeam_.size() - numActive;

    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-active-hyps", numActive);
        clog() << Core::XmlFull("num-terminated-hyps", newBeam_.size() - numActive);
    }

    beam_.swap(newBeam_);
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
        auto const* worstTerminatedHyp = getWorstTerminatedHypothesis();
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
    for (auto& stat : numHypsAfterScorePruning_) {
        stat.clear();
    }
    for (auto& stat : numHypsAfterBeamPruning_) {
        stat.clear();
    }
    numHypsAfterRecombination_.clear();
    numActiveHyps_.clear();
    numTerminatedHyps_.clear();
}

void LexiconfreeLabelsyncBeamSearch::logStatistics() const {
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
    numTerminatedHyps_.write(clog());
}

template<typename Element>
void LexiconfreeLabelsyncBeamSearch::beamSizePruning(std::vector<Element>& hypotheses, size_t maxBeamSize) const {
    if (hypotheses.size() <= maxBeamSize) {
        return;
    }

    // Reorder the hypotheses by associated score value such that the first `maxBeamSize_` elements are the best
    std::nth_element(hypotheses.begin(), hypotheses.begin() + maxBeamSize, hypotheses.end());
    hypotheses.resize(maxBeamSize);  // Get rid of excessive elements
}

template void LexiconfreeLabelsyncBeamSearch::beamSizePruning<LexiconfreeLabelsyncBeamSearch::ExtensionCandidate>(std::vector<LexiconfreeLabelsyncBeamSearch::ExtensionCandidate>&, size_t) const;
template void LexiconfreeLabelsyncBeamSearch::beamSizePruning<LexiconfreeLabelsyncBeamSearch::LabelHypothesis>(std::vector<LexiconfreeLabelsyncBeamSearch::LabelHypothesis>&, size_t) const;

void LexiconfreeLabelsyncBeamSearch::scorePruningExtensions(Score scoreThreshold) {
    if (extensions_.empty()) {
        return;
    }

    // Compute the pruning threshold
    auto bestScore        = std::min_element(extensions_.begin(), extensions_.end())->score;
    auto pruningThreshold = bestScore + scoreThreshold;

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
    auto pruningThreshold = (bestHyp.score + scoreThresholds_[labelScorers_.size() - 1]) / std::pow(bestHyp.length, lengthNormScale_);
    newBeam_.erase(
            std::remove_if(
                    newBeam_.begin(),
                    newBeam_.end(),
                    [&](auto const& hyp) { return hyp.scaledScore > pruningThreshold; }),
            newBeam_.end());
}

void LexiconfreeLabelsyncBeamSearch::recombination() {
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
    tempHypotheses_.reserve(newBeam_.size());

    // Map each unique ScoringContext in `newBeam_` to its hypothesis
    std::unordered_map<RecombinationContext, LabelHypothesis*, RecombinationContextHash> seenScoringContexts;
    for (auto const& hyp : newBeam_) {
        // Use try_emplace to check if the scoring context already exists and create a new entry if not at the same time
        auto [it, inserted] = seenScoringContexts.try_emplace({hyp}, nullptr);

        if (inserted) {
            // First time seeing this scoring context so move it over to `newHypotheses`
            tempHypotheses_.push_back(std::move(hyp));
            it->second = &tempHypotheses_.back();
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

    newBeam_.swap(tempHypotheses_);
}

void LexiconfreeLabelsyncBeamSearch::maximumStableDelayPruning() {
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

size_t LexiconfreeLabelsyncBeamSearch::numActiveHyps() const {
    return std::accumulate(
            newBeam_.begin(),
            newBeam_.end(),
            0ul,
            [](size_t acc, auto const& hyp) { return acc + static_cast<size_t>(hyp.isActive); });
}

}  // namespace Search
