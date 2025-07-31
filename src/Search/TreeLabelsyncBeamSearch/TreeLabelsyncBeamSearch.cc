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

#include "TreeLabelsyncBeamSearch.hh"

#include <algorithm>
#include <strings.h>

#include <Core/CollapsedVector.hh>
#include <Core/XmlStream.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/Module.hh>
#include <Search/Traceback.hh>

namespace Search {

/*
 * =======================
 * === LabelHypothesis ===
 * =======================
 */

TreeLabelsyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContext(),
          currentToken(Core::Type<Nn::LabelIndex>::max),
          currentState(invalidTreeNodeIndex),
          lmHistory(),
          length(0),
          score(0.0),
          scaledScore(0.0),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))),
          isActive(true) {}

TreeLabelsyncBeamSearch::LabelHypothesis::LabelHypothesis(
        TreeLabelsyncBeamSearch::LabelHypothesis const&    base,
        TreeLabelsyncBeamSearch::ExtensionCandidate const& extension,
        Nn::ScoringContextRef const&                       newScoringContext,
        float                                              lengthNormScale)
        : scoringContext(newScoringContext),
          currentToken(extension.nextToken),
          currentState(extension.nextState),
          lmHistory(base.lmHistory),
          length(base.length + 1),
          time(extension.timeframe),
          score(extension.score),
          scaledScore(score / std::pow(length, lengthNormScale)),
          trace(base.trace),
          isActive(extension.transitionType != Nn::LabelScorer::TransitionType::SENTENCE_END) {
}

TreeLabelsyncBeamSearch::LabelHypothesis::LabelHypothesis(
        TreeLabelsyncBeamSearch::LabelHypothesis const& base,
        StateId                                         rootState,
        Bliss::LemmaPronunciation const&                pron,
        Core::Ref<Lm::ScaledLanguageModel const> const& lm,
        float                                           lengthNormScale)
        : scoringContext(base.scoringContext),
          currentToken(base.currentToken),
          currentState(rootState),
          length(base.length),
          time(base.time),
          isActive(base.isActive) {
    auto const* lemma = pron.lemma();
    auto const& sts   = lemma->syntacticTokenSequence();

    auto lmScore = base.trace->score.lm;
    auto amScore = base.score - lmScore;

    lmHistory = base.lmHistory;
    for (auto const* st : sts) {
        lmScore += lm->score(lmHistory, st);
        lmHistory = lm->extendedHistory(lmHistory, st);
    }

    trace = Core::ref(new LatticeTrace(
            base.trace,
            &pron,
            base.time + 1,
            {amScore, lmScore},
            {}));

    score       = amScore + lmScore;
    scaledScore = score / std::pow(base.length, lengthNormScale);
}

std::string TreeLabelsyncBeamSearch::LabelHypothesis::toString() const {
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
 * === TreeLabelsyncBeamSearch ==
 * =====================================
 */

const Core::ParameterInt TreeLabelsyncBeamSearch::paramMaxBeamSize(
        "max-beam-size",
        "Maximum number of hypotheses in the search beam.",
        1, 1);

const Core::ParameterInt TreeLabelsyncBeamSearch::paramMaxWordEndBeamSize(
        "max-word-end-beam-size",
        "Maximum number of word-end hypotheses in the search beam.",
        std::numeric_limits<int>::max(), 0);

const Core::ParameterFloat TreeLabelsyncBeamSearch::paramScoreThreshold(
        "score-threshold",
        "Prune any hypotheses with a score that is at least this much worse than the best hypothesis."
        "If length normalization is enabled, the score threshold is added to the raw score before normalization."
        "If not set, no score pruning will be done.",
        Core::Type<Score>::max, 0);

const Core::ParameterFloat TreeLabelsyncBeamSearch::paramWordEndScoreThreshold(
        "word-end-score-threshold",
        "Prune any word-end hypothesis with a score that is at least this much worse than the best word-end hypothesis.\
        This value is relative to the score-threshold.",
        1.0, 0);

const Core::ParameterInt TreeLabelsyncBeamSearch::paramGlobalMaxBeamSize(
        "global-max-beam-size",
        "Maximum number of total terminated and active hypotheses. If at least this many terminated hypotheses exist, stop the search.\
        If `prune-active` is enabled, active hypotheses are pruned such that the total count is limited to this value, otherwise this is only used as a stopping condition.",
        Core::Type<int>::max, 1);

const Core::ParameterFloat TreeLabelsyncBeamSearch::paramGlobalScoreThreshold(
        "global-score-threshold",
        "Score threshold for terminated and active hypotheses. If no active hypothesis is better than the best terminated plus this threshold, stop the search.\
        This value is relative to the score-threshold.\
        If `prune-active` is enabled, all active hypotheses that do not fall under this threshold are pruned, otherwise this is only used as a stopping condition.",
        Core::Type<Score>::max, 0);

const Core::ParameterBool TreeLabelsyncBeamSearch::paramPruneActiveAgainstTerminated(
        "prune-active-against-terminated",
        "Prune active hypotheses against terminated ones based on `global-max-beam-size` and `global-score-threshold`. If false, these parameters are\
        only used as stopping conditions but no hypotheses are actually pruned.",
        false);

const Core::ParameterFloat TreeLabelsyncBeamSearch::paramLengthNormScale(
        "length-norm-scale",
        "Exponent of length for the hypothesis length normalization. Scaled scores are computed as score / length^length_norm_scale.",
        0.0);

const Core::ParameterFloat TreeLabelsyncBeamSearch::paramMaxLabelsPerTimestep(
        "max-labels-per-timestep",
        "Maximum number of emitted labels per input timestep counted via `addInput`/`addInputs`.",
        1.0);

const Core::ParameterBool TreeLabelsyncBeamSearch::paramSentenceEndFallBack(
        "sentence-end-fall-back",
        "Allow for fallback solution if no active word-end hypothesis exists at the end of a segment.",
        true);

const Core::ParameterBool TreeLabelsyncBeamSearch::paramLogStepwiseStatistics(
        "log-stepwise-statistics",
        "Log statistics about the beam at every search step.",
        false);

const Core::ParameterInt TreeLabelsyncBeamSearch::paramCacheCleanupInterval(
        "cache-cleanup-interval",
        "Interval of search steps after which buffered inputs that are not needed anymore get cleaned up.",
        10);

TreeLabelsyncBeamSearch::TreeLabelsyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          maxBeamSize_(paramMaxBeamSize(config)),
          maxWordEndBeamSize_(paramMaxWordEndBeamSize(config)),
          scoreThreshold_(paramScoreThreshold(config)),
          wordEndScoreThreshold_(paramWordEndScoreThreshold(config)),
          globalMaxBeamSize_(paramGlobalMaxBeamSize(config)),
          globalScoreThreshold_(paramGlobalScoreThreshold(config)),
          pruneActiveAgainstTerminated_(paramPruneActiveAgainstTerminated(config)),
          lengthNormScale_(paramLengthNormScale(config)),
          maxLabelsPerTimestep_(paramMaxLabelsPerTimestep(config)),
          sentenceEndLemma_(nullptr),
          sentenceEndFallback_(paramSentenceEndFallBack(config)),
          logStepwiseStatistics_(paramLogStepwiseStatistics(config)),
          cacheCleanupInterval_(paramCacheCleanupInterval(config)),
          debugChannel_(config, "debug"),
          labelScorer_(),
          beamActive_(),
          beamTerminated_(),
          extensions_(),
          requests_(),
          recombinedHypotheses_(),
          initializationTime_(),
          featureProcessingTime_(),
          scoringTime_(),
          contextExtensionTime_(),
          numHypsAfterScorePruning_("num-hyps-after-score-pruning"),
          numHypsAfterRecombination_("num-hyps-after-recombination"),
          numHypsAfterBeamPruning_("num-hyps-after-beam-pruning"),
          numWordEndHypsAfterScorePruning_("num-word-end-hyps-after-score-pruning"),
          numWordEndHypsAfterRecombination_("num-word-end-hyps-after-recombination"),
          numWordEndHypsAfterBeamPruning_("num-word-end-hyps-after-beam-pruning"),
          numActiveTrees_("num-active-trees"),
          numActiveHyps_("num-active-hyps"),
          numTerminatedHyps_("num-terminated-hyps"),
          currentSearchStep_(0ul),
          totalTimesteps_(0ul),
          finishedSegment_(false) {
    if (wordEndScoreThreshold_ != Core::Type<Score>::max) {
        if (scoreThreshold_ == Core::Type<Score>::max) {
            error() << "Word-end score-threshold is relative to score-threshold, but score-threshold is not set";
        }
        wordEndScoreThreshold_ *= scoreThreshold_;
    }

    if (globalScoreThreshold_ != Core::Type<Score>::max) {
        if (scoreThreshold_ == Core::Type<Score>::max) {
            error() << "Global score-threshold is relative to score-threshold, but score-threshold is not set";
        }
        globalScoreThreshold_ *= scoreThreshold_;
    }
}

Speech::ModelCombination::Mode TreeLabelsyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon | Speech::ModelCombination::useAcousticModel | Speech::ModelCombination::useLanguageModel;
}

Speech::ModelCombination::Mode TreeLabelsyncBeamSearch::requiredAcousticModel() const {
    return Am::AcousticModel::noEmissions;
}

bool TreeLabelsyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_       = modelCombination.lexicon();
    labelScorer_   = modelCombination.labelScorer();
    acousticModel_ = modelCombination.acousticModel();
    languageModel_ = modelCombination.languageModel();

    // Build the search tree
    log() << "Start building search tree";
    network_                                     = Core::ref(new PersistentStateTree(config, acousticModel_, lexicon_, std::bind(&Module_::createTreeBuilder, &Search::Module::instance(), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5)));
    std::unique_ptr<AbstractTreeBuilder> builder = Search::Module::instance().createTreeBuilder(config, *lexicon_, *acousticModel_, *network_);
    builder->build();
    log() << "Building finished";

    sentenceEndLemma_ = lexicon_->specialLemma("sentence-end");
    if (!sentenceEndLemma_) {
        sentenceEndLemma_ = lexicon_->specialLemma("sentence-boundary");
    }
    if (sentenceEndLemma_ == nullptr) {
        error() << "Could not find sentence-end lemma in the lexicon";
    }
    if (sentenceEndLemma_->nPronunciations() == 0) {
        error() << "Sentence-end lemma has no pronunciation so the sentence-end label cannot be determined";
    }
    auto const* sentenceEndPronunciation = sentenceEndLemma_->pronunciations().first->pronunciation();
    if (sentenceEndPronunciation->length() != 1) {
        error() << "Sentence-end lemma pronunciation must contain exactly one label, otherwise the sentence-end label cannot be determined";
    }
    sentenceEndLabelIndex_ = (*sentenceEndPronunciation)[0];
    log() << "Use sentence-end index " << sentenceEndLabelIndex_ << " inferred from lexicon";

    // Create look-ups for state successors and exits of each state
    createSuccessorLookups();

    reset();
    return true;
}

void TreeLabelsyncBeamSearch::reset() {
    initializationTime_.start();

    labelScorer_->reset();

    // Reset beam to a single empty hypothesis
    beamTerminated_.clear();
    beamActive_.clear();
    beamActive_.push_back(LabelHypothesis());
    beamActive_.front().scoringContext = labelScorer_->getInitialScoringContext();
    beamActive_.front().currentState   = network_->rootState;
    beamActive_.front().lmHistory      = languageModel_->startHistory();

    finishedSegment_   = false;
    totalTimesteps_    = 0ul;
    currentSearchStep_ = 0ul;

    initializationTime_.stop();
}

void TreeLabelsyncBeamSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    initializationTime_.start();
    labelScorer_->reset();
    resetStatistics();
    initializationTime_.stop();
    finishedSegment_   = false;
    totalTimesteps_    = 0ul;
    currentSearchStep_ = 0ul;
}

void TreeLabelsyncBeamSearch::finishSegment() {
    featureProcessingTime_.start();
    labelScorer_->signalNoMoreFeatures();
    featureProcessingTime_.stop();
    decodeManySteps();
    logStatistics();
    finishedSegment_ = true;
    finalize();
}

void TreeLabelsyncBeamSearch::putFeature(Nn::DataView const& feature) {
    featureProcessingTime_.start();
    labelScorer_->addInput(feature);
    ++totalTimesteps_;
    featureProcessingTime_.stop();
    finishedSegment_ = false;
}

void TreeLabelsyncBeamSearch::putFeatures(Nn::DataView const& features, size_t nTimesteps) {
    featureProcessingTime_.start();
    labelScorer_->addInputs(features, nTimesteps);
    totalTimesteps_ += nTimesteps;
    featureProcessingTime_.stop();
    finishedSegment_ = false;
}

Core::Ref<const Traceback> TreeLabelsyncBeamSearch::getCurrentBestTraceback() const {
    return getBestHypothesis().trace->performTraceback();
}

Core::Ref<const LatticeAdaptor> TreeLabelsyncBeamSearch::getCurrentBestWordLattice() const {
    auto& bestHypothesis = getBestHypothesis();

    LatticeTrace endTrace(bestHypothesis.trace, 0, bestHypothesis.trace->time + 1, bestHypothesis.trace->score, {});

    std::vector<LabelHypothesis> const& beam = bestHypothesis.isActive ? beamActive_ : beamTerminated_;
    for (auto const& hyp : beam) {
        if (hyp.isActive != bestHypothesis.isActive) {
            continue;
        }
        auto siblingTrace = Core::ref(new LatticeTrace(hyp.trace, 0, hyp.trace->time, hyp.trace->score, {}));
        endTrace.appendSiblingToChain(siblingTrace);
    }

    return endTrace.buildWordLattice(lexicon_);
}

bool TreeLabelsyncBeamSearch::decodeStep() {
    if (finishedSegment_) {
        return false;
    }
    if (currentSearchStep_ >= maxLabelsPerTimestep_ * std::max(totalTimesteps_, 1ul)) {
        warning() << "Terminated search due to reaching max number of labels";
        finishedSegment_ = true;
        return false;
    }

    /*
     * Within-word hypotheses
     */
    createExtensions();

    if (requests_.empty()) {
        finishedSegment_ = true;
        return false;
    }

    if (not scoreExtensions()) {
        return false;
    }

    if (logStepwiseStatistics_) {
        clog() << Core::XmlOpen("search-step-stats");
    }

    scorePruningExtensions();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-score-pruning", extensions_.size());
    }
    numHypsAfterScorePruning_ += extensions_.size();

    createWithinWordHypothesesFromExtensions();

    recombination(withinWordHypotheses_, false);
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-recombination", withinWordHypotheses_.size());
    }

    beamSizePruning(withinWordHypotheses_, maxBeamSize_);
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-beam-pruning", withinWordHypotheses_.size());
    }

    /*
     * Word-end hypotheses
     */
    createWordEndHypotheses();

    scorePruningWordEnds();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-word-end-hyps-after-score-pruning", wordEndHypotheses_.size());
    }

    recombination(wordEndHypotheses_, true);
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-word-end-hyps-after-recombination", wordEndHypotheses_.size());
    }

    beamSizePruning(wordEndHypotheses_, maxWordEndBeamSize_);
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-word-end-hyps-after-beam-pruning", wordEndHypotheses_.size());
    }

    /*
     * New beam preparation and global comparison (+ optionally pruning)
     */
    createNewBeam();

    pruneActiveAgainstTerminatedByScore();
    pruneActiveAgainstTerminatedByLimit();

    finishedSegment_ = stopCriterion();

    /*
     * Logging and statistics
     */
    std::vector<Lm::History> seenHistories;
    for (auto const& hyp : beamActive_) {
        if (std::find(seenHistories.begin(), seenHistories.end(), hyp.lmHistory) == seenHistories.end()) {
            seenHistories.push_back(hyp.lmHistory);
        }
    }
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-active-trees", seenHistories.size());
    }
    numActiveTrees_ += seenHistories.size();

    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-terminated-hyps", beamTerminated_.size());
        clog() << Core::XmlFull("num-active-hyps", beamActive_.size());
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
    numActiveHyps_ += beamActive_.size();
    numTerminatedHyps_ += beamTerminated_.size();

    if (debugChannel_.isOpen()) {
        std::stringstream ss;
        for (size_t hypIdx = 0ul; hypIdx < beamTerminated_.size(); ++hypIdx) {
            ss << "Terminated hypothesis " << hypIdx + 1ul << ":  " << beamTerminated_[hypIdx].toString() << "\n";
        }
        ss << "\n";

        for (size_t hypIdx = 0ul; hypIdx < beamActive_.size(); ++hypIdx) {
            ss << "Active hypothesis " << hypIdx + 1ul << ":  " << beamActive_[hypIdx].toString() << "\n";
        }
        ss << "\n";
        debugChannel_ << ss.str();
    }

    /*
     * Clean up label scorer caches.
     */
    if (++currentSearchStep_ % cacheCleanupInterval_ == 0) {
        Core::CollapsedVector<Nn::ScoringContextRef> activeContexts;
        for (auto const& hyp : beamActive_) {
            activeContexts.push_back(hyp.scoringContext);
        }
        labelScorer_->cleanupCaches(activeContexts);
    }

    return true;
}

TreeLabelsyncBeamSearch::LabelHypothesis const* TreeLabelsyncBeamSearch::getBestTerminatedHypothesis() const {
    LabelHypothesis const* best = nullptr;

    for (auto const& hyp : beamTerminated_) {
        if (best == nullptr or hyp < *best) {
            best = &hyp;
        }
    }

    return best;
}

TreeLabelsyncBeamSearch::LabelHypothesis const* TreeLabelsyncBeamSearch::getWorstTerminatedHypothesis() const {
    LabelHypothesis const* worst = nullptr;

    for (auto const& hyp : beamTerminated_) {
        if (worst == nullptr or *worst < hyp) {
            worst = &hyp;
        }
    }

    return worst;
}

TreeLabelsyncBeamSearch::LabelHypothesis const* TreeLabelsyncBeamSearch::getBestActiveHypothesis() const {
    LabelHypothesis const* best = nullptr;

    for (auto const& hyp : beamActive_) {
        if (best == nullptr or hyp < *best) {
            best = &hyp;
        }
    }

    return best;
}

TreeLabelsyncBeamSearch::LabelHypothesis const* TreeLabelsyncBeamSearch::getWorstActiveHypothesis() const {
    LabelHypothesis const* worst = nullptr;

    for (auto const& hyp : beamActive_) {
        if (worst == nullptr or *worst < hyp) {
            worst = &hyp;
        }
    }

    return worst;
}

TreeLabelsyncBeamSearch::LabelHypothesis const& TreeLabelsyncBeamSearch::getBestHypothesis() const {
    auto const* result = getBestTerminatedHypothesis();
    if (result != nullptr) {
        return *result;
    }
    result = getBestActiveHypothesis();
    verify(result != nullptr);
    return *result;
}

TreeLabelsyncBeamSearch::LabelHypothesis const& TreeLabelsyncBeamSearch::getWorstHypothesis() const {
    auto const* result = getWorstTerminatedHypothesis();
    if (result != nullptr) {
        return *result;
    }
    result = getWorstActiveHypothesis();
    verify(result != nullptr);
    return *result;
}

void TreeLabelsyncBeamSearch::resetStatistics() {
    initializationTime_.reset();
    featureProcessingTime_.reset();
    scoringTime_.reset();
    contextExtensionTime_.reset();
    numHypsAfterScorePruning_.clear();
    numHypsAfterRecombination_.clear();
    numHypsAfterBeamPruning_.clear();
    numWordEndHypsAfterScorePruning_.clear();
    numWordEndHypsAfterRecombination_.clear();
    numWordEndHypsAfterBeamPruning_.clear();
    numActiveTrees_.clear();
    numActiveHyps_.clear();
    numTerminatedHyps_.clear();
}

void TreeLabelsyncBeamSearch::logStatistics() const {
    clog() << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    clog() << Core::XmlOpen("initialization-time") << initializationTime_.elapsedMilliseconds() << Core::XmlClose("initialization-time");
    clog() << Core::XmlOpen("feature-processing-time") << featureProcessingTime_.elapsedMilliseconds() << Core::XmlClose("feature-processing-time");
    clog() << Core::XmlOpen("scoring-time") << scoringTime_.elapsedMilliseconds() << Core::XmlClose("scoring-time");
    clog() << Core::XmlOpen("context-extension-time") << contextExtensionTime_.elapsedMilliseconds() << Core::XmlClose("context-extension-time");
    clog() << Core::XmlClose("timing-statistics");
    numHypsAfterScorePruning_.write(clog());
    numHypsAfterRecombination_.write(clog());
    numHypsAfterBeamPruning_.write(clog());
    numWordEndHypsAfterScorePruning_.write(clog());
    numWordEndHypsAfterRecombination_.write(clog());
    numWordEndHypsAfterBeamPruning_.write(clog());
    numActiveTrees_.write(clog());
    numActiveHyps_.write(clog());
    numTerminatedHyps_.write(clog());
}

void TreeLabelsyncBeamSearch::beamSizePruning(std::vector<LabelHypothesis>& hyps, size_t maxBeamSize) {
    if (hyps.size() <= maxBeamSize_) {
        return;
    }

    // Reorder the hypotheses by associated score value such that the first `beamSizeTerminated_` elements are the best
    std::nth_element(hyps.begin(), hyps.begin() + maxBeamSize, hyps.end());
    hyps.resize(maxBeamSize);  // Get rid of excessive elements
}

void TreeLabelsyncBeamSearch::scorePruningExtensions() {
    if (extensions_.empty() or scoreThreshold_ == Core::Type<Score>::max) {
        return;
    }

    // Compute the pruning threshold
    // Extensions all have the same length so we can compare absolute scores
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

void TreeLabelsyncBeamSearch::scorePruningWordEnds() {
    if (wordEndHypotheses_.empty() or wordEndScoreThreshold_ == Core::Type<Score>::max) {
        return;
    }

    // Compute the pruning threshold
    auto bestHyp = *std::min_element(wordEndHypotheses_.begin(), wordEndHypotheses_.end());

    // Remove elements with score > pruningThreshold
    auto pruningThreshold = (bestHyp.score + wordEndScoreThreshold_) / std::pow(bestHyp.length, lengthNormScale_);
    wordEndHypotheses_.erase(
            std::remove_if(
                    wordEndHypotheses_.begin(),
                    wordEndHypotheses_.end(),
                    [&](auto const& hyp) { return hyp.scaledScore > pruningThreshold; }),
            wordEndHypotheses_.end());
}

void TreeLabelsyncBeamSearch::createSuccessorLookups() {
    stateSuccessorLookup_.resize(network_->structure.stateCount());
    exitLookup_.resize(network_->structure.stateCount());

    for (u32 state = 1; state < network_->structure.stateCount(); ++state) {
        std::vector<StateId>                   stateList;  // Collect the state successors of all nodes
        std::vector<PersistentStateTree::Exit> exitList;   // Collect the exits of all nodes
        for (HMMStateNetwork::SuccessorIterator it = network_->structure.successors(state); it; ++it) {
            if (not it.isLabel()) {
                stateList.push_back(*it);
            }
            else {
                exitList.push_back(network_->exits[it.label()]);
            }
        }
        stateSuccessorLookup_[state] = stateList;
        exitLookup_[state]           = exitList;
    }
}

void TreeLabelsyncBeamSearch::finalize() {
    if (not beamTerminated_.empty()) {
        return;
    }

    warning("No active word-end hypothesis at segment end.");

    if (beamActive_.empty()) {  // There was no active or terminated hypothesis in the beam
        warning("No active word-end hypothesis at segment end.");
    }

    if (sentenceEndFallback_) {
        log() << "Use sentence-end fallback";

        for (auto& hyp : beamActive_) {
            auto sentenceEndScore = languageModel_->sentenceEndScore(hyp.lmHistory);
            hyp.score += sentenceEndScore;
            hyp.isActive = false;
            beamTerminated_.push_back(std::move(hyp));
        }
    }
    else {
        // Construct an empty hypothesis with a lattice containing only one empty pronunciation from start to end
        beamTerminated_.push_back(LabelHypothesis());
        beamTerminated_.front().trace->time          = beamActive_.empty() ? 0 : beamActive_.front().trace->time;  // Retrieve the timeframe from any hyp in the old beam
        beamTerminated_.front().trace->pronunciation = nullptr;
        beamTerminated_.front().trace->predecessor   = Core::ref(new LatticeTrace(0, {0, 0}, {}));
    }
}

void TreeLabelsyncBeamSearch::createExtensions() {
    extensions_.clear();
    requests_.clear();

    for (size_t hypIndex = 0ul; hypIndex < beamActive_.size(); ++hypIndex) {
        auto& hyp = beamActive_[hypIndex];

        // Iterate over the successors of this hypothesis' current state in the tree
        for (const auto& successorState : stateSuccessorLookup_[hyp.currentState]) {
            Nn::LabelIndex tokenIdx = network_->structure.state(successorState).stateDesc.acousticModel;

            auto transitionType = Nn::LabelScorer::TransitionType::LABEL_TO_LABEL;
            if (hyp.currentToken == Core::Type<Nn::LabelIndex>::max) {
                transitionType = Nn::LabelScorer::TransitionType::INITIAL_LABEL;
            }
            if (tokenIdx == sentenceEndLabelIndex_) {
                transitionType = Nn::LabelScorer::TransitionType::SENTENCE_END;
            }

            extensions_.push_back(
                    {tokenIdx,
                     successorState,
                     hyp.score,
                     0,
                     transitionType,
                     hypIndex});
            requests_.push_back({beamActive_[hypIndex].scoringContext, tokenIdx, transitionType});
        }
    }
}

bool TreeLabelsyncBeamSearch::scoreExtensions() {
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

    return true;
}

void TreeLabelsyncBeamSearch::createWithinWordHypothesesFromExtensions() {
    withinWordHypotheses_.clear();
    for (auto const& extension : extensions_) {
        auto const& baseHyp = beamActive_[extension.baseHypIndex];

        auto newScoringContext = labelScorer_->extendedScoringContext(
                {baseHyp.scoringContext,
                 extension.nextToken,
                 extension.transitionType});
        withinWordHypotheses_.push_back({baseHyp, extension, newScoringContext, lengthNormScale_});
    }
}

void TreeLabelsyncBeamSearch::recombination(std::vector<LabelHypothesis>& hyps, bool createTraceSiblings) {
    recombinedHypotheses_.clear();
    recombinedHypotheses_.reserve(hyps.size());  // We need to reserve enough space to avoid pointer invalidation

    // Map each unique recombination context in `withinWordHypotheses_` to its best hypothesis
    std::unordered_map<RecombinationContext, LabelHypothesis*, RecombinationContextHash> seenRecombinationContexts;

    for (auto const& hyp : hyps) {
        // Use try_emplace to check if the combination already exists and create a new entry if not at the same time
        auto [it, inserted] = seenRecombinationContexts.try_emplace({hyp}, nullptr);

        if (inserted) {
            // First time seeing this combination so move it over to `newHypotheses`
            recombinedHypotheses_.push_back(std::move(hyp));
            it->second = &recombinedHypotheses_.back();
        }
        else {
            verify(not hyp.trace->sibling);

            auto* existingHyp = it->second;
            if (hyp.score < existingHyp->score) {
                // New hyp is better -> replace in `newHypotheses` and optionally add existing one as sibling
                if (createTraceSiblings) {
                    hyp.trace->sibling = existingHyp->trace;
                }
                *existingHyp = std::move(hyp);  // Overwrite in-place
            }
            else if (createTraceSiblings) {
                // New hyp is worse -> add to existing one as sibling
                hyp.trace->sibling          = existingHyp->trace->sibling;
                existingHyp->trace->sibling = hyp.trace;
            }
        }
    }

    hyps.swap(recombinedHypotheses_);
}

void TreeLabelsyncBeamSearch::createWordEndHypotheses() {
    wordEndHypotheses_.clear();

    for (auto const& hyp : withinWordHypotheses_) {
        auto const& exitList = exitLookup_[hyp.currentState];
        // Create one word-end hypothesis for each exit
        for (auto const& exit : exitList) {
            auto        rootState = exit.transitState;
            auto const* lemmaPron = lexicon_->lemmaPronunciation(exit.pronunciation);
            wordEndHypotheses_.push_back({hyp, rootState, *lemmaPron, languageModel_, lengthNormScale_});
        }
    }

    // All previously terminated hypotheses are also word-end hypotheses
    for (auto const& hyp : beamTerminated_) {
        wordEndHypotheses_.push_back(std::move(hyp));
    }
}

void TreeLabelsyncBeamSearch::createNewBeam() {
    beamActive_.clear();
    beamTerminated_.clear();

    for (auto const& hyp : withinWordHypotheses_) {
        if (hyp.isActive) {
            beamActive_.push_back(std::move(hyp));
        }
        // For terminated hypotheses we don't need the "within-word" version any more. Only the "word-end" version is kept.
    }

    for (auto const& hyp : wordEndHypotheses_) {
        if (hyp.isActive) {
            beamActive_.push_back(std::move(hyp));
        }
        else {
            beamTerminated_.push_back(std::move(hyp));
        }
    }
}

void TreeLabelsyncBeamSearch::pruneActiveAgainstTerminatedByScore() {
    if (not pruneActiveAgainstTerminated_) {
        return;
    }
    if (globalScoreThreshold_ == Core::Type<Score>::max) {
        return;
    }
    if (beamActive_.empty()) {
        return;
    }

    auto const* bestTerminatedHyp = getBestTerminatedHypothesis();
    if (bestTerminatedHyp == nullptr) {
        return;
    }

    auto pruningThreshold = (bestTerminatedHyp->score + globalScoreThreshold_) / std::pow(bestTerminatedHyp->length, lengthNormScale_);
    beamActive_.erase(
            std::remove_if(
                    beamActive_.begin(),
                    beamActive_.end(),
                    [&](auto const& hyp) { return hyp.scaledScore > pruningThreshold; }),
            beamActive_.end());
}

void TreeLabelsyncBeamSearch::pruneActiveAgainstTerminatedByLimit() {
    if (not pruneActiveAgainstTerminated_) {
        return;
    }
    if (beamActive_.empty()) {
        return;
    }
    if (beamTerminated_.size() >= globalMaxBeamSize_) {
        beamActive_.clear();
        return;
    }
    if (beamTerminated_.size() + beamActive_.size() <= globalMaxBeamSize_) {
        return;
    }

    size_t limit = globalMaxBeamSize_ - beamTerminated_.size();
    std::nth_element(beamActive_.begin(), beamActive_.begin() + limit, beamActive_.end());
    beamActive_.resize(limit);  // Get rid of excessive elements
}

bool TreeLabelsyncBeamSearch::stopCriterion() {
    if (beamActive_.empty()) {
        return true;
    }

    if (beamTerminated_.size() >= globalMaxBeamSize_) {
        return true;
    }

    auto const* bestTerminatedHyp = getBestTerminatedHypothesis();
    if (bestTerminatedHyp == nullptr) {
        return false;
    }

    if (globalScoreThreshold_ != Core::Type<Score>::max) {
        auto threshold = (bestTerminatedHyp->score + globalScoreThreshold_) / std::pow(bestTerminatedHyp->length, lengthNormScale_);
        if (std::all_of(beamActive_.begin(), beamActive_.end(), [&](auto const& hyp) { return hyp.scaledScore > threshold; })) {
            return true;
        }
    }

    return false;
}

}  // namespace Search
