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

#include "TreeTimesyncBeamSearch.hh"

#include <algorithm>
#include <strings.h>

#include <Core/CollapsedVector.hh>
#include <Core/XmlStream.hh>
#include <Lattice/LatticeAdaptor.hh>
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

TreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContext(),
          currentToken(Nn::invalidLabelIndex),
          currentState(invalidTreeNodeIndex),
          lmHistory(),
          score(0.0),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))) {}

TreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        TreeTimesyncBeamSearch::LabelHypothesis const&              base,
        TreeTimesyncBeamSearch::WithinWordExtensionCandidate const& extension,
        Nn::ScoringContextRef const&                                newScoringContext)
        : scoringContext(newScoringContext),
          currentToken(extension.nextToken),
          currentState(extension.nextState),
          lmHistory(base.lmHistory),
          timeframe(extension.timeframe),
          score(extension.score),
          trace(base.trace),
          recentTransitionType(extension.transitionType) {
}

TreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        LabelHypothesis const&                                   base,
        TreeTimesyncBeamSearch::WordEndExtensionCandidate const& extension,
        Lm::History const&                                       newLmHistory)
        : scoringContext(base.scoringContext),
          currentToken(base.currentToken),
          currentState(extension.rootState),
          lmHistory(newLmHistory),
          timeframe(base.timeframe),
          score(extension.score),
          recentTransitionType(base.recentTransitionType) {
    auto newLmScore   = score - base.score;
    auto totalLmScore = base.trace->score.lm + newLmScore;
    auto totalAmScore = score - totalLmScore;

    // Create a successor trace item from base
    trace = Core::ref(new LatticeTrace(
            base.trace,
            extension.pron,
            timeframe + 1,
            {totalAmScore, totalLmScore},
            {}));
}

std::string TreeTimesyncBeamSearch::LabelHypothesis::toString() const {
    std::stringstream ss;
    ss << "Score: " << score << ", current token: " << currentToken << ", current state: " << currentState << ", traceback: ";

    auto traceback = trace->performTraceback();

    for (auto& item : *traceback) {
        if (item.pronunciation and item.pronunciation->lemma()) {
            ss << item.pronunciation->lemma()->symbol() << " ";
        }
    }
    return ss.str();
}

/*
 * ==============================
 * === TreeTimesyncBeamSearch ===
 * ==============================
 */

const Core::ParameterInt TreeTimesyncBeamSearch::paramMaxBeamSize(
        "max-beam-size",
        "Maximum number of within-word hypotheses in the search beam.",
        1, 1);

const Core::ParameterInt TreeTimesyncBeamSearch::paramMaxWordEndBeamSize(
        "max-word-end-beam-size",
        "Maximum number of word-end hypotheses in the search beam. If not set, global beam pruning will be done and word-end hypotheses will not be pruned separately.",
        std::numeric_limits<int>::max(), 0);

const Core::ParameterFloat TreeTimesyncBeamSearch::paramScoreThreshold(
        "score-threshold",
        "Prune any within-word hypothesis with a score that is at least this much worse than the best hypothesis.",
        Core::Type<Score>::max, 0);

const Core::ParameterFloat TreeTimesyncBeamSearch::paramIntermediateScoreThreshold(
        "intermediate-score-threshold",
        "Prune any intermediate hypotheses of sub-scorers with a score that is at least this much worse than the best hypothesis. If not set, no intermediate score pruning will be done.",
        Core::Type<Score>::max, 0);

const Core::ParameterInt TreeTimesyncBeamSearch::paramIntermediateMaxBeamSize(
        "intermediate-max-beam-size",
        "",
        Core::Type<Score>::max, 0);

const Core::ParameterFloat TreeTimesyncBeamSearch::paramWordEndScoreThreshold(
        "word-end-score-threshold",
        "Prune any word-end hypothesis with a score that is at least this much worse than the best word-end hypothesis. This threshold is relative to the score-threshold. \
        If not set, global score pruning will be done and word-end hypotheses will not be pruned separately.",
        Core::Type<Score>::max, 0);

const Core::ParameterBool TreeTimesyncBeamSearch::paramCollapseRepeatedLabels(
        "collapse-repeated-labels",
        "Collapse repeated emission of the same label into one output. If false, every emission is treated like a new output.",
        false);

const Core::ParameterBool TreeTimesyncBeamSearch::paramSentenceEndFallBack(
        "sentence-end-fall-back",
        "Allow for fallback solution if no active word-end hypothesis exists at the end of a segment.",
        true);

const Core::ParameterBool TreeTimesyncBeamSearch::paramLogStepwiseStatistics(
        "log-stepwise-statistics",
        "Log statistics about the beam at every search step.",
        false);

const Core::ParameterBool TreeTimesyncBeamSearch::paramCacheCleanupInterval(
        "cache-cleanup-interval",
        "Interval of search steps after which buffered inputs that are not needed anymore get cleaned up.",
        10);

const Core::ParameterInt TreeTimesyncBeamSearch::paramMaximumStableDelay(
        "maximum-stable-delay",
        "Introduce a cutoff point at `current-time` - `delay`. Every hypothesis that disagrees with the current best anywhere before the cutoff gets pruned. This way words in the traceback become stable after at most `delay` frames."
        "maximum number of frames before results are required to be stable",
        Core::Type<int>::max, 0);

TreeTimesyncBeamSearch::TreeTimesyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          maxBeamSize_(paramMaxBeamSize(config)),
          intermediateMaxBeamSize_(paramIntermediateMaxBeamSize(config)),
          maxWordEndBeamSize_(paramMaxWordEndBeamSize(config)),
          scoreThreshold_(paramScoreThreshold(config)),
          intermediateScoreThreshold_(paramIntermediateScoreThreshold(config)),
          wordEndScoreThreshold_(paramWordEndScoreThreshold(config)),
          cacheCleanupInterval_(paramCacheCleanupInterval(config)),
          useBlank_(),
          collapseRepeatedLabels_(paramCollapseRepeatedLabels(config)),
          sentenceEndFallback_(paramSentenceEndFallBack(config)),
          logStepwiseStatistics_(paramLogStepwiseStatistics(config)),
          labelScorer_(),
          debugChannel_(config, "debug"),
          withinWordExtensions_(),
          beam_(),
          newBeam_(),
          wordEndHypotheses_(),
          requests_(),
          tempHypotheses_(),
          currentSearchStep_(0ul),
          finishedSegment_(false),
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
          numActiveHyps_("num-active-hyps"),
          numActiveTrees_("num-active-trees"),
          stableTraceTracker_(),
          canUpdateStablePrefix_(false),
          maximumStableDelay_(paramMaximumStableDelay(config)) {
    if (scoreThreshold_ == Core::Type<Score>::max and wordEndScoreThreshold_ != Core::Type<Score>::max) {
        error() << "Word-end score-threshold which is relative to the score-threshold is set, but score-threshold is not set";
    }
    useIntermediateScorePruning_ = intermediateScoreThreshold_ != Core::Type<Score>::max;
    useIntermediateBeamPruning_  = intermediateMaxBeamSize_ != Core::Type<int>::max;
    wordEndScoreThreshold_ *= scoreThreshold_;
}

Speech::ModelCombination::Mode TreeTimesyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon | Speech::ModelCombination::useAcousticModel | Speech::ModelCombination::useLanguageModel;
}

Am::AcousticModel::Mode TreeTimesyncBeamSearch::requiredAcousticModel() const {
    return Am::AcousticModel::noEmissions;
}

bool TreeTimesyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_       = modelCombination.lexicon();
    labelScorer_   = modelCombination.labelScorer();
    acousticModel_ = modelCombination.acousticModel();
    languageModel_ = modelCombination.languageModel();

    // Build the search tree
    log() << "Start building search tree";
    network_ = Core::ref(new PersistentStateTree(
            config,
            acousticModel_,
            lexicon_,
            std::bind(
                    &Module_::createTreeBuilder,
                    &Search::Module::instance(),
                    std::placeholders::_1,
                    std::placeholders::_2,
                    std::placeholders::_3,
                    std::placeholders::_4,
                    std::placeholders::_5)));

    std::unique_ptr<AbstractTreeBuilder> builder = Search::Module::instance().createTreeBuilder(config, *lexicon_, *acousticModel_, *network_);
    builder->build();

    if (lexicon_->specialLemma("blank")) {
        blankLabelIndex_ = acousticModel_->emissionIndex(acousticModel_->blankAllophoneStateIndex());
        useBlank_        = true;
        log() << "Use blank label with index " << blankLabelIndex_;
    }
    else {
        blankLabelIndex_ = Nn::invalidLabelIndex;
        useBlank_        = false;
    }

    for (const auto& lemma : {"silence", "blank"}) {
        if (lexicon_->specialLemma(lemma) and (lexicon_->specialLemma(lemma)->syntacticTokenSequence()).size() != 0) {
            warning("Special lemma \"%s\" will be scored by the language model. To prevent the LM from scoring it, set an empty syntactic token sequence for it in the lexicon.", lemma);
        }
    }

    // Create look-ups for state successors and exits of each state
    createSuccessorLookups();

    reset();
    return true;
}

void TreeTimesyncBeamSearch::reset() {
    initializationTime_.start();

    labelScorer_->reset();

    // Reset beam to a single empty hypothesis
    beam_.clear();
    beam_.push_back(LabelHypothesis());
    beam_.front().scoringContext = labelScorer_->getInitialScoringContext();
    beam_.front().currentState   = network_->rootState;
    beam_.front().lmHistory      = languageModel_->startHistory();

    stableTraceTracker_.setTrace(beam_.front().trace);
    canUpdateStablePrefix_ = false;

    currentSearchStep_ = 0ul;
    finishedSegment_   = false;

    initializationTime_.stop();
}

void TreeTimesyncBeamSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    initializationTime_.start();
    labelScorer_->reset();
    if (segment != nullptr) {
        languageModel_->setSegment(segment);
        for (auto& hyp : beam_) {
            hyp.lmHistory = languageModel_->startHistory();
        }
    }
    resetStatistics();
    initializationTime_.stop();
    finishedSegment_ = false;
}

void TreeTimesyncBeamSearch::finishSegment() {
    featureProcessingTime_.start();
    labelScorer_->signalNoMoreFeatures();
    featureProcessingTime_.stop();
    decodeManySteps();
    logStatistics();
    finishedSegment_ = true;
    finalizeLmScoring();
}

void TreeTimesyncBeamSearch::putFeature(Nn::DataView const& feature) {
    featureProcessingTime_.start();
    labelScorer_->addInput(feature);
    featureProcessingTime_.stop();
}

void TreeTimesyncBeamSearch::putFeatures(Nn::DataView const& features, size_t nTimesteps) {
    featureProcessingTime_.start();
    labelScorer_->addInputs(features, nTimesteps);
    featureProcessingTime_.stop();
}

Core::Ref<const Traceback> TreeTimesyncBeamSearch::getCurrentBestTraceback() const {
    return getBestHypothesis().trace->performTraceback();
}

Core::Ref<const Traceback> TreeTimesyncBeamSearch::getCurrentStableTraceback() {
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

Core::Ref<const LatticeAdaptor> TreeTimesyncBeamSearch::getCurrentBestWordLattice() const {
    auto&        bestHypothesis = getBestHypothesis();
    LatticeTrace endTrace(bestHypothesis.trace, 0, bestHypothesis.trace->time + 1, bestHypothesis.trace->score, {});

    for (size_t hypIdx = 1ul; hypIdx < beam_.size(); ++hypIdx) {
        auto& hyp          = beam_[hypIdx];
        auto  siblingTrace = Core::ref(new LatticeTrace(hyp.trace, 0, hyp.trace->time, hyp.trace->score, {}));
        endTrace.appendSiblingToChain(siblingTrace);
    }

    return endTrace.buildWordLattice(lexicon_);
}

bool TreeTimesyncBeamSearch::decodeStep() {
    if (finishedSegment_) {
        return false;
    }

    /*
     * Collect all possible within-word extensions for all hypotheses in the beam.
     * Also create scoring requests for the label scorer.
     * Each extension candidate makes up a request.
     */
    withinWordExtensions_.clear();

    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        auto& hyp = beam_[hypIndex];

        // Iterate over the successors of this hypothesis' current state in the tree
        for (const auto& successorState : stateSuccessorLookup_[hyp.currentState]) {
            Nn::LabelIndex tokenIdx = network_->structure.state(successorState).stateDesc.acousticModel;
            // If we collapse repeated labels, a new word should not start with the same token as the previous word ended (except for blank itself)
            if (collapseRepeatedLabels_ and
                hyp.currentState == network_->rootState and
                tokenIdx == hyp.currentToken and
                (not useBlank_ or tokenIdx != blankLabelIndex_)) {
                continue;
            }
            auto transitionType = inferTransitionType(hyp.currentToken, tokenIdx);
            withinWordExtensions_.push_back(
                    {tokenIdx,
                     successorState,
                     hyp.timeframe,
                     hyp.score,
                     transitionType,
                     hypIndex});
        }
    }

    /*
     * Perform scoring of all the requests with the label scorer.
     */
    if (logStepwiseStatistics_) {
        clog() << Core::XmlOpen("search-step-stats");
    }
    for (size_t subScorerIdx = 0ul; subScorerIdx < labelScorer_->numSubScorers(); ++subScorerIdx) {
        requests_.clear();
        for (auto const& extension : withinWordExtensions_) {
            requests_.push_back({beam_[extension.baseHypIndex].scoringContext, extension.nextToken, extension.transitionType});
        }
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

        for (size_t requestIdx = 0ul; requestIdx < withinWordExtensions_.size(); ++requestIdx) {
            withinWordExtensions_[requestIdx].score += result->scores[requestIdx];
            withinWordExtensions_[requestIdx].timeframe = result->timeframes[requestIdx];
        }

        if (subScorerIdx + 1 < labelScorer_->numSubScorers()) {
            if (useIntermediateScorePruning_) {
                scorePruning(withinWordExtensions_, intermediateScoreThreshold_);

                if (logStepwiseStatistics_) {
                    clog() << Core::XmlFull("num-hyps-after-intermediate-score-pruning-" + std::to_string(subScorerIdx), withinWordExtensions_.size());
                }
            }

            if (useIntermediateBeamPruning_) {
                beamSizePruning(withinWordExtensions_, intermediateMaxBeamSize_);

                if (logStepwiseStatistics_) {
                    clog() << Core::XmlFull("num-hyps-after-intermediate-beam-pruning-" + std::to_string(subScorerIdx), withinWordExtensions_.size());
                }
            }
        }
    }

    /*
     * Prune set of possible within-word extensions by max beam size and possibly also by score.
     */
    scorePruning(withinWordExtensions_, scoreThreshold_);
    numHypsAfterScorePruning_ += withinWordExtensions_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-score-pruning", withinWordExtensions_.size());
    }

    // Create new label hypotheses from extension candidates
    newBeam_.clear();
    for (auto const& extension : withinWordExtensions_) {
        auto const& baseHyp = beam_[extension.baseHypIndex];

        auto newScoringContext = labelScorer_->extendedScoringContext(
                {baseHyp.scoringContext,
                 extension.nextToken,
                 extension.transitionType});

        newBeam_.push_back({baseHyp, extension, newScoringContext});
    }

    // For all hypotheses at the same state and with the same scoring context and LM history
    // keep only the best since they will all develop in the same way
    recombination(newBeam_, false);
    numHypsAfterRecombination_ += newBeam_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-recombination", newBeam_.size());
    }

    beamSizePruning(newBeam_, maxBeamSize_);
    numHypsAfterBeamPruning_ += newBeam_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-beam-pruning", newBeam_.size());
    }

    /*
     * Expand hypotheses to word-end hypotheses and incorporate the language model
     */
    wordEndExtensions_.clear();
    for (size_t hypIndex = 0ul; hypIndex < newBeam_.size(); ++hypIndex) {
        auto const& hyp = newBeam_[hypIndex];

        auto const& exitList = exitLookup_[hyp.currentState];
        if (not exitList.empty()) {
            // Create one word-end hypothesis for each exit
            for (auto const& exit : exitList) {
                auto        rootState = exit.transitState;
                auto const* lemmaPron = lexicon_->lemmaPronunciation(exit.pronunciation);
                auto const* lemma     = lemmaPron->lemma();

                Score       lmScore = 0;
                auto const& sts     = lemma->syntacticTokenSequence();
                if (sts.size() != 0) {
                    require(sts.size() == 1);
                    auto const* st = sts.front();
                    lmScore        = languageModel_->score(hyp.lmHistory, st);
                    wordEndExtensions_.push_back({lemmaPron, rootState, hyp.score + lmScore, hypIndex});
                }
            }
        }
    }

    /*
     * Prune set of word-end extensions by max beam size and possibly also by score.
     */
    scorePruning(wordEndExtensions_, wordEndScoreThreshold_);
    numWordEndHypsAfterScorePruning_ += wordEndExtensions_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-word-end-hyps-after-score-pruning", wordEndExtensions_.size());
    }

    // Create new word-end label hypotheses from word-end extension candidates and update the LM history
    wordEndHypotheses_.clear();
    for (auto& extension : wordEndExtensions_) {
        auto const& baseHyp = newBeam_[extension.baseHypIndex];

        auto const* lemma        = extension.pron->lemma();
        auto        newLmHistory = baseHyp.lmHistory;
        auto const& sts          = lemma->syntacticTokenSequence();
        if (sts.size() != 0) {
            require(sts.size() == 1);
            auto const* st = sts.front();
            newLmHistory   = languageModel_->extendedHistory(newLmHistory, st);
        }

        wordEndHypotheses_.push_back({baseHyp, extension, newLmHistory});
    }

    recombination(wordEndHypotheses_, true);
    numWordEndHypsAfterRecombination_ += wordEndHypotheses_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-word-end-hyps-after-recombination", wordEndHypotheses_.size());
    }

    beamSizePruning(wordEndHypotheses_, maxWordEndBeamSize_);
    numWordEndHypsAfterBeamPruning_ += wordEndHypotheses_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-word-end-hyps-after-beam-pruning", wordEndHypotheses_.size());
    }

    beam_.swap(newBeam_);
    beam_.insert(beam_.end(), wordEndHypotheses_.begin(), wordEndHypotheses_.end());

    canUpdateStablePrefix_ = true;

    numActiveHyps_ += beam_.size();

    /*
     * Clean up label scorer caches and calculate number of active trees
     */
    Core::CollapsedVector<Nn::ScoringContextRef> activeContexts;
    std::vector<Lm::History>                     seenHistories;
    for (auto const& hyp : beam_) {
        activeContexts.push_back(hyp.scoringContext);
        if (std::find(seenHistories.begin(), seenHistories.end(), hyp.lmHistory) == seenHistories.end()) {
            seenHistories.push_back(hyp.lmHistory);
        }
    }
    if (++currentSearchStep_ % cacheCleanupInterval_ == 0) {
        labelScorer_->cleanupCaches(activeContexts);
    }
    numActiveTrees_ += seenHistories.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-active-trees", seenHistories.size());
    }

    /*
     * Log statistics about the new beam.
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

TreeTimesyncBeamSearch::LabelHypothesis const& TreeTimesyncBeamSearch::getBestHypothesis() const {
    verify(not beam_.empty());

    return *std::min_element(beam_.begin(), beam_.end());
}

TreeTimesyncBeamSearch::LabelHypothesis const& TreeTimesyncBeamSearch::getWorstHypothesis() const {
    verify(not beam_.empty());

    return *std::max_element(beam_.begin(), beam_.end());
}

void TreeTimesyncBeamSearch::resetStatistics() {
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
    numActiveHyps_.clear();
    numActiveTrees_.clear();
}

void TreeTimesyncBeamSearch::logStatistics() const {
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
    numActiveHyps_.write(clog());
    numActiveTrees_.write(clog());
}

Nn::LabelScorer::TransitionType TreeTimesyncBeamSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const {
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
void TreeTimesyncBeamSearch::beamSizePruning(std::vector<Element>& hypotheses, size_t maxBeamSize) const {
    if (hypotheses.size() <= maxBeamSize) {
        return;
    }

    // Sort the hypotheses by associated score value such that the first `maxBeamSize` elements are the best
    std::nth_element(hypotheses.begin(), hypotheses.begin() + maxBeamSize, hypotheses.end());
    hypotheses.resize(maxBeamSize);  // Get rid of excessive elements
}

template void TreeTimesyncBeamSearch::beamSizePruning<TreeTimesyncBeamSearch::WithinWordExtensionCandidate>(std::vector<TreeTimesyncBeamSearch::WithinWordExtensionCandidate>&, size_t) const;
template void TreeTimesyncBeamSearch::beamSizePruning<TreeTimesyncBeamSearch::WordEndExtensionCandidate>(std::vector<TreeTimesyncBeamSearch::WordEndExtensionCandidate>&, size_t) const;

template<typename Element>
void TreeTimesyncBeamSearch::scorePruning(std::vector<Element>& hyps, Score scoreThreshold) const {
    if (hyps.empty() or scoreThreshold == Core::Type<Score>::max) {
        return;
    }

    // Compute the pruning threshold
    auto bestScore        = std::min_element(hyps.begin(), hyps.end())->score;
    auto pruningThreshold = bestScore + scoreThreshold;

    // Remove elements with score > pruningThreshold
    hyps.erase(
            std::remove_if(
                    hyps.begin(),
                    hyps.end(),
                    [=](auto const& ext) { return ext.score > pruningThreshold; }),
            hyps.end());
}

template void TreeTimesyncBeamSearch::scorePruning<TreeTimesyncBeamSearch::WithinWordExtensionCandidate>(std::vector<TreeTimesyncBeamSearch::WithinWordExtensionCandidate>&, Score) const;
template void TreeTimesyncBeamSearch::scorePruning<TreeTimesyncBeamSearch::WordEndExtensionCandidate>(std::vector<TreeTimesyncBeamSearch::WordEndExtensionCandidate>&, Score) const;

void TreeTimesyncBeamSearch::recombination(std::vector<TreeTimesyncBeamSearch::LabelHypothesis>& hypotheses, bool createTraceSiblings) {
    // Represents a unique combination of StateId, ScoringContext and LmHistory
    struct RecombinationContext {
        StateId               state;
        Nn::ScoringContextRef scoringContext;
        Lm::History           lmHistory;

        RecombinationContext(LabelHypothesis const& hyp)
                : state(hyp.currentState), scoringContext(hyp.scoringContext), lmHistory(hyp.lmHistory) {}

        bool operator==(const RecombinationContext& other) const {
            return state == other.state && Nn::ScoringContextEq{}(scoringContext, other.scoringContext) && lmHistory == other.lmHistory;
        }
    };
    struct RecombinationContextHash {
        size_t operator()(const RecombinationContext& context) const {
            size_t h1 = context.state;
            size_t h2 = Nn::ScoringContextHash{}(context.scoringContext);
            size_t h3 = Lm::History::Hash{}(context.lmHistory);
            return Core::combineHashes(Core::combineHashes(h1, h2), h3);
        }
    };

    tempHypotheses_.clear();
    // Reserve capacity because future reallocations would break the raw pointer we are storing later
    tempHypotheses_.reserve(hypotheses.size());
    // Map each unique combination of StateId, ScoringContext and LmHistory in newHypotheses to its hypothesis
    std::unordered_map<RecombinationContext, LabelHypothesis*, RecombinationContextHash> seenCombinations;
    for (auto const& hyp : hypotheses) {
        // Use try_emplace to check if the combination already exists and create a new entry if not at the same time
        auto [it, inserted] = seenCombinations.try_emplace({hyp}, nullptr);

        if (inserted) {
            // First time seeing this combination so move it over to `newHypotheses`
            tempHypotheses_.push_back(std::move(hyp));
            it->second = &tempHypotheses_.back();
        }
        else {
            auto* existingHyp = it->second;
            if (hyp.score < existingHyp->score) {
                // New hyp is better
                if (createTraceSiblings) {
                    hyp.trace->sibling = existingHyp->trace;
                }
                // Replace in `newHypotheses`
                *existingHyp = std::move(hyp);  // Overwrite in-place
            }
            else if (createTraceSiblings) {
                // New hyp is worse -> add to existing one as sibling if we are at a word end
                hyp.trace->sibling          = existingHyp->trace->sibling;
                existingHyp->trace->sibling = hyp.trace;
            }
        }
    }

    hypotheses.swap(tempHypotheses_);
}

void TreeTimesyncBeamSearch::createSuccessorLookups() {
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

void TreeTimesyncBeamSearch::finalizeLmScoring() {
    newBeam_.clear();
    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        auto& hyp = beam_[hypIndex];
        // Check if the hypotheses in the beam are at a root state and add the sentence-end LM score
        if (hyp.currentState == network_->rootState or network_->otherRootStates.find(hyp.currentState) != network_->otherRootStates.end()) {
            Lm::Score sentenceEndScore = languageModel_->sentenceEndScore(hyp.lmHistory);
            hyp.score += sentenceEndScore;
            hyp.trace->score.lm += sentenceEndScore;
            newBeam_.push_back(hyp);
        }
    }

    if (newBeam_.empty()) {  // There was no word-end hypothesis in the beam
        warning("No active word-end hypothesis at segment end.");
        if (sentenceEndFallback_) {
            log() << "Use sentence-end fallback";
            // The trace of the unfinished word keeps an empty pronunciation, only the LM score is added
            for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
                auto&     hyp              = beam_[hypIndex];
                Lm::Score sentenceEndScore = languageModel_->sentenceEndScore(hyp.lmHistory);
                hyp.score += sentenceEndScore;
                hyp.trace->score.lm += sentenceEndScore;
                newBeam_.push_back(hyp);
            }
        }
        else {
            // Construct an empty hypothesis with a lattice containing only one empty pronunciation from start to end
            newBeam_.push_back(LabelHypothesis());
            newBeam_.front().trace->time          = beam_.front().trace->time;  // Retrieve the timeframe from any hyp in the old beam
            newBeam_.front().trace->pronunciation = nullptr;
            newBeam_.front().trace->predecessor   = Core::ref(new LatticeTrace(0, {0, 0}, {}));
        }
    }
    beam_.swap(newBeam_);
}

void TreeTimesyncBeamSearch::maximumStableDelayPruning() {
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
