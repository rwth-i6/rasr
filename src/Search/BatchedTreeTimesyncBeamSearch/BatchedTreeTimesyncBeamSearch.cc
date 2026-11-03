/** Copyright 2026 RWTH Aachen University. All rights reserved.
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

#include "BatchedTreeTimesyncBeamSearch.hh"

#include <algorithm>
#include <unordered_set>
#include <strings.h>

#include <Am/ClassicStateModel.hh>
#include <Core/CollapsedVector.hh>
#include <Core/XmlStream.hh>
#include <Lattice/LatticeAdaptor.hh>
#include <Lm/BackingOff.hh>
#include <Lm/Module.hh>
#include <Math/Utilities.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/Module.hh>
#include <Search/Traceback.hh>
#include <Search/TracebackHelper.hh>

namespace {

enum RecombinationMode {
    RecombinationModeOff,
    RecombinationModeOn,
};

}  // namespace

namespace Search {

/*
 * =======================
 * === LabelHypothesis ===
 * =======================
 */

BatchedTreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContexts(),
          currentToken(Nn::invalidLabelIndex),
          currentState(invalidTreeNodeIndex),
          lookahead(),
          lmHistory(),
          lookaheadHistory(),
          timeframe(0),
          score(0.0),
          lookaheadScore(0.0),
          lookaheadBackOff(0.0),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))) {
}

BatchedTreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        BatchedTreeTimesyncBeamSearch::LabelHypothesis const&              base,
        BatchedTreeTimesyncBeamSearch::WithinWordExtensionCandidate const& extension,
        std::vector<Nn::ScoringContextRef> const&                          newScoringContexts)
        : scoringContexts(newScoringContexts),
          currentToken(extension.nextToken),
          currentState(extension.nextState),
          lookahead(extension.lookahead),
          lmHistory(base.lmHistory),
          lookaheadHistory(base.lookaheadHistory),
          timeframe(extension.timeframe),
          score(extension.score),
          lookaheadScore(extension.lookaheadScore),
          lookaheadBackOff(extension.lookaheadBackOff),
          trace(base.trace) {
}

BatchedTreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        LabelHypothesis const&                                          base,
        BatchedTreeTimesyncBeamSearch::WordEndExtensionCandidate const& extension,
        Lm::History const&                                              newLmHistory,
        LanguageModelLookahead::ContextLookaheadReference const         newLookahead,
        Lm::History const&                                              newLookaheadHistory,
        Score                                                           newLookaheadBackOff)
        : scoringContexts(base.scoringContexts),
          currentToken(base.currentToken),
          currentState(extension.rootState),
          lookahead(newLookahead),
          lmHistory(newLmHistory),
          lookaheadHistory(newLookaheadHistory),
          timeframe(base.timeframe),
          score(extension.score),
          lookaheadScore(0.0),
          lookaheadBackOff(newLookaheadBackOff) {
    auto newLmScore   = score - base.score;
    auto totalLmScore = base.trace->score.lm + newLmScore;
    auto totalAmScore = score - totalLmScore;

    // Only increment timeframe when not SENTENCE_END
    auto trace_timeframe = extension.transitionType == Nn::TransitionType::SENTENCE_END ? base.timeframe : base.timeframe + 1;

    // Create a successor trace item from base
    trace = Core::ref(new LatticeTrace(
            base.trace,
            extension.pron,
            trace_timeframe,
            {totalAmScore, totalLmScore},
            {}));
}

std::string BatchedTreeTimesyncBeamSearch::LabelHypothesis::toString() const {
    std::stringstream ss;
    ss << "Score: " << score << ", current state: " << currentState << ", traceback: ";

    auto traceback = trace->performTraceback();

    for (auto& item : *traceback) {
        if (item.pronunciation and item.pronunciation->lemma()) {
            ss << item.pronunciation->lemma()->symbol() << " ";
        }
    }

    return ss.str();
}

/*
 * ====================
 * === SegmentState ===
 * ====================
 */

BatchedTreeTimesyncBeamSearch::SegmentState::SegmentState(SegmentHandle handle, std::string const& name, size_t numLabelScorers)
        : SegmentStateBase(handle, name),
          beam(),
          currentSearchStep(0ul),
          hypIndexToContextIndexMap(),
          withinWordExtensions(),
          wordEndExtensions(),
          newBeam(),
          wordEndHypotheses(),
          finalHypotheses(),
          stepFailed(false),
          stableDelayLimitExceeded(false),
          numInputHyps("num-input-hyps"),
          numWithinWordExtensionsBeforeFirstPruning("num-within-word-extensions-before-first-pruning"),
          numWithinWordHypsAfterIntermediatePruning(numLabelScorers, Core::Statistics<u32>("num-within-word-hyps-after-intermediate-pruning")),
          numWithinWordHypsAfterRecombination("num-within-word-hyps-after-recombination"),
          numWithinWordHypsAfterPruning("num-within-word-hyps-after-pruning"),
          numWordEndExtensionsBeforePruning("num-word-end-extensions-before-pruning"),
          numWordEndHypsAfterScorePruning("num-word-end-hyps-after-score-pruning"),
          numWordEndHypsAfterRecombination("num-word-end-hyps-after-recombination"),
          numWordEndHypsAfterBeamPruning("num-word-end-hyps-after-beam-pruning"),
          numActiveHyps("num-active-hyps"),
          numActiveTrees("num-active-trees") {
}

/*
 * =====================================
 * === BatchedTreeTimesyncBeamSearch ===
 * =====================================
 */

BatchedTreeTimesyncBeamSearch::ThreadBuffers::ThreadBuffers(size_t numHistogramBins)
        : tempHypotheses(),
          wordEndScoreAccessors(),
          scoreHistogram(numHistogramBins),
          scoreReadoutTimes(),
          intermediatePruningTimes(),
          buildWithinWordHypsTime(),
          wordEndExpansionTime(),
          wordEndExtensionTime(),
          lmTime(),
          wordEndScorePruningTime(),
          wordEndHypBuildingTime(),
          lmLookaheadTime(),
          wordEndRecombinationTime(),
          wordEndBeamPruningTime(),
          recombinationTime(),
          beamPruningTime() {}

const Core::ParameterIntVector BatchedTreeTimesyncBeamSearch::paramMaxBeamSizes(
        "max-beam-size",
        "Maximum number of within-word hypotheses in the search beam of each segment. Pruning is applied after each intermediate label scorer.",
        "",
        1);

const Core::ParameterInt BatchedTreeTimesyncBeamSearch::paramMaxWordEndBeamSize(
        "max-word-end-beam-size",
        "Maximum number of word-end hypotheses in the search beam of each segment. If not set, global beam pruning will be done and word-end hypotheses will not be pruned separately.",
        std::numeric_limits<int>::max(), 0);

const Core::ParameterFloatVector BatchedTreeTimesyncBeamSearch::paramScoreThresholds(
        "score-threshold",
        "Prune any within-word hypotheses with a score that is at least this much worse than the best hypothesis of the segment. Pruning is applied after each intermediate label scorer.",
        "",
        0,
        Core::Type<Score>::max);

const Core::ParameterFloat BatchedTreeTimesyncBeamSearch::paramWordEndScoreThreshold(
        "word-end-score-threshold",
        "Prune any word-end hypothesis with a score that is at least this much worse than the best word-end hypothesis. This threshold is relative to the score-threshold."
        " If not set, global score pruning will be done and word-end hypotheses will not be pruned separately.",
        Core::Type<Score>::max, 0);

const Core::ParameterInt BatchedTreeTimesyncBeamSearch::paramNumHistogramBins(
        "num-histogram-bins",
        "Number of bins for histogram pruning of hypotheses (very minor effect).",
        100,
        2);

const Core::ParameterBool BatchedTreeTimesyncBeamSearch::paramEnableHistogramPruning(
        "enable-histogram-pruning",
        "Use histogram pruning for max-beam-size. If false, exactly the max-beam-size best hypotheses are kept.",
        true);

const Core::ParameterBool BatchedTreeTimesyncBeamSearch::paramCollapseRepeatedLabels(
        "collapse-repeated-labels",
        "Collapse repeated emission of the same label into one output. If false, every emission is treated like a new output.",
        false);

const Core::ParameterBool BatchedTreeTimesyncBeamSearch::paramLmLookahead(
        "lm-lookahead",
        "Enable language model lookahead.",
        false);

const Core::ParameterBool BatchedTreeTimesyncBeamSearch::paramSeparateLookaheadLm(
        "separate-lookahead-lm",
        "Use a separate LM for lookahead.",
        false);

const Core::ParameterBool BatchedTreeTimesyncBeamSearch::paramSparseLmLookAhead(
        "sparse-lm-lookahead",
        "Use sparse n-gram LM lookahead.",
        true);

const Core::ParameterBool BatchedTreeTimesyncBeamSearch::paramSentenceEndFallBack(
        "sentence-end-fall-back",
        "Allow for fallback solution if no active word-end hypothesis exists at the end of a segment.",
        true);

const Core::ParameterInt BatchedTreeTimesyncBeamSearch::paramCacheCleanupInterval(
        "cache-cleanup-interval",
        "Interval of search steps after which buffered inputs that are not needed anymore get cleaned up.",
        10,
        1);

const Core::ParameterInt BatchedTreeTimesyncBeamSearch::paramMaximumStableDelay(
        "maximum-stable-delay",
        "Introduce a cutoff point at `current-time` - `delay`. Every hypothesis that disagrees with the current best anywhere before the cutoff gets pruned."
        "This way words in the traceback become stable after at most `delay` frames.",
        Core::Type<int>::max,
        0);

const Core::ParameterInt BatchedTreeTimesyncBeamSearch::paramMaximumStableDelayPruningInterval(
        "maximum-stable-delay-pruning-interval",
        "Interval of search steps after which the maximum-stable-delay-pruning gets applied.",
        10,
        1);

const Core::Choice BatchedTreeTimesyncBeamSearch::choiceRecombinationMode(
        "off", RecombinationModeOff,
        "on", RecombinationModeOn,
        Core::Choice::endMark());

const Core::ParameterChoice BatchedTreeTimesyncBeamSearch::paramRecombinationMode(
        "recombination-mode",
        &choiceRecombinationMode,
        "Whether hypotheses with identical recombination state should be recombined.",
        RecombinationModeOn);

BatchedTreeTimesyncBeamSearch::BatchedTreeTimesyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          BatchedSearchBase(config),
          maxWordEndBeamSize_(paramMaxWordEndBeamSize(config)),
          histogramPruningEnabled_(paramEnableHistogramPruning(config)),
          wordEndScoreThreshold_(paramWordEndScoreThreshold(config)),
          blankLabelIndex_(Nn::invalidLabelIndex),
          silenceLabelIndex_(Nn::invalidLabelIndex),
          sentenceEndLemma_(),
          sentenceEndLabelIndex_(Nn::invalidLabelIndex),
          cacheCleanupInterval_(paramCacheCleanupInterval(config)),
          maximumStableDelay_(paramMaximumStableDelay(config)),
          maximumStableDelayPruningInterval_(paramMaximumStableDelayPruningInterval(config)),
          useBlank_(),
          useSilence_(),
          collapseRepeatedLabels_(paramCollapseRepeatedLabels(config)),
          sentenceEndFallback_(paramSentenceEndFallBack(config)),
          recombinationEnabled_(paramRecombinationMode(config) == RecombinationModeOn),
          nonWordLemmas_(),
          debugChannel_(config, "debug"),
          enableLmLookahead_(paramLmLookahead(config)),
          separateLookaheadLm_(paramSeparateLookaheadLm(config)),
          sparseLmLookahead_(paramSparseLmLookAhead(config)),
          stepSegments_(),
          segmentsToFinalize_(),
          scoringContexts_(),
          threadBuffers_(),
          scoreAndPruneExtensionsTimes_(),
          scoringTimes_(),
          scoreReadoutTimes_(),
          intermediatePruningTimes_(),
          finalizeTime_(),
          finalizeScoringTime_(),
          keepLookaheadTables_(false),
          lookaheadMutex_(),
          lookaheadTablesInUse_(numPipelineGroups()),
          currentGroup_(0ul) {
    if (debugChannel_.isOpen()) {
        disableParallelSearch("debug output is written");
    }
    for (size_t threadIdx = 0ul; threadIdx < numSearchThreads(); ++threadIdx) {
        threadBuffers_.emplace_back(paramNumHistogramBins(config));
    }
    auto maxBeamSizes = paramMaxBeamSizes(config);
    maxBeamSizes_.insert(maxBeamSizes_.begin(), maxBeamSizes.begin(), maxBeamSizes.end());

    auto scoreThresholds = paramScoreThresholds(config);
    scoreThresholds_.insert(scoreThresholds_.begin(), scoreThresholds.begin(), scoreThresholds.end());
    // Fill up with default value
    for (size_t i = scoreThresholds_.size(); i < maxBeamSizes_.size(); ++i) {
        scoreThresholds_.push_back(Core::Type<Score>::max);
    }

    if (scoreThresholds_.back() == Core::Type<Score>::max and wordEndScoreThreshold_ != Core::Type<Score>::max) {
        error() << "Word-end score-threshold which is relative to the score-threshold is set, but score-threshold is not set";
    }
    if (wordEndScoreThreshold_ != Core::Type<Score>::max) {
        log() << "Use absolute word-end score-threshold of " << wordEndScoreThreshold_ * scoreThresholds_.back() << "; computed relative to within-word threshold " << scoreThresholds_.back() << " with factor " << wordEndScoreThreshold_;
        wordEndScoreThreshold_ *= scoreThresholds_.back();
    }
}

Speech::ModelCombination::Mode BatchedTreeTimesyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon | Speech::ModelCombination::useAcousticModel | Speech::ModelCombination::useLanguageModel;
}

Am::AcousticModel::Mode BatchedTreeTimesyncBeamSearch::requiredAcousticModel() const {
    return Am::AcousticModel::noEmissions;
}

bool BatchedTreeTimesyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_       = modelCombination.lexicon();
    acousticModel_ = modelCombination.acousticModel();
    languageModel_ = modelCombination.languageModel();
    setLabelScorers(modelCombination.labelScorers());

    if (labelScorers_.size() > maxBeamSizes_.size()) {
        error() << "Number of label scorers (" << labelScorers_.size() << ") exceeds number of configured max beam sizes (" << maxBeamSizes_.size() << ")";
    }
    if (labelScorers_.size() < maxBeamSizes_.size()) {
        warning() << "Number of label scorers (" << labelScorers_.size() << ") is less than number of configured max beam sizes (" << maxBeamSizes_.size() << ")";
    }

    // Timers are indexed by label scorer
    scoreAndPruneExtensionsTimes_.resize(labelScorers_.size());
    scoringTimes_.resize(labelScorers_.size());
    scoreReadoutTimes_.resize(labelScorers_.size());
    intermediatePruningTimes_.resize(labelScorers_.size());
    for (auto& buffers : threadBuffers_) {
        buffers.scoreReadoutTimes.resize(labelScorers_.size());
        buffers.intermediatePruningTimes.resize(labelScorers_.size());
    }

    nonWordLemmas_ = lexicon_->specialLemmas("nonword");

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

    // Read the search tree from image or build it
    if (not network_->read()) {
        log() << "Persistent search tree image could not be loaded; building it";
        std::unique_ptr<AbstractTreeBuilder> builder = Search::Module::instance().createTreeBuilder(config, *lexicon_, *acousticModel_, *network_);
        builder->build();

        if (network_->write(0)) {
            log() << "Wrote search tree image to file";
        }
        else {
            log() << "Writing search tree image failed";
        }
    }

    if (lexicon_->specialLemma("blank")) {
        blankLabelIndex_ = acousticModel_->emissionIndex(acousticModel_->blankAllophoneStateIndex());
        useBlank_        = true;
        log() << "Use blank label with index " << blankLabelIndex_;
    }
    else {
        blankLabelIndex_ = Nn::invalidLabelIndex;
        useBlank_        = false;
    }

    if (lexicon_->specialLemma("silence")) {
        silenceLabelIndex_ = acousticModel_->emissionIndex(acousticModel_->silenceAllophoneStateIndex());
        useSilence_        = true;
        log() << "Use silence label with index " << silenceLabelIndex_;
    }
    else {
        silenceLabelIndex_ = Nn::invalidLabelIndex;
        useSilence_        = false;
    }

    sentenceEndLemma_ = lexicon_->specialLemma("sentence-end");
    if (not sentenceEndLemma_) {
        sentenceEndLemma_ = lexicon_->specialLemma("sentence-boundary");
    }
    if (sentenceEndLemma_ and sentenceEndLemma_->nPronunciations() != 0 and sentenceEndLemma_->pronunciations().first->pronunciation()->length() > 0) {
        auto const* pron = sentenceEndLemma_->pronunciations().first->pronunciation();
        require(pron->length() == 1);
        Am::Allophone           allo(acousticModel_->phonology()->allophone(*pron, 0),
                                     Am::Allophone::isInitialPhone | Am::Allophone::isFinalPhone);
        Am::AllophoneStateIndex alloStateIdx = acousticModel_->allophoneStateAlphabet()->index(&allo, 0);

        sentenceEndLabelIndex_ = acousticModel_->emissionIndex(alloStateIdx);
        log() << "Use sentence-end label with index " << sentenceEndLabelIndex_;
    }
    else {
        sentenceEndLabelIndex_ = Nn::invalidLabelIndex;
    }

    for (const auto& lemma : {"silence", "blank"}) {
        if (lexicon_->specialLemma(lemma) and (lexicon_->specialLemma(lemma)->syntacticTokenSequence()).size() != 0) {
            warning("Special lemma \"%s\" will be scored by the language model. To prevent the LM from scoring it, set an empty syntactic token sequence for it in the lexicon.", lemma);
        }
    }

    // Create look-ups for state successors and exits of each state
    createSuccessorLookups();

    // Set lookahead LM
    if (enableLmLookahead_) {
        if (separateLookaheadLm_) {
            log() << "Use separate lookahead LM";
            lookaheadLm_ = Lm::Module::instance().createScaledLanguageModel(select("lookahead-lm"), lexicon_);
        }
        else if (languageModel_->lookaheadLanguageModel().get() != nullptr) {
            lookaheadLm_ = Core::Ref<Lm::ScaledLanguageModel>(new Lm::LanguageModelScaling(select("lookahead-lm"),
                                                                                           Core::Ref<Lm::LanguageModel>(const_cast<Lm::LanguageModel*>(languageModel_->lookaheadLanguageModel().get()))));
        }
        else {
            lookaheadLm_ = languageModel_;
        }

        if (sparseLmLookahead_ && !dynamic_cast<const Lm::BackingOffLm*>(lookaheadLm_->unscaled().get())) {
            warning() << "Not using sparse LM lookahead, because the LM is not a backing-off LM.";
            sparseLmLookahead_ = false;
        }

        lmLookahead_ = std::make_unique<LanguageModelLookahead>(Core::Configuration(config, "lm-lookahead"),
                                                                modelCombination.pronunciationScale(),
                                                                lookaheadLm_,
                                                                network_->structure,
                                                                network_->rootState,
                                                                network_->exits,
                                                                acousticModel_);
    }

    // Only backing-off LMs can be queried from several threads at the same time (no mutable state, no reference
    // counting of histories)
    if (not dynamic_cast<Lm::BackingOffLm const*>(languageModel_->unscaled().get()) or
        (enableLmLookahead_ and not dynamic_cast<Lm::BackingOffLm const*>(lookaheadLm_->unscaled().get()))) {
        disableParallelSearch("the language model is not a backing-off LM and can't be used from several threads");
    }
    keepLookaheadTables_ = enableLmLookahead_ and numSearchThreads() > 1ul;

    return true;
}

std::unique_ptr<BatchedSearchBase::SegmentStateBase> BatchedTreeTimesyncBeamSearch::createSegmentState(SegmentHandle handle, std::string const& name) {
    auto state = std::make_unique<SegmentState>(handle, name, labelScorers_.size());

    // Initialize beam with a single empty hypothesis
    state->beam.push_back(LabelHypothesis());
    for (auto& labelScorer : labelScorers_) {
        state->beam.front().scoringContexts.push_back(labelScorer->getInitialSegmentScoringContext(handle));
    }
    state->beam.front().currentState = network_->rootState;
    state->beam.front().lmHistory    = languageModel_->startHistory();

    if (enableLmLookahead_) {
        state->beam.front().lookaheadHistory = lookaheadLm_->startHistory();
    }

    return state;
}

Core::Ref<const Traceback> BatchedTreeTimesyncBeamSearch::getCurrentBestTraceback(SegmentHandle segment) const {
    return getBestHypothesis(getSegment<SegmentState>(segment).beam).trace->performTraceback();
}

Core::Ref<const LatticeAdaptor> BatchedTreeTimesyncBeamSearch::getCurrentBestWordLattice(SegmentHandle segment) const {
    auto const&  beam           = getSegment<SegmentState>(segment).beam;
    auto&        bestHypothesis = getBestHypothesis(beam);
    LatticeTrace endTrace(bestHypothesis.trace, 0, bestHypothesis.trace->time + 1, bestHypothesis.trace->score, {});

    for (auto const& hyp : beam) {
        // The best hypothesis is already represented in endTrace
        if (&hyp == &bestHypothesis) {
            continue;
        }
        auto siblingTrace = Core::ref(new LatticeTrace(hyp.trace, 0, hyp.trace->time, hyp.trace->score, {}));
        endTrace.appendSiblingToChain(siblingTrace);
    }

    return endTrace.buildWordLattice(lexicon_);
}

Core::Ref<const LatticeTrace> BatchedTreeTimesyncBeamSearch::getCurrentBestLatticeTrace(SegmentHandle segment) const {
    return getBestHypothesis(getSegment<SegmentState>(segment).beam).trace;
}

Core::Ref<const LatticeTrace> BatchedTreeTimesyncBeamSearch::getCommonPrefix(SegmentHandle segment) const {
    auto const&                          beam = getSegment<SegmentState>(segment).beam;
    std::vector<Core::Ref<LatticeTrace>> traces(beam.size());
    for (size_t hypIndex = 0ul; hypIndex < beam.size(); ++hypIndex) {
        traces[hypIndex] = beam[hypIndex].trace;
    }

    RootTraceSearcher searcher(traces);
    if (not searcher.rootTrace()) {
        warning("Common prefix of all traces is a sentinel value");
    }

    return Core::Ref<const LatticeTrace>(searcher.rootTrace());
}

bool BatchedTreeTimesyncBeamSearch::decodeGroupStep(size_t group) {
    searchStepTime_.start();
    currentGroup_ = group;

    stepSegments_.clear();
    forEachSegment<SegmentState>([&](SegmentState& segment) {
        if (not segment.finished and segment.group == group) {
            stepSegments_.push_back(&segment);
        }
    });

    /*
     * Scores of the first label scorer: with pipelining, they were requested at the end of the previous step of the
     * group; only the segments whose contexts were scored in that call take part (others join the next step).
     * Otherwise the contexts are collected and scored now.
     */
    scoreAndPruneExtensionsTimes_.front().start();
    scoringTimes_.front().start();
    ScoreAccessors firstScoreAccessors;
    if (auto pending = takePendingScoring(group)) {
        std::unordered_set<SegmentHandle> scoredSegments(pending->segments.begin(), pending->segments.end());
        stepSegments_.erase(std::remove_if(stepSegments_.begin(), stepSegments_.end(), [&](SegmentState* segment) { return not scoredSegments.contains(segment->handle); }),
                            stepSegments_.end());
        firstScoreAccessors = waitForScoring(*pending);
    }
    else if (not stepSegments_.empty()) {
        collectScoringContexts(stepSegments_, scoringContexts_);
        firstScoreAccessors = getScoreAccessors(0ul, scoringContexts_);
    }
    scoringTimes_.front().stop();

    if (stepSegments_.empty()) {
        scoreAndPruneExtensionsTimes_.front().stop();
        if (keepLookaheadTables_) {
            lookaheadTablesInUse_[group].clear();
        }
        startNextScoring(group);
        searchStepTime_.stop();
        return false;
    }

    numSegmentsPerStep_ += stepSegments_.size();

    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlOpen("search-step-stats") + Core::XmlAttribute("step", globalSearchStep_);
    }

    for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
        /*
         * Scores of the contexts of all segments with the label scorer.
         */
        scoreAndPruneExtensionsTimes_[scorerIdx].start();
        ScoreAccessors scoreAccessors;
        if (scorerIdx == 0ul) {
            scoreAccessors = std::move(firstScoreAccessors);
        }
        else {
            Core::StopWatch::Scope scoringTimer(scoringTimes_[scorerIdx]);
            scoreAccessors = getScoreAccessors(scorerIdx, scoringContexts_);
        }

        scoreReadoutTimes_[scorerIdx].start();
        std::vector<std::optional<Nn::DenseScoreSpan>> denseScoreSpans(scoreAccessors.size(), std::nullopt);
        std::vector<Nn::TimeframeIndex>                scoreTimes(scoreAccessors.size(), 0);
        for (size_t accessorIdx = 0ul; accessorIdx < scoreAccessors.size(); ++accessorIdx) {
            if (scoreAccessors[accessorIdx]) {
                denseScoreSpans[accessorIdx] = (*scoreAccessors[accessorIdx])->getDenseScores();
                scoreTimes[accessorIdx]      = (*scoreAccessors[accessorIdx])->getTime();
            }
        }
        scoreReadoutTimes_[scorerIdx].stop();

        parallelForSegments(stepSegments_, [&](SegmentState& segment, size_t threadIdx) {
            if (segment.stepFailed) {
                return;
            }
            auto& buffers = threadBuffers_[threadIdx];

            if (scorerIdx == 0ul) {
                segment.numInputHyps += segment.beam.size();
                if (stepwiseStatisticsChannel_.isOpen()) {
                    stepwiseStatisticsChannel_ << Core::XmlFull("num-input-hyps", segment.beam.size()) + Core::XmlAttribute("segment", segment.name);
                }
            }

            {
                // Lazily computing label scorers do their work here
                Core::StopWatch::Scope readoutTimer(buffers.scoreReadoutTimes[scorerIdx]);
                scoreExtensions(segment, scorerIdx, scoreAccessors, denseScoreSpans, scoreTimes);
            }
            if (segment.withinWordExtensions.empty()) {
                segment.stepFailed = true;
                return;
            }

            if (scorerIdx == 0ul) {
                segment.numWithinWordExtensionsBeforeFirstPruning += segment.withinWordExtensions.size();
                if (stepwiseStatisticsChannel_.isOpen()) {
                    stepwiseStatisticsChannel_ << Core::XmlFull("num-within-word-extensions-before-first-pruning", segment.withinWordExtensions.size()) + Core::XmlAttribute("segment", segment.name);
                }
            }

            /*
             * Prune set of possible within-word extensions by max beam size and possibly also by score.
             */
            Core::StopWatch::Scope pruningTimer(buffers.intermediatePruningTimes[scorerIdx]);
            size_t                 maxBeamSize = segment.withinWordExtensions.size();
            if (scorerIdx < labelScorers_.size() - 1) {
                maxBeamSize = maxBeamSizes_[scorerIdx];
            }
            scorePruning(segment.withinWordExtensions, scoreThresholds_[scorerIdx], maxBeamSize, buffers.scoreHistogram);
            segment.numWithinWordHypsAfterIntermediatePruning[scorerIdx] += segment.withinWordExtensions.size();
            if (stepwiseStatisticsChannel_.isOpen()) {
                stepwiseStatisticsChannel_ << Core::XmlFull("num-within-word-hyps-after-intermediate-pruning", segment.withinWordExtensions.size()) + Core::XmlAttribute("scorer", scorerIdx + 1) + Core::XmlAttribute("segment", segment.name);
            }
            if (segment.withinWordExtensions.empty()) {
                segment.stepFailed = true;
            }
        });

        if (scorerIdx < labelScorers_.size() - 1) {
            // Prepare scoring context list for next iteration
            // Some scoring contexts from the current iteration may not have survived pruning, so we need to recreate the list
            // Use -1 as placeholder to signify that this hyp was not visited yet
            Core::StopWatch::Scope pruningTimer(intermediatePruningTimes_[scorerIdx]);
            scoringContexts_.clear();
            for (auto* segment : stepSegments_) {
                if (segment->stepFailed) {
                    continue;
                }
                segment->hypIndexToContextIndexMap.assign(segment->beam.size(), -1);
                for (auto const& ext : segment->withinWordExtensions) {
                    if (segment->hypIndexToContextIndexMap[ext.baseHypIndex] == -1) {
                        segment->hypIndexToContextIndexMap[ext.baseHypIndex] = scoringContexts_.size();
                        scoringContexts_.push_back(segment->beam[ext.baseHypIndex].scoringContexts[scorerIdx + 1]);
                    }
                }
            }
        }
        scoreAndPruneExtensionsTimes_[scorerIdx].stop();
    }

    /*
     * Create new beams from surviving extensions. Segments that can't make progress anymore are finalized.
     */
    bool anythingChanged = false;
    parallelForSegments(stepSegments_, [&](SegmentState& segment, size_t threadIdx) {
        if (not segment.stepFailed) {
            createNewBeam(segment, threadBuffers_[threadIdx]);
        }
    });
    segmentsToFinalize_.clear();
    for (auto* segment : stepSegments_) {
        if (segment->stepFailed) {
            // The segment can't make progress in this step. If no more features were expected when its contexts were
            // scored, this won't change anymore.
            if (segment->noMoreFeaturesWhenScored) {
                segmentsToFinalize_.push_back(segment);
            }
            continue;
        }
        if (segment->stableDelayLimitExceeded) {
            warning() << "Most recent word in best hypothesis of segment " << segment->name << " is before cutoff point for maximum-stable-delay-pruning so the limit will be surpassed";
        }
        anythingChanged = true;
    }

    if (not segmentsToFinalize_.empty()) {
        Core::StopWatch::Scope finalizeTimer(finalizeTime_);
        finalizeSegments(segmentsToFinalize_);
        anythingChanged = true;
    }

    if (keepLookaheadTables_) {
        // Release all lookahead table references that are not part of a beam (in intermediate buffers) and the
        // references of the tables in use here, i.e. sequentially
        for (auto& buffers : threadBuffers_) {
            buffers.tempHypotheses.clear();
        }
        for (auto* segment : stepSegments_) {
            segment->withinWordExtensions.clear();
            segment->newBeam.clear();
            segment->wordEndHypotheses.clear();
        }
        lookaheadTablesInUse_[group].clear();
    }

    ++globalSearchStep_;

    /*
     * Clean up label scorer caches.
     */
    if (globalSearchStep_ % cacheCleanupInterval_ == 0) {
        for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
            Core::CollapsedVector<Nn::ScoringContextRef> activeContexts;
            forEachSegment<SegmentState>([&](SegmentState const& segment) {
                if (segment.finished) {
                    return;
                }
                for (auto const& hyp : segment.beam) {
                    activeContexts.push_back(hyp.scoringContexts[scorerIdx]);
                }
            });
            cleanupLabelScorerCaches(scorerIdx, activeContexts);
        }
    }

    startNextScoring(group);

    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlClose("search-step-stats");
    }

    searchStepTime_.stop();

    if (numActiveSegments() == 0ul) {
        logGlobalStatistics();
    }

    return anythingChanged;
}

void BatchedTreeTimesyncBeamSearch::collectScoringContexts(std::vector<SegmentState*> const& segments, std::vector<Nn::ScoringContextRef>& contexts) {
    /*
     * The contexts of all hypotheses of all segments are passed to the first label scorer in one list.
     * `segment.hypIndexToContextIndexMap` maps a beam index of the segment to the position of its context in the list.
     */
    contexts.clear();
    for (auto* segment : segments) {
        segment->withinWordExtensions.clear();
        segment->stepFailed               = false;
        segment->noMoreFeaturesWhenScored = segment->noMoreFeatures;
        segment->hypIndexToContextIndexMap.resize(segment->beam.size());
        for (size_t hypIndex = 0ul; hypIndex < segment->beam.size(); ++hypIndex) {
            segment->hypIndexToContextIndexMap[hypIndex] = contexts.size();
            contexts.push_back(segment->beam[hypIndex].scoringContexts.front());
            if (keepLookaheadTables_ and segment->beam[hypIndex].lookahead) {
                lookaheadTablesInUse_[segment->group].push_back(segment->beam[hypIndex].lookahead);
            }
        }
    }
}

void BatchedTreeTimesyncBeamSearch::startNextScoring(size_t group) {
    if (numPipelineGroups() == 1ul) {
        return;
    }
    std::vector<SegmentState*> segments;
    forEachSegment<SegmentState>([&](SegmentState& segment) {
        if (segment.group == group and not segment.finished) {
            segments.push_back(&segment);
        }
    });
    if (segments.empty()) {
        return;
    }
    std::vector<Nn::ScoringContextRef> contexts;
    collectScoringContexts(segments, contexts);
    std::vector<SegmentHandle> handles;
    for (auto* segment : segments) {
        handles.push_back(segment->handle);
    }
    startScoring(group, std::move(handles), std::move(contexts));
}

void BatchedTreeTimesyncBeamSearch::scoreExtensions(SegmentState&                                           segment,
                                                    size_t                                                  scorerIdx,
                                                    std::vector<std::optional<Nn::ScoreAccessorRef>> const& scoreAccessors,
                                                    std::vector<std::optional<Nn::DenseScoreSpan>> const&   denseScoreSpans,
                                                    std::vector<Nn::TimeframeIndex> const&                  scoreTimes) {
    auto const& labelScorer = labelScorers_[scorerIdx];
    auto const& beam        = segment.beam;
    auto&       extensions  = segment.withinWordExtensions;
    auto const& contextMap  = segment.hypIndexToContextIndexMap;

    if (scorerIdx == 0ul) {
        // In the first iteration, create extensions while pre-pruning
        Score currentBestScore = Core::Type<Score>::max;

        for (size_t hypIndex = 0ul; hypIndex < beam.size(); ++hypIndex) {
            auto const& hyp = beam[hypIndex];

            auto const& scoreAccessor = scoreAccessors[contextMap[hypIndex]];
            if (not scoreAccessor) {
                // No extensions for hyps that couldn't be scored
                continue;
            }
            auto const& denseScores = denseScoreSpans[contextMap[hypIndex]];
            auto        scoreTime   = scoreTimes[contextMap[hypIndex]];

            // Iterate over the successors of this hypothesis' current state in the tree
            for (size_t i = stateSuccessorsOffset_[hyp.currentState]; i < stateSuccessorsOffset_[hyp.currentState + 1]; ++i) {
                const StateId  successorState = stateSuccessors_[i];
                Nn::LabelIndex tokenIdx       = network_->structure.state(successorState).stateDesc.acousticModel;
                // If we collapse repeated labels, a new word should not start with the same token as the previous word ended (except for blank or silence)
                if (collapseRepeatedLabels_ and
                    network_->isRoot(hyp.currentState) and
                    tokenIdx == hyp.currentToken and
                    (not useBlank_ or tokenIdx != blankLabelIndex_) and
                    (not useSilence_ or tokenIdx != silenceLabelIndex_)) {
                    continue;
                }
                auto transitionType = inferTransitionType(hyp.currentToken, tokenIdx, hyp.currentState == successorState);
                auto extScore       = hyp.score;
                auto extTime        = hyp.timeframe;
                if (labelScorer->scoresTransition(transitionType)) {
                    if (denseScores and tokenIdx < denseScores->size()) {
                        extScore += (*denseScores)[tokenIdx];
                    }
                    else {
                        extScore += (*scoreAccessor)->getScore(transitionType, tokenIdx);
                    }
                    extTime = std::max(extTime, scoreTime);
                }

                // Pre-prune based on score before creating extension instance and appending to list
                if (scoreThresholds_.front() != Core::Type<Score>::max and extScore > currentBestScore + scoreThresholds_.front()) {
                    continue;
                }
                currentBestScore = std::min(currentBestScore, extScore);
                extensions.push_back(
                        {.nextToken        = tokenIdx,
                         .nextState        = successorState,
                         .timeframe        = extTime,
                         .score            = extScore,
                         .lookaheadScore   = 0,
                         .lookahead        = {},
                         .lookaheadBackOff = 0,
                         .transitionType   = transitionType,
                         .baseHypIndex     = hypIndex});

                // Add the LM lookahead score to the extensions' scores for pruning
                if (enableLmLookahead_) {
                    auto lookaheadScore              = getLmLookaheadScore(beam, extensions.back());
                    extensions.back().lookaheadScore = lookaheadScore;
                    extensions.back().score += lookaheadScore;
                }
            }
        }
    }
    else {
        // Update ext score and timestep
        for (auto& ext : extensions) {
            if (not labelScorer->scoresTransition(ext.transitionType)) {
                continue;
            }
            auto const& scoreAccessor = scoreAccessors[contextMap[ext.baseHypIndex]];

            if (scoreAccessor) {
                auto const& denseScores = denseScoreSpans[contextMap[ext.baseHypIndex]];
                if (denseScores and ext.nextToken < denseScores->size()) {
                    ext.score += (*denseScores)[ext.nextToken];
                }
                else {
                    ext.score += (*scoreAccessor)->getScore(ext.transitionType, ext.nextToken);
                }
                ext.timeframe = std::max(ext.timeframe, scoreTimes[contextMap[ext.baseHypIndex]]);
            }
            else {
                // Extension is not scorable so set the score to max in order to prune it later
                ext.score = Core::Type<Score>::max;
            }
        }
    }
}

void BatchedTreeTimesyncBeamSearch::createNewBeam(SegmentState& segment, ThreadBuffers& buffers) {
    auto& beam              = segment.beam;
    auto& newBeam           = segment.newBeam;
    auto& wordEndExtensions = segment.wordEndExtensions;
    auto& wordEndHypotheses = segment.wordEndHypotheses;

    // Create new label hypotheses from extension candidates
    buffers.buildWithinWordHypsTime.start();
    newBeam.clear();
    for (auto const& extension : segment.withinWordExtensions) {
        auto const& baseHyp = beam[extension.baseHypIndex];

        std::vector<Nn::ScoringContextRef> newScoringContexts;
        for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
            newScoringContexts.push_back(labelScorers_[scorerIdx]->extendedScoringContext(
                    baseHyp.scoringContexts[scorerIdx],
                    extension.nextToken,
                    extension.transitionType));
        }

        newBeam.push_back({baseHyp, extension, newScoringContexts});
    }
    buffers.buildWithinWordHypsTime.stop();

    // For all hypotheses at the same state and with the same scoring context and LM history
    // keep only the best since they will all develop in the same way
    buffers.recombinationTime.start();
    recombination(newBeam, false, buffers.tempHypotheses);
    buffers.recombinationTime.stop();
    segment.numWithinWordHypsAfterRecombination += newBeam.size();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-within-word-hyps-after-recombination", newBeam.size()) + Core::XmlAttribute("segment", segment.name);
    }

    buffers.beamPruningTime.start();
    scorePruning(newBeam, Core::Type<Score>::max, maxBeamSizes_[labelScorers_.size() - 1], buffers.scoreHistogram);
    buffers.beamPruningTime.stop();
    segment.numWithinWordHypsAfterPruning += newBeam.size();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-within-word-hyps-after-pruning", newBeam.size()) + Core::XmlAttribute("scorer", labelScorers_.size()) + Core::XmlAttribute("segment", segment.name);
    }

    /*
     * Expand hypotheses to word-end hypotheses and incorporate the language model
     */
    buffers.wordEndExpansionTime.start();
    buffers.wordEndExtensionTime.start();
    wordEndExtensions.clear();
    for (size_t hypIndex = 0ul; hypIndex < newBeam.size(); ++hypIndex) {
        auto& hyp = newBeam[hypIndex];

        if (enableLmLookahead_) {
            // Subtract the LM lookahead score again
            hyp.score -= hyp.lookaheadScore;
            hyp.lookaheadScore = 0.0;
        }

        // All exits below share this hypothesis' scoring contexts, so each accessor is fetched
        // only once. A scorer that is not ready to score leaves its slot empty and is retried.
        auto& wordEndScoreAccessors = buffers.wordEndScoreAccessors;
        wordEndScoreAccessors.assign(labelScorers_.size(), Nn::ScoreAccessorRef());

        // Create one word-end hypothesis for each exit
        for (size_t i = stateExitsOffset_[hyp.currentState]; i < stateExitsOffset_[hyp.currentState + 1]; ++i) {
            const PersistentStateTree::Exit exit      = stateExits_[i];
            auto const*                     lemmaPron = lexicon_->lemmaPronunciation(exit.pronunciation);
            auto const*                     lemma     = lemmaPron->lemma();

            Score                               lmScore = 0;
            const Bliss::SyntacticTokenSequence sts     = lemma->syntacticTokenSequence();
            if (sts.size() != 0) {
                require(sts.size() == 1);
                buffers.lmTime.start();
                lmScore = languageModel_->score(hyp.lmHistory, sts.front());
                buffers.lmTime.stop();
            }

            Score              penalty               = 0.0;
            Nn::TransitionType wordEndtransitionType = Nn::TransitionType::WORD_EXIT;
            if (lemma == lexicon_->specialLemma("silence")) {
                wordEndtransitionType = Nn::TransitionType::SILENCE_EXIT;
            }
            else if (nonWordLemmas_.contains(lemma)) {
                wordEndtransitionType = Nn::TransitionType::NONWORD_EXIT;
            }
            for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
                if (not labelScorers_[scorerIdx]->scoresTransition(wordEndtransitionType)) {
                    continue;
                }
                auto& scoreAccessor = wordEndScoreAccessors[scorerIdx];
                if (not scoreAccessor) {
                    auto fetched = getScoreAccessor(scorerIdx, hyp.scoringContexts[scorerIdx]);
                    if (not fetched) {
                        continue;
                    }
                    scoreAccessor = *fetched;
                }
                penalty += scoreAccessor->getScore(wordEndtransitionType);
            }

            wordEndExtensions.push_back({
                    .pron           = lemmaPron,
                    .rootState      = exit.transitState,
                    .score          = hyp.score + lmScore + penalty,
                    .transitionType = wordEndtransitionType,
                    .baseHypIndex   = hypIndex,
            });
        }
    }

    buffers.wordEndExtensionTime.stop();

    segment.numWordEndExtensionsBeforePruning += wordEndExtensions.size();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-word-end-extensions-before-pruning", wordEndExtensions.size()) + Core::XmlAttribute("segment", segment.name);
    }

    /*
     * Prune set of word-end extensions by score.
     */
    buffers.wordEndScorePruningTime.start();
    scorePruning(wordEndExtensions, wordEndScoreThreshold_, wordEndExtensions.size(), buffers.scoreHistogram);
    buffers.wordEndScorePruningTime.stop();
    segment.numWordEndHypsAfterScorePruning += wordEndExtensions.size();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-word-end-hyps-after-score-pruning", wordEndExtensions.size()) + Core::XmlAttribute("segment", segment.name);
    }

    // Create new word-end label hypotheses from word-end extension candidates, update the LM history and prepare the new lookahead if its history has changed
    buffers.wordEndHypBuildingTime.start();
    wordEndHypotheses.clear();
    for (auto& extension : wordEndExtensions) {
        auto const& baseHyp = newBeam[extension.baseHypIndex];

        auto        newLmHistory = baseHyp.lmHistory;
        auto const& sts          = extension.pron->lemma()->syntacticTokenSequence();

        LanguageModelLookahead::ContextLookaheadReference newLookahead        = baseHyp.lookahead;
        Lm::History                                       newLookaheadHistory = baseHyp.lookaheadHistory;
        Score                                             newLookaheadBackOff = baseHyp.lookaheadBackOff;

        if (sts.size() != 0) {
            require(sts.size() == 1);
            const Bliss::SyntacticToken* st = sts.front();
            newLmHistory                    = languageModel_->extendedHistory(newLmHistory, st);

            if (enableLmLookahead_) {
                buffers.lmLookaheadTime.start();
                newLookaheadHistory = lookaheadLm_->extendedHistory(baseHyp.lookaheadHistory, st);

                if (!(newLookaheadHistory == baseHyp.lookaheadHistory)) {
                    // The lookahead context changed, so a table the base may have backed off to no
                    // longer applies: start the new word from the table for the new context.
                    getLmLookahead(newLookahead, newLookaheadHistory);
                    newLookaheadBackOff = 0.0;
                }
                buffers.lmLookaheadTime.stop();
            }
        }

        wordEndHypotheses.push_back({baseHyp, extension, newLmHistory, newLookahead, newLookaheadHistory, newLookaheadBackOff});
    }

    buffers.wordEndHypBuildingTime.stop();

    buffers.wordEndRecombinationTime.start();
    recombination(wordEndHypotheses, true, buffers.tempHypotheses);
    buffers.wordEndRecombinationTime.stop();
    segment.numWordEndHypsAfterRecombination += wordEndHypotheses.size();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-word-end-hyps-after-recombination", wordEndHypotheses.size()) + Core::XmlAttribute("segment", segment.name);
    }

    // Prune set of word-end hypotheses by max beam size.
    buffers.wordEndBeamPruningTime.start();
    scorePruning(wordEndHypotheses, Core::Type<Score>::max, maxWordEndBeamSize_, buffers.scoreHistogram);
    buffers.wordEndBeamPruningTime.stop();
    buffers.wordEndExpansionTime.stop();
    segment.numWordEndHypsAfterBeamPruning += wordEndHypotheses.size();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-word-end-hyps-after-beam-pruning", wordEndHypotheses.size()) + Core::XmlAttribute("segment", segment.name);
    }

    beam.swap(newBeam);
    beam.insert(beam.end(), wordEndHypotheses.begin(), wordEndHypotheses.end());

    ++segment.currentSearchStep;

    /*
     * Apply maximum-stable-delay-pruning.
     */
    segment.stableDelayLimitExceeded = false;
    if (segment.currentSearchStep % maximumStableDelayPruningInterval_ == 0) {
        buffers.beamPruningTime.start();
        segment.stableDelayLimitExceeded = not maximumStableDelayPruning(segment, buffers.tempHypotheses);
        buffers.beamPruningTime.stop();
        if (stepwiseStatisticsChannel_.isOpen()) {
            stepwiseStatisticsChannel_ << Core::XmlFull("num-hyps-after-maximum-stable-delay-pruning", beam.size()) + Core::XmlAttribute("segment", segment.name);
        }
    }

    // Counted after all pruning of this step, including maximum-stable-delay pruning
    segment.numActiveHyps += beam.size();

    /*
     * Calculate number of active trees
     */
    std::vector<Lm::History> seenHistories;
    for (auto const& hyp : beam) {
        if (std::find(seenHistories.begin(), seenHistories.end(), hyp.lmHistory) == seenHistories.end()) {
            seenHistories.push_back(hyp.lmHistory);
        }
    }
    segment.numActiveTrees += seenHistories.size();

    /*
     * Log statistics about the new beam.
     */
    if (debugChannel_.isOpen()) {
        std::stringstream ss;
        for (size_t hypIdx = 0ul; hypIdx < beam.size(); ++hypIdx) {
            ss << "Segment " << segment.name << ": Hypothesis " << hypIdx + 1ul << ":  " << beam[hypIdx].toString() << "\n";
        }
        ss << "\n";
        debugChannel_ << ss.str();
    }

    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("active-hyps", beam.size()) + Core::XmlAttribute("segment", segment.name);
        stepwiseStatisticsChannel_ << Core::XmlFull("num-active-trees", seenHistories.size()) + Core::XmlAttribute("segment", segment.name);
        stepwiseStatisticsChannel_ << Core::XmlFull("best-hyp-score", getBestHypothesis(beam).score) + Core::XmlAttribute("segment", segment.name);
        stepwiseStatisticsChannel_ << Core::XmlFull("worst-hyp-score", getWorstHypothesis(beam).score) + Core::XmlAttribute("segment", segment.name);
    }
}

void BatchedTreeTimesyncBeamSearch::finalizeSegments(std::vector<SegmentState*> const& segments) {
    auto& tempHypotheses = threadBuffers_.front().tempHypotheses;  // Finalization runs in the main thread
    // Collect the hypotheses to finalize: those in final states of the search tree or, as fallback, all hypotheses
    for (auto* segment : segments) {
        auto& finalHypotheses = segment->finalHypotheses;
        finalHypotheses.clear();
        for (auto const& hyp : segment->beam) {
            if (network_->finalStates.contains(hyp.currentState)) {
                finalHypotheses.push_back(hyp);
            }
        }

        if (finalHypotheses.empty() and sentenceEndFallback_) {  // There was no valid final hypothesis in the beam
            warning() << "No active word-end hypothesis at the end of segment " << segment->name;
            log() << "Use sentence-end fallback";
            // The trace of the unfinished word keeps an empty pronunciation
            finalHypotheses = segment->beam;
        }

        segment->withinWordExtensions.clear();
        for (size_t hypIndex = 0ul; hypIndex < finalHypotheses.size(); ++hypIndex) {
            auto& hyp = finalHypotheses[hypIndex];
            segment->withinWordExtensions.push_back(
                    {.nextToken        = sentenceEndLabelIndex_,
                     .nextState        = hyp.currentState,
                     .timeframe        = hyp.trace->time,
                     .score            = hyp.score,
                     .lookaheadScore   = 0,
                     .lookahead        = {},
                     .lookaheadBackOff = 0,
                     .transitionType   = Nn::TransitionType::SENTENCE_END,
                     .baseHypIndex     = hypIndex});
        }
    }

    // Score sentence-end with all label scorers (for the final hypotheses of all segments together)
    for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
        if (not labelScorers_[scorerIdx]->scoresTransition(Nn::TransitionType::SENTENCE_END)) {
            continue;
        }

        scoringContexts_.clear();
        for (auto* segment : segments) {
            for (auto const& hyp : segment->finalHypotheses) {
                scoringContexts_.push_back(hyp.scoringContexts[scorerIdx]);
            }
        }
        if (scoringContexts_.empty()) {
            continue;
        }

        finalizeScoringTime_.start();
        auto scoreAccessors = getScoreAccessors(scorerIdx, scoringContexts_);
        finalizeScoringTime_.stop();

        size_t accessorIdx = 0ul;
        for (auto* segment : segments) {
            for (auto& ext : segment->withinWordExtensions) {
                auto const& scoreAccessor = scoreAccessors[accessorIdx++];
                if (not scoreAccessor) {
                    continue;
                }
                auto denseScores = (*scoreAccessor)->getDenseScores();
                if (denseScores and sentenceEndLabelIndex_ < denseScores->size()) {
                    ext.score += (*denseScores)[sentenceEndLabelIndex_];
                }
                else {
                    ext.score += (*scoreAccessor)->getScore(ext.transitionType, sentenceEndLabelIndex_);
                }
                ext.timeframe = std::max(ext.timeframe, (*scoreAccessor)->getTime());
            }
        }
    }

    for (auto* segment : segments) {
        auto& beam            = segment->beam;
        auto& newBeam         = segment->newBeam;
        auto& finalHypotheses = segment->finalHypotheses;

        if (not finalHypotheses.empty()) {
            newBeam.clear();
            for (auto const& ext : segment->withinWordExtensions) {
                auto const& baseHyp = finalHypotheses[ext.baseHypIndex];
                // The scoring context is not updated as no further scoring is done afterwards
                newBeam.push_back({baseHyp, ext, baseHyp.scoringContexts});
            }

            segment->wordEndExtensions.clear();
            for (size_t hypIndex = 0ul; hypIndex < newBeam.size(); ++hypIndex) {
                auto& hyp = newBeam[hypIndex];
                // Add the LM's sentence-end score
                // The LM history is not updated as this is the last LM scoring step
                Lm::Score sentenceEndScore = languageModel_->sentenceEndScore(hyp.lmHistory);
                segment->wordEndExtensions.push_back({
                        .pron           = sentenceEndLemma_->pronunciations().first,
                        .rootState      = hyp.currentState,
                        .score          = hyp.score + sentenceEndScore,
                        .transitionType = Nn::TransitionType::SENTENCE_END,
                        .baseHypIndex   = hypIndex,
                });
            }

            tempHypotheses.clear();
            for (auto const& ext : segment->wordEndExtensions) {
                auto const& baseHyp = newBeam[ext.baseHypIndex];
                tempHypotheses.push_back({baseHyp, ext, baseHyp.lmHistory, baseHyp.lookahead, baseHyp.lookaheadHistory, baseHyp.lookaheadBackOff});
            }
        }
        else {  // No valid final hypotheses and no sentence-end fallback
            // Construct an empty hypothesis with a lattice containing only one empty pronunciation from start to end
            tempHypotheses.clear();
            tempHypotheses.push_back(LabelHypothesis());
            tempHypotheses.front().trace->time          = beam.front().trace->time;  // Retrieve the timeframe from any hyp in the old beam
            tempHypotheses.front().trace->pronunciation = nullptr;
            tempHypotheses.front().trace->predecessor   = Core::ref(new LatticeTrace(0, {0, 0}, {}));
        }

        beam.swap(tempHypotheses);

        segment->numActiveHyps += beam.size();

        // Log statistics about the final beam
        if (stepwiseStatisticsChannel_.isOpen()) {
            stepwiseStatisticsChannel_ << Core::XmlOpen("final-beam-stats") + Core::XmlAttribute("segment", segment->name);
            stepwiseStatisticsChannel_ << Core::XmlFull("active-hyps", beam.size());
            stepwiseStatisticsChannel_ << Core::XmlFull("best-hyp-score", getBestHypothesis(beam).score);
            stepwiseStatisticsChannel_ << Core::XmlFull("worst-hyp-score", getWorstHypothesis(beam).score);
            stepwiseStatisticsChannel_ << Core::XmlClose("final-beam-stats");
        }

        if (debugChannel_.isOpen()) {
            std::stringstream ss;
            for (size_t hypIdx = 0ul; hypIdx < beam.size(); ++hypIdx) {
                ss << "Segment " << segment->name << ": Hypothesis " << hypIdx + 1ul << ":  " << beam[hypIdx].toString() << "\n";
            }
            ss << "\n";
            debugChannel_ << ss.str();
        }

        finishSegmentSearch(*segment);
    }
}

void BatchedTreeTimesyncBeamSearch::releaseSegmentState(SegmentStateBase& segmentBase) {
    auto& segment = static_cast<SegmentState&>(segmentBase);
    // The scoring contexts are not needed anymore
    for (auto& hyp : segment.beam) {
        hyp.scoringContexts.clear();
    }
    segment.withinWordExtensions.clear();
    segment.wordEndExtensions.clear();
    segment.newBeam.clear();
    segment.wordEndHypotheses.clear();
    segment.finalHypotheses.clear();
}

BatchedTreeTimesyncBeamSearch::LabelHypothesis const& BatchedTreeTimesyncBeamSearch::getBestHypothesis(std::vector<LabelHypothesis> const& beam) const {
    verify(not beam.empty());

    return *std::min_element(beam.begin(), beam.end());
}

BatchedTreeTimesyncBeamSearch::LabelHypothesis const& BatchedTreeTimesyncBeamSearch::getWorstHypothesis(std::vector<LabelHypothesis> const& beam) const {
    verify(not beam.empty());

    return *std::max_element(beam.begin(), beam.end());
}

void BatchedTreeTimesyncBeamSearch::logSegmentStatistics(SegmentStateBase const& segmentBase) const {
    if (not statisticsChannel_.isOpen()) {
        return;
    }
    auto const& segment = static_cast<SegmentState const&>(segmentBase);
    auto&       channel = statisticsChannel_;
    channel << Core::XmlOpen("search-statistics") + Core::XmlAttribute("segment", segment.name);
    channel << Core::XmlFull("num-search-steps", segment.currentSearchStep);
    segment.numInputHyps.write(channel);
    segment.numWithinWordExtensionsBeforeFirstPruning.write(channel);
    for (size_t i = 0ul; i < segment.numWithinWordHypsAfterIntermediatePruning.size(); ++i) {
        segment.numWithinWordHypsAfterIntermediatePruning[i].write(channel, {Core::XmlAttribute("scorer", i + 1)});
    }
    segment.numWithinWordHypsAfterRecombination.write(channel);
    segment.numWithinWordHypsAfterPruning.write(channel);
    segment.numWordEndExtensionsBeforePruning.write(channel);
    segment.numWordEndHypsAfterScorePruning.write(channel);
    segment.numWordEndHypsAfterRecombination.write(channel);
    segment.numWordEndHypsAfterBeamPruning.write(channel);
    segment.numActiveHyps.write(channel);
    segment.numActiveTrees.write(channel);
    channel << Core::XmlClose("search-statistics");
}

void BatchedTreeTimesyncBeamSearch::logSearchStepBreakdown() const {
    // Timers of the search threads are summed up
    auto threadSum = [this](auto getTimer) {
        double sum = 0.0;
        for (auto const& buffers : threadBuffers_) {
            sum += getTimer(buffers).elapsedMilliseconds();
        }
        return sum;
    };
    using Timer = Core::StopWatch const&;

    auto& channel = statisticsChannel_;
    for (size_t i = 0ul; i < scoreAndPruneExtensionsTimes_.size(); ++i) {
        channel << Core::XmlOpen("score-and-prune-extensions-time") + Core::XmlAttribute("scorer", i + 1) + Core::XmlAttribute("total", scoreAndPruneExtensionsTimes_[i].elapsedMilliseconds());
        channel << Core::XmlFull("scoring-time", scoringTimes_[i].elapsedMilliseconds());
        channel << Core::XmlFull("score-readout-time", scoreReadoutTimes_[i].elapsedMilliseconds() + threadSum([i](ThreadBuffers const& buffers) -> Timer { return buffers.scoreReadoutTimes[i]; }));
        channel << Core::XmlFull("intermediate-pruning-time", intermediatePruningTimes_[i].elapsedMilliseconds() + threadSum([i](ThreadBuffers const& buffers) -> Timer { return buffers.intermediatePruningTimes[i]; }));
        channel << Core::XmlClose("score-and-prune-extensions-time");
    }
    channel << Core::XmlFull("build-within-word-hyps-time", threadSum([](ThreadBuffers const& buffers) -> Timer { return buffers.buildWithinWordHypsTime; }));
    channel << Core::XmlOpen("word-end-expansion-time") + Core::XmlAttribute("total", threadSum([](ThreadBuffers const& buffers) -> Timer { return buffers.wordEndExpansionTime; }));
    channel << Core::XmlOpen("word-end-extension-time") + Core::XmlAttribute("total", threadSum([](ThreadBuffers const& buffers) -> Timer { return buffers.wordEndExtensionTime; }));
    channel << Core::XmlFull("lm-time", threadSum([](ThreadBuffers const& buffers) -> Timer { return buffers.lmTime; }));
    channel << Core::XmlClose("word-end-extension-time");
    channel << Core::XmlFull("word-end-score-pruning-time", threadSum([](ThreadBuffers const& buffers) -> Timer { return buffers.wordEndScorePruningTime; }));
    channel << Core::XmlOpen("word-end-hyp-building-time") + Core::XmlAttribute("total", threadSum([](ThreadBuffers const& buffers) -> Timer { return buffers.wordEndHypBuildingTime; }));
    channel << Core::XmlFull("lm-lookahead-time", threadSum([](ThreadBuffers const& buffers) -> Timer { return buffers.lmLookaheadTime; }));
    channel << Core::XmlClose("word-end-hyp-building-time");
    channel << Core::XmlFull("word-end-recombination-time", threadSum([](ThreadBuffers const& buffers) -> Timer { return buffers.wordEndRecombinationTime; }));
    channel << Core::XmlFull("word-end-beam-pruning-time", threadSum([](ThreadBuffers const& buffers) -> Timer { return buffers.wordEndBeamPruningTime; }));
    channel << Core::XmlClose("word-end-expansion-time");
    channel << Core::XmlFull("recombination-time", threadSum([](ThreadBuffers const& buffers) -> Timer { return buffers.recombinationTime; }));
    channel << Core::XmlFull("beam-pruning-time", threadSum([](ThreadBuffers const& buffers) -> Timer { return buffers.beamPruningTime; }));
    channel << Core::XmlOpen("finalize-time") + Core::XmlAttribute("total", finalizeTime_.elapsedMilliseconds());
    channel << Core::XmlFull("scoring-time", finalizeScoringTime_.elapsedMilliseconds());
    channel << Core::XmlClose("finalize-time");
}

void BatchedTreeTimesyncBeamSearch::resetSearchStepBreakdown() {
    for (auto* timers : {&scoreAndPruneExtensionsTimes_, &scoringTimes_, &scoreReadoutTimes_, &intermediatePruningTimes_}) {
        for (auto& timer : *timers) {
            timer.reset();
        }
    }
    finalizeTime_.reset();
    finalizeScoringTime_.reset();
    for (auto& buffers : threadBuffers_) {
        for (auto* timers : {&buffers.scoreReadoutTimes, &buffers.intermediatePruningTimes}) {
            for (auto& timer : *timers) {
                timer.reset();
            }
        }
        for (auto* timer : {&buffers.buildWithinWordHypsTime, &buffers.wordEndExpansionTime, &buffers.wordEndExtensionTime, &buffers.lmTime,
                            &buffers.wordEndScorePruningTime, &buffers.wordEndHypBuildingTime, &buffers.lmLookaheadTime, &buffers.wordEndRecombinationTime,
                            &buffers.wordEndBeamPruningTime, &buffers.recombinationTime, &buffers.beamPruningTime}) {
            timer->reset();
        }
    }
}

void BatchedTreeTimesyncBeamSearch::logAdditionalStatistics() const {
    if (enableLmLookahead_) {
        lmLookahead_->logStatistics();
    }
}

Nn::TransitionType BatchedTreeTimesyncBeamSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel, bool isSameState) const {
    bool prevIsBlank = (useBlank_ and prevLabel == blankLabelIndex_);
    bool nextIsBlank = (useBlank_ and nextLabel == blankLabelIndex_);

    bool prevIsSilence = (useSilence_ and prevLabel == silenceLabelIndex_);
    bool nextIsSilence = (useSilence_ and nextLabel == silenceLabelIndex_);

    if (isSameState) {
        if (prevIsBlank) {
            return Nn::TransitionType::BLANK_LOOP;
        }
        if (prevIsSilence) {
            return Nn::TransitionType::SILENCE_LOOP;
        }
        else if (collapseRepeatedLabels_) {
            return Nn::TransitionType::LABEL_LOOP;
        }
    }

    if (prevLabel == Nn::invalidLabelIndex) {
        if (nextIsBlank) {
            return Nn::TransitionType::INITIAL_BLANK;
        }
        else if (nextIsSilence) {
            return Nn::TransitionType::INITIAL_SILENCE;
        }
        else {
            return Nn::TransitionType::INITIAL_LABEL;
        }
    }

    if (prevIsBlank) {
        if (nextIsBlank) {
            return Nn::TransitionType::BLANK_LOOP;
        }
        else {
            return Nn::TransitionType::BLANK_TO_LABEL;
        }
    }
    else if (prevIsSilence) {
        if (nextIsSilence) {
            return Nn::TransitionType::SILENCE_LOOP;
        }
        else {
            return Nn::TransitionType::SILENCE_TO_LABEL;
        }
    }
    else {
        if (nextIsBlank) {
            return Nn::TransitionType::LABEL_TO_BLANK;
        }
        else if (nextIsSilence) {
            return Nn::TransitionType::LABEL_TO_SILENCE;
        }
        else {
            // Assume that we can only have a label-loop if the state in the search tree didn't change
            return Nn::TransitionType::LABEL_TO_LABEL;
        }
    }
}

template<typename Element>
void BatchedTreeTimesyncBeamSearch::scorePruning(std::vector<Element>& hypotheses, Score relativeThreshold, size_t maxBeamSize, Histogram& scoreHistogram) {
    hypotheses.erase(
            std::remove_if(
                    hypotheses.begin(),
                    hypotheses.end(),
                    [](auto const& hyp) {
                        return Math::isinf(hyp.score) or hyp.score >= Core::Type<Score>::max;
                    }),
            hypotheses.end());

    if (hypotheses.empty()) {
        return;
    }

    if (not histogramPruningEnabled_ and hypotheses.size() > maxBeamSize) {
        // Plain max beam size pruning: keep exactly the maxBeamSize best hypotheses
        std::nth_element(hypotheses.begin(), hypotheses.begin() + maxBeamSize, hypotheses.end(), [](auto const& a, auto const& b) { return a.score < b.score; });
        hypotheses.resize(maxBeamSize);
    }

    if (hypotheses.size() <= maxBeamSize and relativeThreshold == Core::Type<Score>::max) {
        // Neither relative score pruning nor max beam size pruning triggers
        return;
    }

    // Find ranges for score histogram and setting absolute threshold
    Score lowerScore = Core::Type<Score>::max;
    Score upperScore = Core::Type<Score>::min;

    for (auto const& hyp : hypotheses) {
        lowerScore = std::min(lowerScore, hyp.score);
        upperScore = std::max(upperScore, hyp.score);
    }

    if (lowerScore == upperScore) {
        // All scores are the same (usually only happens when exactly 1 hyp is active)
        if (hypotheses.size() > maxBeamSize) {
            hypotheses.resize(maxBeamSize);
        }
        return;
    }

    Score absoluteThreshold = upperScore;

    // Pruning by relative score threshold
    if (relativeThreshold != Core::Type<Score>::max) {
        absoluteThreshold = lowerScore + relativeThreshold;
    }

    // Pruning by max beam size
    if (hypotheses.size() > maxBeamSize) {
        scoreHistogram.clear();
        scoreHistogram.setLimits(lowerScore, upperScore);

        for (auto const& hyp : hypotheses) {
            scoreHistogram += hyp.score;
        }

        absoluteThreshold = std::min(absoluteThreshold, scoreHistogram.quantile(maxBeamSize));
    }

    if (absoluteThreshold >= upperScore) {
        // Nothing will be pruned
        return;
    }

    // Remove elements with score > absoluteThreshold
    hypotheses.erase(
            std::remove_if(
                    hypotheses.begin(),
                    hypotheses.end(),
                    [absoluteThreshold](auto const& hyp) { return hyp.score > absoluteThreshold; }),
            hypotheses.end());
}

void BatchedTreeTimesyncBeamSearch::recombination(std::vector<BatchedTreeTimesyncBeamSearch::LabelHypothesis>& hypotheses, bool createTraceSiblings, std::vector<LabelHypothesis>& tempHypotheses) {
    if (not recombinationEnabled_) {
        return;
    }

    // Represents a unique combination of StateId, ScoringContext and LmHistory
    struct RecombinationContext {
        StateId                            state;
        std::vector<Nn::ScoringContextRef> scoringContexts;
        Lm::History                        lmHistory;

        RecombinationContext(LabelHypothesis const& hyp)
                : state(hyp.currentState), scoringContexts(hyp.scoringContexts), lmHistory(hyp.lmHistory) {}

        bool operator==(const RecombinationContext& other) const {
            if (state != other.state) {
                return false;
            }
            if (lmHistory != other.lmHistory) {
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
        size_t operator()(const RecombinationContext& context) const {
            size_t hash = Core::combineHashes(context.state, Lm::History::Hash{}(context.lmHistory));
            for (auto const& scoringContext : context.scoringContexts) {
                hash = Core::combineHashes(hash, Nn::ScoringContextHash{}(scoringContext));
            }
            return hash;
        }
    };

    tempHypotheses.clear();
    // Reserve capacity because future reallocations would break the raw pointer we are storing later
    tempHypotheses.reserve(hypotheses.size());
    // Map each unique combination of StateId, ScoringContext and LmHistory in newHypotheses to its hypothesis
    std::unordered_map<RecombinationContext, LabelHypothesis*, RecombinationContextHash> seenCombinations;
    for (auto const& hyp : hypotheses) {
        // Use try_emplace to check if the combination already exists and create a new entry if not at the same time
        auto [it, inserted] = seenCombinations.try_emplace({hyp}, nullptr);

        if (inserted) {
            // First time seeing this combination so move it over to `newHypotheses`
            tempHypotheses.push_back(std::move(hyp));
            it->second = &tempHypotheses.back();
        }
        else {
            if (network_->isRoot(hyp.currentState)) {
                verify(not hyp.trace->sibling);
            }

            auto* existingHyp = it->second;
            if (hyp.score - hyp.lookaheadScore < existingHyp->score - existingHyp->lookaheadScore) {
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

    hypotheses.swap(tempHypotheses);
}

void BatchedTreeTimesyncBeamSearch::createSuccessorLookups() {
    size_t numStates = network_->structure.stateCount();

    stateSuccessorsOffset_.assign(numStates + 1, 0);
    stateExitsOffset_.assign(numStates + 1, 0);
    stateSuccessors_.clear();
    stateExits_.clear();

    for (u32 state = 1; state < numStates; ++state) {
        // The offset for the next state is the current size of the data vectors
        stateSuccessorsOffset_[state] = stateSuccessors_.size();
        stateExitsOffset_[state]      = stateExits_.size();

        // Add successor/exit data to contiguous vectors
        for (HMMStateNetwork::SuccessorIterator it = network_->structure.successors(state); it; ++it) {
            if (not it.isLabel()) {
                stateSuccessors_.push_back(*it);
            }
            else {
                stateExits_.push_back(network_->exits[it.label()]);
            }
        }
    }
    stateSuccessorsOffset_[numStates] = stateSuccessors_.size();
    stateExitsOffset_[numStates]      = stateExits_.size();
}

void BatchedTreeTimesyncBeamSearch::getLmLookahead(LanguageModelLookahead::ContextLookaheadReference& lookahead, Lm::History history) {
    std::lock_guard<std::mutex> lock(lookaheadMutex_);
    lookahead = lmLookahead_->getLookahead(history);
    lmLookahead_->fill(lookahead, sparseLmLookahead_);
    if (keepLookaheadTables_) {
        lookaheadTablesInUse_[currentGroup_].push_back(lookahead);
    }
}

Score BatchedTreeTimesyncBeamSearch::getLmLookaheadScore(std::vector<LabelHypothesis> const& beam, BatchedTreeTimesyncBeamSearch::WithinWordExtensionCandidate& extension) {
    // The resolved table/back-off belong to `extension.nextState`, not the base hypothesis, so
    // they're stored on the candidate rather than written back to `baseHyp`
    auto const& baseHyp = beam[extension.baseHypIndex];

    extension.lookahead        = baseHyp.lookahead;
    extension.lookaheadBackOff = baseHyp.lookaheadBackOff;
    if (!extension.lookahead) {
        getLmLookahead(extension.lookahead, baseHyp.lookaheadHistory);
        extension.lookaheadBackOff = 0.0;
    }

    Score lookaheadScore = 0;
    bool  scoreFound     = false;
    do {
        if (extension.lookahead->isSparse()) {  // Sparse lookahead
            auto lookaheadHash = lmLookahead_->lookaheadHash(extension.nextState);
            scoreFound         = extension.lookahead->getScoreForLookAheadHashSparse(lookaheadHash, lookaheadScore);
        }
        else {  // Non-sparse lookahead
            auto lookaheadId = lmLookahead_->lookaheadId(extension.nextState);
            lookaheadScore   = extension.lookahead->scoreForLookAheadIdNormal(lookaheadId);
            scoreFound       = true;
        }

        if (!scoreFound) {  // No lookahead table entry, use back-off
            const Lm::BackingOffLm* lm = dynamic_cast<const Lm::BackingOffLm*>(lookaheadLm_->unscaled().get());
            // Accumulated separately from lookaheadScore, since a successful lookup assigns rather than adds to it
            extension.lookaheadBackOff += extension.lookahead->backOffScore();
            // Reduce the active table's history (not the base hypothesis' one, which may differ
            // under history-limit) to avoid re-selecting the same table and double-charging its back-off
            Lm::History tableHistory   = extension.lookahead->history();
            u32         lengthLimit    = std::max(lm->historyLength(tableHistory), 1u) - 1u;
            auto        reducedHistory = lm->reducedHistory(tableHistory, lengthLimit);
            if (reducedHistory == tableHistory) {
                // Additional fail-safe for the loop. If we cannot reduce the history further we abort.
                break;
            }
            getLmLookahead(extension.lookahead, reducedHistory);
        }
    } while (!scoreFound);

    return lookaheadScore + extension.lookaheadBackOff;
}

bool BatchedTreeTimesyncBeamSearch::maximumStableDelayPruning(SegmentState& segment, std::vector<LabelHypothesis>& tempHypotheses) {
    auto& beam = segment.beam;

    if (segment.currentSearchStep + 1 <= maximumStableDelay_) {
        return true;
    }

    auto cutoff = segment.currentSearchStep + 1 - maximumStableDelay_;

    // Find trace of current best hypothesis that has a recent word-end within the limit
    Score                   bestScore = Core::Type<Score>::max;
    Core::Ref<LatticeTrace> root;

    for (auto const& hyp : beam) {
        if (hyp.score < bestScore and hyp.trace->time >= cutoff) {
            bestScore = hyp.score;
            root      = hyp.trace;
        }
    }

    // No Hypothesis with a recent word-end was found so just take the overall best as fallback
    bool withinLimit = true;
    if (not root) {
        root        = getBestHypothesis(beam).trace;
        withinLimit = false;
    }

    // Determine the right predecessor of best trace for pruning. `root->time` should be after the cutoff and `root->predecessor->time` before the cutoff
    Core::Ref<LatticeTrace> preRoot = root->predecessor;

    while (preRoot and preRoot->time >= cutoff) {
        root    = preRoot;
        preRoot = preRoot->predecessor;
    }

    // Perform pruning on root
    tempHypotheses.clear();
    for (auto const& hyp : beam) {
        auto curr = hyp.trace;
        while (curr and curr != root and curr->time > root->time) {
            curr = curr->predecessor;
        }
        if (curr == root) {
            tempHypotheses.push_back(hyp);
        }
    }
    beam.swap(tempHypotheses);

    return withinLimit;
}

}  // namespace Search
