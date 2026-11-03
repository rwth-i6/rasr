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

#include "BatchedLexiconfreeTimesyncBeamSearch.hh"

#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <strings.h>

#include <Core/CollapsedVector.hh>
#include <Core/XmlStream.hh>
#include <Lattice/LatticeAdaptor.hh>
#include <Math/Utilities.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/Histogram.hh>
#include <Search/Traceback.hh>
#include <Search/TracebackHelper.hh>

namespace Search {

namespace {

enum RecombinationMode {
    RecombinationModeOff,
    RecombinationModeOn,
};

}  // namespace

/*
 * =======================
 * === LabelHypothesis ===
 * =======================
 */

BatchedLexiconfreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContexts(),
          currentToken(Nn::invalidLabelIndex),
          score(0.0),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))) {}

BatchedLexiconfreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        BatchedLexiconfreeTimesyncBeamSearch::LabelHypothesis const&    base,
        BatchedLexiconfreeTimesyncBeamSearch::ExtensionCandidate const& extension,
        std::vector<Nn::ScoringContextRef> const&                       newScoringContexts)
        : scoringContexts(newScoringContexts),
          currentToken(extension.nextToken),
          score(extension.score),
          trace() {
    Core::Ref<LatticeTrace> predecessor;
    switch (extension.transitionType) {
        case Nn::TransitionType::LABEL_LOOP:
        case Nn::TransitionType::BLANK_LOOP:
        case Nn::TransitionType::SILENCE_LOOP:
            predecessor = base.trace->predecessor;
            break;
        default:
            predecessor = base.trace;
            break;
    }

    // Only increment timeframe when not SENTENCE_END
    auto timeframe = extension.transitionType == Nn::TransitionType::SENTENCE_END ? extension.timeframe : extension.timeframe + 1;
    trace          = Core::ref(new LatticeTrace(
            predecessor,
            extension.pron,
            timeframe,
            {score, 0},
            {}));
}

std::string BatchedLexiconfreeTimesyncBeamSearch::LabelHypothesis::toString() const {
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
 * ====================
 * === SegmentState ===
 * ====================
 */

BatchedLexiconfreeTimesyncBeamSearch::SegmentState::SegmentState(SegmentHandle handle, std::string const& name, size_t numLabelScorers)
        : SegmentStateBase(handle, name),
          beam(),
          currentSearchStep(0ul),
          hypIndexToContextIndexMap(),
          extensions(),
          newBeam(),
          stepFailed(false),
          stableDelayLimitExceeded(false),
          numInputHyps("num-input-hyps"),
          numExtensionsBeforeFirstPruning("num-extensions-before-first-pruning"),
          numHypsAfterIntermediatePruning(numLabelScorers, Core::Statistics<u32>("num-hyps-after-intermediate-pruning")),
          numHypsAfterRecombination("num-hyps-after-recombination"),
          numHypsAfterPruning("num-hyps-after-pruning"),
          numActiveHyps("num-active-hyps") {
}

/*
 * ============================================
 * === BatchedLexiconfreeTimesyncBeamSearch ===
 * ============================================
 */

BatchedLexiconfreeTimesyncBeamSearch::ThreadBuffers::ThreadBuffers(size_t numHistogramBins)
        : tempHypotheses(),
          selectedTokens(),
          tokenHeap(),
          scoreHistogram(numHistogramBins),
          scoreReadoutTimes(),
          intermediatePruningTimes(),
          buildNewBeamTime(),
          recombinationTime(),
          beamPruningTime() {}

const Core::ParameterIntVector BatchedLexiconfreeTimesyncBeamSearch::paramMaxBeamSizes(
        "max-beam-size",
        "Maximum number of elements in the search beam of each segment. Pruning is applied after each intermediate label scorer.",
        "",
        1);

const Core::ParameterFloatVector BatchedLexiconfreeTimesyncBeamSearch::paramScoreThresholds(
        "score-threshold",
        "Prune any hypotheses with a score that is at least this much worse than the best hypothesis of the segment. Pruning is applied after each intermediate label scorer.",
        "",
        0,
        Core::Type<Score>::max);

const Core::ParameterInt BatchedLexiconfreeTimesyncBeamSearch::paramNumHistogramBins(
        "num-histogram-bins",
        "Number of bins for histogram pruning of hypotheses (very minor effect).",
        100,
        2);

const Core::ParameterBool BatchedLexiconfreeTimesyncBeamSearch::paramEnableHistogramPruning(
        "enable-histogram-pruning",
        "Use histogram pruning for max-beam-size. If false, exactly the max-beam-size best hypotheses are kept.",
        true);

const Core::ParameterInt BatchedLexiconfreeTimesyncBeamSearch::paramBlankLabelIndex(
        "blank-label-index",
        "Index of the blank label in the lexicon. Can also be inferred from lexicon if it has a lemma with `special='blank'`. If not set, the search will not use blank.",
        Nn::invalidLabelIndex);

const Core::ParameterInt BatchedLexiconfreeTimesyncBeamSearch::paramSilenceLabelIndex(
        "silence-label-index",
        "Index of the silence label in the lexicon. Can also be inferred from lexicon if it has a lemma with `special='silence'`. If not set, the search will not use silence.",
        Nn::invalidLabelIndex);

const Core::ParameterInt BatchedLexiconfreeTimesyncBeamSearch::paramSentenceEndLabelIndex(
        "sentence-end-label-index",
        "Index of the sentence end label in the lexicon. Can also be inferred from lexicon if it has a lemma with `special='sentence-end'` or `special='sentence-boundary'`. If not set, the search will not use sentence end.",
        Nn::invalidLabelIndex);

const Core::ParameterBool BatchedLexiconfreeTimesyncBeamSearch::paramCollapseRepeatedLabels(
        "collapse-repeated-labels",
        "Collapse repeated emission of the same label into one output. If false, every emission is treated like a new output.",
        false);

const Core::ParameterInt BatchedLexiconfreeTimesyncBeamSearch::paramMaxExtensionsPerHyp(
        "max-extensions-per-hyp",
        "Maximum number of extensions per hypothesis, taking the best labels according to the first label scorer."
        " If not set, this defaults to the first max-beam-size unless the first score-threshold is set, in which case it is not limited."
        " The default does not change the best hypotheses but can make lattices sparser. Values below the first max-beam-size make the search inexact.",
        Core::Type<s32>::max,
        1);

const Core::ParameterInt BatchedLexiconfreeTimesyncBeamSearch::paramCacheCleanupInterval(
        "cache-cleanup-interval",
        "Interval of search steps after which buffered inputs that are not needed anymore get cleaned up.",
        10,
        1);

const Core::ParameterInt BatchedLexiconfreeTimesyncBeamSearch::paramMaximumStableDelay(
        "maximum-stable-delay",
        "Introduce a cutoff point at `current-time` - `delay`. Every hypothesis that disagrees with the current best anywhere before the cutoff gets pruned."
        "This way words in the traceback become stable after at most `delay` frames.",
        Core::Type<int>::max,
        0);

const Core::ParameterInt BatchedLexiconfreeTimesyncBeamSearch::paramMaximumStableDelayPruningInterval(
        "maximum-stable-delay-pruning-interval",
        "Interval of search steps after which the maximum-stable-delay-pruning gets applied.",
        10,
        1);

const Core::Choice BatchedLexiconfreeTimesyncBeamSearch::choiceRecombinationMode(
        "off", RecombinationModeOff,
        "on", RecombinationModeOn,
        Core::Choice::endMark());

const Core::ParameterChoice BatchedLexiconfreeTimesyncBeamSearch::paramRecombinationMode(
        "recombination-mode",
        &choiceRecombinationMode,
        "Whether hypotheses with identical recombination state should be recombined.",
        RecombinationModeOn);

BatchedLexiconfreeTimesyncBeamSearch::BatchedLexiconfreeTimesyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          BatchedSearchBase(config),
          histogramPruningEnabled_(paramEnableHistogramPruning(config)),
          blankLabelIndex_(paramBlankLabelIndex(config)),
          silenceLabelIndex_(paramSilenceLabelIndex(config)),
          sentenceEndLemma_(),
          sentenceEndLabelIndex_(paramSentenceEndLabelIndex(config)),
          collapseRepeatedLabels_(paramCollapseRepeatedLabels(config)),
          cacheCleanupInterval_(paramCacheCleanupInterval(config)),
          maximumStableDelay_(paramMaximumStableDelay(config)),
          maximumStableDelayPruningInterval_(paramMaximumStableDelayPruningInterval(config)),
          recombinationEnabled_(paramRecombinationMode(config) == RecombinationModeOn),
          maxExtensionsPerHyp_(Core::Type<size_t>::max),
          debugChannel_(config, "debug"),
          stepSegments_(),
          segmentsToFinalize_(),
          scoringContexts_(),
          threadBuffers_(),
          scoreAndPruneExtensionsTimes_(),
          scoringTimes_(),
          scoreReadoutTimes_(),
          intermediatePruningTimes_(),
          finalizeTime_(),
          finalizeScoringTime_() {
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

    useBlank_ = blankLabelIndex_ != Nn::invalidLabelIndex;
    if (useBlank_) {
        log() << "Use blank label with index " << blankLabelIndex_;
    }

    useSilence_ = silenceLabelIndex_ != Nn::invalidLabelIndex;
    if (useSilence_) {
        log() << "Use silence label with index " << silenceLabelIndex_;
    }

    for (size_t i = 0; i < scoreThresholds_.size(); ++i) {
        useScorePruning_.push_back(scoreThresholds_[i] != Core::Type<Score>::max);
    }

    // Extensions are created by the first scorer, so only its settings matter
    if (paramMaxExtensionsPerHyp(config) != Core::Type<s32>::max) {
        maxExtensionsPerHyp_ = paramMaxExtensionsPerHyp(config);
    }
    else if (not maxBeamSizes_.empty() and not useScorePruning_.empty() and not useScorePruning_.front()) {
        maxExtensionsPerHyp_ = maxBeamSizes_.front();
    }

    useSentenceEnd_ = sentenceEndLabelIndex_ != Nn::invalidLabelIndex;
    if (useSentenceEnd_) {
        log() << "Use sentence end label with index " << sentenceEndLabelIndex_;
    }
}

Speech::ModelCombination::Mode BatchedLexiconfreeTimesyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon;
}

bool BatchedLexiconfreeTimesyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_ = modelCombination.lexicon();
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

    auto silenceLemma = lexicon_->specialLemma("silence");
    if (silenceLemma) {
        if (silenceLabelIndex_ == Nn::invalidLabelIndex) {
            silenceLabelIndex_ = silenceLemma->id();
            useSilence_        = true;
            log() << "Use silence index " << silenceLabelIndex_ << " inferred from lexicon";
        }
        else if (silenceLabelIndex_ != static_cast<Nn::LabelIndex>(silenceLemma->id())) {
            warning() << "Silence lemma exists in lexicon with id " << silenceLemma->id() << " but is overwritten by config parameter with value " << silenceLabelIndex_;
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
    else {  // Retrieve sentenceEndLemma_ from the lexicon through its label index
        auto lemmas = lexicon_->lemmas();
        for (auto lemmaIt = lemmas.first; lemmaIt != lemmas.second; ++lemmaIt) {
            const Bliss::Lemma* lemma(*lemmaIt);
            Nn::LabelIndex      tokenIdx = lemma->id();
            if (tokenIdx == sentenceEndLabelIndex_) {
                sentenceEndLemma_ = lemma;
                break;
            }
        }
    }

    return true;
}

std::unique_ptr<BatchedSearchBase::SegmentStateBase> BatchedLexiconfreeTimesyncBeamSearch::createSegmentState(SegmentHandle handle, std::string const& name) {
    auto state = std::make_unique<SegmentState>(handle, name, labelScorers_.size());

    // Initialize beam with a single empty hypothesis
    state->beam.push_back(LabelHypothesis());
    for (auto& labelScorer : labelScorers_) {
        state->beam.front().scoringContexts.push_back(labelScorer->getInitialSegmentScoringContext(handle));
    }

    return state;
}

Core::Ref<const Traceback> BatchedLexiconfreeTimesyncBeamSearch::getCurrentBestTraceback(SegmentHandle segment) const {
    return getBestHypothesis(getSegment<SegmentState>(segment).beam).trace->performTraceback();
}

Core::Ref<const LatticeAdaptor> BatchedLexiconfreeTimesyncBeamSearch::getCurrentBestWordLattice(SegmentHandle segment) const {
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

Core::Ref<const LatticeTrace> BatchedLexiconfreeTimesyncBeamSearch::getCurrentBestLatticeTrace(SegmentHandle segment) const {
    return getBestHypothesis(getSegment<SegmentState>(segment).beam).trace;
}

Core::Ref<const LatticeTrace> BatchedLexiconfreeTimesyncBeamSearch::getCommonPrefix(SegmentHandle segment) const {
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

bool BatchedLexiconfreeTimesyncBeamSearch::decodeGroupStep(size_t group) {
    searchStepTime_.start();

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
        startNextScoring(group);
        searchStepTime_.stop();
        return false;
    }

    numSegmentsPerStep_ += stepSegments_.size();

    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlOpen("search-step-stats") + Core::XmlAttribute("step", globalSearchStep_);
    }

    for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
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
                scoreExtensions(segment, scorerIdx, scoreAccessors, denseScoreSpans, scoreTimes, buffers);
            }
            if (segment.extensions.empty()) {
                segment.stepFailed = true;
                return;
            }

            if (scorerIdx == 0ul) {
                segment.numExtensionsBeforeFirstPruning += segment.extensions.size();
                if (stepwiseStatisticsChannel_.isOpen()) {
                    stepwiseStatisticsChannel_ << Core::XmlFull("num-extensions-before-first-pruning", segment.extensions.size()) + Core::XmlAttribute("segment", segment.name);
                }
            }

            /*
             * Prune set of possible extensions by max beam size and possibly also by score.
             */
            Core::StopWatch::Scope pruningTimer(buffers.intermediatePruningTimes[scorerIdx]);
            size_t                 maxBeamSize = segment.extensions.size();
            if (scorerIdx < labelScorers_.size() - 1) {
                maxBeamSize = maxBeamSizes_[scorerIdx];
            }
            scorePruning(segment.extensions, scoreThresholds_[scorerIdx], maxBeamSize, buffers.scoreHistogram);
            segment.numHypsAfterIntermediatePruning[scorerIdx] += segment.extensions.size();
            if (stepwiseStatisticsChannel_.isOpen()) {
                stepwiseStatisticsChannel_ << Core::XmlFull("num-hyps-after-intermediate-pruning", segment.extensions.size()) + Core::XmlAttribute("scorer", scorerIdx + 1) + Core::XmlAttribute("segment", segment.name);
            }
            if (segment.extensions.empty()) {
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
                for (auto const& ext : segment->extensions) {
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
            warning() << "Most recent label in best hypothesis of segment " << segment->name << " is before cutoff point for maximum-stable-delay-pruning so the limit will be surpassed";
        }
        anythingChanged = true;
    }

    if (not segmentsToFinalize_.empty()) {
        Core::StopWatch::Scope finalizeTimer(finalizeTime_);
        finalizeSegments(segmentsToFinalize_);
        anythingChanged = true;
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

void BatchedLexiconfreeTimesyncBeamSearch::collectScoringContexts(std::vector<SegmentState*> const& segments, std::vector<Nn::ScoringContextRef>& contexts) {
    /*
     * The contexts of all segments are passed to the first label scorer in one list. `segment.hypIndexToContextIndexMap`
     * maps a beam index of the segment to the position of its context in the list, i.e.
     * segment.beam[i].scoringContexts.front() == contexts[segment.hypIndexToContextIndexMap[i]]. For each subsequent
     * scorer the list is rebuilt in `decodeGroupStep`, dropping the contexts of any hypotheses whose extensions were all
     * removed by intermediate pruning.
     */
    contexts.clear();
    for (auto* segment : segments) {
        segment->extensions.clear();
        segment->stepFailed               = false;
        segment->noMoreFeaturesWhenScored = segment->noMoreFeatures;
        segment->hypIndexToContextIndexMap.resize(segment->beam.size());
        for (size_t hypIndex = 0ul; hypIndex < segment->beam.size(); ++hypIndex) {
            segment->hypIndexToContextIndexMap[hypIndex] = contexts.size();
            contexts.push_back(segment->beam[hypIndex].scoringContexts.front());
        }
    }
}

void BatchedLexiconfreeTimesyncBeamSearch::startNextScoring(size_t group) {
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

bool BatchedLexiconfreeTimesyncBeamSearch::selectTokens(std::optional<Nn::DenseScoreSpan> const& denseScores, Nn::LabelIndex currentToken, ThreadBuffers& buffers) const {
    if (maxExtensionsPerHyp_ == Core::Type<size_t>::max or not denseScores) {
        return false;
    }

    auto lemmas            = lexicon_->lemmas();
    auto numTokens         = static_cast<size_t>(lemmas.second - lemmas.first);
    auto numScorableTokens = std::min(denseScores->size(), numTokens);
    if (maxExtensionsPerHyp_ >= numScorableTokens) {
        return false;
    }

    // Max-heap of the best tokens so far
    auto& heap           = buffers.tokenHeap;
    auto& selectedTokens = buffers.selectedTokens;
    heap.clear();
    heap.reserve(maxExtensionsPerHyp_);
    selectedTokens.clear();

    for (Nn::LabelIndex tokenIdx = 0ul; tokenIdx < numScorableTokens; ++tokenIdx) {
        // Sentence-end is never extended
        if (tokenIdx == sentenceEndLabelIndex_) {
            continue;
        }
        // The dense scores are only part of the extension score if the transition is scored, otherwise the token can't be ranked
        if (not labelScorers_.front()->scoresTransition(inferTransitionType(currentToken, tokenIdx))) {
            selectedTokens.push_back(tokenIdx);
            continue;
        }

        auto score = (*denseScores)[tokenIdx];
        if (heap.size() < maxExtensionsPerHyp_) {
            heap.emplace_back(score, tokenIdx);
            std::push_heap(heap.begin(), heap.end());
        }
        else if (score < heap.front().first) {
            // Tokens are scanned in ascending order, so an equal score never beats the worst selected token
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = {score, tokenIdx};
            std::push_heap(heap.begin(), heap.end());
        }
    }

    for (auto const& entry : heap) {
        selectedTokens.push_back(entry.second);
    }

    // Tokens without dense score are always kept
    for (Nn::LabelIndex tokenIdx = numScorableTokens; tokenIdx < numTokens; ++tokenIdx) {
        selectedTokens.push_back(tokenIdx);
    }

    // Keep lexicon order
    std::sort(selectedTokens.begin(), selectedTokens.end());

    return true;
}

void BatchedLexiconfreeTimesyncBeamSearch::scoreExtensions(SegmentState&                                           segment,
                                                           size_t                                                  scorerIdx,
                                                           std::vector<std::optional<Nn::ScoreAccessorRef>> const& scoreAccessors,
                                                           std::vector<std::optional<Nn::DenseScoreSpan>> const&   denseScoreSpans,
                                                           std::vector<Nn::TimeframeIndex> const&                  scoreTimes,
                                                           ThreadBuffers&                                          buffers) {
    auto const& labelScorer = labelScorers_[scorerIdx];
    auto const& beam        = segment.beam;
    auto&       extensions  = segment.extensions;
    auto const& contextMap  = segment.hypIndexToContextIndexMap;

    if (scorerIdx == 0ul) {
        // Assume the output labels are stored as lexicon lemma orth and ordered consistently with NN output index
        auto lemmas = lexicon_->lemmas();

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

            // Extend with the selected tokens if their number is limited and with all lemmas otherwise
            bool   limitExtensions = selectTokens(denseScores, hyp.currentToken, buffers);
            size_t numCandidates   = limitExtensions ? buffers.selectedTokens.size() : lemmas.second - lemmas.first;

            // Iterate over possible successors
            for (size_t candidateIdx = 0ul; candidateIdx < numCandidates; ++candidateIdx) {
                auto const*    lemma    = lemmas.first[limitExtensions ? buffers.selectedTokens[candidateIdx] : candidateIdx];
                Nn::LabelIndex tokenIdx = lemma->id();
                // Don't score the sentence-end token
                if (tokenIdx == sentenceEndLabelIndex_) {
                    continue;
                }
                auto transitionType = inferTransitionType(hyp.currentToken, tokenIdx);
                auto extScore       = hyp.score;
                auto extTime        = hyp.trace->time;
                if (labelScorer->scoresTransition(transitionType)) {
                    extScore += (denseScores and tokenIdx < denseScores->size())
                                        ? (*denseScores)[tokenIdx]
                                        : (*scoreAccessor)->getScore(transitionType, tokenIdx);
                    extTime = std::max(extTime, scoreTime);
                }

                // Pre-prune based on score before creating extension instance and appending to list
                if (useScorePruning_.front() and extScore > currentBestScore + scoreThresholds_.front()) {
                    continue;
                }
                currentBestScore = std::min(currentBestScore, extScore);

                extensions.push_back(
                        {.nextToken      = tokenIdx,
                         .pron           = lemma->pronunciations().first,
                         .score          = extScore,
                         .timeframe      = extTime,
                         .transitionType = transitionType,
                         .baseHypIndex   = hypIndex});
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
                ext.score += (denseScores and ext.nextToken < denseScores->size())
                                     ? (*denseScores)[ext.nextToken]
                                     : (*scoreAccessor)->getScore(ext.transitionType, ext.nextToken);
                ext.timeframe = std::max(ext.timeframe, scoreTimes[contextMap[ext.baseHypIndex]]);
            }
            else {
                // Extension is not scorable so set the score to max in order to prune it later
                ext.score = Core::Type<Score>::max;
            }
        }
    }
}

void BatchedLexiconfreeTimesyncBeamSearch::createNewBeam(SegmentState& segment, ThreadBuffers& buffers) {
    auto& beam    = segment.beam;
    auto& newBeam = segment.newBeam;

    // Create new beam from surviving extensions.
    buffers.buildNewBeamTime.start();
    newBeam.clear();
    for (auto const& extension : segment.extensions) {
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
    buffers.buildNewBeamTime.stop();

    // For all hypotheses with the same scoring context keep only the best since they will all develop in the same way.
    buffers.recombinationTime.start();
    recombination(newBeam, buffers.tempHypotheses);
    buffers.recombinationTime.stop();
    segment.numHypsAfterRecombination += newBeam.size();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-hyps-after-recombination", newBeam.size()) + Core::XmlAttribute("segment", segment.name);
    }

    buffers.beamPruningTime.start();
    scorePruning(newBeam, Core::Type<Score>::max, maxBeamSizes_[labelScorers_.size() - 1], buffers.scoreHistogram);
    buffers.beamPruningTime.stop();
    segment.numHypsAfterPruning += newBeam.size();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-hyps-after-pruning", newBeam.size()) + Core::XmlAttribute("scorer", labelScorers_.size()) + Core::XmlAttribute("segment", segment.name);
    }

    beam.swap(newBeam);

    ++segment.currentSearchStep;

    /*
     * Perform maximum-stable-delay-pruning.
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
     * Log statistics about the new beam after this step.
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
        stepwiseStatisticsChannel_ << Core::XmlFull("best-hyp-score", getBestHypothesis(beam).score) + Core::XmlAttribute("segment", segment.name);
        stepwiseStatisticsChannel_ << Core::XmlFull("worst-hyp-score", getWorstHypothesis(beam).score) + Core::XmlAttribute("segment", segment.name);
    }
}

void BatchedLexiconfreeTimesyncBeamSearch::finalizeSegments(std::vector<SegmentState*> const& segments) {
    if (useSentenceEnd_) {
        // Create extensions for all hypotheses in the beams
        for (auto* segment : segments) {
            segment->extensions.clear();
            for (size_t hypIndex = 0ul; hypIndex < segment->beam.size(); ++hypIndex) {
                auto& hyp = segment->beam[hypIndex];
                segment->extensions.push_back(
                        {.nextToken      = sentenceEndLabelIndex_,
                         .pron           = sentenceEndLemma_->pronunciations().first,
                         .score          = hyp.score,
                         .timeframe      = hyp.trace->time,
                         .transitionType = Nn::TransitionType::SENTENCE_END,
                         .baseHypIndex   = hypIndex});
            }
        }

        // Score sentence-end with all label scorers
        for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
            if (not labelScorers_[scorerIdx]->scoresTransition(Nn::TransitionType::SENTENCE_END)) {
                continue;
            }

            scoringContexts_.clear();
            for (auto* segment : segments) {
                for (auto const& hyp : segment->beam) {
                    scoringContexts_.push_back(hyp.scoringContexts[scorerIdx]);
                }
            }

            finalizeScoringTime_.start();
            auto scoreAccessors = getScoreAccessors(scorerIdx, scoringContexts_);
            finalizeScoringTime_.stop();

            size_t accessorIdx = 0ul;
            for (auto* segment : segments) {
                for (auto& ext : segment->extensions) {
                    auto const& scoreAccessor = scoreAccessors[accessorIdx++];
                    if (not scoreAccessor) {
                        continue;
                    }
                    auto score = (*scoreAccessor)->getScore(ext.transitionType, ext.nextToken);
                    ext.score += score;
                    ext.timeframe = std::max(ext.timeframe, (*scoreAccessor)->getTime());
                }
            }
        }

        auto& tempHypotheses = threadBuffers_.front().tempHypotheses;
        for (auto* segment : segments) {
            tempHypotheses.clear();
            for (auto const& ext : segment->extensions) {
                auto const& baseHyp = segment->beam[ext.baseHypIndex];
                // The scoring context is not updated as no further scoring is done afterwards
                tempHypotheses.push_back({baseHyp, ext, baseHyp.scoringContexts});
            }

            segment->beam.swap(tempHypotheses);

            segment->numActiveHyps += segment->beam.size();

            // Log statistics about the final beam
            if (debugChannel_.isOpen()) {
                std::stringstream ss;
                for (size_t hypIdx = 0ul; hypIdx < segment->beam.size(); ++hypIdx) {
                    ss << "Segment " << segment->name << ": Hypothesis " << hypIdx + 1ul << ":  " << segment->beam[hypIdx].toString() << "\n";
                }
                ss << "\n";
                debugChannel_ << ss.str();
            }

            if (stepwiseStatisticsChannel_.isOpen()) {
                stepwiseStatisticsChannel_ << Core::XmlOpen("final-beam-stats") + Core::XmlAttribute("segment", segment->name);
                stepwiseStatisticsChannel_ << Core::XmlFull("active-hyps", segment->beam.size());
                stepwiseStatisticsChannel_ << Core::XmlFull("best-hyp-score", getBestHypothesis(segment->beam).score);
                stepwiseStatisticsChannel_ << Core::XmlFull("worst-hyp-score", getWorstHypothesis(segment->beam).score);
                stepwiseStatisticsChannel_ << Core::XmlClose("final-beam-stats");
            }
        }
    }

    for (auto* segment : segments) {
        finishSegmentSearch(*segment);
    }
}

void BatchedLexiconfreeTimesyncBeamSearch::releaseSegmentState(SegmentStateBase& segmentBase) {
    auto& segment = static_cast<SegmentState&>(segmentBase);
    // The scoring contexts are not needed anymore
    for (auto& hyp : segment.beam) {
        hyp.scoringContexts.clear();
    }
    segment.extensions.clear();
    segment.newBeam.clear();
}

BatchedLexiconfreeTimesyncBeamSearch::LabelHypothesis const& BatchedLexiconfreeTimesyncBeamSearch::getBestHypothesis(std::vector<LabelHypothesis> const& beam) const {
    verify(not beam.empty());

    return *std::min_element(beam.begin(), beam.end());
}

BatchedLexiconfreeTimesyncBeamSearch::LabelHypothesis const& BatchedLexiconfreeTimesyncBeamSearch::getWorstHypothesis(std::vector<LabelHypothesis> const& beam) const {
    verify(not beam.empty());

    return *std::max_element(beam.begin(), beam.end());
}

void BatchedLexiconfreeTimesyncBeamSearch::logSegmentStatistics(SegmentStateBase const& segmentBase) const {
    if (not statisticsChannel_.isOpen()) {
        return;
    }
    auto const& segment = static_cast<SegmentState const&>(segmentBase);
    auto&       channel = statisticsChannel_;
    channel << Core::XmlOpen("search-statistics") + Core::XmlAttribute("segment", segment.name);
    channel << Core::XmlFull("num-search-steps", segment.currentSearchStep);
    segment.numInputHyps.write(channel);
    segment.numExtensionsBeforeFirstPruning.write(channel);
    for (size_t i = 0ul; i < segment.numHypsAfterIntermediatePruning.size(); ++i) {
        segment.numHypsAfterIntermediatePruning[i].write(channel, {Core::XmlAttribute("scorer", i + 1)});
    }
    segment.numHypsAfterRecombination.write(channel);
    segment.numHypsAfterPruning.write(channel);
    segment.numActiveHyps.write(channel);
    channel << Core::XmlClose("search-statistics");
}

void BatchedLexiconfreeTimesyncBeamSearch::logSearchStepBreakdown() const {
    // Timers of the search threads are summed up
    auto threadSum = [this](auto getTimer) {
        double sum = 0.0;
        for (auto const& buffers : threadBuffers_) {
            sum += getTimer(buffers).elapsedMilliseconds();
        }
        return sum;
    };

    auto& channel = statisticsChannel_;
    for (size_t i = 0ul; i < scoreAndPruneExtensionsTimes_.size(); ++i) {
        channel << Core::XmlOpen("score-and-prune-extensions-time") + Core::XmlAttribute("scorer", i + 1) + Core::XmlAttribute("total", scoreAndPruneExtensionsTimes_[i].elapsedMilliseconds());
        channel << Core::XmlFull("scoring-time", scoringTimes_[i].elapsedMilliseconds());
        channel << Core::XmlFull("score-readout-time", scoreReadoutTimes_[i].elapsedMilliseconds() + threadSum([i](ThreadBuffers const& buffers) -> Core::StopWatch const& { return buffers.scoreReadoutTimes[i]; }));
        channel << Core::XmlFull("intermediate-pruning-time", intermediatePruningTimes_[i].elapsedMilliseconds() + threadSum([i](ThreadBuffers const& buffers) -> Core::StopWatch const& { return buffers.intermediatePruningTimes[i]; }));
        channel << Core::XmlClose("score-and-prune-extensions-time");
    }
    channel << Core::XmlFull("build-new-beam-time", threadSum([](ThreadBuffers const& buffers) -> Core::StopWatch const& { return buffers.buildNewBeamTime; }));
    channel << Core::XmlFull("recombination-time", threadSum([](ThreadBuffers const& buffers) -> Core::StopWatch const& { return buffers.recombinationTime; }));
    channel << Core::XmlFull("beam-pruning-time", threadSum([](ThreadBuffers const& buffers) -> Core::StopWatch const& { return buffers.beamPruningTime; }));
    channel << Core::XmlOpen("finalize-time") + Core::XmlAttribute("total", finalizeTime_.elapsedMilliseconds());
    channel << Core::XmlFull("scoring-time", finalizeScoringTime_.elapsedMilliseconds());
    channel << Core::XmlClose("finalize-time");
}

void BatchedLexiconfreeTimesyncBeamSearch::resetSearchStepBreakdown() {
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
        buffers.buildNewBeamTime.reset();
        buffers.recombinationTime.reset();
        buffers.beamPruningTime.reset();
    }
}

Nn::TransitionType BatchedLexiconfreeTimesyncBeamSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const {
    bool prevIsBlank = (useBlank_ and prevLabel == blankLabelIndex_);
    bool nextIsBlank = (useBlank_ and nextLabel == blankLabelIndex_);

    bool prevIsSilence = (useSilence_ and prevLabel == silenceLabelIndex_);
    bool nextIsSilence = (useSilence_ and nextLabel == silenceLabelIndex_);

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
        else if (collapseRepeatedLabels_ and prevLabel == nextLabel) {
            return Nn::TransitionType::LABEL_LOOP;
        }
        else {
            return Nn::TransitionType::LABEL_TO_LABEL;
        }
    }
}

template<typename Element>
void BatchedLexiconfreeTimesyncBeamSearch::scorePruning(std::vector<Element>& hypotheses, Score relativeThreshold, size_t maxBeamSize, Histogram& scoreHistogram) {
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

void BatchedLexiconfreeTimesyncBeamSearch::recombination(std::vector<LabelHypothesis>& hypotheses, std::vector<LabelHypothesis>& tempHypotheses) {
    if (not recombinationEnabled_) {
        return;
    }

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

    tempHypotheses.clear();
    // Reserve capacity because future reallocations would break the raw pointer we are storing later
    tempHypotheses.reserve(hypotheses.size());
    // Map each unique ScoringContext in newHypotheses to its hypothesis
    std::unordered_map<RecombinationContext, LabelHypothesis*, RecombinationContextHash> seenScoringContexts;
    for (auto const& hyp : hypotheses) {
        // Use try_emplace to check if the scoring context already exists and create a new entry if not at the same time
        auto [it, inserted] = seenScoringContexts.try_emplace({hyp}, nullptr);

        if (inserted) {
            // First time seeing this scoring context so move it over to `newHypotheses`
            tempHypotheses.push_back(std::move(hyp));
            it->second = &tempHypotheses.back();
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

    hypotheses.swap(tempHypotheses);
}

bool BatchedLexiconfreeTimesyncBeamSearch::maximumStableDelayPruning(SegmentState& segment, std::vector<LabelHypothesis>& tempHypotheses) {
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
