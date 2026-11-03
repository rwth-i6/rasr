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

#include "BatchedLexiconfreeLabelsyncBeamSearch.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <strings.h>

#include <Core/CollapsedVector.hh>
#include <Core/XmlStream.hh>
#include <Math/Utilities.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Search/Traceback.hh>
#include <Search/TracebackHelper.hh>

namespace Search {

namespace {

enum RecombinationMode {
    RecombinationModeOff,
    RecombinationModeOn,
};

enum PruningStrategyType {
    PruningStrategyJoint,
    PruningStrategySeparate,
};

}  // namespace

/*
 * =======================
 * === LabelHypothesis ===
 * =======================
 */

BatchedLexiconfreeLabelsyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContexts(),
          currentToken(Nn::invalidLabelIndex),
          length(0),
          score(0.0),
          scaledScore(0.0),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))),
          isActive(true) {}

BatchedLexiconfreeLabelsyncBeamSearch::LabelHypothesis::LabelHypothesis(
        BatchedLexiconfreeLabelsyncBeamSearch::LabelHypothesis const&    base,
        BatchedLexiconfreeLabelsyncBeamSearch::ExtensionCandidate const& extension,
        std::vector<Nn::ScoringContextRef> const&                        newScoringContexts,
        float                                                            lengthNormScale)
        : scoringContexts(newScoringContexts),
          currentToken(extension.nextToken),
          score(extension.score),
          trace(),
          isActive(extension.isActive) {
    switch (extension.transitionType) {
        case Nn::TransitionType::LABEL_TO_LABEL:
        case Nn::TransitionType::BLANK_TO_LABEL:
        case Nn::TransitionType::INITIAL_LABEL:
        case Nn::TransitionType::SENTENCE_END:
            length = base.length + 1;
            break;
        default:
            length = base.length;
    }
    scaledScore = score / std::pow(length, lengthNormScale);

    Core::Ref<LatticeTrace> predecessor;
    switch (extension.transitionType) {
        case Nn::TransitionType::LABEL_LOOP:
        case Nn::TransitionType::BLANK_LOOP:
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

std::string BatchedLexiconfreeLabelsyncBeamSearch::LabelHypothesis::toString() const {
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

BatchedLexiconfreeLabelsyncBeamSearch::SegmentState::SegmentState(SegmentHandle handle, std::string const& name, size_t numLabelScorers)
        : SegmentStateBase(handle, name),
          beam(),
          currentSearchStep(0ul),
          hypIndexToContextIndexMap(),
          extensions(),
          stepFailed(false),
          numInputHyps("num-input-hyps"),
          numExtensionsBeforeFirstPruning("num-extensions-before-first-pruning"),
          numHypsAfterIntermediatePruning(numLabelScorers, Core::Statistics<u32>("num-hyps-after-intermediate-pruning")),
          numTerminatedHypsAfterScorePruning("num-terminated-hyps-after-score-pruning"),
          numTerminatedHypsAfterRecombination("num-terminated-hyps-after-recombination"),
          numTerminatedHypsAfterBeamPruning("num-terminated-hyps-after-beam-pruning"),
          numActiveHypsAfterScorePruning("num-active-hyps-after-score-pruning"),
          numActiveHypsAfterRecombination("num-active-hyps-after-recombination"),
          numActiveHypsAfterBeamPruning("num-active-hyps-after-beam-pruning") {
}

/*
 * =============================================
 * === BatchedLexiconfreeLabelsyncBeamSearch ===
 * =============================================
 */

BatchedLexiconfreeLabelsyncBeamSearch::ThreadBuffers::ThreadBuffers(size_t numHistogramBins)
        : extensions(),
          newBeam(),
          tempHypotheses(),
          selectedTokens(),
          tokenHeap(),
          scoreHistogram(numHistogramBins),
          activeScoreHistogram(numHistogramBins),
          terminatedScoreHistogram(numHistogramBins),
          scoreReadoutTimes(),
          intermediatePruningTimes(),
          buildNewBeamTime(),
          recombinationTime(),
          beamPruningTime() {}

const Core::ParameterIntVector BatchedLexiconfreeLabelsyncBeamSearch::paramMaxBeamSizes(
        "max-beam-size",
        "Maximum number of hypotheses in the search beam of each segment."
        " The pruning-strategy-type determines whether active and terminated hypotheses share this limit or are limited separately."
        " Pruning is applied after each intermediate label scorer.",
        "",
        1);

const Core::ParameterFloatVector BatchedLexiconfreeLabelsyncBeamSearch::paramScoreThresholds(
        "score-threshold",
        "Prune hypotheses with a score that is at least this much worse than the pruning reference."
        " In joint pruning mode, the reference is the best hypothesis."
        " In separate pruning mode, active hypotheses use the overall best hypothesis and terminated hypotheses use the best terminated hypothesis."
        " If length normalization is enabled, the score threshold is added to the raw score before normalization."
        " Pruning is applied after each intermediate label scorer."
        " If not set, no score pruning will be done.",
        "",
        0,
        Core::Type<Score>::max);

const Core::ParameterInt BatchedLexiconfreeLabelsyncBeamSearch::paramNumHistogramBins(
        "num-histogram-bins",
        "Number of bins for histogram pruning of hypotheses (very minor effect).",
        100,
        2);

const Core::ParameterBool BatchedLexiconfreeLabelsyncBeamSearch::paramEnableHistogramPruning(
        "enable-histogram-pruning",
        "Use histogram pruning for max-beam-size. If false, exactly the max-beam-size best hypotheses are kept.",
        true);

const Core::ParameterInt BatchedLexiconfreeLabelsyncBeamSearch::paramSentenceEndLabelIndex(
        "sentence-end-label-index",
        "Index of the sentence-end label in the lexicon."
        "Can also be inferred from lexicon if it has a lemma with `special='sentence-end'` or `special='sentence-boundary'`");

const Core::ParameterFloat BatchedLexiconfreeLabelsyncBeamSearch::paramLengthNormScale(
        "length-norm-scale",
        "Exponent of length for the hypothesis score length normalization. Scaled scores are computed as score / length^length_norm_scale.",
        0.0);

const Core::ParameterFloat BatchedLexiconfreeLabelsyncBeamSearch::paramMaxLabelsPerTimestep(
        "max-labels-per-timestep",
        "Maximum number of emitted labels per input timestep counted via `putFeature`/`putFeatures`."
        " The resulting limit (rounded up) is clamped to the range given by `max-labels-lower-bound` and `max-labels-upper-bound`.",
        1.0);

const Core::ParameterInt BatchedLexiconfreeLabelsyncBeamSearch::paramMaxLabelsLowerBound(
        "max-labels-lower-bound",
        "Lower bound for the maximum number of emitted labels, so that short inputs are not cut off too early."
        " Only raises the limit given by `max-labels-per-timestep`, but does not enforce a minimum output length,"
        " the search still ends as soon as all hypotheses have reached sentence-end.",
        0,
        0);

const Core::ParameterInt BatchedLexiconfreeLabelsyncBeamSearch::paramMaxLabelsUpperBound(
        "max-labels-upper-bound",
        "Upper bound for the maximum number of emitted labels, independent of the input length."
        " If it is smaller than `max-labels-lower-bound`, the upper bound takes precedence.",
        std::numeric_limits<int>::max(),
        0);

const Core::Choice BatchedLexiconfreeLabelsyncBeamSearch::choicePruningStrategyType(
        "joint", PruningStrategyJoint,
        "separate", PruningStrategySeparate,
        Core::Choice::endMark());

const Core::ParameterChoice BatchedLexiconfreeLabelsyncBeamSearch::paramPruningStrategyType(
        "pruning-strategy-type",
        &choicePruningStrategyType,
        "How active and terminated hypotheses are pruned."
        " Joint mode prunes both pools together."
        " Separate mode prunes terminated hypotheses against terminated hypotheses and active hypotheses against the overall best hypothesis.",
        PruningStrategyJoint);

const Core::Choice BatchedLexiconfreeLabelsyncBeamSearch::choiceRecombinationMode(
        "off", RecombinationModeOff,
        "on", RecombinationModeOn,
        Core::Choice::endMark());

const Core::ParameterChoice BatchedLexiconfreeLabelsyncBeamSearch::paramRecombinationMode(
        "recombination-mode",
        &choiceRecombinationMode,
        "Whether hypotheses with identical recombination state should be recombined.",
        RecombinationModeOn);

const Core::ParameterInt BatchedLexiconfreeLabelsyncBeamSearch::paramMaxExtensionsPerHyp(
        "max-extensions-per-hyp",
        "Maximum number of extensions per active hypothesis, taking the best labels according to the first label scorer."
        " If not set, this defaults to the first max-beam-size unless the first score-threshold is set, in which case it is not limited."
        " The default does not change the best hypotheses but can make lattices sparser. Values below the first max-beam-size make the search inexact.",
        Core::Type<s32>::max,
        1);

const Core::ParameterInt BatchedLexiconfreeLabelsyncBeamSearch::paramCacheCleanupInterval(
        "cache-cleanup-interval",
        "Interval of search steps after which buffered inputs that are not needed anymore get cleaned up.",
        10,
        1);

BatchedLexiconfreeLabelsyncBeamSearch::BatchedLexiconfreeLabelsyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          BatchedSearchBase(config),
          histogramPruningEnabled_(paramEnableHistogramPruning(config)),
          lengthNormScale_(paramLengthNormScale(config)),
          maxLabelsPerTimestep_(paramMaxLabelsPerTimestep(config)),
          maxLabelsLowerBound_(paramMaxLabelsLowerBound(config)),
          maxLabelsUpperBound_(paramMaxLabelsUpperBound(config)),
          sentenceEndLabelIndex_(paramSentenceEndLabelIndex(config)),
          pruningStrategyType_(paramPruningStrategyType(config)),
          recombinationEnabled_(paramRecombinationMode(config) == RecombinationModeOn),
          maxExtensionsPerHyp_(Core::Type<size_t>::max),
          cacheCleanupInterval_(paramCacheCleanupInterval(config)),
          debugChannel_(config, "debug"),
          stepSegments_(),
          scoringContexts_(),
          threadBuffers_(),
          scoreAndPruneExtensionsTimes_(),
          scoringTimes_(),
          scoreReadoutTimes_(),
          intermediatePruningTimes_() {
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
    for (size_t i = scoreThresholds_.size(); i < maxBeamSizes_.size(); ++i) {
        scoreThresholds_.push_back(Core::Type<Score>::max);
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

    if (maxLabelsLowerBound_ > maxLabelsUpperBound_) {
        warning() << "max-labels-lower-bound (" << maxLabelsLowerBound_ << ") is larger than max-labels-upper-bound (" << maxLabelsUpperBound_ << "). The upper bound takes precedence";
    }

    if (sentenceEndLabelIndex_ != Core::Type<s32>::max) {
        log() << "Use sentence-end label with index " << sentenceEndLabelIndex_;
    }

    switch (pruningStrategyType_) {
        case PruningStrategyJoint:
            break;
        case PruningStrategySeparate:
            if (lengthNormScale_ == 0.0f) {
                warning() << "pruning-strategy-type=separate with length-norm-scale=0 compares active and terminated hypotheses by raw accumulated scores. This can favor early terminated hypotheses that stay around until later.";
            }
            break;
    }
}

Speech::ModelCombination::Mode BatchedLexiconfreeLabelsyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon;
}

bool BatchedLexiconfreeLabelsyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_ = modelCombination.lexicon();
    setLabelScorers(modelCombination.labelScorers());
    verify(not labelScorers_.empty());

    if (labelScorers_.size() > maxBeamSizes_.size()) {
        error() << "Number of label scorers (" << labelScorers_.size() << ") exceeds number of configured max beam sizes (" << maxBeamSizes_.size() << ")";
    }
    if (labelScorers_.size() < maxBeamSizes_.size()) {
        warning() << "Number of label scorers (" << labelScorers_.size() << ") is less than number of configured max beam sizes (" << maxBeamSizes_.size() << ")";
    }
    if (labelScorers_.size() < scoreThresholds_.size()) {
        warning() << "Number of label scorers (" << labelScorers_.size() << ") is less than number of configured score thresholds (" << scoreThresholds_.size() << "); the surplus thresholds are ignored";
    }

    // The final pruning stage is indexed by the last label scorer, so make sure the pruning parameter
    // vectors are long enough even if the size mismatch reported above is configured to be ignored
    for (size_t i = maxBeamSizes_.size(); i < labelScorers_.size(); ++i) {
        maxBeamSizes_.push_back(Core::Type<size_t>::max);
    }
    for (size_t i = scoreThresholds_.size(); i < labelScorers_.size(); ++i) {
        scoreThresholds_.push_back(Core::Type<Score>::max);
        useScorePruning_.push_back(false);
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

    switch (pruningStrategyType_) {
        case PruningStrategyJoint:
            break;
        case PruningStrategySeparate:
            if (not useScorePruning_[labelScorers_.size() - 1]) {
                error() << "pruning-strategy-type=separate requires a finite final score-threshold. Otherwise, the normal stop criterion can't trigger. Use pruning-strategy-type=joint for pure max-beam-size pruning.";
            }
            break;
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

    return true;
}

std::unique_ptr<BatchedSearchBase::SegmentStateBase> BatchedLexiconfreeLabelsyncBeamSearch::createSegmentState(SegmentHandle handle, std::string const& name) {
    auto state = std::make_unique<SegmentState>(handle, name, labelScorers_.size());

    // Initialize beam with a single empty hypothesis
    state->beam.push_back(LabelHypothesis());
    for (auto& labelScorer : labelScorers_) {
        state->beam.front().scoringContexts.push_back(labelScorer->getInitialSegmentScoringContext(handle));
    }

    return state;
}

Core::Ref<const Traceback> BatchedLexiconfreeLabelsyncBeamSearch::getCurrentBestTraceback(SegmentHandle segment) const {
    return getOutputHypothesis(getSegment<SegmentState>(segment).beam).trace->performTraceback();
}

Core::Ref<const LatticeAdaptor> BatchedLexiconfreeLabelsyncBeamSearch::getCurrentBestWordLattice(SegmentHandle segment) const {
    auto const& beam           = getSegment<SegmentState>(segment).beam;
    auto&       bestHypothesis = getOutputHypothesis(beam);

    LatticeTrace endTrace(bestHypothesis.trace, 0, bestHypothesis.trace->time + 1, bestHypothesis.trace->score, {});

    for (auto const& hyp : beam) {
        if (&hyp == &bestHypothesis or hyp.isActive != bestHypothesis.isActive) {
            continue;
        }
        auto siblingTrace = Core::ref(new LatticeTrace(hyp.trace, 0, hyp.trace->time, hyp.trace->score, {}));
        endTrace.appendSiblingToChain(siblingTrace);
    }

    return endTrace.buildWordLattice(lexicon_);
}

Core::Ref<const LatticeTrace> BatchedLexiconfreeLabelsyncBeamSearch::getCurrentBestLatticeTrace(SegmentHandle segment) const {
    return getOutputHypothesis(getSegment<SegmentState>(segment).beam).trace;
}

Core::Ref<const LatticeTrace> BatchedLexiconfreeLabelsyncBeamSearch::getCommonPrefix(SegmentHandle segment) const {
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

bool BatchedLexiconfreeLabelsyncBeamSearch::decodeGroupStep(size_t group) {
    searchStepTime_.start();

    /*
     * Determine the segments of the group that take part in this step and finish the ones that can't continue.
     */
    bool anythingChanged = false;
    stepSegments_.clear();
    forEachSegment<SegmentState>([&](SegmentState& segment) {
        if (segment.finished or segment.group != group) {
            return;
        }
        size_t const outputLengthLimit = maxOutputLength(segment);
        if (segment.currentSearchStep >= outputLengthLimit) {
            // More features can't raise the limit above the upper bound
            if (segment.noMoreFeatures or segment.currentSearchStep >= maxLabelsUpperBound_) {
                size_t const numInputs = std::max(segment.totalTimesteps, 1ul);
                warning() << "Terminated search of segment " << segment.name << " after " << segment.currentSearchStep << " steps: reached the output length limit of "
                          << outputLengthLimit << " labels (" << maxLabelsPerTimestep_ << " labels per timestep * " << numInputs
                          << " inputs = " << static_cast<size_t>(std::ceil(std::max(maxLabelsPerTimestep_ * numInputs, 0.0f))) << " rounded up, lower bound " << maxLabelsLowerBound_ << ", upper bound "
                          << (maxLabelsUpperBound_ == static_cast<size_t>(std::numeric_limits<int>::max()) ? std::string("none") : std::to_string(maxLabelsUpperBound_)) << ")";
                finishSegmentSearch(segment);
                anythingChanged = true;
            }
            // Otherwise wait for more features
            return;
        }
        if (std::all_of(segment.beam.begin(), segment.beam.end(), [](LabelHypothesis const& hyp) { return not hyp.isActive; })) {
            log() << "Terminated search of segment " << segment.name << " because all hypotheses reached sentence-end";
            finishSegmentSearch(segment);
            anythingChanged = true;
            return;
        }
        stepSegments_.push_back(&segment);
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
        if (numActiveSegments() == 0ul) {
            logGlobalStatistics();
        }
        return anythingChanged;
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

            // The extensions of the first label scorer are created and pruned in a buffer of the thread (which thus
            // stays in the cache); only the surviving extensions are kept per segment
            auto& extensions = scorerIdx == 0ul ? buffers.extensions : segment.extensions;

            {
                // Lazily computing label scorers do their work here
                Core::StopWatch::Scope readoutTimer(buffers.scoreReadoutTimes[scorerIdx]);
                scoreExtensions(segment, extensions, scorerIdx, scoreAccessors, denseScoreSpans, scoreTimes, buffers);
            }
            if (extensions.empty()) {
                segment.stepFailed = true;
                return;
            }

            if (scorerIdx == 0ul) {
                segment.numExtensionsBeforeFirstPruning += extensions.size();
                if (stepwiseStatisticsChannel_.isOpen()) {
                    stepwiseStatisticsChannel_ << Core::XmlFull("num-extensions-before-first-pruning", extensions.size()) + Core::XmlAttribute("segment", segment.name);
                }
            }

            Core::StopWatch::Scope pruningTimer(buffers.intermediatePruningTimes[scorerIdx]);
            pruneExtensions(segment, extensions, scorerIdx, buffers);
            if (scorerIdx == 0ul) {
                segment.extensions.assign(buffers.extensions.begin(), buffers.extensions.end());
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
     * Create new beams from surviving extensions.
     */
    parallelForSegments(stepSegments_, [&](SegmentState& segment, size_t threadIdx) {
        if (not segment.stepFailed) {
            createNewBeam(segment, threadBuffers_[threadIdx]);
        }
    });
    for (auto* segment : stepSegments_) {
        if (segment->stepFailed) {
            // The segment can't make progress in this step. If no more features were expected when its contexts were
            // scored, this won't change anymore.
            if (segment->noMoreFeaturesWhenScored) {
                finishSegmentSearch(*segment);
                anythingChanged = true;
            }
            continue;
        }
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

size_t BatchedLexiconfreeLabelsyncBeamSearch::maxOutputLength(SegmentState const& segment) const {
    size_t const numInputs  = std::max(segment.totalTimesteps, 1ul);
    size_t const ratioLimit = static_cast<size_t>(std::ceil(std::max(maxLabelsPerTimestep_ * numInputs, 0.0f)));
    return std::min(std::max(ratioLimit, maxLabelsLowerBound_), maxLabelsUpperBound_);
}

bool BatchedLexiconfreeLabelsyncBeamSearch::canMakeStep(SegmentState const& segment) const {
    return not segment.finished and
           segment.currentSearchStep < maxOutputLength(segment) and
           std::any_of(segment.beam.begin(), segment.beam.end(), [](LabelHypothesis const& hyp) { return hyp.isActive; });
}

void BatchedLexiconfreeLabelsyncBeamSearch::collectScoringContexts(std::vector<SegmentState*> const& segments, std::vector<Nn::ScoringContextRef>& contexts) {
    /*
     * The contexts of all segments are passed to the first label scorer in one list. `segment.hypIndexToContextIndexMap`
     * maps a beam index of the segment to the position of its context in the list, i.e.
     * segment.beam[i].scoringContexts.front() == contexts[segment.hypIndexToContextIndexMap[i]]. A value of -1 means
     * that the hypothesis contributes no context. Only active hypotheses contribute a context. For each subsequent
     * scorer the list is rebuilt in `decodeGroupStep`, dropping the contexts of any hypotheses whose extensions were all
     * removed by intermediate pruning.
     */
    contexts.clear();
    for (auto* segment : segments) {
        segment->extensions.clear();
        segment->stepFailed               = false;
        segment->noMoreFeaturesWhenScored = segment->noMoreFeatures;
        segment->hypIndexToContextIndexMap.assign(segment->beam.size(), -1);

        for (size_t hypIndex = 0ul; hypIndex < segment->beam.size(); ++hypIndex) {
            auto& hyp = segment->beam[hypIndex];
            if (not hyp.isActive) {
                continue;
            }
            segment->hypIndexToContextIndexMap[hypIndex] = contexts.size();
            contexts.push_back(hyp.scoringContexts.front());
        }
    }
}

void BatchedLexiconfreeLabelsyncBeamSearch::startNextScoring(size_t group) {
    if (numPipelineGroups() == 1ul) {
        return;
    }
    std::vector<SegmentState*> segments;
    forEachSegment<SegmentState>([&](SegmentState& segment) {
        if (segment.group == group and canMakeStep(segment)) {
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

bool BatchedLexiconfreeLabelsyncBeamSearch::selectTokens(std::optional<Nn::DenseScoreSpan> const& denseScores, Nn::LabelIndex currentToken, ThreadBuffers& buffers) const {
    if (maxExtensionsPerHyp_ == Core::Type<size_t>::max or not denseScores) {
        return false;
    }

    auto lemmas            = lexicon_->lemmas();
    auto numTokens         = static_cast<size_t>(lemmas.second - lemmas.first);
    auto numScorableTokens = std::min(denseScores->size(), numTokens);
    if (maxExtensionsPerHyp_ >= numScorableTokens) {
        return false;
    }

    // The dense scores are only part of the extension score if the transition is scored, otherwise the token can't be ranked
    auto const& labelScorer    = labelScorers_.front();
    auto        transitionType = currentToken == Nn::invalidLabelIndex ? Nn::TransitionType::INITIAL_LABEL : Nn::TransitionType::LABEL_TO_LABEL;
    if (not labelScorer->scoresTransition(transitionType)) {
        return false;
    }

    // Max-heap of the best tokens so far
    auto& heap           = buffers.tokenHeap;
    auto& selectedTokens = buffers.selectedTokens;
    heap.clear();
    heap.reserve(maxExtensionsPerHyp_);
    selectedTokens.clear();

    for (Nn::LabelIndex tokenIdx = 0ul; tokenIdx < numScorableTokens; ++tokenIdx) {
        // With separate pruning, sentence-end has its own pool and is always kept
        if (tokenIdx == sentenceEndLabelIndex_ and (pruningStrategyType_ == PruningStrategySeparate or not labelScorer->scoresTransition(Nn::TransitionType::SENTENCE_END))) {
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

void BatchedLexiconfreeLabelsyncBeamSearch::scoreExtensions(SegmentState&                                           segment,
                                                            std::vector<ExtensionCandidate>&                        extensions,
                                                            size_t                                                  scorerIdx,
                                                            std::vector<std::optional<Nn::ScoreAccessorRef>> const& scoreAccessors,
                                                            std::vector<std::optional<Nn::DenseScoreSpan>> const&   denseScoreSpans,
                                                            std::vector<Nn::TimeframeIndex> const&                  scoreTimes,
                                                            ThreadBuffers&                                          buffers) {
    auto const& labelScorer = labelScorers_[scorerIdx];
    auto const& beam        = segment.beam;
    auto const& contextMap  = segment.hypIndexToContextIndexMap;

    if (scorerIdx == 0ul) {
        extensions.clear();

        // Assume the output labels are stored as lexicon lemma orth and ordered consistently with NN output index
        auto lemmas = lexicon_->lemmas();

        // In the first iteration, create extensions while pre-pruning
        Score currentBestScore           = Core::Type<Score>::max;
        Score currentBestTerminatedScore = Core::Type<Score>::max;

        for (size_t hypIndex = 0ul; hypIndex < beam.size(); ++hypIndex) {
            auto const& hyp = beam[hypIndex];

            if (not hyp.isActive) {
                continue;
            }

            auto const& scoreAccessor = scoreAccessors[contextMap[hypIndex]];
            auto const& denseScores   = denseScoreSpans[contextMap[hypIndex]];
            auto        scoreTime     = scoreTimes[contextMap[hypIndex]];

            if (not scoreAccessor) {
                // No extensions for hyps that couldn't be scored
                continue;
            }

            // Extend with the selected tokens if their number is limited and with all lemmas otherwise
            bool   limitExtensions = selectTokens(denseScores, hyp.currentToken, buffers);
            size_t numCandidates   = limitExtensions ? buffers.selectedTokens.size() : lemmas.second - lemmas.first;

            // Iterate over possible successors
            for (size_t candidateIdx = 0ul; candidateIdx < numCandidates; ++candidateIdx) {
                auto const*    lemma      = lemmas.first[limitExtensions ? buffers.selectedTokens[candidateIdx] : candidateIdx];
                Nn::LabelIndex tokenIdx   = lemma->id();
                bool           terminated = false;

                auto transitionType = Nn::TransitionType::LABEL_TO_LABEL;
                if (hyp.currentToken == Nn::invalidLabelIndex) {
                    transitionType = Nn::TransitionType::INITIAL_LABEL;
                }
                if (tokenIdx == sentenceEndLabelIndex_) {
                    transitionType = Nn::TransitionType::SENTENCE_END;
                    terminated     = true;
                }

                auto extScore = hyp.score;
                auto extTime  = hyp.trace->time;
                if (labelScorer->scoresTransition(transitionType)) {
                    extScore += (denseScores and tokenIdx < denseScores->size())
                                        ? (*denseScores)[tokenIdx]
                                        : (*scoreAccessor)->getScore(transitionType, tokenIdx);
                    extTime = std::max(extTime, scoreTime);
                }

                // Pre-prune based on score before creating extension instance and appending to list
                if (useScorePruning_.front()) {
                    Score referenceScore = (pruningStrategyType_ == PruningStrategySeparate and terminated)
                                                   ? currentBestTerminatedScore
                                                   : currentBestScore;
                    if (extScore > referenceScore + scoreThresholds_.front()) {
                        continue;
                    }
                }
                currentBestScore = std::min(currentBestScore, extScore);
                if (terminated) {
                    currentBestTerminatedScore = std::min(currentBestTerminatedScore, extScore);
                }

                extensions.push_back(
                        {.nextToken      = tokenIdx,
                         .pron           = lemma->pronunciations().first,
                         .score          = extScore,
                         .timeframe      = extTime,
                         .transitionType = transitionType,
                         .baseHypIndex   = hypIndex,
                         .isActive       = not terminated});
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

void BatchedLexiconfreeLabelsyncBeamSearch::pruneExtensions(SegmentState& segment, std::vector<ExtensionCandidate>& extensions, size_t scorerIdx, ThreadBuffers& buffers) {
    /*
     * Prune set of possible extensions by max beam size and possibly also by score.
     */
    size_t maxBeamSize = extensions.size();
    if (scorerIdx < labelScorers_.size() - 1) {
        maxBeamSize = maxBeamSizes_[scorerIdx];
    }
    switch (pruningStrategyType_) {
        case PruningStrategyJoint:
            scorePruning(
                    extensions,
                    PruningParams{
                            .relativeThreshold = scoreThresholds_[scorerIdx],
                            .maxBeamSize       = maxBeamSize},
                    buffers.scoreHistogram);
            break;
        case PruningStrategySeparate: {
            auto const* bestExtension = getBestHypothesis(extensions, HypothesisFilter::Any);
            verify(bestExtension != nullptr);

            separateScorePruning(
                    extensions,
                    PruningParams{
                            .relativeThreshold = scoreThresholds_[scorerIdx],
                            .maxBeamSize       = maxBeamSize,
                            .referenceScore    = bestExtension->pruningScore()},
                    PruningParams{
                            .relativeThreshold = scoreThresholds_[scorerIdx],
                            .maxBeamSize       = maxBeamSize},
                    buffers);
            break;
        }
    }
    segment.numHypsAfterIntermediatePruning[scorerIdx] += extensions.size();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-hyps-after-intermediate-pruning", extensions.size()) + Core::XmlAttribute("scorer", scorerIdx + 1) + Core::XmlAttribute("segment", segment.name);
    }
}

void BatchedLexiconfreeLabelsyncBeamSearch::createNewBeam(SegmentState& segment, ThreadBuffers& buffers) {
    auto& beam = segment.beam;
    // The new beam is built in a buffer of the thread; only the surviving hypotheses are moved to the beam of the segment
    auto& newBeam = buffers.newBeam;

    buffers.buildNewBeamTime.start();
    newBeam.clear();

    for (auto const& hyp : beam) {
        if (not hyp.isActive) {
            newBeam.push_back(hyp);
        }
    }

    for (auto const& extension : segment.extensions) {
        auto const& baseHyp = beam[extension.baseHypIndex];

        std::vector<Nn::ScoringContextRef> newScoringContexts;
        newScoringContexts.reserve(labelScorers_.size());
        for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
            newScoringContexts.push_back(labelScorers_[scorerIdx]->extendedScoringContext(
                    baseHyp.scoringContexts[scorerIdx],
                    extension.nextToken,
                    extension.transitionType));
        }
        newBeam.push_back({baseHyp, extension, newScoringContexts, lengthNormScale_});
    }
    buffers.buildNewBeamTime.stop();

    /*
     * Prune hypotheses by score according to the selected pruning strategy.
     */
    size_t const finalScorerIdx = labelScorers_.size() - 1;
    if (useScorePruning_[finalScorerIdx]) {
        buffers.beamPruningTime.start();
        auto        relativeThreshold = scoreThresholds_[finalScorerIdx];
        auto const* bestHypothesis    = getBestHypothesis(newBeam, HypothesisFilter::Any);
        verify(bestHypothesis != nullptr);
        switch (pruningStrategyType_) {
            case PruningStrategyJoint:
                if (lengthNormScale_ != 0) {
                    relativeThreshold /= std::pow(bestHypothesis->length, lengthNormScale_);
                }
                scorePruning(newBeam,
                             PruningParams{
                                     .relativeThreshold = relativeThreshold,
                                     .maxBeamSize       = newBeam.size()},
                             buffers.scoreHistogram);
                break;
            case PruningStrategySeparate: {
                auto                 activeRelativeThreshold     = relativeThreshold;
                auto                 terminatedRelativeThreshold = relativeThreshold;
                std::optional<Score> terminatedReferenceScore;
                auto const*          bestTerminatedHypothesis = getBestHypothesis(newBeam, HypothesisFilter::Terminated);
                if (bestTerminatedHypothesis != nullptr) {
                    terminatedReferenceScore = bestTerminatedHypothesis->pruningScore();
                }
                if (lengthNormScale_ != 0) {
                    activeRelativeThreshold /= std::pow(bestHypothesis->length, lengthNormScale_);
                    if (bestTerminatedHypothesis) {
                        terminatedRelativeThreshold /= std::pow(bestTerminatedHypothesis->length, lengthNormScale_);
                    }
                }
                separateScorePruning(
                        newBeam,
                        PruningParams{
                                .relativeThreshold = activeRelativeThreshold,
                                .maxBeamSize       = newBeam.size(),
                                .referenceScore    = bestHypothesis->pruningScore()},
                        PruningParams{
                                .relativeThreshold = terminatedRelativeThreshold,
                                .maxBeamSize       = newBeam.size(),
                                .referenceScore    = terminatedReferenceScore},
                        buffers);
                break;
            }
        }
        buffers.beamPruningTime.stop();

        size_t numActive     = numActiveHyps(newBeam);
        size_t numTerminated = newBeam.size() - numActive;

        segment.numTerminatedHypsAfterScorePruning += numTerminated;
        segment.numActiveHypsAfterScorePruning += numActive;

        if (stepwiseStatisticsChannel_.isOpen()) {
            stepwiseStatisticsChannel_ << Core::XmlFull("num-terminated-hyps-after-score-pruning", numTerminated) + Core::XmlAttribute("segment", segment.name);
            stepwiseStatisticsChannel_ << Core::XmlFull("num-active-hyps-after-score-pruning", numActive) + Core::XmlAttribute("segment", segment.name);
        }
    }

    /*
     * For all hypotheses with the same scoring context keep only the best since they will
     * all develop in the same way.
     */
    buffers.recombinationTime.start();
    recombination(newBeam, buffers.tempHypotheses);
    buffers.recombinationTime.stop();

    size_t numActive     = numActiveHyps(newBeam);
    size_t numTerminated = newBeam.size() - numActive;

    segment.numTerminatedHypsAfterRecombination += numTerminated;
    segment.numActiveHypsAfterRecombination += numActive;

    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-terminated-hyps-after-recombination", numTerminated) + Core::XmlAttribute("segment", segment.name);
        stepwiseStatisticsChannel_ << Core::XmlFull("num-active-hyps-after-recombination", numActive) + Core::XmlAttribute("segment", segment.name);
    }

    buffers.beamPruningTime.start();
    switch (pruningStrategyType_) {
        case PruningStrategyJoint:
            scorePruning(newBeam,
                         PruningParams{
                                 .relativeThreshold = Core::Type<Score>::max,
                                 .maxBeamSize       = maxBeamSizes_[finalScorerIdx]},
                         buffers.scoreHistogram);
            break;
        case PruningStrategySeparate: {
            auto finalBeamSize = maxBeamSizes_[finalScorerIdx];
            separateScorePruning(
                    newBeam,
                    PruningParams{
                            .relativeThreshold = Core::Type<Score>::max,
                            .maxBeamSize       = finalBeamSize},
                    PruningParams{
                            .relativeThreshold = Core::Type<Score>::max,
                            .maxBeamSize       = finalBeamSize},
                    buffers);
            break;
        }
    }
    buffers.beamPruningTime.stop();

    numActive     = numActiveHyps(newBeam);
    numTerminated = newBeam.size() - numActive;

    segment.numTerminatedHypsAfterBeamPruning += numTerminated;
    segment.numActiveHypsAfterBeamPruning += numActive;

    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-terminated-hyps-after-beam-pruning", numTerminated) + Core::XmlAttribute("segment", segment.name);
        stepwiseStatisticsChannel_ << Core::XmlFull("num-active-hyps-after-beam-pruning", numActive) + Core::XmlAttribute("segment", segment.name);
    }

    ++segment.currentSearchStep;

    beam.clear();
    beam.insert(beam.end(), std::make_move_iterator(newBeam.begin()), std::make_move_iterator(newBeam.end()));
    newBeam.clear();

    /*
     * Log statistics about the new beam after this step.
     */

    if (debugChannel_.isOpen()) {
        std::stringstream ssActive;
        std::stringstream ssTerminated;
        for (size_t hypIdx = 0ul; hypIdx < beam.size(); ++hypIdx) {
            auto const& hyp = beam[hypIdx];
            if (not hyp.isActive) {
                ssTerminated << "Segment " << segment.name << ": Terminated hypothesis " << hypIdx + 1ul << ":  " << beam[hypIdx].toString() << "\n";
            }
            else {
                ssActive << "Segment " << segment.name << ": Active hypothesis " << hypIdx + 1ul << ":  " << beam[hypIdx].toString() << "\n";
            }
        }
        ssActive << "\n";
        ssTerminated << "\n";
        debugChannel_ << ssActive.str() << ssTerminated.str();
    }

    if (stepwiseStatisticsChannel_.isOpen()) {
        auto const* bestTerminatedHyp  = getBestHypothesis(beam, HypothesisFilter::Terminated);
        auto const* worstTerminatedHyp = getWorstHypothesis(beam, HypothesisFilter::Terminated);
        auto const* bestActiveHyp      = getBestHypothesis(beam, HypothesisFilter::Active);
        auto const* worstActiveHyp     = getWorstHypothesis(beam, HypothesisFilter::Active);
        if (bestTerminatedHyp != nullptr) {
            stepwiseStatisticsChannel_ << Core::XmlFull("best-terminated-hyp-score", bestTerminatedHyp->score) + Core::XmlAttribute("segment", segment.name);
            stepwiseStatisticsChannel_ << Core::XmlFull("best-terminated-hyp-normalized-score", bestTerminatedHyp->scaledScore) + Core::XmlAttribute("segment", segment.name);
        }
        if (worstTerminatedHyp != nullptr) {
            stepwiseStatisticsChannel_ << Core::XmlFull("worst-terminated-hyp-score", worstTerminatedHyp->score) + Core::XmlAttribute("segment", segment.name);
            stepwiseStatisticsChannel_ << Core::XmlFull("worst-terminated-hyp-normalized-score", worstTerminatedHyp->scaledScore) + Core::XmlAttribute("segment", segment.name);
        }
        if (bestActiveHyp != nullptr) {
            stepwiseStatisticsChannel_ << Core::XmlFull("best-active-hyp-score", bestActiveHyp->score) + Core::XmlAttribute("segment", segment.name);
            stepwiseStatisticsChannel_ << Core::XmlFull("best-active-hyp-normalized-score", bestActiveHyp->scaledScore) + Core::XmlAttribute("segment", segment.name);
        }
        if (worstActiveHyp != nullptr) {
            stepwiseStatisticsChannel_ << Core::XmlFull("worst-active-hyp-score", worstActiveHyp->score) + Core::XmlAttribute("segment", segment.name);
            stepwiseStatisticsChannel_ << Core::XmlFull("worst-active-hyp-normalized-score", worstActiveHyp->scaledScore) + Core::XmlAttribute("segment", segment.name);
        }
    }
}

void BatchedLexiconfreeLabelsyncBeamSearch::releaseSegmentState(SegmentStateBase& segmentBase) {
    auto& segment = static_cast<SegmentState&>(segmentBase);
    // The scoring contexts are not needed anymore
    for (auto& hyp : segment.beam) {
        hyp.scoringContexts.clear();
    }
    segment.extensions.clear();
}

template<typename Element>
bool BatchedLexiconfreeLabelsyncBeamSearch::matchesHypothesisFilter(Element const& element, HypothesisFilter filter) const {
    switch (filter) {
        case HypothesisFilter::Any:
            return true;
        case HypothesisFilter::Active:
            return element.isActive;
        case HypothesisFilter::Terminated:
            return not element.isActive;
    }
    return false;
}

template<typename Element>
Element const* BatchedLexiconfreeLabelsyncBeamSearch::getBestHypothesis(
        std::vector<Element> const& elements,
        HypothesisFilter            filter) const {
    Element const* best = nullptr;

    for (auto const& element : elements) {
        if (not matchesHypothesisFilter(element, filter)) {
            continue;
        }

        if (best == nullptr or element < *best) {
            best = &element;
        }
    }

    return best;
}

BatchedLexiconfreeLabelsyncBeamSearch::LabelHypothesis const* BatchedLexiconfreeLabelsyncBeamSearch::getWorstHypothesis(
        std::vector<LabelHypothesis> const& hypotheses,
        HypothesisFilter                    filter) const {
    LabelHypothesis const* worst = nullptr;

    for (auto const& hyp : hypotheses) {
        if (not matchesHypothesisFilter(hyp, filter)) {
            continue;
        }

        if (worst == nullptr or hyp > *worst) {
            worst = &hyp;
        }
    }

    return worst;
}

BatchedLexiconfreeLabelsyncBeamSearch::LabelHypothesis const& BatchedLexiconfreeLabelsyncBeamSearch::getOutputHypothesis(std::vector<LabelHypothesis> const& hypotheses) const {
    auto const* result = getBestHypothesis(hypotheses, HypothesisFilter::Terminated);
    if (result != nullptr) {
        return *result;
    }
    result = getBestHypothesis(hypotheses, HypothesisFilter::Active);
    verify(result != nullptr);
    return *result;
}

void BatchedLexiconfreeLabelsyncBeamSearch::logSegmentStatistics(SegmentStateBase const& segmentBase) const {
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
    segment.numTerminatedHypsAfterScorePruning.write(channel);
    segment.numTerminatedHypsAfterRecombination.write(channel);
    segment.numTerminatedHypsAfterBeamPruning.write(channel);
    segment.numActiveHypsAfterScorePruning.write(channel);
    segment.numActiveHypsAfterRecombination.write(channel);
    segment.numActiveHypsAfterBeamPruning.write(channel);
    channel << Core::XmlClose("search-statistics");
}

void BatchedLexiconfreeLabelsyncBeamSearch::logSearchStepBreakdown() const {
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
}

void BatchedLexiconfreeLabelsyncBeamSearch::resetSearchStepBreakdown() {
    for (auto* timers : {&scoreAndPruneExtensionsTimes_, &scoringTimes_, &scoreReadoutTimes_, &intermediatePruningTimes_}) {
        for (auto& timer : *timers) {
            timer.reset();
        }
    }
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

template<typename Element>
void BatchedLexiconfreeLabelsyncBeamSearch::scorePruning(std::vector<Element>& hypotheses, PruningParams const& pruningParams, Histogram& scoreHistogram) {
    if (hypotheses.empty()) {
        return;
    }

    if (not histogramPruningEnabled_ and hypotheses.size() > pruningParams.maxBeamSize) {
        // Plain max beam size pruning: keep exactly the maxBeamSize best hypotheses
        std::nth_element(hypotheses.begin(), hypotheses.begin() + pruningParams.maxBeamSize, hypotheses.end(), [](auto const& a, auto const& b) { return a.pruningScore() < b.pruningScore(); });
        hypotheses.resize(pruningParams.maxBeamSize);
    }

    // Find ranges for score histogram and setting absolute threshold
    Score lowerScore = Core::Type<Score>::max;
    Score upperScore = Core::Type<Score>::min;

    for (auto const& hyp : hypotheses) {
        if (Math::isinf(hyp.score) or hyp.score >= Core::Type<Score>::max) {
            continue;
        }
        lowerScore = std::min(lowerScore, hyp.pruningScore());
        upperScore = std::max(upperScore, hyp.pruningScore());
    }

    if (lowerScore > upperScore) {
        // No hypothesis with finite score is present in the beam
        hypotheses.clear();
        return;
    }

    if (hypotheses.size() <= pruningParams.maxBeamSize and pruningParams.relativeThreshold == Core::Type<Score>::max) {
        // Neither relative score pruning nor max beam size pruning triggers
        return;
    }

    if (pruningParams.relativeThreshold != Core::Type<Score>::max and pruningParams.referenceScore and lowerScore > *pruningParams.referenceScore + pruningParams.relativeThreshold) {
        // Even the lowest score is not in range of `referenceScore`, so everything will be pruned
        hypotheses.clear();
        return;
    }

    if (lowerScore == upperScore) {
        // All scores are the same (usually only happens when exactly 1 hyp is active)
        if (hypotheses.size() > pruningParams.maxBeamSize) {
            hypotheses.resize(pruningParams.maxBeamSize);
        }
        return;
    }

    Score absoluteThreshold = upperScore;

    // Pruning by relative score threshold
    if (pruningParams.relativeThreshold != Core::Type<Score>::max) {
        absoluteThreshold = pruningParams.referenceScore.value_or(lowerScore) + pruningParams.relativeThreshold;
    }

    // Pruning by max beam size
    if (hypotheses.size() > pruningParams.maxBeamSize) {
        scoreHistogram.clear();
        scoreHistogram.setLimits(lowerScore, upperScore);

        for (auto const& hyp : hypotheses) {
            if (Math::isinf(hyp.score) or hyp.score >= Core::Type<Score>::max) {
                continue;
            }
            scoreHistogram += hyp.pruningScore();
        }

        absoluteThreshold = std::min(absoluteThreshold, scoreHistogram.quantile(pruningParams.maxBeamSize));
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
                    [absoluteThreshold](auto const& hyp) { return hyp.pruningScore() > absoluteThreshold; }),
            hypotheses.end());
}

template<typename Element>
void BatchedLexiconfreeLabelsyncBeamSearch::separateScorePruning(
        std::vector<Element>& elements,
        PruningParams const&  activePruningParams,
        PruningParams const&  terminatedPruningParams,
        ThreadBuffers&        buffers) {
    if (elements.empty()) {
        return;
    }

    if (not histogramPruningEnabled_) {
        // Plain max beam size pruning: keep exactly the maxBeamSize best elements of each pool
        auto   byScore   = [](auto const& a, auto const& b) { return a.pruningScore() < b.pruningScore(); };
        size_t numActive = std::partition(elements.begin(), elements.end(), [](auto const& element) { return element.isActive; }) - elements.begin();
        if (elements.size() - numActive > terminatedPruningParams.maxBeamSize) {
            auto keepEnd = elements.begin() + numActive + terminatedPruningParams.maxBeamSize;
            std::nth_element(elements.begin() + numActive, keepEnd, elements.end(), byScore);
            elements.erase(keepEnd, elements.end());
        }
        if (numActive > activePruningParams.maxBeamSize) {
            auto keepEnd = elements.begin() + activePruningParams.maxBeamSize;
            std::nth_element(elements.begin(), keepEnd, elements.begin() + numActive, byScore);
            elements.erase(keepEnd, elements.begin() + numActive);
        }
    }

    static constexpr size_t activeIdx     = 0ul;
    static constexpr size_t terminatedIdx = 1ul;
    static constexpr size_t numPools      = 2ul;

    PruningParams const* pruningParams[numPools] = {&activePruningParams, &terminatedPruningParams};
    Histogram*           histograms[numPools]    = {&buffers.activeScoreHistogram, &buffers.terminatedScoreHistogram};

    size_t numElements[numPools]            = {0ul, 0ul};
    size_t numFiniteScoreElements[numPools] = {0ul, 0ul};
    Score  lowerScores[numPools]            = {Core::Type<Score>::max, Core::Type<Score>::max};
    Score  upperScores[numPools]            = {Core::Type<Score>::min, Core::Type<Score>::min};
    Score  absoluteThresholds[numPools]     = {Core::Type<Score>::max, Core::Type<Score>::max};
    bool   pruneAll[numPools]               = {false, false};
    bool   noPruning[numPools]              = {false, false};
    bool   allScoresEqual[numPools]         = {false, false};

    for (auto const& element : elements) {
        size_t const poolIdx = element.isActive ? activeIdx : terminatedIdx;
        ++numElements[poolIdx];

        if (Math::isinf(element.score) or element.score >= Core::Type<Score>::max) {
            continue;
        }

        ++numFiniteScoreElements[poolIdx];
        auto pruningScore    = element.pruningScore();
        lowerScores[poolIdx] = std::min(lowerScores[poolIdx], pruningScore);
        upperScores[poolIdx] = std::max(upperScores[poolIdx], pruningScore);
    }

    for (size_t poolIdx = 0ul; poolIdx < numPools; ++poolIdx) {
        auto const* params = pruningParams[poolIdx];
        bool const  active = poolIdx == activeIdx;

        if (numElements[poolIdx] == 0ul) {
            continue;
        }

        if (numFiniteScoreElements[poolIdx] == 0ul) {
            // No element with finite score is present in this set
            pruneAll[poolIdx] = true;
            continue;
        }

        if (numElements[poolIdx] <= params->maxBeamSize and params->relativeThreshold == Core::Type<Score>::max) {
            // Neither relative score pruning nor max beam size pruning triggers
            noPruning[poolIdx] = true;
            continue;
        }

        absoluteThresholds[poolIdx] = upperScores[poolIdx];

        if (params->relativeThreshold != Core::Type<Score>::max) {
            absoluteThresholds[poolIdx] = params->referenceScore.value_or(lowerScores[poolIdx]) + params->relativeThreshold;
            if (lowerScores[poolIdx] > absoluteThresholds[poolIdx]) {
                // Even the lowest score is not in range of `referenceScore`, so everything in this pool will be pruned
                pruneAll[poolIdx] = true;
                continue;
            }
        }

        if (numElements[poolIdx] > params->maxBeamSize) {
            if (lowerScores[poolIdx] == upperScores[poolIdx]) {
                // A score threshold cannot separate equal scores, so keep the first maxBeamSize elements in this pool.
                allScoresEqual[poolIdx] = numFiniteScoreElements[poolIdx] > params->maxBeamSize;
                continue;
            }

            if (numFiniteScoreElements[poolIdx] > params->maxBeamSize) {
                histograms[poolIdx]->clear();
                histograms[poolIdx]->setLimits(lowerScores[poolIdx], upperScores[poolIdx]);

                for (auto const& element : elements) {
                    if (element.isActive == active and
                        not Math::isinf(element.score) and
                        element.score < Core::Type<Score>::max) {
                        *histograms[poolIdx] += element.pruningScore();
                    }
                }

                absoluteThresholds[poolIdx] = std::min(
                        absoluteThresholds[poolIdx],
                        histograms[poolIdx]->quantile(params->maxBeamSize));
            }
        }
    }

    size_t numKept[numPools] = {0ul, 0ul};
    auto   writeIt           = elements.begin();
    for (auto readIt = elements.begin(); readIt != elements.end(); ++readIt) {
        size_t const poolIdx = readIt->isActive ? activeIdx : terminatedIdx;

        bool eraseElement;
        if (pruneAll[poolIdx]) {
            eraseElement = true;
        }
        else if (noPruning[poolIdx]) {
            eraseElement = false;
        }
        else if (readIt->pruningScore() > absoluteThresholds[poolIdx]) {
            eraseElement = true;
        }
        else if (allScoresEqual[poolIdx]) {
            eraseElement = numKept[poolIdx] >= pruningParams[poolIdx]->maxBeamSize;
        }
        else {
            eraseElement = false;
        }

        if (not eraseElement) {
            ++numKept[poolIdx];
            if (writeIt != readIt) {
                *writeIt = std::move(*readIt);
            }
            ++writeIt;
        }
    }

    elements.erase(writeIt, elements.end());
}

void BatchedLexiconfreeLabelsyncBeamSearch::recombination(std::vector<LabelHypothesis>& hypotheses, std::vector<LabelHypothesis>& tempHypotheses) {
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
    tempHypotheses.reserve(hypotheses.size());

    // Map each unique ScoringContext in `hypotheses` to its hypothesis
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

    hypotheses.swap(tempHypotheses);
}

size_t BatchedLexiconfreeLabelsyncBeamSearch::numActiveHyps(std::vector<LabelHypothesis> const& hypotheses) const {
    return std::accumulate(
            hypotheses.begin(),
            hypotheses.end(),
            0ul,
            [](size_t acc, auto const& hyp) { return acc + static_cast<size_t>(hyp.isActive); });
}

}  // namespace Search
