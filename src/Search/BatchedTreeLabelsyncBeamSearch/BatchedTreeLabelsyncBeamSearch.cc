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

#include "BatchedTreeLabelsyncBeamSearch.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <strings.h>

#include <Am/ClassicStateModel.hh>
#include <Core/CollapsedVector.hh>
#include <Core/XmlStream.hh>
#include <Lattice/LatticeAdaptor.hh>
#include <Lm/BackingOff.hh>
#include <Math/Utilities.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/Module.hh>
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

BatchedTreeLabelsyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContexts(),
          currentToken(Nn::invalidLabelIndex),
          currentState(invalidTreeNodeIndex),
          lmHistory(),
          timeframe(0),
          length(0),
          score(0.0),
          scaledScore(0.0),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))),
          isActive(true) {
}

BatchedTreeLabelsyncBeamSearch::LabelHypothesis::LabelHypothesis(
        BatchedTreeLabelsyncBeamSearch::LabelHypothesis const&              base,
        BatchedTreeLabelsyncBeamSearch::WithinWordExtensionCandidate const& extension,
        std::vector<Nn::ScoringContextRef> const&                           newScoringContexts,
        float                                                               lengthNormScale)
        : scoringContexts(newScoringContexts),
          currentToken(extension.nextToken),
          currentState(extension.nextState),
          lmHistory(base.lmHistory),
          timeframe(extension.timeframe),
          length(base.length + 1),
          score(extension.score),
          scaledScore(score / std::pow(length, lengthNormScale)),
          trace(base.trace),
          isActive(extension.transitionType != Nn::TransitionType::SENTENCE_END) {
}

BatchedTreeLabelsyncBeamSearch::LabelHypothesis::LabelHypothesis(
        LabelHypothesis const&                                           base,
        BatchedTreeLabelsyncBeamSearch::WordEndExtensionCandidate const& extension,
        Lm::History const&                                               newLmHistory,
        float                                                            lengthNormScale)
        : scoringContexts(base.scoringContexts),
          currentToken(base.currentToken),
          currentState(extension.rootState),
          lmHistory(newLmHistory),
          timeframe(extension.timeframe),
          length(base.length),
          score(extension.score),
          scaledScore(score / std::pow(length, lengthNormScale)),
          trace(),
          isActive(base.isActive) {
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

std::string BatchedTreeLabelsyncBeamSearch::LabelHypothesis::toString() const {
    std::stringstream ss;
    ss << "Score: " << score << ", scaled score: " << scaledScore << ", traceback: ";

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

BatchedTreeLabelsyncBeamSearch::SegmentState::SegmentState(SegmentHandle handle, std::string const& name, size_t numLabelScorers)
        : SegmentStateBase(handle, name),
          beam(),
          currentSearchStep(0ul),
          hypIndexToContextIndexMap(),
          withinWordExtensions(),
          wordEndExtensions(),
          newBeam(),
          wordEndHypotheses(),
          stepFailed(false),
          numInputHyps("num-input-hyps"),
          numWithinWordExtensionsBeforeFirstPruning("num-within-word-extensions-before-first-pruning"),
          numWithinWordHypsAfterIntermediatePruning(numLabelScorers, Core::Statistics<u32>("num-within-word-hyps-after-intermediate-pruning")),
          numTerminatedHypsAfterScorePruning("num-terminated-hyps-after-score-pruning"),
          numTerminatedHypsAfterRecombination("num-terminated-hyps-after-recombination"),
          numTerminatedHypsAfterBeamPruning("num-terminated-hyps-after-beam-pruning"),
          numActiveHypsAfterScorePruning("num-active-hyps-after-score-pruning"),
          numActiveHypsAfterRecombination("num-active-hyps-after-recombination"),
          numActiveHypsAfterBeamPruning("num-active-hyps-after-beam-pruning"),
          numWordEndExtensionsBeforePruning("num-word-end-extensions-before-pruning"),
          numActiveWordEndHypsAfterPruning("num-word-end-hyps-after-pruning"),
          numActiveWordEndHypsAfterScorePruning("num-active-word-end-hyps-after-score-pruning"),
          numActiveWordEndHypsAfterRecombination("num-active-word-end-hyps-after-recombination"),
          numActiveWordEndHypsAfterBeamPruning("num-active-word-end-hyps-after-beam-pruning"),
          numActiveTrees("num-active-trees") {
}

/*
 * ======================================
 * === BatchedTreeLabelsyncBeamSearch ===
 * ======================================
 */

BatchedTreeLabelsyncBeamSearch::ThreadBuffers::ThreadBuffers(size_t numHistogramBins)
        : tempHypotheses(),
          wordEndScoreAccessors(),
          scoreHistogram(numHistogramBins),
          activeScoreHistogram(numHistogramBins),
          terminatedScoreHistogram(numHistogramBins),
          scoreReadoutTimes(),
          intermediatePruningTimes(),
          buildWithinWordHypsTime(),
          wordEndExpansionTime(),
          wordEndExtensionTime(),
          wordEndScorePruningTime(),
          wordEndHypBuildingTime(),
          lmTime(),
          recombinationTime(),
          beamPruningTime() {}

const Core::ParameterIntVector BatchedTreeLabelsyncBeamSearch::paramMaxBeamSizes(
        "max-beam-size",
        "Maximum number of elements in the search beam of each segment."
        " The pruning-strategy-type determines whether active and terminated hypotheses share the final limit or are limited separately."
        " Pruning is applied after each intermediate label scorer.",
        "",
        1);

const Core::ParameterInt BatchedTreeLabelsyncBeamSearch::paramMaxWordEndBeamSize(
        "max-word-end-beam-size",
        "Maximum number of word-end hypotheses in the search beam of each segment. If not set, global beam pruning will be done and word-end hypotheses will not be pruned separately.",
        std::numeric_limits<int>::max(), 0);

const Core::ParameterFloatVector BatchedTreeLabelsyncBeamSearch::paramScoreThresholds(
        "score-threshold",
        "Prune hypotheses with a score that is at least this much worse than the pruning reference."
        " In joint pruning mode, the final combined beam uses the best hypothesis as reference."
        " In separate pruning mode, active hypotheses in the final combined beam use the overall best hypothesis and terminated hypotheses use the best terminated hypothesis."
        " If length normalization is enabled, the score threshold is added to the raw score before normalization."
        " Pruning is applied after each intermediate label scorer."
        " If not set, no score pruning will be done.",
        "",
        0,
        Core::Type<Score>::max);

const Core::ParameterFloat BatchedTreeLabelsyncBeamSearch::paramWordEndScoreThreshold(
        "word-end-score-threshold",
        "Prune any word-end hypothesis with a score that is at least this much worse than the best word-end hypothesis. This threshold is relative to the score-threshold."
        " If not set, global score pruning will be done and word-end hypotheses will not be pruned separately.",
        Core::Type<Score>::max, 0);

const Core::ParameterInt BatchedTreeLabelsyncBeamSearch::paramNumHistogramBins(
        "num-histogram-bins",
        "Number of bins for histogram pruning of hypotheses (very minor effect).",
        100,
        2);

const Core::ParameterBool BatchedTreeLabelsyncBeamSearch::paramEnableHistogramPruning(
        "enable-histogram-pruning",
        "Use histogram pruning for max-beam-size. If false, exactly the max-beam-size best hypotheses are kept.",
        true);

const Core::ParameterFloat BatchedTreeLabelsyncBeamSearch::paramLengthNormScale(
        "length-norm-scale",
        "Exponent of length for the hypothesis length normalization. Scaled scores are computed as score / length^length_norm_scale.",
        0.0);

const Core::ParameterFloat BatchedTreeLabelsyncBeamSearch::paramMaxLabelsPerTimestep(
        "max-labels-per-timestep",
        "Maximum number of emitted labels per input timestep counted via `putFeature`/`putFeatures`."
        " The resulting limit (rounded up) is clamped to the range given by `max-labels-lower-bound` and `max-labels-upper-bound`.",
        1.0);

const Core::ParameterInt BatchedTreeLabelsyncBeamSearch::paramMaxLabelsLowerBound(
        "max-labels-lower-bound",
        "Lower bound for the maximum number of emitted labels, so that short inputs are not cut off too early."
        " Only raises the limit given by `max-labels-per-timestep`, but does not enforce a minimum output length,"
        " the search still ends as soon as all hypotheses have reached sentence-end.",
        0,
        0);

const Core::ParameterInt BatchedTreeLabelsyncBeamSearch::paramMaxLabelsUpperBound(
        "max-labels-upper-bound",
        "Upper bound for the maximum number of emitted labels, independent of the input length."
        " If it is smaller than `max-labels-lower-bound`, the upper bound takes precedence.",
        std::numeric_limits<int>::max(),
        0);

const Core::Choice BatchedTreeLabelsyncBeamSearch::choicePruningStrategyType(
        "joint", PruningStrategyJoint,
        "separate", PruningStrategySeparate,
        Core::Choice::endMark());

const Core::ParameterChoice BatchedTreeLabelsyncBeamSearch::paramPruningStrategyType(
        "pruning-strategy-type",
        &choicePruningStrategyType,
        "How active and terminated hypotheses are pruned in the final combined beam."
        " Joint mode prunes both pools together."
        " Separate mode prunes terminated hypotheses against terminated hypotheses and active hypotheses against the overall best hypothesis.",
        PruningStrategyJoint);

const Core::ParameterBool BatchedTreeLabelsyncBeamSearch::paramSentenceEndFallBack(
        "sentence-end-fall-back",
        "Allow for fallback solution if no active word-end hypothesis exists at the end of a segment.",
        true);

const Core::Choice BatchedTreeLabelsyncBeamSearch::choiceRecombinationMode(
        "off", RecombinationModeOff,
        "on", RecombinationModeOn,
        Core::Choice::endMark());

const Core::ParameterChoice BatchedTreeLabelsyncBeamSearch::paramRecombinationMode(
        "recombination-mode",
        &choiceRecombinationMode,
        "Whether hypotheses with identical recombination state should be recombined.",
        RecombinationModeOn);

const Core::ParameterInt BatchedTreeLabelsyncBeamSearch::paramCacheCleanupInterval(
        "cache-cleanup-interval",
        "Interval of search steps after which buffered inputs that are not needed anymore get cleaned up.",
        10,
        1);

BatchedTreeLabelsyncBeamSearch::BatchedTreeLabelsyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          BatchedSearchBase(config),
          maxWordEndBeamSize_(paramMaxWordEndBeamSize(config)),
          histogramPruningEnabled_(paramEnableHistogramPruning(config)),
          wordEndScoreThresholdFactor_(paramWordEndScoreThreshold(config)),
          wordEndScoreThreshold_(Core::Type<Score>::max),
          lengthNormScale_(paramLengthNormScale(config)),
          maxLabelsPerTimestep_(paramMaxLabelsPerTimestep(config)),
          maxLabelsLowerBound_(paramMaxLabelsLowerBound(config)),
          maxLabelsUpperBound_(paramMaxLabelsUpperBound(config)),
          sentenceEndLemma_(),
          sentenceEndLabelIndex_(Nn::invalidLabelIndex),
          cacheCleanupInterval_(paramCacheCleanupInterval(config)),
          sentenceEndFallback_(paramSentenceEndFallBack(config)),
          pruningStrategyType_(paramPruningStrategyType(config)),
          recombinationEnabled_(paramRecombinationMode(config) == RecombinationModeOn),
          nonWordLemmas_(),
          debugChannel_(config, "debug"),
          stepSegments_(),
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
    for (size_t i = 0; i < scoreThresholds_.size(); ++i) {
        useScorePruning_.push_back(scoreThresholds_[i] != Core::Type<Score>::max);
    }

    if (maxLabelsLowerBound_ > maxLabelsUpperBound_) {
        warning() << "max-labels-lower-bound (" << maxLabelsLowerBound_ << ") is larger than max-labels-upper-bound (" << maxLabelsUpperBound_ << "). The upper bound takes precedence";
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

Speech::ModelCombination::Mode BatchedTreeLabelsyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon | Speech::ModelCombination::useAcousticModel | Speech::ModelCombination::useLanguageModel;
}

Am::AcousticModel::Mode BatchedTreeLabelsyncBeamSearch::requiredAcousticModel() const {
    return Am::AcousticModel::noEmissions;
}

bool BatchedTreeLabelsyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_       = modelCombination.lexicon();
    acousticModel_ = modelCombination.acousticModel();
    languageModel_ = modelCombination.languageModel();
    // Only backing-off LMs can be queried from several threads at the same time (no mutable state, no reference
    // counting of histories)
    if (not dynamic_cast<Lm::BackingOffLm const*>(languageModel_->unscaled().get())) {
        disableParallelSearch("the language model is not a backing-off LM and can't be used from several threads");
    }
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

    size_t const finalScorerIdx = labelScorers_.size() - 1;
    switch (pruningStrategyType_) {
        case PruningStrategyJoint:
            break;
        case PruningStrategySeparate:
            if (not useScorePruning_[finalScorerIdx]) {
                error() << "pruning-strategy-type=separate requires a finite final score-threshold. Otherwise, the normal stop criterion can't trigger. Use pruning-strategy-type=joint for pure max-beam-size pruning.";
            }
            break;
    }

    // The word-end threshold is configured as a factor relative to the within-word threshold of the final label scorer
    if (wordEndScoreThresholdFactor_ != Core::Type<Score>::max) {
        Score const withinWordScoreThreshold = scoreThresholds_[finalScorerIdx];
        if (withinWordScoreThreshold == Core::Type<Score>::max) {
            error() << "Word-end score-threshold is relative to the score-threshold, but no score-threshold is set for the final label scorer";
            wordEndScoreThreshold_ = Core::Type<Score>::max;
        }
        else {
            wordEndScoreThreshold_ = wordEndScoreThresholdFactor_ * withinWordScoreThreshold;
            log() << "Use absolute word-end score-threshold of " << wordEndScoreThreshold_ << "; computed relative to within-word threshold " << withinWordScoreThreshold << " with factor " << wordEndScoreThresholdFactor_;
        }
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
        log() << "Use sentence-end index " << sentenceEndLabelIndex_ << " inferred from lexicon";
    }
    else {
        error() << "No sentence end lemma or pronunciation defined";
    }

    if (lexicon_->specialLemma("silence") and (lexicon_->specialLemma("silence")->syntacticTokenSequence()).size() != 0) {
        warning("Special lemma silence will be scored by the language model. To prevent the LM from scoring it, set an empty syntactic token sequence for it in the lexicon.");
    }

    // Create look-ups for state successors and exits of each state
    createSuccessorLookups();

    return true;
}

std::unique_ptr<BatchedSearchBase::SegmentStateBase> BatchedTreeLabelsyncBeamSearch::createSegmentState(SegmentHandle handle, std::string const& name) {
    auto state = std::make_unique<SegmentState>(handle, name, labelScorers_.size());

    // Initialize beam with a single empty hypothesis
    state->beam.push_back(LabelHypothesis());
    for (auto& labelScorer : labelScorers_) {
        state->beam.front().scoringContexts.push_back(labelScorer->getInitialSegmentScoringContext(handle));
    }
    state->beam.front().currentState = network_->rootState;
    state->beam.front().lmHistory    = languageModel_->startHistory();

    return state;
}

Core::Ref<const Traceback> BatchedTreeLabelsyncBeamSearch::getCurrentBestTraceback(SegmentHandle segment) const {
    return getOutputHypothesis(getSegment<SegmentState>(segment).beam).trace->performTraceback();
}

Core::Ref<const LatticeAdaptor> BatchedTreeLabelsyncBeamSearch::getCurrentBestWordLattice(SegmentHandle segment) const {
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

Core::Ref<const LatticeTrace> BatchedTreeLabelsyncBeamSearch::getCurrentBestLatticeTrace(SegmentHandle segment) const {
    return getOutputHypothesis(getSegment<SegmentState>(segment).beam).trace;
}

Core::Ref<const LatticeTrace> BatchedTreeLabelsyncBeamSearch::getCommonPrefix(SegmentHandle segment) const {
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

bool BatchedTreeLabelsyncBeamSearch::decodeGroupStep(size_t group) {
    searchStepTime_.start();

    /*
     * Determine the segments that take part in this step and finish the ones that can't continue.
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
                finalizeAndFinishSegment(segment);
                anythingChanged = true;
            }
            // Otherwise wait for more features
            return;
        }
        if (std::all_of(segment.beam.begin(), segment.beam.end(), [](LabelHypothesis const& hyp) { return not hyp.isActive; })) {
            log() << "Terminated search of segment " << segment.name << " because all hypotheses reached sentence-end";
            finalizeAndFinishSegment(segment);
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
             * Prune set of possible extensions by max beam size and possibly also by score.
             */
            Core::StopWatch::Scope pruningTimer(buffers.intermediatePruningTimes[scorerIdx]);
            size_t                 maxBeamSize = segment.withinWordExtensions.size();
            if (scorerIdx < labelScorers_.size() - 1) {
                maxBeamSize = maxBeamSizes_[scorerIdx];
            }
            scorePruning(segment.withinWordExtensions,
                         PruningParams{
                                 .relativeThreshold = scoreThresholds_[scorerIdx],
                                 .maxBeamSize       = maxBeamSize},
                         buffers.scoreHistogram);
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
                finalizeAndFinishSegment(*segment);
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

size_t BatchedTreeLabelsyncBeamSearch::maxOutputLength(SegmentState const& segment) const {
    size_t const numInputs  = std::max(segment.totalTimesteps, 1ul);
    size_t const ratioLimit = static_cast<size_t>(std::ceil(std::max(maxLabelsPerTimestep_ * numInputs, 0.0f)));
    return std::min(std::max(ratioLimit, maxLabelsLowerBound_), maxLabelsUpperBound_);
}

bool BatchedTreeLabelsyncBeamSearch::canMakeStep(SegmentState const& segment) const {
    return not segment.finished and
           segment.currentSearchStep < maxOutputLength(segment) and
           std::any_of(segment.beam.begin(), segment.beam.end(), [](LabelHypothesis const& hyp) { return hyp.isActive; });
}

void BatchedTreeLabelsyncBeamSearch::collectScoringContexts(std::vector<SegmentState*> const& segments, std::vector<Nn::ScoringContextRef>& contexts) {
    /*
     * The contexts of the active hypotheses of all segments are passed to the first label scorer in one list.
     * `segment.hypIndexToContextIndexMap` maps a beam index of the segment to the position of its context in the list
     * (-1 if the hypothesis contributes no context).
     */
    contexts.clear();
    for (auto* segment : segments) {
        segment->withinWordExtensions.clear();
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

void BatchedTreeLabelsyncBeamSearch::startNextScoring(size_t group) {
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

void BatchedTreeLabelsyncBeamSearch::scoreExtensions(SegmentState&                                           segment,
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

            // Iterate over the successors of this hypothesis' current state in the tree
            for (size_t i = stateSuccessorsOffset_[hyp.currentState]; i < stateSuccessorsOffset_[hyp.currentState + 1]; ++i) {
                const StateId  successorState = stateSuccessors_[i];
                Nn::LabelIndex tokenIdx       = network_->structure.state(successorState).stateDesc.acousticModel;

                auto transitionType = Nn::TransitionType::LABEL_TO_LABEL;
                if (hyp.currentToken == Nn::invalidLabelIndex) {
                    transitionType = Nn::TransitionType::INITIAL_LABEL;
                }
                if (tokenIdx == sentenceEndLabelIndex_) {
                    transitionType = Nn::TransitionType::SENTENCE_END;
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
                if (useScorePruning_.front() and extScore > currentBestScore + scoreThresholds_.front()) {
                    continue;
                }
                currentBestScore = std::min(currentBestScore, extScore);

                extensions.push_back(
                        {.nextToken      = tokenIdx,
                         .nextState      = successorState,
                         .timeframe      = extTime,
                         .score          = extScore,
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

void BatchedTreeLabelsyncBeamSearch::createNewBeam(SegmentState& segment, ThreadBuffers& buffers) {
    auto& beam              = segment.beam;
    auto& newBeam           = segment.newBeam;
    auto& wordEndExtensions = segment.wordEndExtensions;
    auto& wordEndHypotheses = segment.wordEndHypotheses;

    /*
     * Create new beam from surviving extensions.
     */
    buffers.buildWithinWordHypsTime.start();
    newBeam.clear();

    for (auto const& hyp : beam) {
        if (not hyp.isActive) {
            newBeam.push_back(hyp);
        }
    }

    for (auto const& extension : segment.withinWordExtensions) {
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
    buffers.buildWithinWordHypsTime.stop();

    /*
     * Expand hypotheses to word-end hypotheses and incorporate the language model
     */
    buffers.wordEndExpansionTime.start();
    buffers.wordEndExtensionTime.start();
    wordEndExtensions.clear();
    for (size_t hypIndex = 0ul; hypIndex < newBeam.size(); ++hypIndex) {
        auto& hyp = newBeam[hypIndex];

        // Terminated hypotheses that are already in a root state have been expanded to word-end
        // hypotheses at termination time and can't yield new ones
        if (not hyp.isActive and network_->isRoot(hyp.currentState)) {
            continue;
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

            if (not hyp.isActive) {
                // Expand freshly terminated hypothesis to a word-end hypothesis with the sentence-end LM score
                buffers.lmTime.start();
                Score sentenceEndScore = languageModel_->sentenceEndScore(hyp.lmHistory);
                buffers.lmTime.stop();

                wordEndExtensions.push_back({
                        .pron           = lemmaPron,
                        .rootState      = exit.transitState,
                        .score          = hyp.score + sentenceEndScore,
                        .timeframe      = hyp.timeframe,
                        .transitionType = Nn::TransitionType::SENTENCE_END,
                        .baseHypIndex   = hypIndex,
                        .isActive       = false,
                });
                continue;
            }

            Score       lmScore = 0;
            auto const& sts     = lemma->syntacticTokenSequence();
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
                    .timeframe      = hyp.timeframe,
                    .transitionType = wordEndtransitionType,
                    .baseHypIndex   = hypIndex,
                    .isActive       = true,
            });
        }
    }

    buffers.wordEndExtensionTime.stop();

    segment.numWordEndExtensionsBeforePruning += wordEndExtensions.size();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-word-end-extensions-before-pruning", wordEndExtensions.size()) + Core::XmlAttribute("segment", segment.name);
    }

    /*
     * Prune set of word-end hypotheses by max beam size and possibly also by score
     * according to the selected pruning strategy.
     */
    buffers.wordEndScorePruningTime.start();
    switch (pruningStrategyType_) {
        case PruningStrategyJoint:
            scorePruning(wordEndExtensions,
                         PruningParams{
                                 .relativeThreshold = wordEndScoreThreshold_,
                                 .maxBeamSize       = maxWordEndBeamSize_},
                         buffers.scoreHistogram);
            break;
        case PruningStrategySeparate: {
            // There are no word-end extensions at all in most steps, so the best one is optional here
            std::optional<Score> activeReferenceScore;
            auto const*          bestExtension = getBestHypothesis(wordEndExtensions, HypothesisFilter::Any);
            if (bestExtension != nullptr) {
                activeReferenceScore = bestExtension->pruningScore();
            }

            separateScorePruning(
                    wordEndExtensions,
                    PruningParams{
                            .relativeThreshold = wordEndScoreThreshold_,
                            .maxBeamSize       = maxWordEndBeamSize_,
                            .referenceScore    = activeReferenceScore},
                    PruningParams{
                            .relativeThreshold = wordEndScoreThreshold_,
                            .maxBeamSize       = maxWordEndBeamSize_},
                    buffers);
            break;
        }
    }
    buffers.wordEndScorePruningTime.stop();
    segment.numActiveWordEndHypsAfterPruning += wordEndExtensions.size();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-word-end-hyps-after-pruning", wordEndExtensions.size()) + Core::XmlAttribute("segment", segment.name);
    }

    // Create new word-end label hypotheses from word-end extension candidates and update the LM history
    buffers.wordEndHypBuildingTime.start();
    wordEndHypotheses.clear();
    for (auto& extension : wordEndExtensions) {
        auto const& baseHyp = newBeam[extension.baseHypIndex];

        auto        newLmHistory = baseHyp.lmHistory;
        auto const& sts          = extension.pron->lemma()->syntacticTokenSequence();

        // The LM history is not extended for terminated hypotheses since sentence-end is the last LM scoring step
        if (baseHyp.isActive and sts.size() != 0) {
            require(sts.size() == 1);
            const Bliss::SyntacticToken* st = sts.front();
            newLmHistory                    = languageModel_->extendedHistory(newLmHistory, st);
        }

        wordEndHypotheses.push_back({baseHyp, extension, newLmHistory, lengthNormScale_});
    }

    // Freshly terminated hypotheses have been expanded to word-end hypotheses, so the base hypotheses are removed
    newBeam.erase(
            std::remove_if(
                    newBeam.begin(),
                    newBeam.end(),
                    [this](LabelHypothesis const& hyp) { return not hyp.isActive and not network_->isRoot(hyp.currentState); }),
            newBeam.end());

    newBeam.insert(newBeam.end(), wordEndHypotheses.begin(), wordEndHypotheses.end());
    buffers.wordEndHypBuildingTime.stop();
    buffers.wordEndExpansionTime.stop();

    /*
     * Prune terminated and active hypotheses by score according to the selected pruning strategy.
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
                    if (bestTerminatedHypothesis != nullptr) {
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

        size_t numActive        = numActiveHyps(newBeam);
        size_t numActiveWordEnd = numActiveWordEndHyps(newBeam);
        segment.numTerminatedHypsAfterScorePruning += newBeam.size() - numActive;
        segment.numActiveHypsAfterScorePruning += numActive;
        segment.numActiveWordEndHypsAfterScorePruning += numActiveWordEnd;
        if (stepwiseStatisticsChannel_.isOpen()) {
            stepwiseStatisticsChannel_ << Core::XmlFull("num-terminated-hyps-after-score-pruning", newBeam.size() - numActive) + Core::XmlAttribute("segment", segment.name);
            stepwiseStatisticsChannel_ << Core::XmlFull("num-active-hyps-after-score-pruning", numActive) + Core::XmlAttribute("segment", segment.name);
            stepwiseStatisticsChannel_ << Core::XmlFull("num-active-word-end-hyps-after-score-pruning", numActiveWordEnd) + Core::XmlAttribute("segment", segment.name);
        }
    }

    /*
     * For all hypotheses with the same activity status at the same state and with the same
     * scoring context and LM history keep only the best since they will all develop in the same way.
     */
    buffers.recombinationTime.start();
    recombination(newBeam, buffers.tempHypotheses);
    buffers.recombinationTime.stop();

    size_t numActive        = numActiveHyps(newBeam);
    size_t numActiveWordEnd = numActiveWordEndHyps(newBeam);
    segment.numTerminatedHypsAfterRecombination += newBeam.size() - numActive;
    segment.numActiveHypsAfterRecombination += numActive;
    segment.numActiveWordEndHypsAfterRecombination += numActiveWordEnd;
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-terminated-hyps-after-recombination", newBeam.size() - numActive) + Core::XmlAttribute("segment", segment.name);
        stepwiseStatisticsChannel_ << Core::XmlFull("num-active-hyps-after-recombination", numActive) + Core::XmlAttribute("segment", segment.name);
        stepwiseStatisticsChannel_ << Core::XmlFull("num-active-word-end-hyps-after-recombination", numActiveWordEnd) + Core::XmlAttribute("segment", segment.name);
    }

    /*
     * Prune new beam by max beam size according to the selected pruning strategy.
     */
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

    numActive        = numActiveHyps(newBeam);
    numActiveWordEnd = numActiveWordEndHyps(newBeam);
    segment.numTerminatedHypsAfterBeamPruning += newBeam.size() - numActive;
    segment.numActiveHypsAfterBeamPruning += numActive;
    segment.numActiveWordEndHypsAfterBeamPruning += numActiveWordEnd;
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-terminated-hyps-after-beam-pruning", newBeam.size() - numActive) + Core::XmlAttribute("segment", segment.name);
        stepwiseStatisticsChannel_ << Core::XmlFull("num-active-hyps-after-beam-pruning", numActive) + Core::XmlAttribute("segment", segment.name);
        stepwiseStatisticsChannel_ << Core::XmlFull("num-active-word-end-hyps-after-beam-pruning", numActiveWordEnd) + Core::XmlAttribute("segment", segment.name);
    }

    beam.swap(newBeam);
    ++segment.currentSearchStep;

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
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-active-trees", seenHistories.size()) + Core::XmlAttribute("segment", segment.name);
    }

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
        stepwiseStatisticsChannel_ << Core::XmlFull("terminated-hyps", beam.size() - numActive) + Core::XmlAttribute("segment", segment.name);
        stepwiseStatisticsChannel_ << Core::XmlFull("active-hyps", numActive) + Core::XmlAttribute("segment", segment.name);
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

void BatchedTreeLabelsyncBeamSearch::finalizeAndFinishSegment(SegmentState& segment) {
    {
        Core::StopWatch::Scope finalizeTimer(finalizeTime_);
        finalizeHypotheses(segment);
    }
    finishSegmentSearch(segment);
}

void BatchedTreeLabelsyncBeamSearch::releaseSegmentState(SegmentStateBase& segmentBase) {
    auto& segment = static_cast<SegmentState&>(segmentBase);
    // The scoring contexts are not needed anymore
    for (auto& hyp : segment.beam) {
        hyp.scoringContexts.clear();
    }
    segment.withinWordExtensions.clear();
    segment.wordEndExtensions.clear();
    segment.newBeam.clear();
    segment.wordEndHypotheses.clear();
}

template<typename Element>
bool BatchedTreeLabelsyncBeamSearch::matchesHypothesisFilter(Element const& element, HypothesisFilter filter) const {
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
Element const* BatchedTreeLabelsyncBeamSearch::getBestHypothesis(
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

BatchedTreeLabelsyncBeamSearch::LabelHypothesis const* BatchedTreeLabelsyncBeamSearch::getWorstHypothesis(
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

BatchedTreeLabelsyncBeamSearch::LabelHypothesis const& BatchedTreeLabelsyncBeamSearch::getOutputHypothesis(std::vector<LabelHypothesis> const& hypotheses) const {
    auto const* result = getBestHypothesis(hypotheses, HypothesisFilter::Terminated);
    if (result != nullptr) {
        return *result;
    }
    result = getBestHypothesis(hypotheses, HypothesisFilter::Active);
    verify(result != nullptr);
    return *result;
}

void BatchedTreeLabelsyncBeamSearch::logSegmentStatistics(SegmentStateBase const& segmentBase) const {
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
    segment.numTerminatedHypsAfterScorePruning.write(channel);
    segment.numTerminatedHypsAfterRecombination.write(channel);
    segment.numTerminatedHypsAfterBeamPruning.write(channel);
    segment.numActiveHypsAfterScorePruning.write(channel);
    segment.numActiveHypsAfterRecombination.write(channel);
    segment.numActiveHypsAfterBeamPruning.write(channel);
    segment.numWordEndExtensionsBeforePruning.write(channel);
    segment.numActiveWordEndHypsAfterPruning.write(channel);
    segment.numActiveWordEndHypsAfterScorePruning.write(channel);
    segment.numActiveWordEndHypsAfterRecombination.write(channel);
    segment.numActiveWordEndHypsAfterBeamPruning.write(channel);
    segment.numActiveTrees.write(channel);
    channel << Core::XmlClose("search-statistics");
}

void BatchedTreeLabelsyncBeamSearch::logSearchStepBreakdown() const {
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
    channel << Core::XmlFull("word-end-hyp-building-time", threadSum([](ThreadBuffers const& buffers) -> Timer { return buffers.wordEndHypBuildingTime; }));
    channel << Core::XmlClose("word-end-expansion-time");
    channel << Core::XmlFull("recombination-time", threadSum([](ThreadBuffers const& buffers) -> Timer { return buffers.recombinationTime; }));
    channel << Core::XmlFull("beam-pruning-time", threadSum([](ThreadBuffers const& buffers) -> Timer { return buffers.beamPruningTime; }));
    channel << Core::XmlOpen("finalize-time") + Core::XmlAttribute("total", finalizeTime_.elapsedMilliseconds());
    channel << Core::XmlFull("scoring-time", finalizeScoringTime_.elapsedMilliseconds());
    channel << Core::XmlClose("finalize-time");
}

void BatchedTreeLabelsyncBeamSearch::resetSearchStepBreakdown() {
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
        for (auto* timer : {&buffers.buildWithinWordHypsTime, &buffers.wordEndExpansionTime, &buffers.wordEndExtensionTime, &buffers.wordEndScorePruningTime,
                            &buffers.wordEndHypBuildingTime, &buffers.lmTime, &buffers.recombinationTime, &buffers.beamPruningTime}) {
            timer->reset();
        }
    }
}

template<typename Element>
void BatchedTreeLabelsyncBeamSearch::scorePruning(std::vector<Element>& hypotheses, PruningParams const& pruningParams, Histogram& scoreHistogram) {
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

    // Remove elements with pruningScore() > absoluteThreshold
    hypotheses.erase(
            std::remove_if(
                    hypotheses.begin(),
                    hypotheses.end(),
                    [absoluteThreshold](auto const& hyp) { return hyp.pruningScore() > absoluteThreshold; }),
            hypotheses.end());
}

template<typename Element>
void BatchedTreeLabelsyncBeamSearch::separateScorePruning(
        std::vector<Element>& hypotheses,
        PruningParams const&  activePruningParams,
        PruningParams const&  terminatedPruningParams,
        ThreadBuffers&        buffers) {
    if (hypotheses.empty()) {
        return;
    }

    if (not histogramPruningEnabled_) {
        // Plain max beam size pruning: keep exactly the maxBeamSize best elements of each pool
        auto   byScore   = [](auto const& a, auto const& b) { return a.pruningScore() < b.pruningScore(); };
        size_t numActive = std::partition(hypotheses.begin(), hypotheses.end(), [](auto const& element) { return element.isActive; }) - hypotheses.begin();
        if (hypotheses.size() - numActive > terminatedPruningParams.maxBeamSize) {
            auto keepEnd = hypotheses.begin() + numActive + terminatedPruningParams.maxBeamSize;
            std::nth_element(hypotheses.begin() + numActive, keepEnd, hypotheses.end(), byScore);
            hypotheses.erase(keepEnd, hypotheses.end());
        }
        if (numActive > activePruningParams.maxBeamSize) {
            auto keepEnd = hypotheses.begin() + activePruningParams.maxBeamSize;
            std::nth_element(hypotheses.begin(), keepEnd, hypotheses.begin() + numActive, byScore);
            hypotheses.erase(keepEnd, hypotheses.begin() + numActive);
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

    for (auto const& hyp : hypotheses) {
        size_t const poolIdx = hyp.isActive ? activeIdx : terminatedIdx;
        ++numElements[poolIdx];

        if (Math::isinf(hyp.score) or hyp.score >= Core::Type<Score>::max) {
            continue;
        }

        ++numFiniteScoreElements[poolIdx];
        auto pruningScore    = hyp.pruningScore();
        lowerScores[poolIdx] = std::min(lowerScores[poolIdx], pruningScore);
        upperScores[poolIdx] = std::max(upperScores[poolIdx], pruningScore);
    }

    for (size_t poolIdx = 0ul; poolIdx < numPools; ++poolIdx) {
        auto const& params = pruningParams[poolIdx];
        bool const  active = poolIdx == activeIdx;

        if (numElements[poolIdx] == 0ul) {
            continue;
        }

        if (numFiniteScoreElements[poolIdx] == 0ul) {
            // No hypothesis with finite score is present in this pool
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
                // A score threshold cannot separate equal scores, so keep the first maxBeamSize hypotheses in this pool.
                allScoresEqual[poolIdx] = numFiniteScoreElements[poolIdx] > params->maxBeamSize;
                continue;
            }

            if (numFiniteScoreElements[poolIdx] > params->maxBeamSize) {
                histograms[poolIdx]->clear();
                histograms[poolIdx]->setLimits(lowerScores[poolIdx], upperScores[poolIdx]);

                for (auto const& hyp : hypotheses) {
                    if (hyp.isActive == active and
                        not Math::isinf(hyp.score) and
                        hyp.score < Core::Type<Score>::max) {
                        *histograms[poolIdx] += hyp.pruningScore();
                    }
                }

                absoluteThresholds[poolIdx] = std::min(
                        absoluteThresholds[poolIdx],
                        histograms[poolIdx]->quantile(params->maxBeamSize));
            }
        }
    }

    size_t numKept[numPools] = {0ul, 0ul};
    auto   writeIt           = hypotheses.begin();
    for (auto readIt = hypotheses.begin(); readIt != hypotheses.end(); ++readIt) {
        size_t const poolIdx = readIt->isActive ? activeIdx : terminatedIdx;

        bool eraseHyp;
        if (pruneAll[poolIdx]) {
            eraseHyp = true;
        }
        else if (noPruning[poolIdx]) {
            eraseHyp = false;
        }
        else if (readIt->pruningScore() > absoluteThresholds[poolIdx]) {
            eraseHyp = true;
        }
        else if (allScoresEqual[poolIdx]) {
            eraseHyp = numKept[poolIdx] >= pruningParams[poolIdx]->maxBeamSize;
        }
        else {
            eraseHyp = false;
        }

        if (not eraseHyp) {
            ++numKept[poolIdx];
            if (writeIt != readIt) {
                *writeIt = std::move(*readIt);
            }
            ++writeIt;
        }
    }

    hypotheses.erase(writeIt, hypotheses.end());
}

void BatchedTreeLabelsyncBeamSearch::recombination(std::vector<BatchedTreeLabelsyncBeamSearch::LabelHypothesis>& hypotheses, std::vector<LabelHypothesis>& tempHypotheses) {
    if (not recombinationEnabled_) {
        return;
    }

    // Represents a unique combination of StateId, ScoringContext, LmHistory and activity status
    struct RecombinationContext {
        StateId                            state;
        std::vector<Nn::ScoringContextRef> scoringContexts;
        Lm::History                        lmHistory;
        bool                               isActive;

        RecombinationContext(LabelHypothesis const& hyp)
                : state(hyp.currentState), scoringContexts(hyp.scoringContexts), lmHistory(hyp.lmHistory), isActive(hyp.isActive) {}

        bool operator==(const RecombinationContext& other) const {
            if (state != other.state) {
                return false;
            }
            if (isActive != other.isActive) {
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
            hash        = Core::combineHashes(hash, static_cast<size_t>(context.isActive));
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

        bool const isWordEnd = network_->isRoot(hyp.currentState);

        if (inserted) {
            // First time seeing this combination so move it over to `newHypotheses`
            tempHypotheses.push_back(std::move(hyp));
            it->second = &tempHypotheses.back();
        }
        else {
            if (isWordEnd) {
                verify(not hyp.trace->sibling);
            }

            auto* existingHyp = it->second;
            if (hyp.score < existingHyp->score) {
                // New hyp is better
                if (isWordEnd) {
                    hyp.trace->sibling = existingHyp->trace;
                }
                // Replace in `newHypotheses`
                *existingHyp = std::move(hyp);  // Overwrite in-place
            }
            else if (isWordEnd) {
                // New hyp is worse -> add to existing one as sibling
                hyp.trace->sibling          = existingHyp->trace->sibling;
                existingHyp->trace->sibling = hyp.trace;
            }
        }
    }

    hypotheses.swap(tempHypotheses);
}

size_t BatchedTreeLabelsyncBeamSearch::numActiveHyps(std::vector<LabelHypothesis> const& hypotheses) const {
    return std::accumulate(
            hypotheses.begin(),
            hypotheses.end(),
            0ul,
            [](size_t acc, auto const& hyp) { return acc + static_cast<size_t>(hyp.isActive); });
}

size_t BatchedTreeLabelsyncBeamSearch::numActiveWordEndHyps(std::vector<LabelHypothesis> const& hypotheses) const {
    return std::accumulate(
            hypotheses.begin(),
            hypotheses.end(),
            0ul,
            [this](size_t acc, auto const& hyp) { return acc + static_cast<size_t>(hyp.isActive and network_->isRoot(hyp.currentState)); });
}

void BatchedTreeLabelsyncBeamSearch::createSuccessorLookups() {
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

void BatchedTreeLabelsyncBeamSearch::finalizeHypotheses(SegmentState& segment) {
    auto& beam                 = segment.beam;
    auto& newBeam              = segment.newBeam;
    auto& withinWordExtensions = segment.withinWordExtensions;
    auto& wordEndExtensions    = segment.wordEndExtensions;
    auto& tempHypotheses       = threadBuffers_.front().tempHypotheses;  // Finalization runs in the main thread

    // Only keep terminated hypotheses (they already contain the sentence-end AM and LM score)
    newBeam.clear();
    for (auto const& hyp : beam) {
        if (not hyp.isActive) {
            newBeam.push_back(hyp);
        }
    }

    if (not newBeam.empty()) {
        beam.swap(newBeam);
        return;
    }
    warning() << "No terminated hypothesis at the end of segment " << segment.name;

    // Fall back to active hypotheses at a word end
    tempHypotheses.clear();
    for (auto const& hyp : beam) {
        if (network_->isRoot(hyp.currentState)) {
            tempHypotheses.push_back(hyp);
        }
    }

    if (tempHypotheses.empty()) {  // There was also no word-end hypothesis in the beam
        warning() << "No active word-end hypothesis at the end of segment " << segment.name;

        if (sentenceEndFallback_) {
            log() << "Use sentence-end fallback";
            // The trace of the unfinished word keeps an empty pronunciation
            tempHypotheses = beam;
        }
        else {  // No valid final hypotheses and no sentence-end fallback
            // Construct an empty hypothesis with a lattice containing only one empty pronunciation from start to end
            newBeam.push_back(LabelHypothesis());
            newBeam.front().trace->time          = beam.front().trace->time;  // Retrieve the timeframe from any hyp in the old beam
            newBeam.front().trace->pronunciation = nullptr;
            newBeam.front().trace->predecessor   = Core::ref(new LatticeTrace(0, {0, 0}, {}));
            beam.swap(newBeam);
            return;
        }
    }

    // Terminate the fallback hypotheses like a normal sentence-end prediction
    withinWordExtensions.clear();
    for (size_t hypIndex = 0ul; hypIndex < tempHypotheses.size(); ++hypIndex) {
        auto const& hyp = tempHypotheses[hypIndex];
        withinWordExtensions.push_back(
                {.nextToken      = sentenceEndLabelIndex_,
                 .nextState      = hyp.currentState,
                 .timeframe      = hyp.trace->time,
                 .score          = hyp.score,
                 .transitionType = Nn::TransitionType::SENTENCE_END,
                 .baseHypIndex   = hypIndex});
    }

    // Score sentence-end with all label scorers
    for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
        if (not labelScorers_[scorerIdx]->scoresTransition(Nn::TransitionType::SENTENCE_END)) {
            continue;
        }

        scoringContexts_.clear();
        for (auto const& hyp : tempHypotheses) {
            scoringContexts_.push_back(hyp.scoringContexts[scorerIdx]);
        }

        finalizeScoringTime_.start();
        auto scoreAccessors = getScoreAccessors(scorerIdx, scoringContexts_);
        finalizeScoringTime_.stop();
        std::vector<std::optional<Nn::DenseScoreSpan>> denseScoreSpans(scoreAccessors.size(), std::nullopt);
        std::vector<Nn::TimeframeIndex>                scoreTimes(scoreAccessors.size(), 0);
        for (size_t accessorIdx = 0ul; accessorIdx < scoreAccessors.size(); ++accessorIdx) {
            if (scoreAccessors[accessorIdx]) {
                denseScoreSpans[accessorIdx] = (*scoreAccessors[accessorIdx])->getDenseScores();
                scoreTimes[accessorIdx]      = (*scoreAccessors[accessorIdx])->getTime();
            }
        }

        for (size_t extensionIdx = 0ul; extensionIdx < withinWordExtensions.size(); ++extensionIdx) {
            if (not scoreAccessors[extensionIdx]) {
                continue;
            }
            auto& ext = withinWordExtensions[extensionIdx];
            if (denseScoreSpans[extensionIdx] and sentenceEndLabelIndex_ < denseScoreSpans[extensionIdx]->size()) {
                ext.score += (*denseScoreSpans[extensionIdx])[sentenceEndLabelIndex_];
            }
            else {
                ext.score += (*scoreAccessors[extensionIdx])->getScore(ext.transitionType, sentenceEndLabelIndex_);
            }
            ext.timeframe = std::max(ext.timeframe, scoreTimes[extensionIdx]);
        }
    }

    newBeam.clear();
    for (size_t extensionIdx = 0ul; extensionIdx < withinWordExtensions.size(); ++extensionIdx) {
        auto&       ext     = withinWordExtensions[extensionIdx];
        auto const& baseHyp = tempHypotheses[ext.baseHypIndex];
        // The scoring context is not updated as no further scoring is done afterwards
        newBeam.push_back({baseHyp, ext, baseHyp.scoringContexts, lengthNormScale_});
    }

    wordEndExtensions.clear();
    for (size_t hypIndex = 0ul; hypIndex < newBeam.size(); ++hypIndex) {
        auto const& hyp = newBeam[hypIndex];
        // Add the LM's sentence-end score
        // The LM history is not updated as this is the last LM scoring step
        Lm::Score sentenceEndScore = languageModel_->sentenceEndScore(hyp.lmHistory);
        wordEndExtensions.push_back({
                .pron           = sentenceEndLemma_->pronunciations().first,
                .rootState      = hyp.currentState,
                .score          = hyp.score + sentenceEndScore,
                .timeframe      = hyp.timeframe,
                .transitionType = Nn::TransitionType::SENTENCE_END,
                .baseHypIndex   = hypIndex,
        });
    }

    tempHypotheses.clear();
    for (size_t extensionIdx = 0ul; extensionIdx < wordEndExtensions.size(); ++extensionIdx) {
        auto&       ext     = wordEndExtensions[extensionIdx];
        auto const& baseHyp = newBeam[ext.baseHypIndex];
        tempHypotheses.push_back({baseHyp, ext, baseHyp.lmHistory, lengthNormScale_});
    }

    beam.swap(tempHypotheses);
}

}  // namespace Search
