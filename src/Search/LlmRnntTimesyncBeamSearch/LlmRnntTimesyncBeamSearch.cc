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

#include "LlmRnntTimesyncBeamSearch.hh"

#include <algorithm>
#include <cmath>

#include <Core/CollapsedVector.hh>
#include <Core/XmlStream.hh>
#include <Lattice/LatticeAdaptor.hh>
#include <Math/Utilities.hh>
#include <Search/TracebackHelper.hh>

namespace Search {

namespace {

enum RecombinationMode {
    RecombinationModeOff,
    RecombinationModeSum,
    RecombinationModeViterbi,
};

// Copy of `trace` and its sibling chain with `delta` added to every score
Core::Ref<LatticeTrace> traceWithAddedScore(Core::Ref<LatticeTrace> const& trace, ScoreVector delta) {
    auto result = Core::ref(new LatticeTrace(trace->predecessor, trace->pronunciation, trace->time, trace->score + delta, trace->transit));
    if (trace->sibling) {
        result->sibling = traceWithAddedScore(trace->sibling, delta);
    }
    return result;
}

}  // namespace

/*
 * =======================
 * === LabelHypothesis ===
 * =======================
 */

LlmRnntTimesyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContexts(),
          currentToken(Nn::invalidLabelIndex),
          length(1),
          score(0.0),
          scaledScore(0.0),
          outputTokens(),
          outputTokensHash(0ul),
          words({WordAssembler::noWord, 0u, 0.0}),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))),
          reachedSentenceEnd(false) {}

LlmRnntTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        LlmRnntTimesyncBeamSearch::LabelHypothesis const&    base,
        LlmRnntTimesyncBeamSearch::ExtensionCandidate const& extension,
        WordState const&                                     newWords,
        std::vector<Nn::ScoringContextRef> const&            newScoringContexts,
        float                                                lengthNormScale)
        : scoringContexts(newScoringContexts),
          currentToken(extension.nextToken),
          length(base.length),
          score(extension.score),
          scaledScore(),
          outputTokens(base.outputTokens),
          outputTokensHash(base.outputTokensHash),
          words(newWords),
          trace(),
          reachedSentenceEnd(base.reachedSentenceEnd or extension.transitionType == Nn::SENTENCE_END) {
    switch (extension.transitionType) {
        case Nn::INITIAL_LABEL:
        case Nn::LABEL_TO_LABEL:
        case Nn::BLANK_TO_LABEL:
            length += 1;
            outputTokens.push_back(currentToken);
            outputTokensHash = Core::combineHashes(outputTokensHash, std::hash<uint32_t>()(outputTokens.back()));
            break;
        default:
            break;
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
    trace = Core::ref(new LatticeTrace(predecessor, extension.pron, extension.timeframe, {score - words.llmScore, words.llmScore}, {}));
}

std::string LlmRnntTimesyncBeamSearch::LabelHypothesis::toString(WordAssembler const& wordAssembler) const {
    std::stringstream ss;
    ss << "Score: " << score << ", LLM score: " << words.llmScore << ", LLM history: " << words.llmHistory
       << ", pending word: \"" << wordAssembler.spelling(words.pendingWord) << "\", traceback: ";
    for (auto& item : *trace->performTraceback()) {
        if (item.pronunciation and item.pronunciation->lemma()) {
            ss << item.pronunciation->lemma()->symbol() << " ";
        }
    }
    return ss.str();
}

/*
 * =================================
 * === LlmRnntTimesyncBeamSearch ===
 * =================================
 */

const Core::ParameterIntVector LlmRnntTimesyncBeamSearch::paramMaxBeamSizes(
        "max-beam-size",
        "Maximum number of elements in the search beam. Pruning is applied after each intermediate label scorer.",
        "",
        1);

const Core::ParameterFloatVector LlmRnntTimesyncBeamSearch::paramScoreThresholds(
        "score-threshold",
        "Prune any hypotheses with a score that is at least this much worse than the best hypothesis. Pruning is applied after each intermediate label scorer.",
        "",
        0,
        Core::Type<Score>::max);

const Core::ParameterInt LlmRnntTimesyncBeamSearch::paramPreLlmMaxBeamSize(
        "pre-llm-max-beam-size",
        "Maximum number of label extensions per symbol step whose finished words are scored by the LLM. 0 means the max-beam-size of the last label scorer.",
        0,
        0);

const Core::ParameterFloat LlmRnntTimesyncBeamSearch::paramPreLlmScoreThreshold(
        "pre-llm-score-threshold",
        "Prune label extensions worse than the best one by more than this before their finished words are scored by the LLM.",
        Core::Type<Score>::max,
        0);

const Core::ParameterInt LlmRnntTimesyncBeamSearch::paramNumHistogramBins(
        "num-histogram-bins",
        "Number of bins for histogram pruning of hypotheses (very minor effect).",
        100,
        2);

const Core::ParameterFloat LlmRnntTimesyncBeamSearch::paramLengthNormScale(
        "length-norm-scale",
        "Exponent of length for the hypothesis length normalization. Scaled scores are computed as score / length^length_norm_scale.",
        0.0);

const Core::ParameterInt LlmRnntTimesyncBeamSearch::paramMaxLabelsPerFrame(
        "max-labels-per-timeframe",
        "Maximum number of non-blank label predictions per hypothesis in one timestep.",
        10, 0);

const Core::ParameterInt LlmRnntTimesyncBeamSearch::paramBlankLabelIndex(
        "blank-label-index",
        "Index of the blank label in the lexicon. Can also be inferred from lexicon if it has a lemma with `special='blank'`.",
        Nn::invalidLabelIndex);

const Core::ParameterInt LlmRnntTimesyncBeamSearch::paramSentenceEndLabelIndex(
        "sentence-end-label-index",
        "Index of the sentence end label in the lexicon. Can also be inferred from lexicon if it has a lemma with `special='sentence-end'` or `special='sentence-boundary'`. If not set, the search will not use sentence end.",
        Nn::invalidLabelIndex);

const Core::ParameterBool LlmRnntTimesyncBeamSearch::paramSentenceEndFallBack(
        "sentence-end-fall-back",
        "Allow for fallback solution if no active word-end hypothesis exists at the end of a segment.",
        true);

const Core::ParameterBool LlmRnntTimesyncBeamSearch::paramCollapseRepeatedLabels(
        "collapse-repeated-labels",
        "Collapse repeated emission of the same label into one output. If false, every emission is treated like a new output.",
        false);

const Core::ParameterFloat LlmRnntTimesyncBeamSearch::paramLlmScale(
        "llm-scale",
        "Scale of the LLM costs.",
        1.0);

const Core::ParameterFloat LlmRnntTimesyncBeamSearch::paramWordPenalty(
        "word-penalty",
        "Score added for every finished word. Negative values reward words.",
        0.0);

const Core::ParameterInt LlmRnntTimesyncBeamSearch::paramCacheCleanupInterval(
        "cache-cleanup-interval",
        "Interval of search steps after which buffered inputs and LLM states that are not needed anymore get cleaned up.",
        10,
        1);

const Core::ParameterInt LlmRnntTimesyncBeamSearch::paramMaximumStableDelay(
        "maximum-stable-delay",
        "Introduce a cutoff point at `current-time` - `delay`. Every hypothesis that disagrees with the current best anywhere before the cutoff gets pruned."
        "This way words in the traceback become stable after at most `delay` frames.",
        Core::Type<int>::max,
        0);

const Core::ParameterInt LlmRnntTimesyncBeamSearch::paramMaximumStableDelayPruningInterval(
        "maximum-stable-delay-pruning-interval",
        "Interval of search steps after which the maximum-stable-delay-pruning gets applied.",
        10,
        1);

const Core::Choice LlmRnntTimesyncBeamSearch::choiceRecombinationMode(
        "off", RecombinationModeOff,
        "sum", RecombinationModeSum,
        "viterbi", RecombinationModeViterbi,
        Core::Choice::endMark());

const Core::ParameterChoice LlmRnntTimesyncBeamSearch::paramRecombinationMode(
        "recombination-mode",
        &choiceRecombinationMode,
        "Whether hypotheses with identical recombination state should be recombined and how: by summing "
        "their probabilities via log-sum-exp (sum) or by keeping only the better-scoring one (Viterbi).",
        RecombinationModeSum);

LlmRnntTimesyncBeamSearch::LlmRnntTimesyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          preLlmMaxBeamSize_(paramPreLlmMaxBeamSize(config)),
          preLlmScoreThreshold_(paramPreLlmScoreThreshold(config)),
          scoreHistogram_(paramNumHistogramBins(config)),
          lengthNormScale_(paramLengthNormScale(config)),
          maxLabelsPerFrame_(paramMaxLabelsPerFrame(config)),
          blankLabelIndex_(paramBlankLabelIndex(config)),
          blankPron_(nullptr),
          sentenceEndLemma_(),
          sentenceEndLabelIndex_(paramSentenceEndLabelIndex(config)),
          sentenceEndFallback_(paramSentenceEndFallBack(config)),
          collapseRepeatedLabels_(paramCollapseRepeatedLabels(config)),
          llmScale_(paramLlmScale(config)),
          wordPenalty_(paramWordPenalty(config)),
          cacheCleanupInterval_(paramCacheCleanupInterval(config)),
          maximumStableDelay_(paramMaximumStableDelay(config)),
          maximumStableDelayPruningInterval_(paramMaximumStableDelayPruningInterval(config)),
          recombinationMode_(paramRecombinationMode(config)),
          statisticsChannel_(config, "statistics"),
          stepwiseStatisticsChannel_(config, "stepwise-statistics"),
          debugChannel_(config, "debug"),
          wordAssembler_(config),
          llmWordScorer_(select("llm")),
          numSymbolSteps_("num-symbol-steps"),
          numInnerHyps_("num-inner-hyps"),
          numOuterHyps_("num-outer-hyps"),
          numHypsBeforeLlm_("num-hyps-before-llm"),
          numFinishedWords_("num-finished-words"),
          numActiveHyps_("num-active-hyps"),
          currentSearchStep_(0ul),
          finishedSegment_(false) {
    auto maxBeamSizes = paramMaxBeamSizes(config);
    maxBeamSizes_.insert(maxBeamSizes_.begin(), maxBeamSizes.begin(), maxBeamSizes.end());

    auto scoreThresholds = paramScoreThresholds(config);
    scoreThresholds_.insert(scoreThresholds_.begin(), scoreThresholds.begin(), scoreThresholds.end());
    scoreThresholds_.resize(std::max(scoreThresholds_.size(), maxBeamSizes_.size()), Core::Type<Score>::max);
    for (Score threshold : scoreThresholds_) {
        useScorePruning_.push_back(threshold != Core::Type<Score>::max);
    }
    useSentenceEnd_ = sentenceEndLabelIndex_ != Nn::invalidLabelIndex;
}

Speech::ModelCombination::Mode LlmRnntTimesyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon;
}

bool LlmRnntTimesyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_      = modelCombination.lexicon();
    labelScorers_ = modelCombination.labelScorers();

    if (labelScorers_.size() != maxBeamSizes_.size()) {
        error() << "Number of label scorers (" << labelScorers_.size() << ") differs from the number of configured max beam sizes (" << maxBeamSizes_.size() << ")";
    }
    if (preLlmMaxBeamSize_ == 0ul) {
        preLlmMaxBeamSize_ = maxBeamSizes_[labelScorers_.size() - 1];
    }

    resolveSpecialLabels();
    for (auto lemmas = lexicon_->lemmas(); lemmas.first != lemmas.second; ++lemmas.first) {
        if (static_cast<Nn::LabelIndex>((*lemmas.first)->id()) == blankLabelIndex_ and (*lemmas.first)->nPronunciations() > 0) {
            blankPron_ = (*lemmas.first)->pronunciations().first;
        }
    }
    wordAssembler_.setLexicon(lexicon_);
    return true;
}

void LlmRnntTimesyncBeamSearch::resolveSpecialLabels() {
    auto blankLemma = lexicon_->specialLemma("blank");
    if (blankLemma) {
        if (blankLabelIndex_ == Nn::invalidLabelIndex) {
            blankLabelIndex_ = blankLemma->id();
            log() << "Use blank index " << blankLabelIndex_ << " inferred from lexicon";
        }
        else if (blankLabelIndex_ != static_cast<Nn::LabelIndex>(blankLemma->id())) {
            warning() << "Blank lemma exists in lexicon with id " << blankLemma->id() << " but is overwritten by config parameter with value " << blankLabelIndex_;
        }
    }
    if (blankLabelIndex_ == Nn::invalidLabelIndex) {
        error() << "Blank label index is not defined and cannot be inferred from the lexicon";
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
}

void LlmRnntTimesyncBeamSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    resetStatistics();
    initializationTime_.start();

    for (auto& labelScorer : labelScorers_) {
        labelScorer->reset();
    }
    llmWordScorer_.reset();

    beam_.assign(1ul, LabelHypothesis());
    for (auto& labelScorer : labelScorers_) {
        beam_.front().scoringContexts.push_back(labelScorer->getInitialScoringContext());
    }
    beam_.front().words.llmHistory = llmWordScorer_.initialHistory();

    currentSearchStep_ = 0ul;
    finishedSegment_   = false;

    initializationTime_.stop();

    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlOpen("search-steps");
    }
}

void LlmRnntTimesyncBeamSearch::finishSegment() {
    featureProcessingTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->signalNoMoreFeatures();
    }
    featureProcessingTime_.stop();
    decodeManySteps();
    finalizeHypotheses();
    finishedSegment_ = true;
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlClose("search-steps");
    }
    logStatistics();
}

void LlmRnntTimesyncBeamSearch::putFeature(Nn::DataView const& feature) {
    Core::StopWatch::Scope timer(featureProcessingTime_);
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInput(feature);
    }
}

void LlmRnntTimesyncBeamSearch::putFeatures(Nn::DataView const& features, size_t nTimesteps) {
    Core::StopWatch::Scope timer(featureProcessingTime_);
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInputs(features, nTimesteps);
    }
}

Core::Ref<const Traceback> LlmRnntTimesyncBeamSearch::getCurrentBestTraceback() const {
    return getBestHypothesis().trace->performTraceback();
}

Core::Ref<const LatticeAdaptor> LlmRnntTimesyncBeamSearch::getCurrentBestWordLattice() const {
    auto&        bestHypothesis = getBestHypothesis();
    LatticeTrace endTrace(bestHypothesis.trace, 0, bestHypothesis.trace->time + 1, bestHypothesis.trace->score, {});

    for (auto const& hyp : beam_) {
        // The best hypothesis is already represented in endTrace
        if (&hyp == &bestHypothesis) {
            continue;
        }
        auto siblingTrace = Core::ref(new LatticeTrace(hyp.trace, 0, hyp.trace->time, hyp.trace->score, {}));
        endTrace.appendSiblingToChain(siblingTrace);
    }

    return endTrace.buildWordLattice(lexicon_);
}

Core::Ref<const LatticeTrace> LlmRnntTimesyncBeamSearch::getCurrentBestLatticeTrace() const {
    return getBestHypothesis().trace;
}

Core::Ref<const LatticeTrace> LlmRnntTimesyncBeamSearch::getCommonPrefix() const {
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

bool LlmRnntTimesyncBeamSearch::decodeStep() {
    if (finishedSegment_) {
        return false;
    }

    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlOpen("search-step-stats") + Core::XmlAttribute("step", currentSearchStep_);
    }

    recognitionTime_.start();
    bool advanced = advanceTimestep();
    recognitionTime_.stop();

    if (advanced) {
        logBeamStatistics();
    }
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlClose("search-step-stats");
    }
    return advanced;
}

bool LlmRnntTimesyncBeamSearch::advanceTimestep() {
    innerHyps_ = beam_;
    outerHyps_.clear();

    size_t symbolStep = 0ul;
    for (; not innerHyps_.empty(); ++symbolStep) {
        if (stepwiseStatisticsChannel_.isOpen()) {
            stepwiseStatisticsChannel_ << Core::XmlOpen("symbol-step-stats") + Core::XmlAttribute("symbol-step", symbolStep);
        }
        bool advanced = advanceSymbolStep(symbolStep);
        if (stepwiseStatisticsChannel_.isOpen()) {
            stepwiseStatisticsChannel_ << Core::XmlClose("symbol-step-stats");
        }
        if (not advanced) {
            // The label scorers need more input; the timestep is repeated in the next call
            return false;
        }
    }
    numSymbolSteps_ += symbolStep;

    pruneOuterHyps();
    beam_.swap(outerHyps_);
    ++currentSearchStep_;

    if (currentSearchStep_ % cacheCleanupInterval_ == 0) {
        cleanupCaches();
    }
    if (currentSearchStep_ % maximumStableDelayPruningInterval_ == 0) {
        Core::StopWatch::Scope timer(beamPruningTime_);
        maximumStableDelayPruning();
    }
    numActiveHyps_ += beam_.size();
    return true;
}

bool LlmRnntTimesyncBeamSearch::advanceSymbolStep(size_t symbolStep) {
    if (not scoreInnerHyps()) {
        return false;
    }

    extendWithBlank();
    {
        Core::StopWatch::Scope timer(recombinationTime_);
        recombination(outerHyps_);
    }
    numOuterHyps_ += outerHyps_.size();

    if (symbolStep < maxLabelsPerFrame_) {
        extendWithLabels();
        if (pruneBeforeLlm()) {
            applyLlm();
        }
        buildInnerHyps();
    }
    else {
        innerHyps_.clear();
    }
    numInnerHyps_ += innerHyps_.size();

    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-outer-hyps", outerHyps_.size());
        stepwiseStatisticsChannel_ << Core::XmlFull("num-inner-hyps", innerHyps_.size());
    }
    return true;
}

bool LlmRnntTimesyncBeamSearch::scoreInnerHyps() {
    Core::StopWatch::Scope timer(scoringTime_);
    scoringContexts_.clear();
    for (auto const& hyp : innerHyps_) {
        scoringContexts_.push_back(hyp.scoringContexts.front());
    }
    scoreAccessors_ = labelScorers_.front()->getScoreAccessors(scoringContexts_);
    return std::any_of(scoreAccessors_.begin(), scoreAccessors_.end(), [](auto const& accessor) { return accessor.has_value(); });
}

void LlmRnntTimesyncBeamSearch::extendWithBlank() {
    Core::StopWatch::Scope timer(blankExtensionTime_);
    extensions_.clear();
    for (size_t hypIndex = 0ul; hypIndex < innerHyps_.size(); ++hypIndex) {
        auto const& accessor = scoreAccessors_[hypIndex];
        if (not accessor) {
            continue;
        }
        auto  transitionType = inferTransitionType(innerHyps_[hypIndex].currentToken, blankLabelIndex_);
        Score score          = innerHyps_[hypIndex].score;
        if (labelScorers_.front()->scoresTransition(transitionType)) {
            score += (*accessor)->getScore(transitionType, blankLabelIndex_);
        }
        extensions_.push_back({.nextToken      = blankLabelIndex_,
                               .pron           = blankPron_,
                               .score          = score,
                               .timeframe      = (*accessor)->getTime(),
                               .transitionType = transitionType,
                               .baseHypIndex   = hypIndex});
    }

    // With one blank extension per hypothesis there is no need to prune after the first label scorer
    scoreWithRemainingLabelScorers();

    for (auto const& ext : extensions_) {
        auto const& baseHyp = innerHyps_[ext.baseHypIndex];
        outerHyps_.push_back({baseHyp, ext, baseHyp.words, extendedScoringContexts(baseHyp, ext), lengthNormScale_});
    }
}

void LlmRnntTimesyncBeamSearch::extendWithLabels() {
    Core::StopWatch::Scope timer(labelExtensionTime_);
    extensions_.clear();
    for (size_t hypIndex = 0ul; hypIndex < innerHyps_.size(); ++hypIndex) {
        auto const& accessor = scoreAccessors_[hypIndex];
        if (not accessor) {
            continue;
        }
        auto const& hyp         = innerHyps_[hypIndex];
        auto        denseScores = (*accessor)->getDenseScores();
        auto        scoreTime   = (*accessor)->getTime();
        for (auto lemmas = lexicon_->lemmas(); lemmas.first != lemmas.second; ++lemmas.first) {
            Bliss::Lemma const* lemma = *lemmas.first;
            Nn::LabelIndex      token = lemma->id();
            if (token == blankLabelIndex_) {
                continue;
            }
            auto  transitionType = inferTransitionType(hyp.currentToken, token);
            Score score          = hyp.score;
            if (labelScorers_.front()->scoresTransition(transitionType)) {
                score += (denseScores and token < denseScores->size()) ? (*denseScores)[token] : (*accessor)->getScore(transitionType, token);
            }
            extensions_.push_back({.nextToken      = token,
                                   .pron           = lemma->pronunciations().first,
                                   .score          = score,
                                   .timeframe      = scoreTime,
                                   .transitionType = transitionType,
                                   .baseHypIndex   = hypIndex});
        }
    }

    auto rawScore = [](ExtensionCandidate const& ext) { return ext.score; };
    scorePruning(extensions_, scoreThresholds_.front(), labelScorers_.size() > 1ul ? maxBeamSizes_.front() : extensions_.size(), rawScore);
    scoreWithRemainingLabelScorers();
    // Within a timestep raw scores are compared, as in torchaudio's RNNTBeamSearch
    scorePruning(extensions_, Core::Type<Score>::max, maxBeamSizes_.back(), rawScore);
}

void LlmRnntTimesyncBeamSearch::scoreWithRemainingLabelScorers() {
    for (size_t scorerIdx = 1ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
        auto const& labelScorer = labelScorers_[scorerIdx];

        // Contexts of this scorer for the base hypotheses of the remaining extensions
        scoringContexts_.clear();
        hypIndexToContextIndexMap_.assign(innerHyps_.size(), -1);
        for (auto const& ext : extensions_) {
            if (hypIndexToContextIndexMap_[ext.baseHypIndex] == -1) {
                hypIndexToContextIndexMap_[ext.baseHypIndex] = scoringContexts_.size();
                scoringContexts_.push_back(innerHyps_[ext.baseHypIndex].scoringContexts[scorerIdx]);
            }
        }
        auto accessors = labelScorer->getScoreAccessors(scoringContexts_);

        for (auto& ext : extensions_) {
            if (not labelScorer->scoresTransition(ext.transitionType)) {
                continue;
            }
            auto const& accessor = accessors[hypIndexToContextIndexMap_[ext.baseHypIndex]];
            if (not accessor) {
                // Not scorable, so it gets pruned
                ext.score = Core::Type<Score>::max;
                continue;
            }
            auto denseScores = (*accessor)->getDenseScores();
            ext.score += (denseScores and ext.nextToken < denseScores->size()) ? (*denseScores)[ext.nextToken] : (*accessor)->getScore(ext.transitionType, ext.nextToken);
            ext.timeframe = std::max(ext.timeframe, (*accessor)->getTime());
        }

        bool isLastScorer = scorerIdx + 1ul == labelScorers_.size();
        scorePruning(extensions_, scoreThresholds_[scorerIdx], isLastScorer ? extensions_.size() : maxBeamSizes_[scorerIdx], [](ExtensionCandidate const& ext) { return ext.score; });
    }
}

bool LlmRnntTimesyncBeamSearch::pruneBeforeLlm() {
    Core::StopWatch::Scope timer(preLlmPruningTime_);
    scorePruning(extensions_, preLlmScoreThreshold_, preLlmMaxBeamSize_, [](ExtensionCandidate const& ext) { return ext.score; });
    numHypsBeforeLlm_ += extensions_.size();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-hyps-before-llm", extensions_.size());
    }
    return not extensions_.empty();
}

void LlmRnntTimesyncBeamSearch::applyLlm() {
    Core::StopWatch::Scope timer(llmTime_);
    {
        Core::StopWatch::Scope assemblyTimer(wordAssemblyTime_);
        assembleWords();
    }
    {
        Core::StopWatch::Scope scoringTimer(wordScoringTime_);
        scoreFinishedWords();
    }
    numFinishedWords_ += wordRequests_.size();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-finished-words", wordRequests_.size());
    }
}

void LlmRnntTimesyncBeamSearch::assembleWords() {
    extensionWords_.clear();
    wordRequests_.clear();
    wordRequestExtensions_.clear();
    for (size_t extIdx = 0ul; extIdx < extensions_.size(); ++extIdx) {
        auto const& ext   = extensions_[extIdx];
        WordState   words = innerHyps_[ext.baseHypIndex].words;
        if (emitsPiece(ext.transitionType)) {
            auto step         = wordAssembler_.extend(words.pendingWord, ext.nextToken);
            words.pendingWord = step.pending;
            if (step.finished != WordAssembler::noWord) {
                wordRequests_.push_back({.history = words.llmHistory, .word = &wordAssembler_.spelling(step.finished), .sentenceEnd = false});
                wordRequestExtensions_.push_back(extIdx);
            }
        }
        extensionWords_.push_back(words);
    }
}

void LlmRnntTimesyncBeamSearch::scoreFinishedWords() {
    if (wordRequests_.empty()) {
        return;
    }
    llmWordScorer_.score(wordRequests_, wordResults_);
    for (size_t i = 0ul; i < wordRequests_.size(); ++i) {
        size_t extIdx                      = wordRequestExtensions_[i];
        Score  delta                       = llmScale_ * wordResults_[i].cost + wordPenalty_;
        extensionWords_[extIdx].llmHistory = wordResults_[i].history;
        extensionWords_[extIdx].llmScore += delta;
        extensions_[extIdx].score += delta;
    }
}

void LlmRnntTimesyncBeamSearch::buildInnerHyps() {
    Core::StopWatch::Scope timer(buildInnerHypsTime_);
    newHyps_.clear();
    for (size_t extIdx = 0ul; extIdx < extensions_.size(); ++extIdx) {
        auto const& ext     = extensions_[extIdx];
        auto const& baseHyp = innerHyps_[ext.baseHypIndex];
        newHyps_.push_back({baseHyp, ext, extensionWords_[extIdx], extendedScoringContexts(baseHyp, ext), lengthNormScale_});
    }

    // Inner hypotheses which are already worse than the max-beam-size-th best outer one cannot survive the timestep
    size_t maxBeamSize = maxBeamSizes_.back();
    if (outerHyps_.size() < maxBeamSize) {
        innerHyps_.swap(newHyps_);
        return;
    }
    auto kth = outerHyps_.begin() + (maxBeamSize - 1);
    std::nth_element(outerHyps_.begin(), kth, outerHyps_.end(), [](auto const& a, auto const& b) { return a.score < b.score; });
    Score threshold = kth->score;
    innerHyps_.clear();
    for (auto& hyp : newHyps_) {
        if (hyp.score < threshold) {
            innerHyps_.push_back(std::move(hyp));
        }
    }
}

void LlmRnntTimesyncBeamSearch::pruneOuterHyps() {
    Core::StopWatch::Scope timer(beamPruningTime_);
    auto                   scaledScore = [](LabelHypothesis const& hyp) { return hyp.scaledScore; };
    if (useScorePruning_.back() and not outerHyps_.empty()) {
        // The score threshold is converted to a gap of length-normalized scores at the length of the best hypothesis
        Score threshold = scoreThresholds_.back();
        if (lengthNormScale_ != 0.0f) {
            threshold /= std::pow(std::min_element(outerHyps_.begin(), outerHyps_.end())->length, lengthNormScale_);
        }
        scorePruning(outerHyps_, threshold, outerHyps_.size(), scaledScore);
    }
    scorePruning(outerHyps_, Core::Type<Score>::max, maxBeamSizes_.back(), scaledScore);
}

std::vector<Nn::ScoringContextRef> LlmRnntTimesyncBeamSearch::extendedScoringContexts(LabelHypothesis const& baseHyp, ExtensionCandidate const& extension) const {
    std::vector<Nn::ScoringContextRef> contexts;
    contexts.reserve(labelScorers_.size());
    for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
        contexts.push_back(labelScorers_[scorerIdx]->extendedScoringContext(baseHyp.scoringContexts[scorerIdx], extension.nextToken, extension.transitionType));
    }
    return contexts;
}

void LlmRnntTimesyncBeamSearch::cleanupCaches() {
    Core::StopWatch::Scope timer(cleanupTime_);
    for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
        Core::CollapsedVector<Nn::ScoringContextRef> activeContexts;
        for (auto const& hyp : beam_) {
            activeContexts.push_back(hyp.scoringContexts[scorerIdx]);
        }
        labelScorers_[scorerIdx]->cleanupCaches(activeContexts);
    }

    activeHistories_.clear();
    for (auto const& hyp : beam_) {
        activeHistories_.push_back(hyp.words.llmHistory);
    }
    std::sort(activeHistories_.begin(), activeHistories_.end());
    activeHistories_.erase(std::unique(activeHistories_.begin(), activeHistories_.end()), activeHistories_.end());
    llmWordScorer_.cleanup(activeHistories_);
}

bool LlmRnntTimesyncBeamSearch::emitsPiece(Nn::TransitionType transitionType) {
    switch (transitionType) {
        case Nn::TransitionType::INITIAL_LABEL:
        case Nn::TransitionType::LABEL_TO_LABEL:
        case Nn::TransitionType::BLANK_TO_LABEL:
            return true;
        default:
            return false;
    }
}

LlmRnntTimesyncBeamSearch::LabelHypothesis const& LlmRnntTimesyncBeamSearch::getBestHypothesis() const {
    verify(not beam_.empty());

    return *std::min_element(beam_.begin(), beam_.end());
}

LlmRnntTimesyncBeamSearch::LabelHypothesis const& LlmRnntTimesyncBeamSearch::getWorstHypothesis() const {
    verify(not beam_.empty());

    return *std::max_element(beam_.begin(), beam_.end());
}

void LlmRnntTimesyncBeamSearch::resetStatistics() {
    for (auto* timer : {&initializationTime_, &featureProcessingTime_, &recognitionTime_, &scoringTime_, &blankExtensionTime_, &labelExtensionTime_,
                        &preLlmPruningTime_, &llmTime_, &wordAssemblyTime_, &wordScoringTime_, &buildInnerHypsTime_, &recombinationTime_,
                        &beamPruningTime_, &cleanupTime_, &finalizeTime_, &finalizeLlmTime_}) {
        timer->reset();
    }
    for (auto* stat : {&numSymbolSteps_, &numInnerHyps_, &numOuterHyps_, &numHypsBeforeLlm_, &numFinishedWords_, &numActiveHyps_}) {
        stat->clear();
    }
}

void LlmRnntTimesyncBeamSearch::logStatistics() const {
    logOwnStatistics();
    for (auto const& labelScorer : labelScorers_) {
        labelScorer->logStatistics();
    }
    llmWordScorer_.logStatistics();
}

void LlmRnntTimesyncBeamSearch::logOwnStatistics() const {
    if (not statisticsChannel_.isOpen()) {
        return;
    }
    logTimingStatistics();
    logSearchStatistics();
}

void LlmRnntTimesyncBeamSearch::logTimingStatistics() const {
    auto& channel = statisticsChannel_;
    channel << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    channel << Core::XmlFull("initialization-time", initializationTime_.elapsedMilliseconds());
    channel << Core::XmlFull("feature-processing-time", featureProcessingTime_.elapsedMilliseconds());

    channel << Core::XmlOpen("recognition-time") + Core::XmlAttribute("total", recognitionTime_.elapsedMilliseconds());
    channel << Core::XmlFull("scoring-time", scoringTime_.elapsedMilliseconds());
    channel << Core::XmlFull("blank-extension-time", blankExtensionTime_.elapsedMilliseconds());
    channel << Core::XmlFull("label-extension-time", labelExtensionTime_.elapsedMilliseconds());
    channel << Core::XmlFull("pre-llm-pruning-time", preLlmPruningTime_.elapsedMilliseconds());
    channel << Core::XmlOpen("llm-time") + Core::XmlAttribute("total", llmTime_.elapsedMilliseconds());
    channel << Core::XmlFull("word-assembly-time", wordAssemblyTime_.elapsedMilliseconds());
    channel << Core::XmlFull("word-scoring-time", wordScoringTime_.elapsedMilliseconds());
    channel << Core::XmlClose("llm-time");
    channel << Core::XmlFull("build-inner-hyps-time", buildInnerHypsTime_.elapsedMilliseconds());
    channel << Core::XmlFull("recombination-time", recombinationTime_.elapsedMilliseconds());
    channel << Core::XmlFull("beam-pruning-time", beamPruningTime_.elapsedMilliseconds());
    channel << Core::XmlFull("cleanup-time", cleanupTime_.elapsedMilliseconds());
    channel << Core::XmlClose("recognition-time");

    channel << Core::XmlOpen("finalize-time") + Core::XmlAttribute("total", finalizeTime_.elapsedMilliseconds());
    channel << Core::XmlFull("llm-time", finalizeLlmTime_.elapsedMilliseconds());
    channel << Core::XmlClose("finalize-time");
    channel << Core::XmlClose("timing-statistics");
}

void LlmRnntTimesyncBeamSearch::logSearchStatistics() const {
    auto& channel = statisticsChannel_;
    channel << Core::XmlOpen("search-statistics");
    numSymbolSteps_.write(channel);
    numInnerHyps_.write(channel);
    numOuterHyps_.write(channel);
    numHypsBeforeLlm_.write(channel);
    numFinishedWords_.write(channel);
    numActiveHyps_.write(channel);
    channel << Core::XmlClose("search-statistics");
}

void LlmRnntTimesyncBeamSearch::logBeamStatistics() {
    if (debugChannel_.isOpen()) {
        std::stringstream ss;
        for (size_t hypIdx = 0ul; hypIdx < beam_.size(); ++hypIdx) {
            ss << "Hypothesis " << hypIdx + 1ul << ":  " << beam_[hypIdx].toString(wordAssembler_) << "\n";
        }
        ss << "\n";
        debugChannel_ << ss.str();
    }
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("active-hyps", beam_.size());
        stepwiseStatisticsChannel_ << Core::XmlFull("best-hyp-score", getBestHypothesis().score);
        stepwiseStatisticsChannel_ << Core::XmlFull("worst-hyp-score", getWorstHypothesis().score);
    }
}

Nn::TransitionType LlmRnntTimesyncBeamSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const {
    bool prevIsBlank       = prevLabel == blankLabelIndex_;
    bool nextIsBlank       = nextLabel == blankLabelIndex_;
    bool nextIsSentenceEnd = (useSentenceEnd_ and nextLabel == sentenceEndLabelIndex_);

    if (prevLabel == Nn::invalidLabelIndex) {
        if (nextIsBlank) {
            return Nn::TransitionType::INITIAL_BLANK;
        }
        else if (nextIsSentenceEnd) {
            return Nn::TransitionType::SENTENCE_END;
        }
        else {
            return Nn::TransitionType::INITIAL_LABEL;
        }
    }

    if (prevIsBlank) {
        if (nextIsBlank) {
            return Nn::TransitionType::BLANK_LOOP;
        }
        else if (nextIsSentenceEnd) {
            return Nn::TransitionType::SENTENCE_END;
        }
        else {
            return Nn::TransitionType::BLANK_TO_LABEL;
        }
    }
    else {
        if (nextIsBlank) {
            return Nn::TransitionType::LABEL_TO_BLANK;
        }
        else if (collapseRepeatedLabels_ and prevLabel == nextLabel) {
            return Nn::TransitionType::LABEL_LOOP;
        }
        else if (nextIsSentenceEnd) {
            return Nn::TransitionType::SENTENCE_END;
        }
        else {
            return Nn::TransitionType::LABEL_TO_LABEL;
        }
    }
}

template<typename Element, typename ScoreFn>
void LlmRnntTimesyncBeamSearch::scorePruning(std::vector<Element>& elements, Score relativeThreshold, size_t maxBeamSize, ScoreFn scoreFn) {
    // Remove elements that could not be scored by some label scorer
    elements.erase(
            std::remove_if(
                    elements.begin(),
                    elements.end(),
                    [](auto const& elem) { return Math::isinf(elem.score) or elem.score >= Core::Type<Score>::max; }),
            elements.end());

    if (elements.empty()) {
        return;
    }

    if (elements.size() <= maxBeamSize and relativeThreshold == Core::Type<Score>::max) {
        // Neither relative score pruning nor max beam size pruning triggers
        return;
    }

    Score lowerScore = Core::Type<Score>::max;
    Score upperScore = Core::Type<Score>::min;
    for (auto const& elem : elements) {
        lowerScore = std::min(lowerScore, scoreFn(elem));
        upperScore = std::max(upperScore, scoreFn(elem));
    }

    if (lowerScore == upperScore) {
        // All scores are the same (usually only happens when exactly 1 element is active)
        if (elements.size() > maxBeamSize) {
            elements.resize(maxBeamSize);
        }
        return;
    }

    Score absoluteThreshold = upperScore;

    // Prune by relative score threshold
    if (relativeThreshold != Core::Type<Score>::max) {
        absoluteThreshold = lowerScore + relativeThreshold;
    }

    // Prune by max beam size, approximated via a histogram instead of an exact nth_element
    if (elements.size() > maxBeamSize) {
        scoreHistogram_.clear();
        scoreHistogram_.setLimits(lowerScore, upperScore);
        for (auto const& elem : elements) {
            scoreHistogram_ += scoreFn(elem);
        }
        absoluteThreshold = std::min(absoluteThreshold, scoreHistogram_.quantile(maxBeamSize));
    }

    if (absoluteThreshold >= upperScore) {
        // Nothing will be pruned
        return;
    }

    // Remove elements with scoreFn(elem) > absoluteThreshold
    elements.erase(
            std::remove_if(
                    elements.begin(),
                    elements.end(),
                    [&scoreFn, absoluteThreshold](auto const& elem) { return scoreFn(elem) > absoluteThreshold; }),
            elements.end());
}

void LlmRnntTimesyncBeamSearch::recombination(std::vector<LlmRnntTimesyncBeamSearch::LabelHypothesis>& hypotheses) {
    if (recombinationMode_ == RecombinationModeOff) {
        return;
    }

    // Unique combination of currentToken, scoringContexts, LLM history, pending word and, in sum mode only, outputTokens (sum mode merges
    // by summing probabilities, which is only valid for hyps with the same output label sequence, while Viterbi
    // mode just keeps the better hyp, so it stays exact without that check and gets more merges by skipping it).
    // Points at a LabelHypothesis instead of copying it, to avoid copying/rehashing the growing outputTokens.
    struct RecombinationContext {
        LabelHypothesis const* hyp;
        bool                   ignoreOutputTokens;

        bool operator==(RecombinationContext const& other) const {
            if (hyp->currentToken != other.hyp->currentToken or hyp->words.llmHistory != other.hyp->words.llmHistory or
                hyp->words.pendingWord != other.hyp->words.pendingWord) {
                return false;
            }
            if (not ignoreOutputTokens and (hyp->outputTokensHash != other.hyp->outputTokensHash or hyp->outputTokens != other.hyp->outputTokens)) {
                return false;
            }
            if (hyp->scoringContexts.size() != other.hyp->scoringContexts.size()) {
                return false;
            }
            for (size_t i = 0ul; i < hyp->scoringContexts.size(); ++i) {
                if (not Nn::ScoringContextEq{}(hyp->scoringContexts[i], other.hyp->scoringContexts[i])) {
                    return false;
                }
            }
            return true;
        }
    };
    struct RecombinationContextHash {
        size_t operator()(RecombinationContext const& context) const {
            size_t h1 = Core::combineHashes(Core::combineHashes(context.hyp->currentToken, context.hyp->words.llmHistory), context.hyp->words.pendingWord);
            size_t h2 = 0;
            for (auto const& scoringContext : context.hyp->scoringContexts) {
                h2 = Core::combineHashes(h2, Nn::ScoringContextHash{}(scoringContext));
            }
            size_t h3 = context.ignoreOutputTokens ? 0ul : context.hyp->outputTokensHash;
            return Core::combineHashes(Core::combineHashes(h1, h2), h3);
        }
    };

    bool ignoreOutputTokens = recombinationMode_ == RecombinationModeViterbi;

    tempHypotheses_.clear();
    tempHypotheses_.reserve(hypotheses.size());  // Reserve capacity because future reallocations would break the raw pointers
    // Map each unique RecombinationContext in `hypotheses` to its representative hyp
    std::unordered_map<RecombinationContext, LabelHypothesis*, RecombinationContextHash> seenScoringContexts;

    for (auto& hyp : hypotheses) {
        // Probe with a key pointing at `hyp` in its original location. That pointer is only dereferenced
        // for this lookup and never stored, since `hyp` may be moved-from right after.
        auto it = seenScoringContexts.find(RecombinationContext{&hyp, ignoreOutputTokens});

        if (it == seenScoringContexts.end()) {
            // First time seeing this context -> keep this hyp as representative.
            // Move it into the stable tempHypotheses_ storage first and key the map entry off of that
            // address, so the stored pointer stays valid even though `hyp` itself is now moved-from
            tempHypotheses_.push_back(std::move(hyp));
            LabelHypothesis* stableHyp = &tempHypotheses_.back();
            seenScoringContexts.emplace(RecombinationContext{stableHyp, ignoreOutputTokens}, stableHyp);
        }
        else {
            verify(not hyp.trace->sibling);

            auto* existingHyp = it->second;

            // In sum mode, merge scores in probability space (log-sum-exp of the two path scores)
            // Numerically stable form: min(a, b) - log1p(exp(-|a - b|)), which is <= min(a, b)
            // In viterbi mode, no merging happens, the better hyp's own score is kept as-is
            bool  mergeScores = recombinationMode_ == RecombinationModeSum;
            Score mergedScore = mergeScores ? std::min(existingHyp->score, hyp.score) - std::log1p(std::exp(-std::fabs(existingHyp->score - hyp.score))) : 0.0f;

            if (hyp.score < existingHyp->score) {
                // New hyp is better -> keep it as representative and add existing one as sibling
                hyp.trace->sibling = existingHyp->trace;
                *existingHyp       = std::move(hyp);  // Overwrite in-place
            }
            else {
                // New hyp is worse -> add it as sibling to the existing representative
                hyp.trace->sibling          = existingHyp->trace->sibling;
                existingHyp->trace->sibling = hyp.trace;
            }

            if (mergeScores) {
                // Recompute scaled score from the merged score
                existingHyp->score       = mergedScore;
                const auto len           = std::max<std::size_t>(1, existingHyp->length);
                existingHyp->scaledScore = existingHyp->score / std::pow(static_cast<double>(len), static_cast<double>(lengthNormScale_));
            }
        }
    }

    hypotheses.swap(tempHypotheses_);
}

void LlmRnntTimesyncBeamSearch::maximumStableDelayPruning() {
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
        warning() << "Most recent label in best hypothesis is before cutoff point for maximum-stable-delay-pruning so the limit will be surpassed";
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

void LlmRnntTimesyncBeamSearch::finalizeHypotheses() {
    {
        Core::StopWatch::Scope timer(finalizeTime_);
        keepSentenceEndHypotheses();
        finishWordsAtSegmentEnd();
    }
    numActiveHyps_ += beam_.size();

    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlOpen("final-beam-stats");
    }
    logBeamStatistics();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlClose("final-beam-stats");
    }
}

void LlmRnntTimesyncBeamSearch::keepSentenceEndHypotheses() {
    if (not useSentenceEnd_) {
        return;
    }

    newHyps_.clear();
    for (auto const& hyp : beam_) {
        if (hyp.reachedSentenceEnd) {
            newHyps_.push_back(hyp);
        }
    }

    if (newHyps_.empty()) {  // There was no valid final hypothesis in the beam
        warning("No hypothesis has produced sentence-end by the end of the segment.");
        if (sentenceEndFallback_) {
            log() << "Use sentence-end fallback";
            // Keep `beam_` as it is
        }
        else {
            newHyps_.push_back(LabelHypothesis());
            newHyps_.front().trace->time          = beam_.front().trace->time;  // Retrieve the timeframe from any hyp in the old beam
            newHyps_.front().trace->pronunciation = nullptr;
            newHyps_.front().trace->predecessor   = Core::ref(new LatticeTrace(0, {0, 0}, {}));
            newHyps_.front().reachedSentenceEnd   = true;
            newHyps_.front().words.llmHistory     = llmWordScorer_.initialHistory();
            beam_.swap(newHyps_);
        }
    }
    else {
        newHyps_.swap(beam_);
    }
}

void LlmRnntTimesyncBeamSearch::finishWordsAtSegmentEnd() {
    Core::StopWatch::Scope timer(finalizeLlmTime_);
    wordRequests_.clear();
    for (auto const& hyp : beam_) {
        bool pending = hyp.words.pendingWord != WordAssembler::noWord;
        wordRequests_.push_back({.history = hyp.words.llmHistory, .word = pending ? &wordAssembler_.spelling(hyp.words.pendingWord) : nullptr, .sentenceEnd = true});
    }
    llmWordScorer_.score(wordRequests_, wordResults_);

    // There is no trace item after the last label, so the final scores go to the last one
    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        auto& hyp   = beam_[hypIndex];
        Score delta = llmScale_ * wordResults_[hypIndex].cost + (wordRequests_[hypIndex].word ? wordPenalty_ : 0.0);
        hyp.score += delta;
        hyp.scaledScore = hyp.score / std::pow(hyp.length, lengthNormScale_);
        hyp.words       = {WordAssembler::noWord, wordResults_[hypIndex].history, hyp.words.llmScore + delta};
        hyp.trace       = traceWithAddedScore(hyp.trace, {0.0, delta});
    }
}

}  // namespace Search
