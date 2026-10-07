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

#include "LlmTimesyncBeamSearch.hh"

#include <algorithm>
#include <numeric>

#include <Core/CollapsedVector.hh>
#include <Core/XmlStream.hh>
#include <Lattice/LatticeAdaptor.hh>
#include <Math/Utilities.hh>
#include <Search/TracebackHelper.hh>

namespace Search {

namespace {

enum RecombinationMode {
    RecombinationModeOff,
    RecombinationModeOn,
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

LlmTimesyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContexts(),
          currentToken(Nn::invalidLabelIndex),
          score(0.0),
          words({WordAssembler::noWord, 0u, 0.0}),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))) {}

LlmTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        LlmTimesyncBeamSearch::LabelHypothesis const&    base,
        LlmTimesyncBeamSearch::ExtensionCandidate const& extension,
        WordState const&                                 newWords,
        std::vector<Nn::ScoringContextRef> const&        newScoringContexts)
        : scoringContexts(newScoringContexts),
          currentToken(extension.nextToken),
          score(extension.score),
          words(newWords),
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
    trace          = Core::ref(new LatticeTrace(predecessor, extension.pron, timeframe, {score - words.llmScore, words.llmScore}, {}));
}

std::string LlmTimesyncBeamSearch::LabelHypothesis::toString(WordAssembler const& wordAssembler) const {
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
 * =============================
 * === LlmTimesyncBeamSearch ===
 * =============================
 */

const Core::ParameterIntVector LlmTimesyncBeamSearch::paramMaxBeamSizes(
        "max-beam-size",
        "Maximum number of elements in the search beam. Pruning is applied after each intermediate label scorer.",
        "",
        1);

const Core::ParameterFloatVector LlmTimesyncBeamSearch::paramScoreThresholds(
        "score-threshold",
        "Prune any hypotheses with a score that is at least this much worse than the best hypothesis. Pruning is applied after each intermediate label scorer.",
        "",
        0,
        Core::Type<Score>::max);

const Core::ParameterInt LlmTimesyncBeamSearch::paramPreLlmMaxBeamSize(
        "pre-llm-max-beam-size",
        "Maximum number of extensions whose finished words are scored by the LLM in each step. 0 means the max-beam-size of the last label scorer.",
        0,
        0);

const Core::ParameterFloat LlmTimesyncBeamSearch::paramPreLlmScoreThreshold(
        "pre-llm-score-threshold",
        "Prune extensions worse than the best one by more than this before their finished words are scored by the LLM.",
        Core::Type<Score>::max,
        0);

const Core::ParameterInt LlmTimesyncBeamSearch::paramNumHistogramBins(
        "num-histogram-bins",
        "Number of bins for histogram pruning of hypotheses (very minor effect).",
        100,
        2);

const Core::ParameterInt LlmTimesyncBeamSearch::paramBlankLabelIndex(
        "blank-label-index",
        "Index of the blank label in the lexicon. Can also be inferred from lexicon if it has a lemma with `special='blank'`. If not set, the search will not use blank.",
        Nn::invalidLabelIndex);

const Core::ParameterInt LlmTimesyncBeamSearch::paramSilenceLabelIndex(
        "silence-label-index",
        "Index of the silence label in the lexicon. Can also be inferred from lexicon if it has a lemma with `special='silence'`. If not set, the search will not use silence.",
        Nn::invalidLabelIndex);

const Core::ParameterInt LlmTimesyncBeamSearch::paramSentenceEndLabelIndex(
        "sentence-end-label-index",
        "Index of the sentence end label in the lexicon. Can also be inferred from lexicon if it has a lemma with `special='sentence-end'` or `special='sentence-boundary'`. If not set, the search will not use sentence end.",
        Nn::invalidLabelIndex);

const Core::ParameterBool LlmTimesyncBeamSearch::paramCollapseRepeatedLabels(
        "collapse-repeated-labels",
        "Collapse repeated emission of the same label into one output. If false, every emission is treated like a new output.",
        false);

const Core::ParameterFloat LlmTimesyncBeamSearch::paramLlmScale(
        "llm-scale",
        "Scale of the LLM costs.",
        1.0);

const Core::ParameterFloat LlmTimesyncBeamSearch::paramWordPenalty(
        "word-penalty",
        "Score added for every finished word. Negative values reward words.",
        0.0);

const Core::ParameterInt LlmTimesyncBeamSearch::paramCacheCleanupInterval(
        "cache-cleanup-interval",
        "Interval of search steps after which buffered inputs and LLM states that are not needed anymore get cleaned up.",
        10,
        1);

const Core::ParameterInt LlmTimesyncBeamSearch::paramMaximumStableDelay(
        "maximum-stable-delay",
        "Introduce a cutoff point at `current-time` - `delay`. Every hypothesis that disagrees with the current best anywhere before the cutoff gets pruned."
        "This way words in the traceback become stable after at most `delay` frames.",
        Core::Type<int>::max,
        0);

const Core::ParameterInt LlmTimesyncBeamSearch::paramMaximumStableDelayPruningInterval(
        "maximum-stable-delay-pruning-interval",
        "Interval of search steps after which the maximum-stable-delay-pruning gets applied.",
        10,
        1);

const Core::Choice LlmTimesyncBeamSearch::choiceRecombinationMode(
        "off", RecombinationModeOff,
        "on", RecombinationModeOn,
        Core::Choice::endMark());

const Core::ParameterChoice LlmTimesyncBeamSearch::paramRecombinationMode(
        "recombination-mode",
        &choiceRecombinationMode,
        "Whether hypotheses with identical recombination state should be recombined.",
        RecombinationModeOn);

LlmTimesyncBeamSearch::LlmTimesyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          preLlmMaxBeamSize_(paramPreLlmMaxBeamSize(config)),
          preLlmScoreThreshold_(paramPreLlmScoreThreshold(config)),
          scoreHistogram_(paramNumHistogramBins(config)),
          blankLabelIndex_(paramBlankLabelIndex(config)),
          silenceLabelIndex_(paramSilenceLabelIndex(config)),
          sentenceEndLemma_(),
          sentenceEndLabelIndex_(paramSentenceEndLabelIndex(config)),
          collapseRepeatedLabels_(paramCollapseRepeatedLabels(config)),
          llmScale_(paramLlmScale(config)),
          wordPenalty_(paramWordPenalty(config)),
          cacheCleanupInterval_(paramCacheCleanupInterval(config)),
          maximumStableDelay_(paramMaximumStableDelay(config)),
          maximumStableDelayPruningInterval_(paramMaximumStableDelayPruningInterval(config)),
          recombinationEnabled_(paramRecombinationMode(config) == RecombinationModeOn),
          statisticsChannel_(config, "statistics"),
          stepwiseStatisticsChannel_(config, "stepwise-statistics"),
          debugChannel_(config, "debug"),
          wordAssembler_(config),
          llmWordScorer_(select("llm")),
          numInputHyps_("num-input-hyps"),
          numExtensionsBeforeFirstPruning_("num-extensions-before-first-pruning"),
          numHypsBeforeLlm_("num-hyps-before-llm"),
          numFinishedWords_("num-finished-words"),
          numHypsAfterRecombination_("num-hyps-after-recombination"),
          numHypsAfterPruning_("num-hyps-after-pruning"),
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

    useBlank_       = blankLabelIndex_ != Nn::invalidLabelIndex;
    useSilence_     = silenceLabelIndex_ != Nn::invalidLabelIndex;
    useSentenceEnd_ = sentenceEndLabelIndex_ != Nn::invalidLabelIndex;
}

Speech::ModelCombination::Mode LlmTimesyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon;
}

bool LlmTimesyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_      = modelCombination.lexicon();
    labelScorers_ = modelCombination.labelScorers();

    if (labelScorers_.size() != maxBeamSizes_.size()) {
        error() << "Number of label scorers (" << labelScorers_.size() << ") differs from the number of configured max beam sizes (" << maxBeamSizes_.size() << ")";
    }
    if (preLlmMaxBeamSize_ == 0ul) {
        preLlmMaxBeamSize_ = maxBeamSizes_[labelScorers_.size() - 1];
    }

    // Per-scorer timers and statistics
    numHypsAfterIntermediatePruning_.assign(labelScorers_.size(), Core::Statistics<u32>("num-hyps-after-intermediate-pruning"));
    for (auto* timers : {&scoreAndPruneExtensionsTimes_, &scoringTimes_, &scoreReadoutTimes_, &intermediatePruningTimes_}) {
        timers->assign(labelScorers_.size(), Core::StopWatch());
    }

    resolveSpecialLabels();
    wordAssembler_.setLexicon(lexicon_);
    return true;
}

void LlmTimesyncBeamSearch::resolveSpecialLabels() {
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
}

void LlmTimesyncBeamSearch::enterSegment(Bliss::SpeechSegment const* segment) {
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

void LlmTimesyncBeamSearch::finishSegment() {
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

void LlmTimesyncBeamSearch::putFeature(Nn::DataView const& feature) {
    Core::StopWatch::Scope timer(featureProcessingTime_);
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInput(feature);
    }
}

void LlmTimesyncBeamSearch::putFeatures(Nn::DataView const& features, size_t nTimesteps) {
    Core::StopWatch::Scope timer(featureProcessingTime_);
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInputs(features, nTimesteps);
    }
}

Core::Ref<const Traceback> LlmTimesyncBeamSearch::getCurrentBestTraceback() const {
    return getBestHypothesis().trace->performTraceback();
}

Core::Ref<const LatticeAdaptor> LlmTimesyncBeamSearch::getCurrentBestWordLattice() const {
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

Core::Ref<const LatticeTrace> LlmTimesyncBeamSearch::getCurrentBestLatticeTrace() const {
    return getBestHypothesis().trace;
}

Core::Ref<const LatticeTrace> LlmTimesyncBeamSearch::getCommonPrefix() const {
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

bool LlmTimesyncBeamSearch::decodeStep() {
    if (finishedSegment_) {
        return false;
    }

    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlOpen("search-step-stats") + Core::XmlAttribute("step", currentSearchStep_);
    }

    recognitionTime_.start();
    bool advanced = advanceBeam();
    recognitionTime_.stop();

    if (advanced) {
        logBeamStatistics();
    }
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlClose("search-step-stats");
    }
    return advanced;
}

bool LlmTimesyncBeamSearch::advanceBeam() {
    if (not scoreAndPruneExtensions() or not pruneBeforeLlm()) {
        return false;
    }
    applyLlm();

    {
        Core::StopWatch::Scope timer(buildNewBeamTime_);
        buildNewBeamFromExtensions();
    }
    {
        Core::StopWatch::Scope timer(recombinationTime_);
        recombination(newBeam_);
    }
    numHypsAfterRecombination_ += newBeam_.size();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-hyps-after-recombination", newBeam_.size());
    }
    {
        // The LLM scores changed the ranking, so the pruning of the last label scorer is applied again
        Core::StopWatch::Scope timer(beamPruningTime_);
        scorePruning(newBeam_, scoreThresholds_[labelScorers_.size() - 1], maxBeamSizes_[labelScorers_.size() - 1]);
    }
    numHypsAfterPruning_ += newBeam_.size();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-hyps-after-pruning", newBeam_.size());
    }

    beam_.swap(newBeam_);
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

bool LlmTimesyncBeamSearch::scoreAndPruneExtensions() {
    // `scoringContexts_[hypIndexToContextIndexMap_[i]]` is the context of `beam_[i]` for the current label scorer
    scoringContexts_.clear();
    for (auto const& hyp : beam_) {
        scoringContexts_.push_back(hyp.scoringContexts.front());
    }
    hypIndexToContextIndexMap_.resize(beam_.size());
    std::iota(hypIndexToContextIndexMap_.begin(), hypIndexToContextIndexMap_.end(), 0);

    numInputHyps_ += beam_.size();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-input-hyps", beam_.size());
    }

    for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
        Core::StopWatch::Scope timer(scoreAndPruneExtensionsTimes_[scorerIdx]);
        bool                   isLastScorer = scorerIdx + 1ul == labelScorers_.size();

        std::vector<std::optional<Nn::ScoreAccessorRef>> scoreAccessors;
        {
            Core::StopWatch::Scope scoringTimer(scoringTimes_[scorerIdx]);
            scoreAccessors = labelScorers_[scorerIdx]->getScoreAccessors(scoringContexts_);
        }
        {
            // Lazily computing scorers do their work here
            Core::StopWatch::Scope readoutTimer(scoreReadoutTimes_[scorerIdx]);
            readOutScoreAccessors(scoreAccessors);
            if (scorerIdx == 0ul) {
                createExtensions(scoreAccessors);
            }
            else {
                updateExtensionScores(scorerIdx, scoreAccessors);
            }
        }

        Core::StopWatch::Scope pruningTimer(intermediatePruningTimes_[scorerIdx]);
        scorePruning(extensions_, scoreThresholds_[scorerIdx], isLastScorer ? extensions_.size() : maxBeamSizes_[scorerIdx]);
        numHypsAfterIntermediatePruning_[scorerIdx] += extensions_.size();
        if (stepwiseStatisticsChannel_.isOpen()) {
            stepwiseStatisticsChannel_ << Core::XmlFull("num-hyps-after-intermediate-pruning", extensions_.size()) + Core::XmlAttribute("scorer", scorerIdx + 1);
        }
        if (extensions_.empty()) {
            return false;
        }
        if (not isLastScorer) {
            prepareNextScoringContexts(scorerIdx);
        }
    }
    return true;
}

void LlmTimesyncBeamSearch::readOutScoreAccessors(std::vector<std::optional<Nn::ScoreAccessorRef>> const& scoreAccessors) {
    denseScoreSpans_.assign(scoreAccessors.size(), std::nullopt);
    scoreTimes_.assign(scoreAccessors.size(), 0);
    for (size_t i = 0ul; i < scoreAccessors.size(); ++i) {
        if (scoreAccessors[i]) {
            denseScoreSpans_[i] = (*scoreAccessors[i])->getDenseScores();
            scoreTimes_[i]      = (*scoreAccessors[i])->getTime();
        }
    }
}

Score LlmTimesyncBeamSearch::labelScore(Nn::ScoreAccessorRef const& accessor, size_t contextIndex, Nn::TransitionType transitionType, Nn::LabelIndex token) const {
    auto const& dense = denseScoreSpans_[contextIndex];
    return (dense and token < dense->size()) ? (*dense)[token] : accessor->getScore(transitionType, token);
}

void LlmTimesyncBeamSearch::createExtensions(std::vector<std::optional<Nn::ScoreAccessorRef>> const& scoreAccessors) {
    extensions_.clear();
    Score bestScore = Core::Type<Score>::max;
    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        auto const& hyp          = beam_[hypIndex];
        size_t      contextIndex = hypIndexToContextIndexMap_[hypIndex];
        if (not scoreAccessors[contextIndex]) {
            continue;
        }
        for (auto lemmas = lexicon_->lemmas(); lemmas.first != lemmas.second; ++lemmas.first) {
            Bliss::Lemma const* lemma = *lemmas.first;
            Nn::LabelIndex      token = lemma->id();
            if (token == sentenceEndLabelIndex_) {
                continue;
            }
            auto  transitionType = inferTransitionType(hyp.currentToken, token);
            Score score          = hyp.score;
            auto  timeframe      = hyp.trace->time;
            if (labelScorers_.front()->scoresTransition(transitionType)) {
                score += labelScore(*scoreAccessors[contextIndex], contextIndex, transitionType, token);
                timeframe = std::max(timeframe, scoreTimes_[contextIndex]);
            }
            // Pre-prune before creating the extension
            if (useScorePruning_.front() and score > bestScore + scoreThresholds_.front()) {
                continue;
            }
            bestScore = std::min(bestScore, score);
            extensions_.push_back({.nextToken      = token,
                                   .pron           = lemma->pronunciations().first,
                                   .score          = score,
                                   .timeframe      = timeframe,
                                   .transitionType = transitionType,
                                   .baseHypIndex   = hypIndex});
        }
    }

    numExtensionsBeforeFirstPruning_ += extensions_.size();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-extensions-before-first-pruning", extensions_.size());
    }
}

void LlmTimesyncBeamSearch::updateExtensionScores(size_t scorerIdx, std::vector<std::optional<Nn::ScoreAccessorRef>> const& scoreAccessors) {
    for (auto& ext : extensions_) {
        if (not labelScorers_[scorerIdx]->scoresTransition(ext.transitionType)) {
            continue;
        }
        size_t contextIndex = hypIndexToContextIndexMap_[ext.baseHypIndex];
        if (not scoreAccessors[contextIndex]) {
            // Not scorable, so it gets pruned
            ext.score = Core::Type<Score>::max;
            continue;
        }
        ext.score += labelScore(*scoreAccessors[contextIndex], contextIndex, ext.transitionType, ext.nextToken);
        ext.timeframe = std::max(ext.timeframe, scoreTimes_[contextIndex]);
    }
}

void LlmTimesyncBeamSearch::prepareNextScoringContexts(size_t scorerIdx) {
    // Only the hypotheses which still have extensions are scored by the next label scorer
    scoringContexts_.clear();
    hypIndexToContextIndexMap_.assign(beam_.size(), -1);
    for (auto const& ext : extensions_) {
        if (hypIndexToContextIndexMap_[ext.baseHypIndex] == -1) {
            hypIndexToContextIndexMap_[ext.baseHypIndex] = scoringContexts_.size();
            scoringContexts_.push_back(beam_[ext.baseHypIndex].scoringContexts[scorerIdx + 1]);
        }
    }
}

bool LlmTimesyncBeamSearch::pruneBeforeLlm() {
    Core::StopWatch::Scope timer(preLlmPruningTime_);
    scorePruning(extensions_, preLlmScoreThreshold_, preLlmMaxBeamSize_);
    numHypsBeforeLlm_ += extensions_.size();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-hyps-before-llm", extensions_.size());
    }
    return not extensions_.empty();
}

void LlmTimesyncBeamSearch::applyLlm() {
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

void LlmTimesyncBeamSearch::assembleWords() {
    extensionWords_.clear();
    wordRequests_.clear();
    wordRequestExtensions_.clear();
    for (size_t extIdx = 0ul; extIdx < extensions_.size(); ++extIdx) {
        auto const& ext   = extensions_[extIdx];
        WordState   words = beam_[ext.baseHypIndex].words;
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

void LlmTimesyncBeamSearch::scoreFinishedWords() {
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

void LlmTimesyncBeamSearch::buildNewBeamFromExtensions() {
    newBeam_.clear();
    for (size_t extIdx = 0ul; extIdx < extensions_.size(); ++extIdx) {
        auto const& extension = extensions_[extIdx];
        auto const& baseHyp   = beam_[extension.baseHypIndex];

        std::vector<Nn::ScoringContextRef> newScoringContexts;
        for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
            newScoringContexts.push_back(labelScorers_[scorerIdx]->extendedScoringContext(
                    baseHyp.scoringContexts[scorerIdx],
                    extension.nextToken,
                    extension.transitionType));
        }

        newBeam_.push_back({baseHyp, extension, extensionWords_[extIdx], newScoringContexts});
    }
}

void LlmTimesyncBeamSearch::cleanupCaches() {
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

bool LlmTimesyncBeamSearch::emitsPiece(Nn::TransitionType transitionType) {
    switch (transitionType) {
        case Nn::TransitionType::LABEL_TO_LABEL:
        case Nn::TransitionType::BLANK_TO_LABEL:
        case Nn::TransitionType::SILENCE_TO_LABEL:
        case Nn::TransitionType::INITIAL_LABEL:
            return true;
        default:
            return false;
    }
}

LlmTimesyncBeamSearch::LabelHypothesis const& LlmTimesyncBeamSearch::getBestHypothesis() const {
    verify(not beam_.empty());

    return *std::min_element(beam_.begin(), beam_.end());
}

LlmTimesyncBeamSearch::LabelHypothesis const& LlmTimesyncBeamSearch::getWorstHypothesis() const {
    verify(not beam_.empty());

    return *std::max_element(beam_.begin(), beam_.end());
}

void LlmTimesyncBeamSearch::resetStatistics() {
    for (auto* timer : {&initializationTime_, &featureProcessingTime_, &recognitionTime_, &preLlmPruningTime_, &llmTime_, &wordAssemblyTime_,
                        &wordScoringTime_, &buildNewBeamTime_, &recombinationTime_, &beamPruningTime_, &cleanupTime_, &finalizeTime_,
                        &finalizeLlmTime_, &finalizeScoringTime_}) {
        timer->reset();
    }
    for (auto* timers : {&scoreAndPruneExtensionsTimes_, &scoringTimes_, &scoreReadoutTimes_, &intermediatePruningTimes_}) {
        for (auto& timer : *timers) {
            timer.reset();
        }
    }
    for (auto& stat : numHypsAfterIntermediatePruning_) {
        stat.clear();
    }
    for (auto* stat : {&numInputHyps_, &numExtensionsBeforeFirstPruning_, &numHypsBeforeLlm_, &numFinishedWords_, &numHypsAfterRecombination_,
                       &numHypsAfterPruning_, &numActiveHyps_}) {
        stat->clear();
    }
}

void LlmTimesyncBeamSearch::logStatistics() const {
    logOwnStatistics();
    for (auto const& labelScorer : labelScorers_) {
        labelScorer->logStatistics();
    }
    llmWordScorer_.logStatistics();
}

void LlmTimesyncBeamSearch::logOwnStatistics() const {
    if (not statisticsChannel_.isOpen()) {
        return;
    }
    logTimingStatistics();
    logSearchStatistics();
}

void LlmTimesyncBeamSearch::logTimingStatistics() const {
    auto& channel = statisticsChannel_;
    channel << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    channel << Core::XmlFull("initialization-time", initializationTime_.elapsedMilliseconds());
    channel << Core::XmlFull("feature-processing-time", featureProcessingTime_.elapsedMilliseconds());

    channel << Core::XmlOpen("recognition-time") + Core::XmlAttribute("total", recognitionTime_.elapsedMilliseconds());
    for (size_t i = 0ul; i < scoreAndPruneExtensionsTimes_.size(); ++i) {
        channel << Core::XmlOpen("score-and-prune-extensions-time") + Core::XmlAttribute("scorer", i + 1) + Core::XmlAttribute("total", scoreAndPruneExtensionsTimes_[i].elapsedMilliseconds());
        channel << Core::XmlFull("scoring-time", scoringTimes_[i].elapsedMilliseconds());
        channel << Core::XmlFull("score-readout-time", scoreReadoutTimes_[i].elapsedMilliseconds());
        channel << Core::XmlFull("intermediate-pruning-time", intermediatePruningTimes_[i].elapsedMilliseconds());
        channel << Core::XmlClose("score-and-prune-extensions-time");
    }
    channel << Core::XmlFull("pre-llm-pruning-time", preLlmPruningTime_.elapsedMilliseconds());
    channel << Core::XmlOpen("llm-time") + Core::XmlAttribute("total", llmTime_.elapsedMilliseconds());
    channel << Core::XmlFull("word-assembly-time", wordAssemblyTime_.elapsedMilliseconds());
    channel << Core::XmlFull("word-scoring-time", wordScoringTime_.elapsedMilliseconds());
    channel << Core::XmlClose("llm-time");
    channel << Core::XmlFull("build-new-beam-time", buildNewBeamTime_.elapsedMilliseconds());
    channel << Core::XmlFull("recombination-time", recombinationTime_.elapsedMilliseconds());
    channel << Core::XmlFull("beam-pruning-time", beamPruningTime_.elapsedMilliseconds());
    channel << Core::XmlFull("cleanup-time", cleanupTime_.elapsedMilliseconds());
    channel << Core::XmlClose("recognition-time");

    channel << Core::XmlOpen("finalize-time") + Core::XmlAttribute("total", finalizeTime_.elapsedMilliseconds());
    channel << Core::XmlFull("llm-time", finalizeLlmTime_.elapsedMilliseconds());
    channel << Core::XmlFull("scoring-time", finalizeScoringTime_.elapsedMilliseconds());
    channel << Core::XmlClose("finalize-time");
    channel << Core::XmlClose("timing-statistics");
}

void LlmTimesyncBeamSearch::logSearchStatistics() const {
    auto& channel = statisticsChannel_;
    channel << Core::XmlOpen("search-statistics");
    numInputHyps_.write(channel);
    numExtensionsBeforeFirstPruning_.write(channel);
    for (size_t i = 0ul; i < numHypsAfterIntermediatePruning_.size(); ++i) {
        numHypsAfterIntermediatePruning_[i].write(channel, {Core::XmlAttribute("scorer", i + 1)});
    }
    numHypsBeforeLlm_.write(channel);
    numFinishedWords_.write(channel);
    numHypsAfterRecombination_.write(channel);
    numHypsAfterPruning_.write(channel);
    numActiveHyps_.write(channel);
    channel << Core::XmlClose("search-statistics");
}

void LlmTimesyncBeamSearch::logBeamStatistics() {
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

Nn::TransitionType LlmTimesyncBeamSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const {
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
void LlmTimesyncBeamSearch::scorePruning(std::vector<Element>& hypotheses, Score relativeThreshold, size_t maxBeamSize) {
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
        scoreHistogram_.clear();
        scoreHistogram_.setLimits(lowerScore, upperScore);

        for (auto const& hyp : hypotheses) {
            scoreHistogram_ += hyp.score;
        }

        absoluteThreshold = std::min(absoluteThreshold, scoreHistogram_.quantile(maxBeamSize));
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

template void LlmTimesyncBeamSearch::scorePruning<LlmTimesyncBeamSearch::ExtensionCandidate>(std::vector<LlmTimesyncBeamSearch::ExtensionCandidate>&, Score, size_t);
template void LlmTimesyncBeamSearch::scorePruning<LlmTimesyncBeamSearch::LabelHypothesis>(std::vector<LlmTimesyncBeamSearch::LabelHypothesis>&, Score, size_t);

void LlmTimesyncBeamSearch::recombination(std::vector<LlmTimesyncBeamSearch::LabelHypothesis>& hypotheses) {
    if (not recombinationEnabled_) {
        return;
    }

    // Represents a unique combination of currentToken, scoringContexts, LLM history and pending word
    struct RecombinationContext {
        Nn::LabelIndex                     currentToken;
        LlmHistory                         llmHistory;
        WordAssembler::WordId              pendingWord;
        std::vector<Nn::ScoringContextRef> scoringContexts;

        RecombinationContext(LabelHypothesis const& hyp)
                : currentToken(hyp.currentToken), llmHistory(hyp.words.llmHistory), pendingWord(hyp.words.pendingWord), scoringContexts(hyp.scoringContexts) {}

        bool operator==(RecombinationContext const& other) const {
            if (currentToken != other.currentToken or llmHistory != other.llmHistory or pendingWord != other.pendingWord) {
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
            size_t hash = Core::combineHashes(Core::combineHashes(context.currentToken, context.llmHistory), context.pendingWord);
            for (auto const& scoringContext : context.scoringContexts) {
                hash = Core::combineHashes(hash, Nn::ScoringContextHash{}(scoringContext));
            }
            return hash;
        }
    };

    tempHypotheses_.clear();
    // Reserve capacity because future reallocations would break the raw pointer we are storing later
    tempHypotheses_.reserve(hypotheses.size());
    // Map each unique ScoringContext in newHypotheses to its hypothesis
    std::unordered_map<RecombinationContext, LabelHypothesis*, RecombinationContextHash> seenScoringContexts;
    for (auto const& hyp : hypotheses) {
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

    hypotheses.swap(tempHypotheses_);
}

void LlmTimesyncBeamSearch::finalizeHypotheses() {
    {
        Core::StopWatch::Scope timer(finalizeTime_);
        finishWordsAtSegmentEnd();
        scoreSentenceEnd();
        buildFinalHypotheses();
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

void LlmTimesyncBeamSearch::finishWordsAtSegmentEnd() {
    Core::StopWatch::Scope timer(finalizeLlmTime_);
    wordRequests_.clear();
    for (auto const& hyp : beam_) {
        bool pending = hyp.words.pendingWord != WordAssembler::noWord;
        wordRequests_.push_back({.history = hyp.words.llmHistory, .word = pending ? &wordAssembler_.spelling(hyp.words.pendingWord) : nullptr, .sentenceEnd = true});
    }
    llmWordScorer_.score(wordRequests_, wordResults_);

    Bliss::LemmaPronunciation const* sentenceEndPron = nullptr;
    if (sentenceEndLemma_) {
        sentenceEndPron = sentenceEndLemma_->pronunciations().first;
    }

    extensions_.clear();
    extensionWords_.clear();
    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        auto const& hyp   = beam_[hypIndex];
        Score       delta = llmScale_ * wordResults_[hypIndex].cost + (wordRequests_[hypIndex].word ? wordPenalty_ : 0.0);
        extensions_.push_back({.nextToken      = sentenceEndLabelIndex_,
                               .pron           = sentenceEndPron,
                               .score          = hyp.score + delta,
                               .timeframe      = hyp.trace->time,
                               .transitionType = Nn::TransitionType::SENTENCE_END,
                               .baseHypIndex   = hypIndex});
        extensionWords_.push_back({WordAssembler::noWord, wordResults_[hypIndex].history, hyp.words.llmScore + delta});
    }
}

void LlmTimesyncBeamSearch::scoreSentenceEnd() {
    if (not useSentenceEnd_) {
        return;
    }
    for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
        if (not labelScorers_[scorerIdx]->scoresTransition(Nn::TransitionType::SENTENCE_END)) {
            continue;
        }

        scoringContexts_.clear();
        for (auto const& hyp : beam_) {
            scoringContexts_.push_back(hyp.scoringContexts[scorerIdx]);
        }

        finalizeScoringTime_.start();
        auto scoreAccessors = labelScorers_[scorerIdx]->getScoreAccessors(scoringContexts_);
        finalizeScoringTime_.stop();

        for (size_t extIdx = 0ul; extIdx < extensions_.size(); ++extIdx) {
            if (not scoreAccessors[extIdx]) {
                continue;
            }
            auto& ext = extensions_[extIdx];
            ext.score += (*scoreAccessors[extIdx])->getScore(ext.transitionType, ext.nextToken);
            ext.timeframe = std::max(ext.timeframe, (*scoreAccessors[extIdx])->getTime());
        }
    }
}

void LlmTimesyncBeamSearch::buildFinalHypotheses() {
    tempHypotheses_.clear();
    for (size_t extIdx = 0ul; extIdx < extensions_.size(); ++extIdx) {
        auto const& ext     = extensions_[extIdx];
        auto const& words   = extensionWords_[extIdx];
        auto const& baseHyp = beam_[ext.baseHypIndex];
        if (ext.pron != nullptr) {
            // The scoring context is not updated as no further scoring is done afterwards
            tempHypotheses_.push_back({baseHyp, ext, words, baseHyp.scoringContexts});
            continue;
        }
        // Without a sentence-end lemma the final scores go to the last trace item, so that every traceback contains them
        Score llmDelta = words.llmScore - baseHyp.words.llmScore;
        tempHypotheses_.push_back(baseHyp);
        tempHypotheses_.back().score = ext.score;
        tempHypotheses_.back().words = words;
        tempHypotheses_.back().trace = traceWithAddedScore(baseHyp.trace, {ext.score - baseHyp.score - llmDelta, llmDelta});
    }
    beam_.swap(tempHypotheses_);
}

void LlmTimesyncBeamSearch::maximumStableDelayPruning() {
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

}  // namespace Search
