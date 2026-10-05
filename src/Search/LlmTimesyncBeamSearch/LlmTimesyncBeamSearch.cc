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
#include <strings.h>

#include <Core/CollapsedVector.hh>
#include <Core/XmlStream.hh>
#include <Lattice/LatticeAdaptor.hh>
#include <Math/Utilities.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/Histogram.hh>
#include <Search/Module.hh>
#include <Search/Traceback.hh>
#include <Search/TracebackHelper.hh>

namespace Search {

namespace {

enum RecombinationMode {
    RecombinationModeOff,
    RecombinationModeOn,
};

/*
 * Copy of `trace` and its sibling chain with `delta` added to every score
 */
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
          lmScore(0.0),
          lmHistory(0u),
          pendingWord(),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))) {}

LlmTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        LlmTimesyncBeamSearch::LabelHypothesis const&    base,
        LlmTimesyncBeamSearch::ExtensionCandidate const& extension,
        std::vector<Nn::ScoringContextRef> const&        newScoringContexts)
        : scoringContexts(newScoringContexts),
          currentToken(extension.nextToken),
          score(extension.score),
          lmScore(extension.lmScore),
          lmHistory(extension.lmHistory),
          pendingWord(extension.pendingWord),
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
            {score - lmScore, lmScore},
            {}));
}

std::string LlmTimesyncBeamSearch::LabelHypothesis::toString() const {
    std::stringstream ss;
    ss << "Score: " << score << ", LM score: " << lmScore << ", LLM history: " << lmHistory << ", pending word: \"" << pendingWord << "\", traceback: ";

    auto traceback = trace->performTraceback();

    for (auto& item : *traceback) {
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
        "Maximum number of extensions that are passed on to the LLM in each step. 0 means the max-beam-size of the last label scorer.",
        0,
        0);

const Core::ParameterFloat LlmTimesyncBeamSearch::paramPreLlmScoreThreshold(
        "pre-llm-score-threshold",
        "Prune extensions with a score that is at least this much worse than the best one before passing them on to the LLM.",
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

const Core::ParameterString LlmTimesyncBeamSearch::paramWordStartMarker(
        "word-start-marker",
        "Prefix of the orthography of word pieces that begin a new word.",
        "\xE2\x96\x81");  // U+2581 "▁", as used by SentencePiece

const Core::ParameterString LlmTimesyncBeamSearch::paramWordSeparator(
        "word-separator",
        "Text that is put in front of every finished word except the first one before it is tokenized by the LLM.",
        " ");

const Core::ParameterFloat LlmTimesyncBeamSearch::paramLlmScale(
        "llm-scale",
        "Scale of the LLM costs.",
        1.0);

const Core::ParameterFloat LlmTimesyncBeamSearch::paramWordPenalty(
        "word-penalty",
        "Score added for every finished word. Negative values reward words.",
        0.0);

const Core::ParameterBool LlmTimesyncBeamSearch::paramLogStepwiseStatistics(
        "log-stepwise-statistics",
        "Log statistics about the beam at every search step.",
        false);

const Core::ParameterInt LlmTimesyncBeamSearch::paramCacheCleanupInterval(
        "cache-cleanup-interval",
        "Interval of search steps after which buffered inputs that are not needed anymore get cleaned up.",
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
          wordStartMarker_(paramWordStartMarker(config)),
          wordSeparator_(paramWordSeparator(config)),
          llmScale_(paramLlmScale(config)),
          wordPenalty_(paramWordPenalty(config)),
          cacheCleanupInterval_(paramCacheCleanupInterval(config)),
          maximumStableDelay_(paramMaximumStableDelay(config)),
          maximumStableDelayPruningInterval_(paramMaximumStableDelayPruningInterval(config)),
          recombinationEnabled_(paramRecombinationMode(config) == RecombinationModeOn),
          logStepwiseStatistics_(paramLogStepwiseStatistics(config)),
          debugChannel_(config, "debug"),
          labelScorers_(),
          beam_(),
          pieceTexts_(),
          pieceStartsWord_(),
          llmCache_(),
          hypIndexToContextIndexMap_(),
          extensions_(),
          newBeam_(),
          scoringContexts_(),
          tempHypotheses_(),
          initializationTime_(),
          featureProcessingTime_(),
          scoringTime_(),
          llmTime_(),
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

    for (size_t i = 1ul; i <= maxBeamSizes_.size(); ++i) {
        numHypsAfterIntermediatePruning_.push_back({"num-hyps-after-intermediate-pruning-" + std::to_string(i)});
    }

    useSentenceEnd_ = sentenceEndLabelIndex_ != Nn::invalidLabelIndex;
    if (useSentenceEnd_) {
        log() << "Use sentence end label with index " << sentenceEndLabelIndex_;
    }

    if (wordStartMarker_.empty()) {
        error() << "The word-start marker must not be empty";
    }

    llmCache_.setScorer(Search::Module::instance().llmScorerFactory().createLlmScorer(select("llm")));
}

Speech::ModelCombination::Mode LlmTimesyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon;
}

bool LlmTimesyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_      = modelCombination.lexicon();
    labelScorers_ = modelCombination.labelScorers();

    if (labelScorers_.size() > maxBeamSizes_.size()) {
        error() << "Number of label scorers (" << labelScorers_.size() << ") exceeds number of configured max beam sizes (" << maxBeamSizes_.size() << ")";
    }
    if (labelScorers_.size() < maxBeamSizes_.size()) {
        warning() << "Number of label scorers (" << labelScorers_.size() << ") is less than number of configured max beam sizes (" << maxBeamSizes_.size() << ")";
    }
    if (preLlmMaxBeamSize_ == 0ul) {
        preLlmMaxBeamSize_ = maxBeamSizes_[labelScorers_.size() - 1];
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

    // Split the orthography of every label into the word-start marker and the text it contributes to the word
    pieceTexts_.assign(lexicon_->nLemmas(), std::string());
    pieceStartsWord_.assign(lexicon_->nLemmas(), false);
    auto lemmas = lexicon_->lemmas();
    for (auto lemmaIt = lemmas.first; lemmaIt != lemmas.second; ++lemmaIt) {
        Bliss::Lemma const* lemma(*lemmaIt);
        if (lemma->nOrthographicForms() == 0) {
            continue;
        }
        std::string text = lemma->preferredOrthographicForm().str();
        if (text.compare(0, wordStartMarker_.size(), wordStartMarker_) == 0) {
            pieceStartsWord_[lemma->id()] = true;
            text.erase(0, wordStartMarker_.size());
        }
        pieceTexts_[lemma->id()] = std::move(text);
    }

    return true;
}

void LlmTimesyncBeamSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    initializationTime_.reset();
    featureProcessingTime_.reset();
    scoringTime_.reset();
    llmTime_.reset();
    for (auto& stat : numHypsAfterIntermediatePruning_) {
        stat.clear();
    }
    numHypsBeforeLlm_.clear();
    numFinishedWords_.clear();
    numHypsAfterRecombination_.clear();
    numHypsAfterPruning_.clear();
    numActiveHyps_.clear();
    llmCache_.clearStatistics();

    initializationTime_.start();

    for (auto& labelScorer : labelScorers_) {
        labelScorer->reset();
    }

    llmTime_.start();
    llmCache_.reset();
    llmTime_.stop();

    // Reset beam to a single empty hypothesis
    beam_.clear();
    beam_.push_back(LabelHypothesis());
    beam_.front().scoringContexts.clear();
    for (auto& labelScorer : labelScorers_) {
        beam_.front().scoringContexts.push_back(labelScorer->getInitialScoringContext());
    }
    beam_.front().lmHistory = llmCache_.initialHistory();

    currentSearchStep_ = 0ul;
    finishedSegment_   = false;

    initializationTime_.stop();
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
    logStatistics();
}

void LlmTimesyncBeamSearch::putFeature(Nn::DataView const& feature) {
    featureProcessingTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInput(feature);
    }
    featureProcessingTime_.stop();
}

void LlmTimesyncBeamSearch::putFeatures(Nn::DataView const& features, size_t nTimesteps) {
    featureProcessingTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInputs(features, nTimesteps);
    }
    featureProcessingTime_.stop();
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

    // Assume the output labels are stored as lexicon lemma orth and ordered consistently with NN output index
    auto lemmas = lexicon_->lemmas();

    /*
     * We collect the scoring contexts that need to be passed to the LabelScorer into `scoringContexts_`.
     * `hypIndexToContextIndexMap_` maps a beam index to the position of its context in `scoringContexts_`,
     * i.e. beam_[i].scoringContexts.front() == scoringContexts_[hypIndexToContextIndexMap_[i]].
     * For the first label scorer this is just the identity mapping (hypIndexToContextIndexMap_[i] == i). For each
     * subsequent scorer the list is rebuilt, dropping the contexts of any hypotheses whose extensions were all
     * removed by intermediate pruning.
     */
    extensions_.clear();
    scoringContexts_.clear();
    scoringContexts_.reserve(beam_.size());
    hypIndexToContextIndexMap_.resize(beam_.size());
    std::iota(hypIndexToContextIndexMap_.begin(), hypIndexToContextIndexMap_.end(), 0ul);

    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        scoringContexts_.push_back(beam_[hypIndex].scoringContexts.front());
    }

    if (logStepwiseStatistics_) {
        clog() << Core::XmlOpen("search-step-stats");
    }

    for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
        auto const& labelScorer = labelScorers_[scorerIdx];
        scoringTime_.start();
        auto scoreAccessors = labelScorer->getScoreAccessors(scoringContexts_);
        scoringTime_.stop();
        std::vector<std::optional<Nn::DenseScoreSpan>> denseScoreSpans(scoreAccessors.size(), std::nullopt);
        std::vector<Nn::TimeframeIndex>                scoreTimes(scoreAccessors.size(), 0);
        for (size_t accessorIdx = 0ul; accessorIdx < scoreAccessors.size(); ++accessorIdx) {
            if (scoreAccessors[accessorIdx]) {
                denseScoreSpans[accessorIdx] = (*scoreAccessors[accessorIdx])->getDenseScores();
                scoreTimes[accessorIdx]      = (*scoreAccessors[accessorIdx])->getTime();
            }
        }

        if (scorerIdx == 0ul) {
            // In the first iteration, create extensions while pre-pruning
            Score currentBestScore = Core::Type<Score>::max;

            for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
                auto const& hyp = beam_[hypIndex];

                auto const& scoreAccessor = scoreAccessors[hypIndexToContextIndexMap_[hypIndex]];
                if (not scoreAccessor) {
                    // No extensions for hyps that couldn't be scored
                    continue;
                }
                auto const& denseScores = denseScoreSpans[hypIndexToContextIndexMap_[hypIndex]];
                auto        scoreTime   = scoreTimes[hypIndexToContextIndexMap_[hypIndex]];

                // Iterate over possible successors (all lemmas)
                for (auto lemmaIt = lemmas.first; lemmaIt != lemmas.second; ++lemmaIt) {
                    Bliss::Lemma const* lemma(*lemmaIt);
                    Nn::LabelIndex      tokenIdx = lemma->id();
                    // Don't score the sentence-end token
                    if (tokenIdx == sentenceEndLabelIndex_) {
                        continue;
                    }
                    auto transitionType = inferTransitionType(hyp.currentToken, tokenIdx);
                    auto extScore       = hyp.score;
                    auto extTime        = hyp.trace->time;
                    if (labelScorers_[scorerIdx]->scoresTransition(transitionType)) {
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

                    extensions_.push_back(
                            {.nextToken      = tokenIdx,
                             .pron           = lemma->pronunciations().first,
                             .score          = extScore,
                             .timeframe      = extTime,
                             .transitionType = transitionType,
                             .baseHypIndex   = hypIndex,
                             .lmScore        = 0.0,
                             .lmHistory      = 0u,
                             .pendingWord    = {}});
                }
            }
        }
        else {
            // Update ext score and timestep
            for (auto& ext : extensions_) {
                if (not labelScorer->scoresTransition(ext.transitionType)) {
                    continue;
                }
                auto const& scoreAccessor = scoreAccessors[hypIndexToContextIndexMap_[ext.baseHypIndex]];

                if (scoreAccessor) {
                    auto const& denseScores = denseScoreSpans[hypIndexToContextIndexMap_[ext.baseHypIndex]];
                    ext.score += (denseScores and ext.nextToken < denseScores->size())
                                         ? (*denseScores)[ext.nextToken]
                                         : (*scoreAccessor)->getScore(ext.transitionType, ext.nextToken);
                    ext.timeframe = std::max(ext.timeframe, scoreTimes[hypIndexToContextIndexMap_[ext.baseHypIndex]]);
                }
                else {
                    // Extension is not scorable so set the score to max in order to prune it later
                    ext.score = Core::Type<Score>::max;
                }
            }
        }

        if (extensions_.empty()) {
            if (logStepwiseStatistics_) {
                clog() << Core::XmlClose("search-step-stats");
            }
            return false;
        }

        /*
         * Prune set of possible extensions by max beam size and possibly also by score.
         */
        size_t maxBeamSize = extensions_.size();
        if (scorerIdx < labelScorers_.size() - 1) {
            maxBeamSize = maxBeamSizes_[scorerIdx];
        }
        scorePruning(extensions_, scoreThresholds_[scorerIdx], maxBeamSize);
        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("num-hyps-after-intermediate-pruning-" + std::to_string(scorerIdx + 1), extensions_.size());
        }
        numHypsAfterIntermediatePruning_[scorerIdx] += extensions_.size();
        if (extensions_.empty()) {
            if (logStepwiseStatistics_) {
                clog() << Core::XmlClose("search-step-stats");
            }
            return false;
        }

        if (scorerIdx < labelScorers_.size() - 1) {
            // Prepare scoring context list for next iteration
            // Some scoring contexts from the current iteration may not have survived pruning, so we need to recreate the list
            // Use -1 as placeholder to signify that this hyp was not visited yet
            scoringContexts_.clear();
            hypIndexToContextIndexMap_.assign(beam_.size(), -1);
            for (auto& ext : extensions_) {
                if (hypIndexToContextIndexMap_[ext.baseHypIndex] == -1) {
                    hypIndexToContextIndexMap_[ext.baseHypIndex] = scoringContexts_.size();
                    scoringContexts_.push_back(beam_[ext.baseHypIndex].scoringContexts[scorerIdx + 1]);
                }
            }
        }
    }

    /*
     * Prune once more so that the LLM is only asked about extensions that may survive, then add the LLM scores.
     */
    scorePruning(extensions_, preLlmScoreThreshold_, preLlmMaxBeamSize_);
    numHypsBeforeLlm_ += extensions_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-before-llm", extensions_.size());
    }

    if (extensions_.empty()) {
        if (logStepwiseStatistics_) {
            clog() << Core::XmlClose("search-step-stats");
        }
        return false;
    }

    applyLlm();

    // Create new beam from surviving extensions.
    newBeam_.clear();
    for (auto const& extension : extensions_) {
        auto const& baseHyp = beam_[extension.baseHypIndex];

        std::vector<Nn::ScoringContextRef> newScoringContexts;
        for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
            newScoringContexts.push_back(labelScorers_[scorerIdx]->extendedScoringContext(
                    baseHyp.scoringContexts[scorerIdx],
                    extension.nextToken,
                    extension.transitionType));
        }

        newBeam_.push_back({baseHyp, extension, newScoringContexts});
    }

    // For all hypotheses with the same scoring context, LLM history and pending word keep only the best since they will all develop in the same way.
    recombination(newBeam_);
    numHypsAfterRecombination_ += newBeam_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-recombination", newBeam_.size());
    }

    // The LLM scores changed the ranking, so apply the pruning of the last label scorer once more
    scorePruning(newBeam_, scoreThresholds_[labelScorers_.size() - 1], maxBeamSizes_[labelScorers_.size() - 1]);
    numHypsAfterPruning_ += newBeam_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-pruning", newBeam_.size());
    }

    numActiveHyps_ += newBeam_.size();

    beam_.swap(newBeam_);

    ++currentSearchStep_;

    /*
     * Clean up label scorer caches.
     */
    if (currentSearchStep_ % cacheCleanupInterval_ == 0) {
        for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
            Core::CollapsedVector<Nn::ScoringContextRef> activeContexts;
            for (auto const& hyp : beam_) {
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

std::string LlmTimesyncBeamSearch::wordText(LlmHistory history, std::string const& word) const {
    if (llmCache_.historyLength(history) == 0u) {
        return word;
    }
    return wordSeparator_ + word;
}

void LlmTimesyncBeamSearch::scoreCheapestVariants(std::vector<LlmHistory> const&                      histories,
                                                  std::vector<LlmTokenSequenceVariants const*> const& variants,
                                                  LlmTokenSequence const&                             suffix) {
    llmRequests_.clear();
    llmRequestOffsets_.clear();
    for (size_t i = 0ul; i < histories.size(); ++i) {
        llmRequestOffsets_.push_back(llmRequests_.size());
        if (variants[i] == nullptr) {
            llmRequests_.push_back({.history = histories[i], .tokens = suffix});
            continue;
        }
        for (auto const& variant : *variants[i]) {
            llmRequests_.push_back({.history = histories[i], .tokens = variant});
            llmRequests_.back().tokens.insert(llmRequests_.back().tokens.end(), suffix.begin(), suffix.end());
        }
    }
    llmRequestOffsets_.push_back(llmRequests_.size());

    llmCache_.score(llmRequests_, llmResults_);

    // Greedily keep the cheapest variant; on ties the first one, so the order of the variants decides
    bestLlmResults_.clear();
    for (size_t i = 0ul; i < histories.size(); ++i) {
        auto best = std::min_element(llmResults_.begin() + llmRequestOffsets_[i], llmResults_.begin() + llmRequestOffsets_[i + 1],
                                     [](auto const& a, auto const& b) { return a.cost < b.cost; });
        bestLlmResults_.push_back(*best);
    }
}

void LlmTimesyncBeamSearch::applyLlm() {
    llmTime_.start();

    // Update the pending words and collect the words that are finished by the extensions
    wordTexts_.clear();
    wordOwners_.clear();
    wordHistories_.clear();
    for (size_t extIdx = 0ul; extIdx < extensions_.size(); ++extIdx) {
        auto&       ext     = extensions_[extIdx];
        auto const& baseHyp = beam_[ext.baseHypIndex];

        ext.lmScore     = baseHyp.lmScore;
        ext.lmHistory   = baseHyp.lmHistory;
        ext.pendingWord = baseHyp.pendingWord;

        if (not emitsPiece(ext.transitionType)) {
            continue;
        }
        if (pieceStartsWord_[ext.nextToken] and not baseHyp.pendingWord.empty()) {
            wordTexts_.push_back(wordText(baseHyp.lmHistory, baseHyp.pendingWord));
            wordOwners_.push_back(extIdx);
            wordHistories_.push_back(baseHyp.lmHistory);
            ext.pendingWord = pieceTexts_[ext.nextToken];
        }
        else {
            ext.pendingWord += pieceTexts_[ext.nextToken];
        }
    }

    numFinishedWords_ += wordTexts_.size();
    if (wordTexts_.empty()) {
        llmTime_.stop();
        return;
    }

    llmCache_.tokenize(wordTexts_, wordTokenizations_);
    scoreCheapestVariants(wordHistories_, wordTokenizations_, {});

    for (size_t wordIdx = 0ul; wordIdx < wordOwners_.size(); ++wordIdx) {
        auto& ext     = extensions_[wordOwners_[wordIdx]];
        Score lmDelta = llmScale_ * bestLlmResults_[wordIdx].cost + wordPenalty_;
        ext.lmHistory = bestLlmResults_[wordIdx].history;
        ext.lmScore += lmDelta;
        ext.score += lmDelta;
    }

    llmTime_.stop();
}

LlmTimesyncBeamSearch::LabelHypothesis const& LlmTimesyncBeamSearch::getBestHypothesis() const {
    verify(not beam_.empty());

    return *std::min_element(beam_.begin(), beam_.end());
}

LlmTimesyncBeamSearch::LabelHypothesis const& LlmTimesyncBeamSearch::getWorstHypothesis() const {
    verify(not beam_.empty());

    return *std::max_element(beam_.begin(), beam_.end());
}

void LlmTimesyncBeamSearch::logStatistics() const {
    clog() << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    clog() << Core::XmlOpen("initialization-time") << initializationTime_.elapsedMilliseconds() << Core::XmlClose("initialization-time");
    clog() << Core::XmlOpen("feature-processing-time") << featureProcessingTime_.elapsedMilliseconds() << Core::XmlClose("feature-processing-time");
    clog() << Core::XmlOpen("scoring-time") << scoringTime_.elapsedMilliseconds() << Core::XmlClose("scoring-time");
    clog() << Core::XmlOpen("llm-time") << llmTime_.elapsedMilliseconds() << Core::XmlClose("llm-time");
    clog() << Core::XmlClose("timing-statistics");
    for (auto const& stat : numHypsAfterIntermediatePruning_) {
        stat.write(clog());
    }
    numHypsBeforeLlm_.write(clog());
    numFinishedWords_.write(clog());
    numHypsAfterRecombination_.write(clog());
    numHypsAfterPruning_.write(clog());
    numActiveHyps_.write(clog());
    llmCache_.logStatistics(clog());
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
        std::vector<Nn::ScoringContextRef> scoringContexts;
        LlmHistory                         lmHistory;
        std::string const*                 pendingWord;  // Points into the hypothesis, which outlives the context

        RecombinationContext(LabelHypothesis const& hyp)
                : currentToken(hyp.currentToken), scoringContexts(hyp.scoringContexts), lmHistory(hyp.lmHistory), pendingWord(&hyp.pendingWord) {}

        bool operator==(RecombinationContext const& other) const {
            if (currentToken != other.currentToken or lmHistory != other.lmHistory or *pendingWord != *other.pendingWord) {
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
            size_t hash = Core::combineHashes(context.currentToken, context.lmHistory);
            hash        = Core::combineHashes(hash, std::hash<std::string>{}(*context.pendingWord));
            for (auto const& scoringContext : context.scoringContexts) {
                hash = Core::combineHashes(hash, Nn::ScoringContextHash{}(scoringContext));
            }
            return hash;
        }
    };

    tempHypotheses_.clear();
    // Reserve capacity because future reallocations would break the raw pointer we are storing later
    tempHypotheses_.reserve(hypotheses.size());
    // Map each unique recombination context to its hypothesis in `tempHypotheses_`. The context of an entry points
    // at the pending word of that hypothesis, which keeps the same value when the hypothesis is overwritten by a
    // better one with the same context.
    std::unordered_map<RecombinationContext, LabelHypothesis*, RecombinationContextHash> seenContexts;
    for (auto& hyp : hypotheses) {
        auto existing = seenContexts.find(RecombinationContext(hyp));

        if (existing == seenContexts.end()) {
            // First time seeing this context so move it over to `tempHypotheses_`
            tempHypotheses_.push_back(std::move(hyp));
            seenContexts.emplace(RecombinationContext(tempHypotheses_.back()), &tempHypotheses_.back());
        }
        else {
            verify(not hyp.trace->sibling);

            auto* existingHyp = existing->second;
            if (hyp.score < existingHyp->score) {
                // New hyp is better -> replace in `tempHypotheses_` and add existing one as sibling
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
    llmTime_.start();

    // Finish the pending word of every hypothesis and score the LLM's sentence end, both in one request. The variant
    // of the last word is chosen by the cost of word and sentence end together, since nothing else follows.
    wordTexts_.clear();
    wordOwners_.clear();
    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        auto const& hyp = beam_[hypIndex];
        if (not hyp.pendingWord.empty()) {
            wordTexts_.push_back(wordText(hyp.lmHistory, hyp.pendingWord));
            wordOwners_.push_back(hypIndex);
        }
    }
    numFinishedWords_ += wordTexts_.size();
    llmCache_.tokenize(wordTexts_, wordTokenizations_);

    // Variants per hypothesis, null for hypotheses without a pending word
    std::vector<LlmTokenSequenceVariants const*> hypVariants(beam_.size(), nullptr);
    for (size_t wordIdx = 0ul; wordIdx < wordOwners_.size(); ++wordIdx) {
        hypVariants[wordOwners_[wordIdx]] = wordTokenizations_[wordIdx];
    }
    wordHistories_.clear();
    for (auto const& hyp : beam_) {
        wordHistories_.push_back(hyp.lmHistory);
    }
    scoreCheapestVariants(wordHistories_, hypVariants, llmCache_.sentenceEndTokens());

    Bliss::LemmaPronunciation const* sentenceEndPron = nullptr;
    if (sentenceEndLemma_) {
        sentenceEndPron = sentenceEndLemma_->pronunciations().first;
    }

    extensions_.clear();
    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        auto const& hyp     = beam_[hypIndex];
        Score       lmDelta = llmScale_ * bestLlmResults_[hypIndex].cost;
        if (not hyp.pendingWord.empty()) {
            lmDelta += wordPenalty_;
        }
        extensions_.push_back(
                {.nextToken      = sentenceEndLabelIndex_,
                 .pron           = sentenceEndPron,
                 .score          = hyp.score + lmDelta,
                 .timeframe      = hyp.trace->time,
                 .transitionType = Nn::TransitionType::SENTENCE_END,
                 .baseHypIndex   = hypIndex,
                 .lmScore        = hyp.lmScore + lmDelta,
                 .lmHistory      = bestLlmResults_[hypIndex].history,
                 .pendingWord    = {}});
    }

    llmTime_.stop();

    // Score sentence-end with all label scorers
    for (size_t scorerIdx = 0ul; useSentenceEnd_ and scorerIdx < labelScorers_.size(); ++scorerIdx) {
        if (not labelScorers_[scorerIdx]->scoresTransition(Nn::TransitionType::SENTENCE_END)) {
            continue;
        }

        scoringContexts_.clear();
        for (auto const& hyp : beam_) {
            scoringContexts_.push_back(hyp.scoringContexts[scorerIdx]);
        }

        scoringTime_.start();
        auto scoreAccessors = labelScorers_[scorerIdx]->getScoreAccessors(scoringContexts_);
        scoringTime_.stop();

        for (size_t extensionIdx = 0ul; extensionIdx < extensions_.size(); ++extensionIdx) {
            if (not scoreAccessors[extensionIdx]) {
                continue;
            }
            auto& ext   = extensions_[extensionIdx];
            auto  score = (*scoreAccessors[extensionIdx])->getScore(ext.transitionType, ext.nextToken);
            ext.score += score;
            ext.timeframe = std::max(ext.timeframe, (*scoreAccessors[extensionIdx])->getTime());
        }
    }

    tempHypotheses_.clear();
    for (size_t extensionIdx = 0ul; extensionIdx < extensions_.size(); ++extensionIdx) {
        auto&       ext     = extensions_[extensionIdx];
        auto const& baseHyp = beam_[ext.baseHypIndex];
        if (sentenceEndPron) {
            // The scoring context is not updated as no further scoring is done afterwards
            tempHypotheses_.push_back({baseHyp, ext, baseHyp.scoringContexts});
        }
        else {
            // Without a sentence-end lemma there is nothing to attach a new trace item to. Add the final scores to the
            // last item instead, so that they are part of every traceback.
            tempHypotheses_.push_back(baseHyp);
            auto& hyp       = tempHypotheses_.back();
            hyp.score       = ext.score;
            hyp.lmScore     = ext.lmScore;
            hyp.lmHistory   = ext.lmHistory;
            hyp.pendingWord = ext.pendingWord;
            hyp.trace       = traceWithAddedScore(baseHyp.trace, {(ext.score - baseHyp.score) - (ext.lmScore - baseHyp.lmScore), ext.lmScore - baseHyp.lmScore});
        }
    }

    beam_.swap(tempHypotheses_);

    numActiveHyps_ += beam_.size();

    // Log statistics about the final beam
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
    }
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
