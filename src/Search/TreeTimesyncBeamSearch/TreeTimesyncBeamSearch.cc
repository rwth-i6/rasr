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
#include <cmath>
#include <memory>
#include <strings.h>

#include <Am/ClassicStateModel.hh>
#include <Core/CollapsedVector.hh>
#include <Core/XmlStream.hh>
#include <Lattice/LatticeAdaptor.hh>
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

TreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContexts(),
          currentToken(Nn::invalidLabelIndex),
          currentState(invalidTreeNodeIndex),
          lmHistory(),
          timeframe(0),
          score(0.0),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))),
          oov()
#ifdef SEARCHV2_DEBUG
          ,
          tokenSequence(),
          tokenScoreDeltas(),
          tokenTimeframes()
#endif
{
}

TreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        TreeTimesyncBeamSearch::LabelHypothesis const&              base,
        TreeTimesyncBeamSearch::WithinWordExtensionCandidate const& extension,
        std::vector<Nn::ScoringContextRef> const&                   newScoringContexts)
        : scoringContexts(newScoringContexts),
          currentToken(extension.nextToken),
          currentState(extension.nextState),
          lmHistory(base.lmHistory),
          timeframe(extension.timeframe),
          score(extension.score),
          trace(base.trace),
          oov(base.oov)
#ifdef SEARCHV2_DEBUG
          ,
          tokenSequence(base.tokenSequence),
          tokenScoreDeltas(base.tokenScoreDeltas),
          tokenTimeframes(base.tokenTimeframes)
#endif
{
#ifdef SEARCHV2_DEBUG
    tokenSequence.push_back(extension.nextToken);
    tokenScoreDeltas.push_back(extension.score - base.score);
    tokenTimeframes.push_back(extension.timeframe);
#endif
}

TreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        LabelHypothesis const&                                   base,
        TreeTimesyncBeamSearch::WordEndExtensionCandidate const& extension,
        Lm::History const&                                       newLmHistory)
        : scoringContexts(base.scoringContexts),
          currentToken(base.currentToken),
          currentState(extension.rootState),
          lmHistory(newLmHistory),
          timeframe(base.timeframe),
          score(extension.score),
          oov(extension.oov)
#ifdef SEARCHV2_DEBUG
          ,
          tokenSequence(base.tokenSequence),
          tokenScoreDeltas(base.tokenScoreDeltas),
          tokenTimeframes(base.tokenTimeframes)
#endif
{
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

std::string TreeTimesyncBeamSearch::LabelHypothesis::toString() const {
    std::stringstream ss;
    ss << "Score: " << score << ", current state: " << currentState;
    if (oov and oov->wordPending()) {
        ss << ", pending fallback word of " << oov->numPieces << " piece(s) ("
           << (oov->diverged ? "diverged from every known pronunciation" : "still matching a known prefix") << ")";
    }
    ss << ", traceback: ";

    auto traceback = trace->performTraceback();

    for (auto& item : *traceback) {
        if (item.pronunciation and item.pronunciation->lemma()) {
            ss << item.pronunciation->lemma()->symbol() << " ";
        }
    }

#ifdef SEARCHV2_DEBUG
    ss << ", tokens: ";
    for (auto token : tokenSequence) {
        ss << token << " ";
    }
    ss << ", token score deltas: ";
    for (auto scoreDelta : tokenScoreDeltas) {
        ss << scoreDelta << " ";
    }
    ss << ", token timeframes: ";
    for (auto tf : tokenTimeframes) {
        ss << tf << " ";
    }
#endif

    return ss.str();
}

/*
 * ==============================
 * === TreeTimesyncBeamSearch ===
 * ==============================
 */

const Core::ParameterIntVector TreeTimesyncBeamSearch::paramMaxBeamSizes(
        "max-beam-size",
        "Maximum number of within-word hypotheses in the search beam. Pruning is applied after each intermediate label scorer.",
        "",
        1);

const Core::ParameterInt TreeTimesyncBeamSearch::paramMaxWordEndBeamSize(
        "max-word-end-beam-size",
        "Maximum number of word-end hypotheses in the search beam. If not set, global beam pruning will be done and word-end hypotheses will not be pruned separately.",
        std::numeric_limits<int>::max(), 0);

const Core::ParameterFloatVector TreeTimesyncBeamSearch::paramScoreThresholds(
        "score-threshold",
        "Prune any within-word hypotheses with a score that is at least this much worse than the best hypothesis. Pruning is applied after each intermediate label scorer.",
        "",
        0,
        Core::Type<Score>::max);

const Core::ParameterFloat TreeTimesyncBeamSearch::paramWordEndScoreThreshold(
        "word-end-score-threshold",
        "Prune any word-end hypothesis with a score that is at least this much worse than the best word-end hypothesis. This threshold is relative to the score-threshold. \
        If not set, global score pruning will be done and word-end hypotheses will not be pruned separately.",
        Core::Type<Score>::max, 0);

const Core::ParameterInt TreeTimesyncBeamSearch::paramNumHistogramBins(
        "num-histogram-bins",
        "Number of bins for histogram pruning of hypotheses (very minor effect).",
        100,
        2);

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

const Core::ParameterInt TreeTimesyncBeamSearch::paramCacheCleanupInterval(
        "cache-cleanup-interval",
        "Interval of search steps after which buffered inputs that are not needed anymore get cleaned up.",
        10,
        1);

const Core::ParameterInt TreeTimesyncBeamSearch::paramMaximumStableDelay(
        "maximum-stable-delay",
        "Introduce a cutoff point at `current-time` - `delay`. Every hypothesis that disagrees with the current best anywhere before the cutoff gets pruned."
        "This way words in the traceback become stable after at most `delay` frames.",
        Core::Type<int>::max,
        0);

const Core::ParameterInt TreeTimesyncBeamSearch::paramMaximumStableDelayPruningInterval(
        "maximum-stable-delay-pruning-interval",
        "Interval of search steps after which the maximum-stable-delay-pruning gets applied.",
        10,
        1);

const Core::Choice TreeTimesyncBeamSearch::choiceRecombinationMode(
        "off", RecombinationModeOff,
        "on", RecombinationModeOn,
        Core::Choice::endMark());

const Core::ParameterChoice TreeTimesyncBeamSearch::paramRecombinationMode(
        "recombination-mode",
        &choiceRecombinationMode,
        "Whether hypotheses with identical recombination state should be recombined.",
        RecombinationModeOn);

TreeTimesyncBeamSearch::TreeTimesyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          maxWordEndBeamSize_(paramMaxWordEndBeamSize(config)),
          wordEndScoreThreshold_(paramWordEndScoreThreshold(config)),
          scoreHistogram_(paramNumHistogramBins(config)),
          blankLabelIndex_(Nn::invalidLabelIndex),
          silenceLabelIndex_(Nn::invalidLabelIndex),
          blankLemma_(nullptr),
          silenceLemma_(nullptr),
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
          logStepwiseStatistics_(paramLogStepwiseStatistics(config)),
          labelScorers_(),
          nonWordLemmas_(),
          debugChannel_(config, "debug"),
          unknownWordFallback_(),
          excludeKnownWordsFromFallback_(false),
          unknownWordRoot_(invalidTreeNodeIndex),
          unknownSyntacticToken_(nullptr),
          unknownWordPenalty_(0.0),
          initialOovState_(),
          hypIndexToContextIndexMap_(),
          withinWordExtensions_(),
          wordEndExtensions_(),
          beam_(),
          newBeam_(),
          wordEndHypotheses_(),
          scoringContexts_(),
          tempHypotheses_(),
          currentSearchStep_(0ul),
          finishedSegment_(false),
          initializationTime_(),
          featureProcessingTime_(),
          scoringTime_(),
          numHypsAfterRecombination_("num-hyps-after-recombination"),
          numHypsAfterPruning_("num-hyps-after-pruning"),
          numWordEndHypsAfterScorePruning_("num-word-end-hyps-after-score-pruning"),
          numWordEndHypsAfterRecombination_("num-word-end-hyps-after-recombination"),
          numWordEndHypsAfterBeamPruning_("num-word-end-hyps-after-beam-pruning"),
          numActiveHyps_("num-active-hyps"),
          numActiveTrees_("num-active-trees"),
          numUnknownWordEvents_("num-unknown-word-events"),
          numKnownResolvedFallbackWords_("num-known-resolved-fallback-words") {
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

    for (size_t i = 1ul; i <= maxBeamSizes_.size(); ++i) {
        numHypsAfterIntermediatePruning_.push_back({"num-hyps-after-intermediate-pruning-" + std::to_string(i)});
    }
}

Speech::ModelCombination::Mode TreeTimesyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon | Speech::ModelCombination::useAcousticModel | Speech::ModelCombination::useLanguageModel;
}

Am::AcousticModel::Mode TreeTimesyncBeamSearch::requiredAcousticModel() const {
    return Am::AcousticModel::noEmissions;
}

bool TreeTimesyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_       = modelCombination.lexicon();
    labelScorers_  = modelCombination.labelScorers();
    acousticModel_ = modelCombination.acousticModel();
    languageModel_ = modelCombination.languageModel();

    if (labelScorers_.size() > maxBeamSizes_.size()) {
        error() << "Number of label scorers (" << labelScorers_.size() << ") exceeds number of configured max beam sizes (" << maxBeamSizes_.size() << ")";
    }
    if (labelScorers_.size() < maxBeamSizes_.size()) {
        warning() << "Number of label scorers (" << labelScorers_.size() << ") is less than number of configured max beam sizes (" << maxBeamSizes_.size() << ")";
    }

    nonWordLemmas_ = lexicon_->specialLemmas("nonword");

    // The word-end expansion can apply only one syntactic token per exit. Report the
    // offending lemma up front instead of failing on a `require` mid-segment.
    for (auto lemmaIters = lexicon_->lemmas(); lemmaIters.first != lemmaIters.second; ++lemmaIters.first) {
        auto const* lemma = *lemmaIters.first;
        if (lemma->syntacticTokenSequence().size() > 1) {
            error() << "Lemma \"" << lemma->name().str() << "\" has " << lemma->syntacticTokenSequence().size()
                    << " syntactic tokens. " << name() << " supports at most one syntactic token per lexical exit.";
        }
    }

    unknownWordFallback_           = std::make_unique<UnknownWordFallback>(config, *lexicon_);
    excludeKnownWordsFromFallback_ = unknownWordFallback_->excludesKnownWords();
    unknownSyntacticToken_         = unknownWordFallback_->unknownSyntacticToken();
    unknownWordPenalty_            = unknownWordFallback_->unknownWordPenalty();

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

    blankLemma_   = lexicon_->specialLemma("blank");
    silenceLemma_ = lexicon_->specialLemma("silence");

    if (blankLemma_) {
        blankLabelIndex_ = acousticModel_->emissionIndex(acousticModel_->blankAllophoneStateIndex());
        useBlank_        = true;
        log() << "Use blank label with index " << blankLabelIndex_;
    }
    else {
        blankLabelIndex_ = Nn::invalidLabelIndex;
        useBlank_        = false;
    }

    if (silenceLemma_) {
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

    if (excludeKnownWordsFromFallback_) {
        unknownWordRoot_ = network_->unknownWordRoot;
        if (unknownWordRoot_ == invalidTreeNodeIndex) {
            criticalError("The search tree was built without an open-vocabulary fallback root, "
                          "but unknown-word-fallback is \"known-excluding\".");
        }
        if (lexicon_->specialLemma("word-boundary") != nullptr) {
            // Known-prefix tracking starts at the ordinary root. With a word-boundary
            // lemma an ordinary word exit transits to the word-boundary root instead,
            // so the prefix walk and the ordinary path would disagree.
            criticalError("unknown-word-fallback \"known-excluding\" does not support a \"word-boundary\" special lemma.");
        }
        if (unknownSyntacticToken_ == nullptr) {
            criticalError("No unknown syntactic token available for the open-vocabulary fallback.");
        }
        else {
            // A finite additive bias cannot revive a zero-probability unknown token, so
            // refuse to decode rather than silently never taking the fallback.
            Lm::Score unknownScore = languageModel_->score(languageModel_->startHistory(), unknownSyntacticToken_);
            if (not Math::isinf(unknownScore) and not std::isnan(unknownScore)) {
                log() << "Unknown word cost of the configured word LM in the start history: " << unknownScore;
            }
            else {
                criticalError() << "The configured word LM assigns no probability to the unknown token \""
                                << unknownSyntacticToken_->symbol().str() << "\". Use an LM which models it, or "
                                << "configure an explicit smoothing/floor policy for it.";
            }
        }

        createPronunciationTrie();

        auto initial         = Core::ref(new OovState());
        initial->prefixNodes = {0u};  // the trie root
        initialOovState_     = initial;
    }

    return true;
}

void TreeTimesyncBeamSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    initializationTime_.reset();
    featureProcessingTime_.reset();
    scoringTime_.reset();
    for (auto& stat : numHypsAfterIntermediatePruning_) {
        stat.clear();
    }
    numHypsAfterRecombination_.clear();
    numHypsAfterPruning_.clear();
    numWordEndHypsAfterScorePruning_.clear();
    numWordEndHypsAfterRecombination_.clear();
    numWordEndHypsAfterBeamPruning_.clear();
    numActiveHyps_.clear();
    numActiveTrees_.clear();
    numUnknownWordEvents_.clear();
    numKnownResolvedFallbackWords_.clear();

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
    beam_.front().currentState = network_->rootState;
    beam_.front().lmHistory    = languageModel_->startHistory();
    beam_.front().oov          = initialOovState_;

    if (excludeKnownWordsFromFallback_ and
        unknownWordFallback_->tokenization() == UnknownWordFallback::WordStartMarked) {
        // Ordinary pronunciations and the fallback both start a word with a word-start
        // piece, so a first piece without that marker would be unreachable. Seed a
        // second hypothesis inside the pending-word root, where the still empty
        // pending word can be opened by any piece. An empty pending word produces no
        // word event, so this costs nothing if the first piece is word-start marked
        // after all: both hypotheses then recombine immediately.
        beam_.push_back(beam_.front());
        beam_.back().currentState = unknownWordRoot_;
    }

    currentSearchStep_ = 0ul;
    finishedSegment_   = false;

    initializationTime_.stop();
    if (segment != nullptr) {
        languageModel_->setSegment(segment);
        for (auto& hyp : beam_) {
            hyp.lmHistory = languageModel_->startHistory();
        }
    }
}

void TreeTimesyncBeamSearch::finishSegment() {
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

void TreeTimesyncBeamSearch::putFeature(Nn::DataView const& feature) {
    featureProcessingTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInput(feature);
    }
    featureProcessingTime_.stop();
}

void TreeTimesyncBeamSearch::putFeatures(Nn::DataView const& features, size_t nTimesteps) {
    featureProcessingTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInputs(features, nTimesteps);
    }
    featureProcessingTime_.stop();
}

Core::Ref<const Traceback> TreeTimesyncBeamSearch::getCurrentBestTraceback() const {
    return getBestHypothesis().trace->performTraceback();
}

Core::Ref<const LatticeAdaptor> TreeTimesyncBeamSearch::getCurrentBestWordLattice() const {
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

Core::Ref<const LatticeTrace> TreeTimesyncBeamSearch::getCurrentBestLatticeTrace() const {
    return getBestHypothesis().trace;
}

Core::Ref<const LatticeTrace> TreeTimesyncBeamSearch::getCommonPrefix() const {
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

bool TreeTimesyncBeamSearch::decodeStep() {
    if (finishedSegment_) {
        return false;
    }

    /*
     * Collect all possible extensions for all hypotheses in the beam.
     * We build a list of all scoring contexts that need to be passed to the LabelScorer for scoring scored inside `scoringContexts_`.
     * `hypIndexToContextIndexMap_` stores the mapping, i.e. beam_[i].scoringContext = scoringContexts_[hypIndexToScoringContextMap_[i]].
     * In the first iteration, this is just an identity mapping, i.e. hypIndexToContextIndexMap_[i] = i but for later label scorers
     * some scoring contexts become no longer relevant when all extensions using them have been pruned.
     */
    withinWordExtensions_.clear();
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

        /*
         * Perform scoring of all the scoring contexts with the label scorer.
         */
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
                auto const hyp = beam_[hypIndex];

                auto const& scoreAccessor = scoreAccessors[hypIndexToContextIndexMap_[hypIndex]];
                if (not scoreAccessor) {
                    // No extensions for hyps that couldn't be scored
                    continue;
                }
                auto const& denseScores = denseScoreSpans[hypIndexToContextIndexMap_[hypIndex]];
                auto        scoreTime   = scoreTimes[hypIndexToContextIndexMap_[hypIndex]];

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
                    if (labelScorers_[scorerIdx]->scoresTransition(transitionType)) {
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

                    withinWordExtensions_.push_back(
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
            for (auto& ext : withinWordExtensions_) {
                if (not labelScorer->scoresTransition(ext.transitionType)) {
                    continue;
                }
                auto const& scoreAccessor = scoreAccessors[hypIndexToContextIndexMap_[ext.baseHypIndex]];

                if (scoreAccessor) {
                    auto const& denseScores = denseScoreSpans[hypIndexToContextIndexMap_[ext.baseHypIndex]];
                    if (denseScores and ext.nextToken < denseScores->size()) {
                        ext.score += (*denseScores)[ext.nextToken];
                    }
                    else {
                        ext.score += (*scoreAccessor)->getScore(ext.transitionType, ext.nextToken);
                    }
                    ext.timeframe = std::max(ext.timeframe, scoreTimes[hypIndexToContextIndexMap_[ext.baseHypIndex]]);
                }
                else {
                    // Extension is not scorable so set the score to max in order to prune it later
                    ext.score = Core::Type<Score>::max;
                }
            }
        }

        if (withinWordExtensions_.empty()) {
            clog() << Core::XmlClose("search-step-stats");
            return false;
        }

        /*
         * Prune set of possible within-word extensions by max beam size and possibly also by score.
         */
        size_t maxBeamSize = withinWordExtensions_.size();
        if (scorerIdx < labelScorers_.size() - 1) {
            maxBeamSize = maxBeamSizes_[scorerIdx];
        }
        scorePruning(withinWordExtensions_, scoreThresholds_[scorerIdx], maxBeamSize);
        numHypsAfterIntermediatePruning_[scorerIdx] += withinWordExtensions_.size();
        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("num-hyps-after-intermediate-pruning-" + std::to_string(scorerIdx + 1), withinWordExtensions_.size());
        }
        if (withinWordExtensions_.empty()) {
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
            for (auto& ext : withinWordExtensions_) {
                if (hypIndexToContextIndexMap_[ext.baseHypIndex] == -1) {
                    hypIndexToContextIndexMap_[ext.baseHypIndex] = scoringContexts_.size();
                    scoringContexts_.push_back(beam_[ext.baseHypIndex].scoringContexts[scorerIdx + 1]);
                }
            }
        }
    }

    // Create new label hypotheses from extension candidates
    newBeam_.clear();
    for (auto const& extension : withinWordExtensions_) {
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

    // For all hypotheses at the same state and with the same scoring context and LM history
    // keep only the best since they will all develop in the same way
    recombination(newBeam_, false);
    numHypsAfterRecombination_ += newBeam_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-recombination", newBeam_.size());
    }

    scorePruning(newBeam_, Core::Type<Score>::max, maxBeamSizes_[labelScorers_.size() - 1]);
    numHypsAfterPruning_ += newBeam_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-pruning-" + std::to_string(labelScorers_.size()), newBeam_.size());
    }

    /*
     * Expand hypotheses to word-end hypotheses and incorporate the language model
     */
    wordEndExtensions_.clear();
    for (size_t hypIndex = 0ul; hypIndex < newBeam_.size(); ++hypIndex) {
        auto& hyp = newBeam_[hypIndex];

        // Create one word-end hypothesis for each exit
        for (size_t i = stateExitsOffset_[hyp.currentState]; i < stateExitsOffset_[hyp.currentState + 1]; ++i) {
            const PersistentStateTree::Exit exit      = stateExits_[i];
            auto const*                     lemmaPron = lexicon_->lemmaPronunciation(exit.pronunciation);
            auto const*                     lemma     = lemmaPron->lemma();

            // In known-excluding mode the fallback pieces do not carry their word-LM
            // event themselves: the search decides where the word boundary is and
            // whether the completed piece sequence is an exact known pronunciation.
            UnknownWordFallback::PieceRole const role =
                    excludeKnownWordsFromFallback_ ? unknownWordFallback_->roleOf(lemma) : UnknownWordFallback::NotFallback;
            if (role != UnknownWordFallback::NotFallback) {
                expandFallbackExit(hyp, hypIndex, exit, lemmaPron, role);
                continue;
            }

            WordLmEvent                         lmEvent;
            Score                               lmScore = 0;
            const Bliss::SyntacticTokenSequence sts     = lemma->syntacticTokenSequence();
            if (sts.size() != 0) {
                require(sts.size() == 1);
                lmEvent.token = sts.front();
                lmScore       = languageModel_->score(hyp.lmHistory, lmEvent.token);
            }

            Nn::TransitionType wordEndtransitionType = Nn::TransitionType::WORD_EXIT;
            if (lemma == blankLemma_) {
                wordEndtransitionType = Nn::TransitionType::BLANK_EXIT;
            }
            else if (lemma == silenceLemma_) {
                wordEndtransitionType = Nn::TransitionType::SILENCE_EXIT;
            }
            else if (nonWordLemmas_.contains(lemma)) {
                wordEndtransitionType = Nn::TransitionType::NONWORD_EXIT;
            }
            Score penalty = wordEndTransitionScore(hyp, wordEndtransitionType);

            // A completed ordinary word clears any fallback bookkeeping; blank,
            // silence and neutral exits leave a pending fallback word untouched.
            OovStateRef newOov = hyp.oov;
            if (lmEvent.token != nullptr and excludeKnownWordsFromFallback_) {
                newOov = emptyOovState();
            }

            wordEndExtensions_.push_back({
                    .pron           = lemmaPron,
                    .rootState      = exit.transitState,
                    .score          = hyp.score + lmScore + penalty,
                    .transitionType = wordEndtransitionType,
                    .baseHypIndex   = hypIndex,
                    .lmEvent        = lmEvent,
                    .oov            = newOov,
            });
        }
    }

    /*
     * Prune set of word-end extensions by score.
     */
    scorePruning(wordEndExtensions_, wordEndScoreThreshold_, wordEndExtensions_.size());
    numWordEndHypsAfterScorePruning_ += wordEndExtensions_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-word-end-hyps-after-score-pruning", wordEndExtensions_.size());
    }

    // Create new word-end label hypotheses from word-end extension candidates and update the LM history.
    // The history is advanced with exactly the token that was scored above, so score and
    // successor history can never disagree.
    wordEndHypotheses_.clear();
    for (auto& extension : wordEndExtensions_) {
        auto const& baseHyp = newBeam_[extension.baseHypIndex];

        auto newLmHistory = baseHyp.lmHistory;
        if (extension.lmEvent.token != nullptr) {
            newLmHistory = languageModel_->extendedHistory(newLmHistory, extension.lmEvent.token);
        }

        if (extension.oov and extension.lmEvent.token != nullptr) {
            if (extension.lmEvent.isUnknown) {
                numUnknownWordEvents_ += 1;
            }
            else if (unknownWordFallback_->isFallbackLemma(extension.pron->lemma())) {
                numKnownResolvedFallbackWords_ += 1;
            }
        }

        wordEndHypotheses_.push_back({baseHyp, extension, newLmHistory});
    }

    recombination(wordEndHypotheses_, true);
    numWordEndHypsAfterRecombination_ += wordEndHypotheses_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-word-end-hyps-after-recombination", wordEndHypotheses_.size());
    }

    // Prune set of word-end hypotheses by max beam size.
    scorePruning(wordEndHypotheses_, Core::Type<Score>::max, maxWordEndBeamSize_);
    numWordEndHypsAfterBeamPruning_ += wordEndHypotheses_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-word-end-hyps-after-beam-pruning", wordEndHypotheses_.size());
    }

    beam_.swap(newBeam_);
    beam_.insert(beam_.end(), wordEndHypotheses_.begin(), wordEndHypotheses_.end());

    numActiveHyps_ += beam_.size();

    ++currentSearchStep_;

    /*
     * Clean up label scorer caches and calculate number of active trees
     */
    std::vector<Lm::History> seenHistories;
    for (auto const& hyp : beam_) {
        if (std::find(seenHistories.begin(), seenHistories.end(), hyp.lmHistory) == seenHistories.end()) {
            seenHistories.push_back(hyp.lmHistory);
        }
    }
    if (currentSearchStep_ % cacheCleanupInterval_ == 0) {
        for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
            Core::CollapsedVector<Nn::ScoringContextRef> activeContexts;
            for (auto const& hyp : beam_) {
                activeContexts.push_back(hyp.scoringContexts[scorerIdx]);
            }
            labelScorers_[scorerIdx]->cleanupCaches(activeContexts);
        }
    }
    numActiveTrees_ += seenHistories.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-active-trees", seenHistories.size());
    }

    /*
     * Apply maximum-stable-delay-pruning.
     */
    if (currentSearchStep_ % maximumStableDelayPruningInterval_ == 0) {
        maximumStableDelayPruning();
        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("num-hyps-after-maximum-stable-delay-pruning", beam_.size());
        }
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

void TreeTimesyncBeamSearch::logStatistics() const {
    clog() << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    clog() << Core::XmlOpen("initialization-time") << initializationTime_.elapsedMilliseconds() << Core::XmlClose("initialization-time");
    clog() << Core::XmlOpen("feature-processing-time") << featureProcessingTime_.elapsedMilliseconds() << Core::XmlClose("feature-processing-time");
    clog() << Core::XmlOpen("scoring-time") << scoringTime_.elapsedMilliseconds() << Core::XmlClose("scoring-time");
    clog() << Core::XmlClose("timing-statistics");
    for (auto const& stat : numHypsAfterIntermediatePruning_) {
        stat.write(clog());
    }
    numHypsAfterRecombination_.write(clog());
    numHypsAfterPruning_.write(clog());
    numWordEndHypsAfterScorePruning_.write(clog());
    numWordEndHypsAfterRecombination_.write(clog());
    numWordEndHypsAfterBeamPruning_.write(clog());
    numActiveHyps_.write(clog());
    numActiveTrees_.write(clog());
    if (excludeKnownWordsFromFallback_) {
        // Counted over the word-end extensions that survived score pruning, plus the
        // segment-end finalizations. `num-known-resolved-fallback-words` are piece
        // sequences that spell an exact known pronunciation and were therefore scored
        // with their known LM token; in known-excluding mode none of them can reach
        // the unknown route.
        numUnknownWordEvents_.write(clog());
        numKnownResolvedFallbackWords_.write(clog());
    }
}

Nn::TransitionType TreeTimesyncBeamSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel, bool isSameState) const {
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
void TreeTimesyncBeamSearch::scorePruning(std::vector<Element>& hypotheses, Score relativeThreshold, size_t maxBeamSize) {
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

template void TreeTimesyncBeamSearch::scorePruning<TreeTimesyncBeamSearch::WithinWordExtensionCandidate>(std::vector<TreeTimesyncBeamSearch::WithinWordExtensionCandidate>&, Score, size_t);
template void TreeTimesyncBeamSearch::scorePruning<TreeTimesyncBeamSearch::WordEndExtensionCandidate>(std::vector<TreeTimesyncBeamSearch::WordEndExtensionCandidate>&, Score, size_t);

void TreeTimesyncBeamSearch::recombination(std::vector<TreeTimesyncBeamSearch::LabelHypothesis>& hypotheses, bool createTraceSiblings) {
    if (not recombinationEnabled_) {
        return;
    }

    /*
     * Represents a unique combination of StateId, ScoringContext, LmHistory and
     * open-vocabulary fallback state.
     *
     * The fallback state has to be part of the key because it decides which
     * continuations are legal (which pieces may follow, and whether the segment may
     * end here) and how the pending word will be scored once it is closed. Two
     * hypotheses that agree on all of this are interchangeable for everything that
     * follows, so under Viterbi semantics the better-scoring one survives and its
     * traceback stays a valid path -- even if the two spell their pending fallback
     * word differently.
     */
    struct RecombinationContext {
        StateId                            state;
        std::vector<Nn::ScoringContextRef> scoringContexts;
        Lm::History                        lmHistory;
        OovStateRef                        oov;

        RecombinationContext(LabelHypothesis const& hyp)
                : state(hyp.currentState), scoringContexts(hyp.scoringContexts), lmHistory(hyp.lmHistory), oov(hyp.oov) {}

        bool operator==(const RecombinationContext& other) const {
            if (state != other.state) {
                return false;
            }
            if (lmHistory != other.lmHistory) {
                return false;
            }
            if (oov.get() != other.oov.get() and not(oov and other.oov and *oov == *other.oov)) {
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
            if (context.oov) {
                hash = Core::combineHashes(hash, context.oov->numPieces * 2ul + (context.oov->diverged ? 1ul : 0ul));
                for (u32 node : context.oov->prefixNodes) {
                    hash = Core::combineHashes(hash, node);
                }
            }
            return hash;
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
            if (network_->isRoot(hyp.currentState)) {
                verify(not hyp.trace->sibling);
            }

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
    size_t numStates = network_->structure.stateCount();

    stateSuccessorsOffset_.assign(numStates + 1, 0);
    stateExitsOffset_.assign(numStates + 1, 0);

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

void TreeTimesyncBeamSearch::createPronunciationTrie() {
    pronunciationTrie_.assign(1ul, PronunciationTrieNode());  // node 0 is the root

    auto const* sentenceBeginLemma = lexicon_->specialLemma("sentence-begin");

    size_t numEntries = 0ul;
    for (auto iters = lexicon_->lemmaPronunciations(); iters.first != iters.second; ++iters.first) {
        auto const* lemmaPron = *iters.first;
        auto const* lemma     = lemmaPron->lemma();

        // Only ordinary lexical entries define known pronunciations. The fallback
        // pieces, blank, silence and the sentence boundaries do not.
        if (unknownWordFallback_->isExcludedFromOrdinaryTree(lemma) or lemma == blankLemma_ or
            lemma == silenceLemma_ or lemma == sentenceEndLemma_ or lemma == sentenceBeginLemma) {
            continue;
        }

        auto const* pron = lemmaPron->pronunciation();
        if (pron == nullptr or pron->length() == 0) {
            continue;
        }

        u32 node = 0u;
        for (u32 i = 0u; i < pron->length(); ++i) {
            Bliss::Phoneme::Id const phoneme  = (*pron)[i];
            auto&                    children = pronunciationTrie_[node].children;
            auto                     it       = std::lower_bound(children.begin(), children.end(), phoneme,
                                                                 [](std::pair<Bliss::Phoneme::Id, u32> const& child, Bliss::Phoneme::Id id) {
                                           return child.first < id;
                                       });
            if (it != children.end() and it->first == phoneme) {
                node = it->second;
            }
            else {
                u32 const child = static_cast<u32>(pronunciationTrie_.size());
                pronunciationTrie_.emplace_back();
                // `children` may dangle after the reallocation above, so look it up again.
                auto& parentChildren = pronunciationTrie_[node].children;
                parentChildren.insert(
                        std::lower_bound(parentChildren.begin(), parentChildren.end(), phoneme,
                                         [](std::pair<Bliss::Phoneme::Id, u32> const& c, Bliss::Phoneme::Id id) { return c.first < id; }),
                        {phoneme, child});
                node = child;
            }
        }

        auto& completed = pronunciationTrie_[node].completedLemmas;
        if (std::find(completed.begin(), completed.end(), lemma) == completed.end()) {
            completed.push_back(lemma);
        }
        ++numEntries;
    }

    log() << "Built known-pronunciation trie with " << pronunciationTrie_.size() << " nodes from " << numEntries << " lexical entries";
}

TreeTimesyncBeamSearch::OovStateRef TreeTimesyncBeamSearch::advanceOovState(OovStateRef const& base, Bliss::Pronunciation const& piece) const {
    auto next       = Core::ref(new OovState());
    next->numPieces = base->numPieces + 1u;

    for (u32 node : base->prefixNodes) {
        u32  current = node;
        bool alive   = true;
        for (u32 i = 0u; alive and i < piece.length(); ++i) {
            Bliss::Phoneme::Id const phoneme  = piece[i];
            auto const&              children = pronunciationTrie_[current].children;
            auto                     it       = std::lower_bound(children.begin(), children.end(), phoneme,
                                                                 [](std::pair<Bliss::Phoneme::Id, u32> const& c, Bliss::Phoneme::Id id) { return c.first < id; });
            if (it != children.end() and it->first == phoneme) {
                current = it->second;
            }
            else {
                alive = false;
            }
        }
        if (alive) {
            next->prefixNodes.push_back(current);
        }
    }

    std::sort(next->prefixNodes.begin(), next->prefixNodes.end());
    next->prefixNodes.erase(std::unique(next->prefixNodes.begin(), next->prefixNodes.end()), next->prefixNodes.end());

    // Once no known pronunciation spells the pending pieces any more, none ever will:
    // a later piece cannot re-enter the trie in the middle of a word.
    next->diverged = base->diverged or next->prefixNodes.empty();
    return next;
}

void TreeTimesyncBeamSearch::collectKnownLemmas(OovState const& oov, std::vector<Bliss::Lemma const*>& knownLemmas) const {
    knownLemmas.clear();
    if (oov.diverged or not oov.wordPending()) {
        return;
    }

    for (u32 node : oov.prefixNodes) {
        for (auto const* lemma : pronunciationTrie_[node].completedLemmas) {
            knownLemmas.push_back(lemma);
        }
    }
}

void TreeTimesyncBeamSearch::resolveWordLmEvents(OovState const& oov, std::vector<WordLmEvent>& events) const {
    events.clear();
    if (not oov.wordPending()) {
        // An empty pending word is not a word: trailing separators and empty input
        // must not produce an unknown word.
        return;
    }

    collectKnownLemmas(oov, knownLemmaBuffer_);
    for (auto const* lemma : knownLemmaBuffer_) {
        auto const& sts = lemma->syntacticTokenSequence();
        WordLmEvent event;
        event.token = sts.size() == 1 ? sts.front() : nullptr;
        // Several known interpretations can carry the same LM token (pronunciation
        // variants, homophones). They would produce identical hypotheses, so keep one.
        if (std::none_of(events.begin(), events.end(), [&](WordLmEvent const& seen) { return seen.token == event.token; })) {
            events.push_back(event);
        }
    }

    if (not events.empty()) {
        // The piece sequence is an exact known pronunciation, so the unknown route is
        // disallowed for it regardless of LM scale or unknown reward, and the fallback
        // hypothesis is resolved into its known lexical interpretation(s) instead.
        return;
    }

    WordLmEvent unknownEvent;
    unknownEvent.token       = unknownSyntacticToken_;
    unknownEvent.unknownBias = unknownWordPenalty_;
    unknownEvent.isUnknown   = true;
    events.push_back(unknownEvent);
}

Score TreeTimesyncBeamSearch::wordEndTransitionScore(LabelHypothesis const& hyp, Nn::TransitionType transitionType) const {
    Score score = 0.0;
    for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
        if (not labelScorers_[scorerIdx]->scoresTransition(transitionType)) {
            continue;
        }
        auto scoreAccessor = labelScorers_[scorerIdx]->getScoreAccessor(hyp.scoringContexts[scorerIdx]);
        if (not scoreAccessor) {
            continue;
        }
        score += (*scoreAccessor)->getScore(transitionType);
    }
    return score;
}

void TreeTimesyncBeamSearch::expandFallbackExit(LabelHypothesis const&           hyp,
                                                size_t                           hypIndex,
                                                PersistentStateTree::Exit const& exit,
                                                Bliss::LemmaPronunciation const* lemmaPron,
                                                UnknownWordFallback::PieceRole   role) {
    Bliss::Pronunciation const& piece = *lemmaPron->pronunciation();

    // Under the word-start-marked convention a word-start piece belongs to the *next*
    // word and therefore closes the pending one before being consumed. Under the
    // continuation-marked convention a final piece closes the word it belongs to.
    bool const closesWordBefore = unknownWordFallback_->closesWordBefore(role) and hyp.oov->wordPending();
    bool const closesWordAfter  = unknownWordFallback_->closesWordAfter(role);

    OovStateRef eventOov;  // the pending word this exit completes, if any
    OovStateRef pendingAfterExit;

    if (closesWordBefore) {
        eventOov         = hyp.oov;
        pendingAfterExit = advanceOovState(emptyOovState(), piece);
    }
    else {
        pendingAfterExit = advanceOovState(hyp.oov, piece);
        if (closesWordAfter) {
            eventOov         = pendingAfterExit;
            pendingAfterExit = emptyOovState();
        }
    }

    if (eventOov) {
        resolveWordLmEvents(*eventOov, wordLmEventBuffer_);
    }
    else {
        wordLmEventBuffer_.assign(1ul, WordLmEvent{});
    }

    // Only a completed word is a word: an exit which merely appends a piece to the
    // pending word must not collect the word-exit reward a second time. A completed
    // word gets it on both routes alike, also when it resolved to a known lemma whose
    // syntactic token sequence is empty.
    Nn::TransitionType const transitionType  = eventOov ? Nn::TransitionType::WORD_EXIT : Nn::TransitionType::NONWORD_EXIT;
    Score const              transitionScore = wordEndTransitionScore(hyp, transitionType);

    for (auto const& event : wordLmEventBuffer_) {
        Score lmScore = 0.0;
        if (event.token != nullptr) {
            lmScore = languageModel_->score(hyp.lmHistory, event.token);
        }

        wordEndExtensions_.push_back({
                .pron           = lemmaPron,
                .rootState      = exit.transitState,
                .score          = hyp.score + lmScore + event.unknownBias + transitionScore,
                .transitionType = transitionType,
                .baseHypIndex   = hypIndex,
                .lmEvent        = event,
                .oov            = pendingAfterExit,
        });
    }
}

void TreeTimesyncBeamSearch::finalizeHypotheses() {
    tempHypotheses_.clear();
    for (auto const& hyp : beam_) {
        if (network_->finalStates.contains(hyp.currentState)) {
            tempHypotheses_.push_back(hyp);
        }
    }

    if (tempHypotheses_.empty() and sentenceEndFallback_) {  // There was no valid final hypothesis in the beam
        warning("No active word-end hypothesis at segment end.");
        log() << "Use sentence-end fallback";
        // The trace of the unfinished word keeps an empty pronunciation
        tempHypotheses_ = beam_;
    }

    if (not tempHypotheses_.empty()) {
        withinWordExtensions_.clear();
        for (size_t hypIndex = 0ul; hypIndex < tempHypotheses_.size(); ++hypIndex) {
            auto& hyp = tempHypotheses_[hypIndex];
            withinWordExtensions_.push_back(
                    {sentenceEndLabelIndex_,
                     hyp.currentState,
                     hyp.trace->time,
                     hyp.score,
                     Nn::TransitionType::SENTENCE_END,
                     hypIndex});
        }

        // Score sentence-end with all label scorers
        for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
            if (not labelScorers_[scorerIdx]->scoresTransition(Nn::TransitionType::SENTENCE_END)) {
                continue;
            }

            scoringContexts_.clear();
            for (auto const& hyp : tempHypotheses_) {
                scoringContexts_.push_back(hyp.scoringContexts[scorerIdx]);
            }

            scoringTime_.start();
            auto scoreAccessors = labelScorers_[scorerIdx]->getScoreAccessors(scoringContexts_);
            scoringTime_.stop();
            std::vector<std::optional<Nn::DenseScoreSpan>> denseScoreSpans(scoreAccessors.size(), std::nullopt);
            std::vector<Nn::TimeframeIndex>                scoreTimes(scoreAccessors.size(), 0);
            for (size_t accessorIdx = 0ul; accessorIdx < scoreAccessors.size(); ++accessorIdx) {
                if (scoreAccessors[accessorIdx]) {
                    denseScoreSpans[accessorIdx] = (*scoreAccessors[accessorIdx])->getDenseScores();
                    scoreTimes[accessorIdx]      = (*scoreAccessors[accessorIdx])->getTime();
                }
            }

            for (size_t extensionIdx = 0ul; extensionIdx < withinWordExtensions_.size(); ++extensionIdx) {
                if (not scoreAccessors[extensionIdx]) {
                    continue;
                }
                auto& ext = withinWordExtensions_[extensionIdx];
                if (denseScoreSpans[extensionIdx] and sentenceEndLabelIndex_ < denseScoreSpans[extensionIdx]->size()) {
                    ext.score += (*denseScoreSpans[extensionIdx])[sentenceEndLabelIndex_];
                }
                else {
                    ext.score += (*scoreAccessors[extensionIdx])->getScore(ext.transitionType, sentenceEndLabelIndex_);
                }
                ext.timeframe = std::max(ext.timeframe, scoreTimes[extensionIdx]);
            }
        }

        newBeam_.clear();
        for (size_t extensionIdx = 0ul; extensionIdx < withinWordExtensions_.size(); ++extensionIdx) {
            auto&       ext     = withinWordExtensions_[extensionIdx];
            auto const& baseHyp = tempHypotheses_[ext.baseHypIndex];
            // The scoring context is not updated as no further scoring is done afterwards
            newBeam_.push_back({baseHyp, ext, baseHyp.scoringContexts});
        }

        wordEndExtensions_.clear();
        for (size_t hypIndex = 0ul; hypIndex < newBeam_.size(); ++hypIndex) {
            auto& hyp = newBeam_[hypIndex];

            // A fallback word which is still pending is closed by the segment end,
            // exactly once, before the sentence-end score is taken. A pending word of
            // zero pieces is not a word, so a trailing separator or empty input adds
            // no unknown word here.
            if (hyp.oov and hyp.oov->wordPending()) {
                resolveWordLmEvents(*hyp.oov, wordLmEventBuffer_);
                for (auto const& event : wordLmEventBuffer_) {
                    Lm::History pendingHistory = hyp.lmHistory;
                    Score       pendingScore   = event.unknownBias;
                    if (event.token != nullptr) {
                        pendingScore += languageModel_->score(hyp.lmHistory, event.token);
                        pendingHistory = languageModel_->extendedHistory(hyp.lmHistory, event.token);
                    }
                    if (event.isUnknown) {
                        numUnknownWordEvents_ += 1;
                    }
                    else {
                        numKnownResolvedFallbackWords_ += 1;
                    }

                    wordEndExtensions_.push_back({
                            .pron           = sentenceEndLemma_->pronunciations().first,
                            .rootState      = hyp.currentState,
                            .score          = hyp.score + pendingScore + languageModel_->sentenceEndScore(pendingHistory),
                            .transitionType = Nn::TransitionType::SENTENCE_END,
                            .baseHypIndex   = hypIndex,
                            .lmEvent        = event,
                            .oov            = emptyOovState(),
                    });
                }
                continue;
            }

            // Add the LM's sentence-end score
            // The LM history is not updated as this is the last LM scoring step
            Lm::Score sentenceEndScore = languageModel_->sentenceEndScore(hyp.lmHistory);
            wordEndExtensions_.push_back({
                    .pron           = sentenceEndLemma_->pronunciations().first,
                    .rootState      = hyp.currentState,
                    .score          = hyp.score + sentenceEndScore,
                    .transitionType = Nn::TransitionType::SENTENCE_END,
                    .baseHypIndex   = hypIndex,
                    .lmEvent        = WordLmEvent{},
                    .oov            = hyp.oov,
            });
        }

        tempHypotheses_.clear();
        for (size_t extensionIdx = 0ul; extensionIdx < wordEndExtensions_.size(); ++extensionIdx) {
            auto&       ext     = wordEndExtensions_[extensionIdx];
            auto const& baseHyp = newBeam_[ext.baseHypIndex];
            // The LM history is not advanced any further: this is the last LM scoring
            // step. The whole difference to the base score (the finalization of a
            // pending fallback word plus the sentence-end score) is attributed to the
            // language model in the resulting trace.
            tempHypotheses_.push_back({baseHyp, ext, baseHyp.lmHistory});
        }
    }
    else {  // No valid final hypotheses and no sentence-end fallback
        // Construct an empty hypothesis with a lattice containing only one empty pronunciation from start to end
        tempHypotheses_.push_back(LabelHypothesis());
        tempHypotheses_.front().trace->time          = beam_.front().trace->time;  // Retrieve the timeframe from any hyp in the old beam
        tempHypotheses_.front().trace->pronunciation = nullptr;
        tempHypotheses_.front().trace->predecessor   = Core::ref(new LatticeTrace(0, {0, 0}, {}));
    }

    beam_.swap(tempHypotheses_);

    numActiveHyps_ += beam_.size();

    // Log statistics about the final beam
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("active-hyps", beam_.size());
        clog() << Core::XmlFull("best-hyp-score", getBestHypothesis().score);
        clog() << Core::XmlFull("worst-hyp-score", getWorstHypothesis().score);
        clog() << Core::XmlClose("search-step-stats");
    }

    if (debugChannel_.isOpen()) {
        std::stringstream ss;
        for (size_t hypIdx = 0ul; hypIdx < beam_.size(); ++hypIdx) {
            ss << "Hypothesis " << hypIdx + 1ul << ":  " << beam_[hypIdx].toString() << "\n";
        }
        ss << "\n";
        debugChannel_ << ss.str();
    }
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
