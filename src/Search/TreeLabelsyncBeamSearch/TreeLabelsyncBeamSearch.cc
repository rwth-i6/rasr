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
#include <numeric>
#include <strings.h>

#include <Am/ClassicStateModel.hh>
#include <Core/CollapsedVector.hh>
#include <Core/XmlStream.hh>
#include <Lattice/LatticeAdaptor.hh>
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

}  // namespace

/*
 * =======================
 * === LabelHypothesis ===
 * =======================
 */

TreeLabelsyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContexts(),
          currentToken(Nn::invalidLabelIndex),
          currentState(invalidTreeNodeIndex),
          lmHistory(),
          timeframe(0),
          length(0),
          score(0.0),
          scaledScore(0.0),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))),
          isActive(true) {}

TreeLabelsyncBeamSearch::LabelHypothesis::LabelHypothesis(
        TreeLabelsyncBeamSearch::LabelHypothesis const&              base,
        TreeLabelsyncBeamSearch::WithinWordExtensionCandidate const& extension,
        std::vector<Nn::ScoringContextRef> const&                    newScoringContexts,
        float                                                        lengthNormScale)
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

TreeLabelsyncBeamSearch::LabelHypothesis::LabelHypothesis(
        LabelHypothesis const&                                    base,
        TreeLabelsyncBeamSearch::WordEndExtensionCandidate const& extension,
        Lm::History const&                                        newLmHistory,
        float                                                     lengthNormScale)
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

const Core::ParameterIntVector TreeLabelsyncBeamSearch::paramMaxBeamSizes(
        "max-beam-size",
        "Maximum number of elements in the search beam. Pruning is applied after each intermediate label scorer.",
        "",
        1);

const Core::ParameterInt TreeLabelsyncBeamSearch::paramMaxWordEndBeamSize(
        "max-word-end-beam-size",
        "Maximum number of word-end hypotheses in the search beam. If not set, global beam pruning will be done and word-end hypotheses will not be pruned separately.",
        std::numeric_limits<int>::max(), 0);

const Core::ParameterFloatVector TreeLabelsyncBeamSearch::paramScoreThresholds(
        "score-threshold",
        "Prune any hypotheses with a score that is at least this much worse than the best hypothesis."
        "If length normalization is enabled, the score threshold is added to the raw score before normalization."
        "Pruning is applied after each intermediate label scorer."
        "If not set, no score pruning will be done.",
        "",
        0,
        Core::Type<Score>::max);

const Core::ParameterFloat TreeLabelsyncBeamSearch::paramWordEndScoreThreshold(
        "word-end-score-threshold",
        "Prune any word-end hypothesis with a score that is at least this much worse than the best word-end hypothesis. This threshold is relative to the score-threshold. \
        If not set, global score pruning will be done and word-end hypotheses will not be pruned separately.",
        Core::Type<Score>::max, 0);

const Core::ParameterInt TreeLabelsyncBeamSearch::paramNumHistogramBins(
        "num-histogram-bins",
        "Number of bins for histogram pruning of hypotheses (very minor effect).",
        100,
        2);

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

const Core::Choice TreeLabelsyncBeamSearch::choiceRecombinationMode(
        "off", RecombinationModeOff,
        "on", RecombinationModeOn,
        Core::Choice::endMark());

const Core::ParameterChoice TreeLabelsyncBeamSearch::paramRecombinationMode(
        "recombination-mode",
        &choiceRecombinationMode,
        "Whether hypotheses with identical recombination state should be recombined.",
        RecombinationModeOn);

const Core::ParameterBool TreeLabelsyncBeamSearch::paramLogStepwiseStatistics(
        "log-stepwise-statistics",
        "Log statistics about the beam at every search step.",
        false);

const Core::ParameterInt TreeLabelsyncBeamSearch::paramCacheCleanupInterval(
        "cache-cleanup-interval",
        "Interval of search steps after which buffered inputs that are not needed anymore get cleaned up.",
        10);

const Core::ParameterInt TreeLabelsyncBeamSearch::paramMaximumStableDelay(
        "maximum-stable-delay",
        "Introduce a cutoff point at `current-time` - `delay`. Every hypothesis that disagrees with the current best anywhere before the cutoff gets pruned."
        "This way words in the traceback become stable after at most `delay` frames.",
        Core::Type<int>::max,
        0);

const Core::ParameterInt TreeLabelsyncBeamSearch::paramMaximumStableDelayPruningInterval(
        "maximum-stable-delay-pruning-interval",
        "Interval of search steps after which the maximum-stable-delay-pruning gets applied.",
        10,
        1);

TreeLabelsyncBeamSearch::TreeLabelsyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          maxWordEndBeamSize_(paramMaxWordEndBeamSize(config)),
          wordEndScoreThreshold_(paramWordEndScoreThreshold(config)),
          scoreHistogram_(paramNumHistogramBins(config)),
          lengthNormScale_(paramLengthNormScale(config)),
          maxLabelsPerTimestep_(paramMaxLabelsPerTimestep(config)),
          sentenceEndLabelIndex_(Nn::invalidLabelIndex),
          cacheCleanupInterval_(paramCacheCleanupInterval(config)),
          maximumStableDelay_(paramMaximumStableDelay(config)),
          maximumStableDelayPruningInterval_(paramMaximumStableDelayPruningInterval(config)),
          sentenceEndFallback_(paramSentenceEndFallBack(config)),
          recombinationEnabled_(paramRecombinationMode(config) == RecombinationModeOn),
          logStepwiseStatistics_(paramLogStepwiseStatistics(config)),
          labelScorers_(),
          nonWordLemmas_(),
          debugChannel_(config, "debug"),
          hypIndexToContextIndexMap_(),
          withinWordExtensions_(),
          wordEndExtensions_(),
          beam_(),
          newBeam_(),
          wordEndHypotheses_(),
          scoringContexts_(),
          tempHypotheses_(),
          currentSearchStep_(0ul),
          totalTimesteps_(0ul),
          finishedSegment_(false),
          initializationTime_(),
          featureProcessingTime_(),
          scoringTime_(),
          numHypsAfterIntermediatePruning_(),
          numTerminatedHypsAfterScorePruning_("num-termianted-hyps-after-score-pruning"),
          numTerminatedHypsAfterRecombination_("num-terminated-hyps-after-recombination"),
          numTerminatedHypsAfterBeamPruning_("num-terminated-hyps-after-beam-pruning"),
          numActiveHypsAfterScorePruning_("num-active-hyps-after-score-pruning"),
          numActiveHypsAfterRecombination_("num-active-hyps-after-recombination"),
          numActiveHypsAfterBeamPruning_("num-active-hyps-after-beam-pruning"),
          numActiveWordEndHypsAfterIntermediatePruning_("num-word-end-hyps-after-intermediate-pruning"),
          numActiveWordEndHypsAfterScorePruning_("num-active-word-end-hyps-after-score-pruning"),
          numActiveWordEndHypsAfterRecombination_("num-active-word-end-hyps-after-recombination"),
          numActiveWordEndHypsAfterBeamPruning_("num-active-word-end-hyps-after-beam-pruning"),
          numActiveTrees_("num-active-trees") {
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

Speech::ModelCombination::Mode TreeLabelsyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon | Speech::ModelCombination::useAcousticModel | Speech::ModelCombination::useLanguageModel;
}

Speech::ModelCombination::Mode TreeLabelsyncBeamSearch::requiredAcousticModel() const {
    return Am::AcousticModel::noEmissions;
}

bool TreeLabelsyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
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

    auto sentenceEndLemma = lexicon_->specialLemma("sentence-end");
    if (not sentenceEndLemma) {
        sentenceEndLemma = lexicon_->specialLemma("sentence-boundary");
    }
    if (sentenceEndLemma and sentenceEndLemma->nPronunciations() != 0 and sentenceEndLemma->pronunciations().first->pronunciation()->length() > 0) {
        auto const* pron = sentenceEndLemma->pronunciations().first->pronunciation();
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

    for (const auto& lemma : {"silence"}) {
        if (lexicon_->specialLemma(lemma) and (lexicon_->specialLemma(lemma)->syntacticTokenSequence()).size() != 0) {
            warning("Special lemma \"%s\" will be scored by the language model. To prevent the LM from scoring it, set an empty syntactic token sequence for it in the lexicon.", lemma);
        }
    }

    // Create look-ups for state successors and exits of each state
    createSuccessorLookups();

    return true;
}

void TreeLabelsyncBeamSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    initializationTime_.reset();
    featureProcessingTime_.reset();
    scoringTime_.reset();
    for (auto& stat : numHypsAfterIntermediatePruning_) {
        stat.clear();
    }
    numTerminatedHypsAfterScorePruning_.clear();
    numTerminatedHypsAfterRecombination_.clear();
    numTerminatedHypsAfterBeamPruning_.clear();
    numActiveHypsAfterScorePruning_.clear();
    numActiveHypsAfterRecombination_.clear();
    numActiveHypsAfterBeamPruning_.clear();
    numActiveWordEndHypsAfterIntermediatePruning_.clear();
    numActiveWordEndHypsAfterScorePruning_.clear();
    numActiveWordEndHypsAfterRecombination_.clear();
    numActiveWordEndHypsAfterBeamPruning_.clear();
    numActiveTrees_.clear();

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

    if (segment != nullptr) {
        languageModel_->setSegment(segment);
        for (auto& hyp : beam_) {
            hyp.lmHistory = languageModel_->startHistory();
        }
    }

    finishedSegment_   = false;
    totalTimesteps_    = 0ul;
    currentSearchStep_ = 0ul;

    initializationTime_.stop();
}

void TreeLabelsyncBeamSearch::finishSegment() {
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

void TreeLabelsyncBeamSearch::putFeature(Nn::DataView const& feature) {
    featureProcessingTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInput(feature);
    }
    ++totalTimesteps_;
    featureProcessingTime_.stop();
}

void TreeLabelsyncBeamSearch::putFeatures(Nn::DataView const& features, size_t nTimesteps) {
    featureProcessingTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInputs(features, nTimesteps);
    }
    totalTimesteps_ += nTimesteps;
    featureProcessingTime_.stop();
}

Core::Ref<const Traceback> TreeLabelsyncBeamSearch::getCurrentBestTraceback() const {
    return getBestHypothesis(beam_).trace->performTraceback();
}

Core::Ref<const LatticeAdaptor> TreeLabelsyncBeamSearch::getCurrentBestWordLattice() const {
    auto& bestHypothesis = getBestHypothesis(beam_);

    LatticeTrace endTrace(bestHypothesis.trace, 0, bestHypothesis.trace->time + 1, bestHypothesis.trace->score, {});

    for (auto const& hyp : beam_) {
        if (&hyp == &bestHypothesis or hyp.isActive != bestHypothesis.isActive) {
            continue;
        }
        auto siblingTrace = Core::ref(new LatticeTrace(hyp.trace, 0, hyp.trace->time, hyp.trace->score, {}));
        endTrace.appendSiblingToChain(siblingTrace);
    }

    return endTrace.buildWordLattice(lexicon_);
}

Core::Ref<const LatticeTrace> TreeLabelsyncBeamSearch::getCurrentBestLatticeTrace() const {
    return getBestHypothesis(beam_).trace;
}

Core::Ref<const LatticeTrace> TreeLabelsyncBeamSearch::getCommonPrefix() const {
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

bool TreeLabelsyncBeamSearch::decodeStep() {
    if (finishedSegment_) {
        return false;
    }
    if (currentSearchStep_ >= maxLabelsPerTimestep_ * std::max(totalTimesteps_, 1ul)) {
        warning() << "Terminated search due to reaching max number of label outputs given input count";
        finishedSegment_ = true;
        return false;
    }
    if (std::all_of(beam_.begin(), beam_.end(), [](LabelHypothesis const& hyp) { return not hyp.isActive; })) {
        log() << "Terminated search because all hypotheses reached sentence-end";
        finishedSegment_ = true;
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

    // Only the scoring contexts of active hypotheses need to be forwarded; collect them here
    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        auto& hyp = beam_[hypIndex];

        if (not hyp.isActive) {
            continue;
        }

        hypIndexToContextIndexMap_[hypIndex] = scoringContexts_.size();
        scoringContexts_.push_back(hyp.scoringContexts.front());
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

        if (scorerIdx == 0ul) {
            // In the first iteration, create extensions while pre-pruning
            Score currentBestScore = Core::Type<Score>::max;

            for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
                auto const hyp = beam_[hypIndex];

                if (not hyp.isActive) {
                    continue;
                }

                auto const& scoreAccessor = scoreAccessors[hypIndexToContextIndexMap_[hypIndex]];

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
                        extScore += (*scoreAccessor)->getScore(transitionType, tokenIdx);
                        extTime = std::max(extTime, (*scoreAccessor)->getTime());
                    }

                    // Pre-prune based on score before creating extension instance and appending to list
                    if (useScorePruning_.front() and extScore > currentBestScore + scoreThresholds_.front()) {
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
                    ext.score += (*scoreAccessor)->getScore(ext.transitionType, ext.nextToken);
                    ext.timeframe = std::max(ext.timeframe, (*scoreAccessor)->getTime());
                }
                else {
                    // Extension is not scorable so set the score to max in order to prune it later
                    ext.score = Core::Type<Score>::max;
                }
            }
        }

        if (withinWordExtensions_.empty()) {
            if (logStepwiseStatistics_) {
                clog() << Core::XmlClose("search-step-stats");
            }
            return false;
        }

        /*
         * Prune set of possible extensions by max beam size and possibly also by score.
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

        if (scorerIdx < labelScorers_.size() - 1) {
            // Prepare scoring context list for next iteration
            // Some scoring contexts from the current iteration may not have survived pruning, so we need to recreate the list
            // Use -1 as placeholder to signify that this hyp was not visited yet
            scoringContexts_.clear();
            hypIndexToContextIndexMap_.assign(beam_.size(), -1);
            for (auto const& ext : withinWordExtensions_) {
                if (hypIndexToContextIndexMap_[ext.baseHypIndex] == -1) {
                    hypIndexToContextIndexMap_[ext.baseHypIndex] = scoringContexts_.size();
                    scoringContexts_.push_back(beam_[ext.baseHypIndex].scoringContexts[scorerIdx + 1]);
                }
            }
        }
    }

    /*
     * Create new beam from surviving extensions.
     */
    newBeam_.clear();

    for (auto const& hyp : beam_) {
        if (not hyp.isActive) {
            newBeam_.push_back(hyp);
        }
    }

    for (auto const& extension : withinWordExtensions_) {
        auto const& baseHyp = beam_[extension.baseHypIndex];

        std::vector<Nn::ScoringContextRef> newScoringContexts;
        newScoringContexts.reserve(labelScorers_.size());
        for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
            newScoringContexts.push_back(labelScorers_[scorerIdx]->extendedScoringContext(
                    baseHyp.scoringContexts[scorerIdx],
                    extension.nextToken,
                    extension.transitionType));
        }
        newBeam_.push_back({baseHyp, extension, newScoringContexts, lengthNormScale_});
    }

    /*
     * Expand hypotheses to word-end hypotheses and incorporate the language model
     */
    wordEndExtensions_.clear();
    for (size_t hypIndex = 0ul; hypIndex < newBeam_.size(); ++hypIndex) {
        auto& hyp = newBeam_[hypIndex];

        // Terminated hypothesis can't yield word-end hypotheses
        if (not hyp.isActive) {
            continue;
        }

        // Create one word-end hypothesis for each exit
        for (size_t i = stateExitsOffset_[hyp.currentState]; i < stateExitsOffset_[hyp.currentState + 1]; ++i) {
            const PersistentStateTree::Exit exit      = stateExits_[i];
            auto const*                     lemmaPron = lexicon_->lemmaPronunciation(exit.pronunciation);
            auto const*                     lemma     = lemmaPron->lemma();

            Score                               lmScore = 0;
            const Bliss::SyntacticTokenSequence sts     = lemma->syntacticTokenSequence();
            if (sts.size() != 0) {
                require(sts.size() == 1);
                auto const* st = sts.front();
                lmScore        = languageModel_->score(hyp.lmHistory, st);
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
                auto scoreAccessor = labelScorers_[scorerIdx]->getScoreAccessor(hyp.scoringContexts[scorerIdx]);
                if (not scoreAccessor) {
                    continue;
                }
                penalty += (*scoreAccessor)->getScore(wordEndtransitionType);
            }

            wordEndExtensions_.push_back({
                    .pron           = lemmaPron,
                    .rootState      = exit.transitState,
                    .score          = hyp.score + lmScore + penalty,
                    .timeframe      = hyp.timeframe,
                    .transitionType = wordEndtransitionType,
                    .baseHypIndex   = hypIndex,
            });
        }
    }

    /*
     * Prune set of word-end hypotheses by max beam size and possibly also by score.
     */
    scorePruning(wordEndExtensions_, wordEndScoreThreshold_, maxWordEndBeamSize_);
    numActiveWordEndHypsAfterIntermediatePruning_ += wordEndExtensions_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-word-end-hyps-after-intermediate-pruning", wordEndExtensions_.size());
    }

    // Create new word-end label hypotheses from word-end extension candidates and update the LM history
    wordEndHypotheses_.clear();
    for (auto& extension : wordEndExtensions_) {
        auto const& baseHyp = newBeam_[extension.baseHypIndex];

        auto        newLmHistory = baseHyp.lmHistory;
        auto const& sts          = extension.pron->lemma()->syntacticTokenSequence();

        if (sts.size() != 0) {
            require(sts.size() == 1);
            const Bliss::SyntacticToken* st = sts.front();
            newLmHistory                    = languageModel_->extendedHistory(newLmHistory, st);
        }

        wordEndHypotheses_.push_back({baseHyp, extension, newLmHistory, lengthNormScale_});
    }

    newBeam_.insert(newBeam_.end(), wordEndHypotheses_.begin(), wordEndHypotheses_.end());

    /*
     * Jointly prune terminated and active hypotheses by score
     */
    if (not useScorePruning_.empty() and useScorePruning_.back()) {
        auto relativeThreshold = scoreThresholds_.back();
        if (lengthNormScale_ != 0) {
            relativeThreshold /= std::pow(getBestHypothesis(newBeam_).length, lengthNormScale_);
        }
        scorePruning(newBeam_, relativeThreshold, newBeam_.size());

        size_t numActive        = numActiveHyps();
        size_t numActiveWordEnd = numActiveWordEndHyps();
        numTerminatedHypsAfterScorePruning_ += newBeam_.size() - numActive;
        numActiveHypsAfterScorePruning_ += numActive;
        numActiveWordEndHypsAfterScorePruning_ += numActiveWordEnd;
        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("num-terminated-hyps-after-score-pruning", newBeam_.size() - numActive);
            clog() << Core::XmlFull("num-active-hyps-after-score-pruning", numActive);
            clog() << Core::XmlFull("num-active-word-end-hyps-after-score-pruning", numActiveWordEnd);
        }
    }

    /*
     * For all hypotheses at the same state and with the same scoring context and LM history keep
     * only the best since they will all develop in the same way.
     */
    recombination(newBeam_);

    size_t numActive        = numActiveHyps();
    size_t numActiveWordEnd = numActiveWordEndHyps();
    numTerminatedHypsAfterRecombination_ += newBeam_.size() - numActive;
    numActiveHypsAfterRecombination_ += numActive;
    numActiveWordEndHypsAfterRecombination_ += numActiveWordEnd;
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-terminated-hyps-after-recombination", newBeam_.size() - numActive);
        clog() << Core::XmlFull("num-active-hyps-after-recombination", numActive);
        clog() << Core::XmlFull("num-active-word-end-hyps-after-recombination", numActiveWordEnd);
    }

    scorePruning(newBeam_, Core::Type<Score>::max, maxBeamSizes_.back());

    numActive        = numActiveHyps();
    numActiveWordEnd = numActiveWordEndHyps();
    numTerminatedHypsAfterBeamPruning_ += newBeam_.size() - numActive;
    numActiveHypsAfterBeamPruning_ += numActive;
    numActiveWordEndHypsAfterBeamPruning_ += numActiveWordEnd;
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-terminated-hyps-after-beam-pruning", newBeam_.size() - numActive);
        clog() << Core::XmlFull("num-active-hyps-after-beam-pruning", numActive);
        clog() << Core::XmlFull("num-active-word-end-hyps-after-beam-pruning", numActiveWordEnd);
    }

    beam_.swap(newBeam_);
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
     * Log statistics about the new beam after this step.
     */

    if (debugChannel_.isOpen()) {
        std::stringstream ssActive;
        std::stringstream ssTerminated;
        for (size_t hypIdx = 0ul; hypIdx < beam_.size(); ++hypIdx) {
            auto const& hyp = beam_[hypIdx];
            if (not hyp.isActive) {
                ssTerminated << "Terminated hypothesis " << hypIdx + 1ul << ":  " << beam_[hypIdx].toString() << "\n";
            }
            else {
                ssActive << "Active hypothesis " << hypIdx + 1ul << ":  " << beam_[hypIdx].toString() << "\n";
            }
        }
        ssActive << "\n";
        ssTerminated << "\n";
        debugChannel_ << ssActive.str() << ssTerminated.str();
    }

    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("terminated-hyps", beam_.size() - numActive);
        clog() << Core::XmlFull("active-hyps", numActive);
        auto const* bestTerminatedHyp  = getBestTerminatedHypothesis(beam_);
        auto const* worstTerminatedHyp = getWorstTerminatedHypothesis(beam_);
        auto const* bestActiveHyp      = getBestActiveHypothesis(beam_);
        auto const* worstActiveHyp     = getWorstActiveHypothesis(beam_);
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

    return true;
}

TreeLabelsyncBeamSearch::LabelHypothesis const* TreeLabelsyncBeamSearch::getBestTerminatedHypothesis(std::vector<LabelHypothesis> const& hypotheses) const {
    LabelHypothesis const* best = nullptr;

    for (auto const& hyp : hypotheses) {
        if (not hyp.isActive) {
            if (best == nullptr or hyp < *best) {
                best = &hyp;
            }
        }
    }

    return best;
}

TreeLabelsyncBeamSearch::LabelHypothesis const* TreeLabelsyncBeamSearch::getWorstTerminatedHypothesis(std::vector<LabelHypothesis> const& hypotheses) const {
    LabelHypothesis const* worst = nullptr;

    for (auto const& hyp : hypotheses) {
        if (not hyp.isActive) {
            if (worst == nullptr or hyp > *worst) {
                worst = &hyp;
            }
        }
    }

    return worst;
}

TreeLabelsyncBeamSearch::LabelHypothesis const* TreeLabelsyncBeamSearch::getBestActiveHypothesis(std::vector<LabelHypothesis> const& hypotheses) const {
    LabelHypothesis const* best = nullptr;

    for (auto const& hyp : hypotheses) {
        if (hyp.isActive) {
            if (best == nullptr or hyp < *best) {
                best = &hyp;
            }
        }
    }

    return best;
}

TreeLabelsyncBeamSearch::LabelHypothesis const* TreeLabelsyncBeamSearch::getWorstActiveHypothesis(std::vector<LabelHypothesis> const& hypotheses) const {
    LabelHypothesis const* worst = nullptr;

    for (auto const& hyp : hypotheses) {
        if (hyp.isActive) {
            if (worst == nullptr or hyp > *worst) {
                worst = &hyp;
            }
        }
    }

    return worst;
}

TreeLabelsyncBeamSearch::LabelHypothesis const& TreeLabelsyncBeamSearch::getBestHypothesis(std::vector<LabelHypothesis> const& hypotheses) const {
    auto const* result = getBestTerminatedHypothesis(hypotheses);
    if (result != nullptr) {
        return *result;
    }
    result = getBestActiveHypothesis(hypotheses);
    verify(result != nullptr);
    return *result;
}

TreeLabelsyncBeamSearch::LabelHypothesis const& TreeLabelsyncBeamSearch::getWorstHypothesis(std::vector<LabelHypothesis> const& hypotheses) const {
    auto const* result = getWorstTerminatedHypothesis(hypotheses);
    if (result != nullptr) {
        return *result;
    }
    result = getWorstActiveHypothesis(hypotheses);
    verify(result != nullptr);
    return *result;
}

void TreeLabelsyncBeamSearch::logStatistics() const {
    clog() << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    clog() << Core::XmlOpen("initialization-time") << initializationTime_.elapsedMilliseconds() << Core::XmlClose("initialization-time");
    clog() << Core::XmlOpen("feature-processing-time") << featureProcessingTime_.elapsedMilliseconds() << Core::XmlClose("feature-processing-time");
    clog() << Core::XmlOpen("scoring-time") << scoringTime_.elapsedMilliseconds() << Core::XmlClose("scoring-time");
    clog() << Core::XmlClose("timing-statistics");
    for (auto const& stat : numHypsAfterIntermediatePruning_) {
        stat.write(clog());
    }
    numTerminatedHypsAfterScorePruning_.write(clog());
    numTerminatedHypsAfterRecombination_.write(clog());
    numTerminatedHypsAfterBeamPruning_.write(clog());
    numActiveHypsAfterScorePruning_.write(clog());
    numActiveHypsAfterRecombination_.write(clog());
    numActiveHypsAfterBeamPruning_.write(clog());
    numActiveWordEndHypsAfterIntermediatePruning_.write(clog());
    numActiveWordEndHypsAfterScorePruning_.write(clog());
    numActiveWordEndHypsAfterRecombination_.write(clog());
    numActiveWordEndHypsAfterBeamPruning_.write(clog());
    numActiveTrees_.write(clog());
}

template<typename Element>
void TreeLabelsyncBeamSearch::scorePruning(std::vector<Element>& hypotheses, Score relativeThreshold, size_t maxBeamSize) {
    if (hypotheses.size() <= maxBeamSize and relativeThreshold == Core::Type<Score>::max) {
        // Neither relative score pruning nor max beam size pruning triggers
        return;
    }

    // Find ranges for score histogram and setting absolute threshold
    Score lowerScore = Core::Type<Score>::max;
    Score upperScore = Core::Type<Score>::min;

    for (auto const& hyp : hypotheses) {
        lowerScore = std::min(lowerScore, hyp.pruningScore());
        upperScore = std::max(upperScore, hyp.pruningScore());
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
            scoreHistogram_ += hyp.pruningScore();
        }

        absoluteThreshold = std::min(absoluteThreshold, scoreHistogram_.quantile(maxBeamSize));
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

template void TreeLabelsyncBeamSearch::scorePruning<TreeLabelsyncBeamSearch::WithinWordExtensionCandidate>(std::vector<TreeLabelsyncBeamSearch::WithinWordExtensionCandidate>&, Score, size_t);
template void TreeLabelsyncBeamSearch::scorePruning<TreeLabelsyncBeamSearch::WordEndExtensionCandidate>(std::vector<TreeLabelsyncBeamSearch::WordEndExtensionCandidate>&, Score, size_t);
template void TreeLabelsyncBeamSearch::scorePruning<TreeLabelsyncBeamSearch::LabelHypothesis>(std::vector<TreeLabelsyncBeamSearch::LabelHypothesis>&, Score, size_t);

void TreeLabelsyncBeamSearch::recombination(std::vector<TreeLabelsyncBeamSearch::LabelHypothesis>& hypotheses) {
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

    tempHypotheses_.clear();
    // Reserve capacity because future reallocations would break the raw pointer we are storing later
    tempHypotheses_.reserve(hypotheses.size());
    // Map each unique combination of StateId, ScoringContext and LmHistory in newHypotheses to its hypothesis
    std::unordered_map<RecombinationContext, LabelHypothesis*, RecombinationContextHash> seenCombinations;
    for (auto const& hyp : hypotheses) {
        // Use try_emplace to check if the combination already exists and create a new entry if not at the same time
        auto [it, inserted] = seenCombinations.try_emplace({hyp}, nullptr);

        bool const isWordEnd = hyp.isActive and network_->isRoot(hyp.currentState);

        if (inserted) {
            // First time seeing this combination so move it over to `newHypotheses`
            tempHypotheses_.push_back(std::move(hyp));
            it->second = &tempHypotheses_.back();
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

    hypotheses.swap(tempHypotheses_);
}

size_t TreeLabelsyncBeamSearch::numActiveHyps() const {
    return std::accumulate(
            newBeam_.begin(),
            newBeam_.end(),
            0ul,
            [](size_t acc, auto const& hyp) { return acc + static_cast<size_t>(hyp.isActive); });
}

size_t TreeLabelsyncBeamSearch::numActiveWordEndHyps() const {
    return std::accumulate(
            newBeam_.begin(),
            newBeam_.end(),
            0ul,
            [this](size_t acc, auto const& hyp) { return acc + static_cast<size_t>(hyp.isActive and network_->isRoot(hyp.currentState)); });
}

void TreeLabelsyncBeamSearch::createSuccessorLookups() {
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

void TreeLabelsyncBeamSearch::finalizeHypotheses() {
    newBeam_.clear();
    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        auto& hyp = beam_[hypIndex];
        // Check if the hypotheses in the beam are either terminated or at a root state and add the sentence-end LM score
        if (not hyp.isActive or network_->isRoot(hyp.currentState)) {
            Lm::Score sentenceEndScore = languageModel_->sentenceEndScore(hyp.lmHistory);
            hyp.score += sentenceEndScore;
            hyp.trace->score.lm += sentenceEndScore;
            newBeam_.push_back(hyp);
        }
    }

    if (newBeam_.empty()) {  // There was no terminated and no word-end hypothesis in the beam
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

void TreeLabelsyncBeamSearch::maximumStableDelayPruning() {
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
        root = getBestHypothesis(beam_).trace;
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