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

#include "TreeRNNTTimesyncBeamSearch.hh"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_map>

#include <Am/ClassicStateModel.hh>
#include <Core/CollapsedVector.hh>
#include <Core/XmlStream.hh>
#include <Lattice/LatticeAdaptor.hh>
#include <Math/Utilities.hh>
#include <Search/Helpers.hh>
#include <Search/Module.hh>
#include <Search/StableDelayPruning.hh>
#include <Search/TracebackHelper.hh>

namespace Search {

namespace {

enum RecombinationMode {
    RecombinationModeOff,
    RecombinationModeSum,
    RecombinationModeViterbi,
};

}  // namespace

/*
 * =======================
 * === LabelHypothesis ===
 * =======================
 */

TreeRNNTTimesyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContexts(),
          currentToken(Nn::invalidLabelIndex),
          currentState(invalidTreeNodeIndex),
          lmHistory(),
          timeframe(0),
          length(1),
          score(0.0),
          scaledScore(0.0),
          outputWords(),
          outputWordsHash(0ul),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))) {}

TreeRNNTTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        TreeRNNTTimesyncBeamSearch::LabelHypothesis const&              base,
        TreeRNNTTimesyncBeamSearch::WithinWordExtensionCandidate const& extension,
        std::vector<Nn::ScoringContextRef> const&                       newScoringContexts,
        float                                                           lengthNormScale)
        : scoringContexts(newScoringContexts),
          currentToken(extension.nextToken),
          currentState(extension.nextState),
          lmHistory(base.lmHistory),
          timeframe(extension.timeframe),
          length(base.length),
          score(extension.score),
          scaledScore(),
          outputWords(base.outputWords),
          outputWordsHash(base.outputWordsHash),
          trace(base.trace) {
    switch (extension.transitionType) {
        case Nn::TransitionType::INITIAL_LABEL:
        case Nn::TransitionType::LABEL_TO_LABEL:
        case Nn::TransitionType::BLANK_TO_LABEL:
            length += 1;
            break;
        // Only blank consumes the frame, labels are emitted within it
        case Nn::TransitionType::INITIAL_BLANK:
        case Nn::TransitionType::LABEL_TO_BLANK:
        case Nn::TransitionType::BLANK_LOOP:
            timeframe += 1;
            break;
        default:
            break;
    }
    scaledScore = score / std::pow(length, lengthNormScale);
}

TreeRNNTTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        TreeRNNTTimesyncBeamSearch::LabelHypothesis const&           base,
        TreeRNNTTimesyncBeamSearch::WordEndExtensionCandidate const& extension,
        Lm::History const&                                           newLmHistory,
        Bliss::Lemma const*                                          blankLemma,
        float                                                        lengthNormScale)
        : scoringContexts(base.scoringContexts),
          currentToken(base.currentToken),
          currentState(extension.rootState),
          lmHistory(newLmHistory),
          timeframe(base.timeframe),
          length(base.length),
          score(extension.score),
          scaledScore(score / std::pow(length, lengthNormScale)),
          outputWords(base.outputWords),
          outputWordsHash(base.outputWordsHash) {
    if (extension.pron and extension.pron->lemma() != blankLemma) {
        outputWords.push_back(extension.pron->lemma());
        outputWordsHash = Core::combineHashes(outputWordsHash, std::hash<Bliss::Lemma const*>()(outputWords.back()));
    }

    auto newLmScore   = score - base.score;
    auto totalLmScore = base.trace->score.lm + newLmScore;
    auto totalAmScore = score - totalLmScore;

    // The word ends with its last token
    trace = Core::ref(new LatticeTrace(
            base.trace,
            extension.pron,
            timeframe,
            {totalAmScore, totalLmScore},
            {}));
}

std::string TreeRNNTTimesyncBeamSearch::LabelHypothesis::toString() const {
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
 * ==================================
 * === TreeRNNTTimesyncBeamSearch ===
 * ==================================
 */

const Core::ParameterIntVector TreeRNNTTimesyncBeamSearch::paramMaxBeamSizes(
        "max-beam-size",
        "Maximum number of elements in the search beam. Pruning is applied after each intermediate label scorer.",
        "",
        1);

const Core::ParameterInt TreeRNNTTimesyncBeamSearch::paramMaxWordEndBeamSize(
        "max-word-end-beam-size",
        "Maximum number of word-end hypotheses created per symbol step.",
        std::numeric_limits<int>::max(), 0);

const Core::ParameterFloatVector TreeRNNTTimesyncBeamSearch::paramScoreThresholds(
        "score-threshold",
        "Prune any hypotheses with a score that is at least this much worse than the best hypothesis. Pruning is applied after each intermediate label scorer.",
        "",
        0,
        Core::Type<Score>::max);

const Core::ParameterFloat TreeRNNTTimesyncBeamSearch::paramWordEndScoreThreshold(
        "word-end-score-threshold",
        "Prune any word-end hypothesis with a score that is at least this much worse than the best word-end hypothesis. This threshold is relative to the score-threshold.",
        Core::Type<Score>::max, 0);

const Core::ParameterInt TreeRNNTTimesyncBeamSearch::paramNumHistogramBins(
        "num-histogram-bins",
        "Number of bins for histogram pruning of hypotheses (very minor effect).",
        100,
        2);

const Core::ParameterFloat TreeRNNTTimesyncBeamSearch::paramLengthNormScale(
        "length-norm-scale",
        "Exponent of length for the hypothesis length normalization. Scaled scores are computed as score / length^length_norm_scale.",
        0.0);

const Core::ParameterInt TreeRNNTTimesyncBeamSearch::paramMaxLabelsPerFrame(
        "max-labels-per-timeframe",
        "Maximum number of non-blank label predictions per hypothesis in one timestep.",
        10,
        0);

const Core::ParameterBool TreeRNNTTimesyncBeamSearch::paramSentenceEndFallBack(
        "sentence-end-fall-back",
        "Allow for fallback solution if no active word-end hypothesis exists at the end of a segment.",
        true);

const Core::ParameterInt TreeRNNTTimesyncBeamSearch::paramCacheCleanupInterval(
        "cache-cleanup-interval",
        "Interval of search steps after which buffered inputs that are not needed anymore get cleaned up.",
        10,
        1);

const Core::ParameterInt TreeRNNTTimesyncBeamSearch::paramMaximumStableDelay(
        "maximum-stable-delay",
        "Introduce a cutoff point at `current-time` - `delay`. Every hypothesis that disagrees with the current best anywhere before the cutoff gets pruned."
        "This way words in the traceback become stable after at most `delay` frames.",
        Core::Type<int>::max,
        0);

const Core::ParameterInt TreeRNNTTimesyncBeamSearch::paramMaximumStableDelayPruningInterval(
        "maximum-stable-delay-pruning-interval",
        "Interval of search steps after which the maximum-stable-delay-pruning gets applied.",
        10,
        1);

const Core::Choice TreeRNNTTimesyncBeamSearch::choiceRecombinationMode(
        "off", RecombinationModeOff,
        "sum", RecombinationModeSum,
        "viterbi", RecombinationModeViterbi,
        Core::Choice::endMark());

const Core::ParameterChoice TreeRNNTTimesyncBeamSearch::paramRecombinationMode(
        "recombination-mode",
        &choiceRecombinationMode,
        "Whether hypotheses with identical recombination state should be recombined and how: by summing "
        "their probabilities via log-sum-exp (sum) or by keeping only the better-scoring one (Viterbi).",
        RecombinationModeSum);

TreeRNNTTimesyncBeamSearch::TreeRNNTTimesyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          maxWordEndBeamSize_(paramMaxWordEndBeamSize(config)),
          wordEndScoreThreshold_(paramWordEndScoreThreshold(config)),
          scoreHistogram_(paramNumHistogramBins(config)),
          lengthNormScale_(paramLengthNormScale(config)),
          maxLabelsPerFrame_(paramMaxLabelsPerFrame(config)),
          blankLabelIndex_(Nn::invalidLabelIndex),
          blankLemma_(),
          sentenceEndLemma_(),
          sentenceEndLabelIndex_(Nn::invalidLabelIndex),
          sentenceEndFallback_(paramSentenceEndFallBack(config)),
          cacheCleanupInterval_(paramCacheCleanupInterval(config)),
          maximumStableDelay_(paramMaximumStableDelay(config)),
          maximumStableDelayPruningInterval_(paramMaximumStableDelayPruningInterval(config)),
          recombinationMode_(paramRecombinationMode(config)),
          statisticsChannel_(config, "statistics"),
          stepwiseStatisticsChannel_(config, "stepwise-statistics"),
          debugChannel_(config, "debug"),
          labelScorers_(),
          beam_(),
          innerHyps_(),
          outerHyps_(),
          hypIndexToContextIndexMap_(),
          extensions_(),
          wordEndExtensions_(),
          newBeam_(),
          wordEndHypotheses_(),
          scoringContexts_(),
          tempHypotheses_(),
          initializationTime_(),
          featureProcessingTime_(),
          numInputHyps_("num-input-hyps"),
          numWordEndHyps_("num-word-end-hyps"),
          numOuterHyps_("num-outer-hyps"),
          numInnerHyps_("num-inner-hyps"),
          numInnerAndOuterHyps_("num-inner-and-outer-hyps"),
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

    for (size_t i = 0ul; i < scoreThresholds_.size(); ++i) {
        useScorePruning_.push_back(scoreThresholds_[i] != Core::Type<Score>::max);
    }

    if (scoreThresholds_.back() == Core::Type<Score>::max and wordEndScoreThreshold_ != Core::Type<Score>::max) {
        error() << "Word-end score-threshold which is relative to the score-threshold is set, but score-threshold is not set";
    }
    if (wordEndScoreThreshold_ != Core::Type<Score>::max) {
        wordEndScoreThreshold_ *= scoreThresholds_.back();
        log() << "Use absolute word-end score-threshold of " << wordEndScoreThreshold_;
    }
}

Speech::ModelCombination::Mode TreeRNNTTimesyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon | Speech::ModelCombination::useAcousticModel | Speech::ModelCombination::useLanguageModel;
}

Am::AcousticModel::Mode TreeRNNTTimesyncBeamSearch::requiredAcousticModel() const {
    return Am::AcousticModel::noEmissions;
}

bool TreeRNNTTimesyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
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

    // Per-scorer timers and statistics are indexed by label scorer. All statistics entries
    // share one name; the scorer is written as an attribute.
    numHypsAfterIntermediatePruning_.resize(labelScorers_.size(), Core::Statistics<u32>("num-hyps-after-intermediate-pruning"));
    scoreAndPruneExtensionsTimes_.resize(labelScorers_.size());
    scoringTimes_.resize(labelScorers_.size());
    scoreReadoutTimes_.resize(labelScorers_.size());
    intermediatePruningTimes_.resize(labelScorers_.size());

    blankLemma_ = lexicon_->specialLemma("blank");
    if (not blankLemma_) {
        criticalError() << "Lexicon has no blank lemma";
    }

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

    blankLabelIndex_ = acousticModel_->emissionIndex(acousticModel_->blankAllophoneStateIndex());
    log() << "Use blank label with index " << blankLabelIndex_;

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

    if (blankLemma_->syntacticTokenSequence().size() != 0) {
        warning("Special lemma \"blank\" will be scored by the language model. To prevent the LM from scoring it, set an empty syntactic token sequence for it in the lexicon.");
    }

    // Create look-ups for state successors and exits of each state
    createSuccessorLookups();

    return true;
}

void TreeRNNTTimesyncBeamSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    initializationTime_.reset();
    featureProcessingTime_.reset();
    recognitionTime_.reset();
    for (auto* timers : {&scoreAndPruneExtensionsTimes_, &scoringTimes_, &scoreReadoutTimes_, &intermediatePruningTimes_}) {
        for (auto& timer : *timers) {
            timer.reset();
        }
    }
    buildNewBeamTime_.reset();
    wordEndExpansionTime_.reset();
    recombinationTime_.reset();
    beamPruningTime_.reset();
    finalizeTime_.reset();
    for (auto& stat : numHypsAfterIntermediatePruning_) {
        stat.clear();
    }
    numInputHyps_.clear();
    numWordEndHyps_.clear();
    numOuterHyps_.clear();
    numInnerHyps_.clear();
    numInnerAndOuterHyps_.clear();
    numHypsAfterPruning_.clear();
    numActiveHyps_.clear();

    initializationTime_.start();

    for (auto& labelScorer : labelScorers_) {
        labelScorer->reset();
    }

    if (segment != nullptr) {
        languageModel_->setSegment(segment);
    }

    // Reset beam to a single empty hypothesis
    beam_.clear();
    beam_.push_back(LabelHypothesis());
    for (auto& labelScorer : labelScorers_) {
        beam_.front().scoringContexts.push_back(labelScorer->getInitialScoringContext());
    }
    beam_.front().currentState = network_->rootState;
    beam_.front().lmHistory    = languageModel_->startHistory();

    currentSearchStep_  = 0ul;
    stepStatisticsOpen_ = false;
    finishedSegment_    = false;

    initializationTime_.stop();

    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlOpen("search-steps");
    }
}

void TreeRNNTTimesyncBeamSearch::finishSegment() {
    featureProcessingTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->signalNoMoreFeatures();
    }
    featureProcessingTime_.stop();
    decodeManySteps();
    finalizeTime_.start();
    finalizeHypotheses();
    finalizeTime_.stop();
    finishedSegment_ = true;
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlClose("search-steps");
    }
    logStatistics();
}

void TreeRNNTTimesyncBeamSearch::putFeature(Nn::DataView const& feature) {
    featureProcessingTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInput(feature);
    }
    featureProcessingTime_.stop();
}

void TreeRNNTTimesyncBeamSearch::putFeatures(Nn::DataView const& features, size_t nTimesteps) {
    featureProcessingTime_.start();
    for (auto& labelScorer : labelScorers_) {
        labelScorer->addInputs(features, nTimesteps);
    }
    featureProcessingTime_.stop();
}

Core::Ref<const Traceback> TreeRNNTTimesyncBeamSearch::getCurrentBestTraceback() const {
    return getBestHypothesis().trace->performTraceback();
}

Core::Ref<const LatticeAdaptor> TreeRNNTTimesyncBeamSearch::getCurrentBestWordLattice() const {
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

Core::Ref<const LatticeTrace> TreeRNNTTimesyncBeamSearch::getCurrentBestLatticeTrace() const {
    return getBestHypothesis().trace;
}

Core::Ref<const LatticeTrace> TreeRNNTTimesyncBeamSearch::getCommonPrefix() const {
    std::vector<Core::Ref<LatticeTrace>> traces(beam_.size());
    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        traces[hypIndex] = beam_[hypIndex].trace;
    }

    // The head trace of a hypothesis is its last word end, which is never replaced
    RootTraceSearcher searcher(traces, true);
    if (not searcher.rootTrace()) {
        warning("Common prefix of all traces is a sentinel value");
    }

    return Core::Ref<const LatticeTrace>(searcher.rootTrace());
}

bool TreeRNNTTimesyncBeamSearch::decodeStep() {
    if (finishedSegment_) {
        return false;
    }

    // Without hypotheses there is nothing to score, so no input can be consumed and the step would never progress
    if (beam_.empty()) {
        return false;
    }

    recognitionTime_.start();

    // Discard the partial work of this timestep if a label scorer can't score it yet. `beam_` stays intact,
    // so the timestep is redone from scratch once the label scorers have received more input
    auto postponeTimestep = [this](bool symbolStepStatisticsOpen) {
        if (symbolStepStatisticsOpen and stepwiseStatisticsChannel_.isOpen()) {
            stepwiseStatisticsChannel_ << Core::XmlClose("symbol-step-stats");
        }
        outerHyps_.clear();
        closeStepStatistics();
        recognitionTime_.stop();
        return false;
    };

    // Start timestep with beam of previous timestep
    innerHyps_ = beam_;

    size_t symbolStep = 0;

    // Start inner loop of this timestep
    while (true) {
        // Early stopping if no inner hyps left
        // This happens if all inner hyps were worse than the worst outer hyp
        if (innerHyps_.empty()) {
            break;
        }

        /*
         * Get score accessors of the first label scorer for all inner hyps. These get reused below for
         * both the blank and the non-blank extension candidates
         */
        scoringContexts_.clear();
        for (auto const& hyp : innerHyps_) {
            scoringContexts_.push_back(hyp.scoringContexts.front());
        }

        std::vector<std::optional<Nn::ScoreAccessorRef>> scoreAccessors;
        {
            Core::StopWatch::Scope scoreAndPruneTimer(scoreAndPruneExtensionsTimes_.front());
            Core::StopWatch::Scope scoringTimer(scoringTimes_.front());
            scoreAccessors = labelScorers_.front()->getScoreAccessors(scoringContexts_);
        }

        bool anyScored = std::any_of(scoreAccessors.begin(), scoreAccessors.end(),
                                     [](auto const& a) { return a.has_value(); });
        if (not anyScored) {
            return postponeTimestep(false);
        }

        if (symbolStep == 0ul) {
            // Counted only now, so that a step which can't be scored yet leaves no trace in the
            // statistics and doesn't claim a step number
            openStepStatistics();
            numInputHyps_ += innerHyps_.size();
            if (stepwiseStatisticsChannel_.isOpen()) {
                stepwiseStatisticsChannel_ << Core::XmlFull("num-input-hyps", innerHyps_.size());
            }
        }
        if (stepwiseStatisticsChannel_.isOpen()) {
            stepwiseStatisticsChannel_ << Core::XmlOpen("symbol-step-stats") + Core::XmlAttribute("step", symbolStep);
        }

        /*
         * Extend inner hyps with the blank label, so they become outer hyps
         */
        {
            Core::StopWatch::Scope scoreAndPruneTimer(scoreAndPruneExtensionsTimes_.front());
            Core::StopWatch::Scope readoutTimer(scoreReadoutTimes_.front());
            createBlankExtensions(scoreAccessors);
        }

        // Blank extensions are not pruned after the first label scorer, there is at most one per inner hyp
        if (not scoreAndPruneWithRemainingLabelScorers(false)) {
            return postponeTimestep(true);
        }

        buildNewBeamTime_.start();
        newBeam_.clear();
        buildHypsFromExtensions(newBeam_);
        buildNewBeamTime_.stop();

        // Hyps in the blank lemma also exit it to the root
        wordEndExpansionTime_.start();
        expandWordEnds(newBeam_);
        wordEndExpansionTime_.stop();

        outerHyps_.insert(outerHyps_.end(), std::make_move_iterator(newBeam_.begin()), std::make_move_iterator(newBeam_.end()));

        recombinationTime_.start();
        recombination(outerHyps_);
        recombinationTime_.stop();

        numOuterHyps_ += outerHyps_.size();
        if (stepwiseStatisticsChannel_.isOpen()) {
            stepwiseStatisticsChannel_ << Core::XmlFull("num-outer-hyps", outerHyps_.size());
        }

        // Finish this step if the maximum number of output symbols has been reached
        if (symbolStep >= maxLabelsPerFrame_) {
            if (stepwiseStatisticsChannel_.isOpen()) {
                stepwiseStatisticsChannel_ << Core::XmlClose("symbol-step-stats");
            }
            break;
        }

        /*
         * Extend inner hyps with non-blank labels
         */
        {
            Core::StopWatch::Scope scoreAndPruneTimer(scoreAndPruneExtensionsTimes_.front());
            Core::StopWatch::Scope readoutTimer(scoreReadoutTimes_.front());
            createExtensions(scoreAccessors);
        }

        // Score/beam-size-prune extension candidates after the first label scorer
        {
            Core::StopWatch::Scope scoreAndPruneTimer(scoreAndPruneExtensionsTimes_.front());
            Core::StopWatch::Scope pruningTimer(intermediatePruningTimes_.front());
            size_t                 maxBeamSize = extensions_.size();
            if (labelScorers_.size() > 1ul) {
                maxBeamSize = maxBeamSizes_.front();
            }
            scorePruning(extensions_, scoreThresholds_.front(), maxBeamSize, [](auto const& ext) { return ext.score; });
        }
        numHypsAfterIntermediatePruning_.front() += extensions_.size();
        if (stepwiseStatisticsChannel_.isOpen()) {
            stepwiseStatisticsChannel_ << Core::XmlFull("num-hyps-after-intermediate-pruning", extensions_.size()) + Core::XmlAttribute("scorer", 1);
        }

        if (not scoreAndPruneWithRemainingLabelScorers(true)) {
            return postponeTimestep(true);
        }

        // Prune extension candidates down to maxBeamSize based on the raw score
        beamPruningTime_.start();
        scorePruning(extensions_, Core::Type<Score>::max, maxBeamSizes_[labelScorers_.size() - 1], [](auto const& ext) { return ext.score; });
        beamPruningTime_.stop();

        buildNewBeamTime_.start();
        newBeam_.clear();
        buildHypsFromExtensions(newBeam_);
        buildNewBeamTime_.stop();

        recombinationTime_.start();
        recombination(newBeam_);
        recombinationTime_.stop();

        // Word-end hyps can continue with the next word or blank in this timestep
        wordEndExpansionTime_.start();
        expandWordEnds(newBeam_);
        wordEndExpansionTime_.stop();

        // If there are already more than maxBeamSize outer hyps,
        // remove all inner hyps with a score that is worse than the worst score of the max-beam-size best outer hyps
        beamPruningTime_.start();
        if (outerHyps_.size() >= maxBeamSizes_[labelScorers_.size() - 1]) {
            auto kth = outerHyps_.begin() + (maxBeamSizes_[labelScorers_.size() - 1] - 1);
            std::nth_element(outerHyps_.begin(), kth, outerHyps_.end(),
                             [](auto const& a, auto const& b) { return a.score < b.score; });
            Score outerHypsThreshold = kth->score;

            innerHyps_.clear();
            for (auto& hyp : newBeam_) {
                if (hyp.score < outerHypsThreshold) {
                    innerHyps_.push_back(std::move(hyp));
                }
            }
        }
        else {
            innerHyps_.swap(newBeam_);
        }
        beamPruningTime_.stop();

        numInnerHyps_ += innerHyps_.size();
        numInnerAndOuterHyps_ += (innerHyps_.size() + outerHyps_.size());
        if (stepwiseStatisticsChannel_.isOpen()) {
            stepwiseStatisticsChannel_ << Core::XmlFull("num-inner-hyps", innerHyps_.size());
            stepwiseStatisticsChannel_ << Core::XmlFull("num-inner-and-outer-hyps", innerHyps_.size() + outerHyps_.size());
            stepwiseStatisticsChannel_ << Core::XmlClose("symbol-step-stats");
        }

        ++symbolStep;

    }  // end of inner loop

    // Prune all hyps of this timestep at the end of this timestep based on the length-normalized score
    beamPruningTime_.start();
    Score relativeThreshold = scoreThresholds_[labelScorers_.size() - 1];
    if (useScorePruning_[labelScorers_.size() - 1] and lengthNormScale_ != 0.0f and not outerHyps_.empty()) {
        // Convert score-threshold to an equivalent length-normalized-score gap using the current best hypothesis's length
        auto const& bestHyp = *std::min_element(outerHyps_.begin(), outerHyps_.end());
        relativeThreshold /= std::pow(bestHyp.length, lengthNormScale_);
    }
    scorePruning(outerHyps_, relativeThreshold, maxBeamSizes_[labelScorers_.size() - 1], [](auto const& hyp) { return hyp.scaledScore; });
    beamPruningTime_.stop();
    numHypsAfterPruning_ += outerHyps_.size();
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("num-hyps-after-pruning", outerHyps_.size()) + Core::XmlAttribute("scorer", labelScorers_.size());
    }

    // The leftover outer hyps of this timestep will be the inner hyps to start with in the next timestep
    beam_ = std::move(outerHyps_);
    outerHyps_.clear();

    ++currentSearchStep_;

    if (currentSearchStep_ % cacheCleanupInterval_ == 0) {
        for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
            Core::CollapsedVector<Nn::ScoringContextRef> activeContexts;
            for (auto const& hyp : beam_) {
                activeContexts.push_back(hyp.scoringContexts[scorerIdx]);
            }
            labelScorers_[scorerIdx]->cleanupCaches(activeContexts);
        }
    }
    if (currentSearchStep_ % maximumStableDelayPruningInterval_ == 0) {
        beamPruningTime_.start();
        maximumStableDelayPruning();
        beamPruningTime_.stop();
        if (stepwiseStatisticsChannel_.isOpen()) {
            stepwiseStatisticsChannel_ << Core::XmlFull("num-hyps-after-maximum-stable-delay-pruning", beam_.size());
        }
    }

    // Counted after all pruning of this step, including maximum-stable-delay pruning
    numActiveHyps_ += beam_.size();

    recognitionTime_.stop();

    logStepStatistics();
    return true;
}

TreeRNNTTimesyncBeamSearch::LabelHypothesis const& TreeRNNTTimesyncBeamSearch::getBestHypothesis() const {
    verify(not beam_.empty());

    return *std::min_element(beam_.begin(), beam_.end());
}

TreeRNNTTimesyncBeamSearch::LabelHypothesis const& TreeRNNTTimesyncBeamSearch::getWorstHypothesis() const {
    verify(not beam_.empty());

    return *std::max_element(beam_.begin(), beam_.end());
}

void TreeRNNTTimesyncBeamSearch::logStatistics() const {
    logOwnStatistics();

    for (auto const& labelScorer : labelScorers_) {
        labelScorer->logStatistics();
    }
}

void TreeRNNTTimesyncBeamSearch::logOwnStatistics() const {
    if (not statisticsChannel_.isOpen()) {
        return;
    }

    statisticsChannel_ << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    statisticsChannel_ << Core::XmlFull("initialization-time", initializationTime_.elapsedMilliseconds());
    statisticsChannel_ << Core::XmlFull("feature-processing-time", featureProcessingTime_.elapsedMilliseconds());
    statisticsChannel_ << Core::XmlOpen("recognition-time") + Core::XmlAttribute("total", recognitionTime_.elapsedMilliseconds());
    for (size_t i = 0ul; i < scoreAndPruneExtensionsTimes_.size(); ++i) {
        statisticsChannel_ << Core::XmlOpen("score-and-prune-extensions-time") + Core::XmlAttribute("scorer", i + 1) + Core::XmlAttribute("total", scoreAndPruneExtensionsTimes_[i].elapsedMilliseconds());
        statisticsChannel_ << Core::XmlFull("scoring-time", scoringTimes_[i].elapsedMilliseconds());
        statisticsChannel_ << Core::XmlFull("score-readout-time", scoreReadoutTimes_[i].elapsedMilliseconds());
        statisticsChannel_ << Core::XmlFull("intermediate-pruning-time", intermediatePruningTimes_[i].elapsedMilliseconds());
        statisticsChannel_ << Core::XmlClose("score-and-prune-extensions-time");
    }
    statisticsChannel_ << Core::XmlFull("build-new-beam-time", buildNewBeamTime_.elapsedMilliseconds());
    statisticsChannel_ << Core::XmlFull("word-end-expansion-time", wordEndExpansionTime_.elapsedMilliseconds());
    statisticsChannel_ << Core::XmlFull("recombination-time", recombinationTime_.elapsedMilliseconds());
    statisticsChannel_ << Core::XmlFull("beam-pruning-time", beamPruningTime_.elapsedMilliseconds());
    statisticsChannel_ << Core::XmlClose("recognition-time");
    statisticsChannel_ << Core::XmlFull("finalize-time", finalizeTime_.elapsedMilliseconds());
    statisticsChannel_ << Core::XmlClose("timing-statistics");
    statisticsChannel_ << Core::XmlOpen("search-statistics");
    numInputHyps_.write(statisticsChannel_);
    for (size_t i = 0ul; i < numHypsAfterIntermediatePruning_.size(); ++i) {
        numHypsAfterIntermediatePruning_[i].write(statisticsChannel_, {Core::XmlAttribute("scorer", i + 1)});
    }
    numWordEndHyps_.write(statisticsChannel_);
    numOuterHyps_.write(statisticsChannel_);
    numInnerHyps_.write(statisticsChannel_);
    numInnerAndOuterHyps_.write(statisticsChannel_);
    numHypsAfterPruning_.write(statisticsChannel_);
    numActiveHyps_.write(statisticsChannel_);
    statisticsChannel_ << Core::XmlClose("search-statistics");
}

Nn::TransitionType TreeRNNTTimesyncBeamSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const {
    bool prevIsBlank = prevLabel == blankLabelIndex_;
    bool nextIsBlank = nextLabel == blankLabelIndex_;

    if (prevLabel == Nn::invalidLabelIndex) {
        return nextIsBlank ? Nn::TransitionType::INITIAL_BLANK : Nn::TransitionType::INITIAL_LABEL;
    }
    if (prevIsBlank) {
        return nextIsBlank ? Nn::TransitionType::BLANK_LOOP : Nn::TransitionType::BLANK_TO_LABEL;
    }
    return nextIsBlank ? Nn::TransitionType::LABEL_TO_BLANK : Nn::TransitionType::LABEL_TO_LABEL;
}

void TreeRNNTTimesyncBeamSearch::createBlankExtensions(std::vector<std::optional<Nn::ScoreAccessorRef>> const& scoreAccessors) {
    extensions_.clear();
    for (size_t hypIndex = 0ul; hypIndex < innerHyps_.size(); ++hypIndex) {
        auto const& hyp = innerHyps_[hypIndex];

        auto const& scoreAccessor = scoreAccessors[hypIndex];
        if (not scoreAccessor) {
            continue;
        }

        // Right after a blank exit, don't start a new blank lemma, the hyp still in the blank lemma continues it
        if (network_->isRoot(hyp.currentState) and hyp.currentToken == blankLabelIndex_) {
            continue;
        }

        // Blank follows the blank successor of the current state (if any)
        for (size_t i = stateSuccessorsOffset_[hyp.currentState]; i < stateSuccessorsOffset_[hyp.currentState + 1]; ++i) {
            StateId successorState = stateSuccessors_[i];
            if (network_->structure.state(successorState).stateDesc.acousticModel != blankLabelIndex_) {
                continue;
            }

            auto  transitionType = inferTransitionType(hyp.currentToken, blankLabelIndex_);
            Score extScore       = hyp.score;
            if (labelScorers_.front()->scoresTransition(transitionType)) {
                extScore += (*scoreAccessor)->getScore(transitionType, blankLabelIndex_);
            }
            extensions_.push_back(
                    {.nextToken      = blankLabelIndex_,
                     .nextState      = successorState,
                     .timeframe      = (*scoreAccessor)->getTime(),
                     .score          = extScore,
                     .transitionType = transitionType,
                     .baseHypIndex   = hypIndex});
        }
    }
}

void TreeRNNTTimesyncBeamSearch::createExtensions(std::vector<std::optional<Nn::ScoreAccessorRef>> const& scoreAccessors) {
    Score currentBestScore = Core::Type<Score>::max;

    extensions_.clear();
    for (size_t hypIndex = 0ul; hypIndex < innerHyps_.size(); ++hypIndex) {
        auto const& hyp = innerHyps_[hypIndex];

        auto const& scoreAccessor = scoreAccessors[hypIndex];
        if (not scoreAccessor) {
            continue;
        }

        auto denseScores = (*scoreAccessor)->getDenseScores();
        auto scoreTime   = (*scoreAccessor)->getTime();

        for (size_t i = stateSuccessorsOffset_[hyp.currentState]; i < stateSuccessorsOffset_[hyp.currentState + 1]; ++i) {
            StateId        successorState = stateSuccessors_[i];
            Nn::LabelIndex tokenIdx       = network_->structure.state(successorState).stateDesc.acousticModel;

            // Blank is handled in `createBlankExtensions`
            if (tokenIdx == blankLabelIndex_) {
                continue;
            }

            auto  transitionType = inferTransitionType(hyp.currentToken, tokenIdx);
            Score extScore       = hyp.score;
            if (labelScorers_.front()->scoresTransition(transitionType)) {
                extScore += (denseScores and tokenIdx < denseScores->size())
                                    ? (*denseScores)[tokenIdx]
                                    : (*scoreAccessor)->getScore(transitionType, tokenIdx);
            }

            // Pre-prune based on score before creating extension instance and appending to list
            if (useScorePruning_.front() and extScore > currentBestScore + scoreThresholds_.front()) {
                continue;
            }
            currentBestScore = std::min(currentBestScore, extScore);

            extensions_.push_back(
                    {.nextToken      = tokenIdx,
                     .nextState      = successorState,
                     .timeframe      = scoreTime,
                     .score          = extScore,
                     .transitionType = transitionType,
                     .baseHypIndex   = hypIndex});
        }
    }
}

bool TreeRNNTTimesyncBeamSearch::scoreAndPruneWithRemainingLabelScorers(bool countStatistics) {
    for (size_t scorerIdx = 1ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
        Core::StopWatch::Scope scoreAndPruneTimer(scoreAndPruneExtensionsTimes_[scorerIdx]);
        auto const&            labelScorer = labelScorers_[scorerIdx];

        // Collect the scoring contexts of this scorer needed for the surviving extensions
        scoringContexts_.clear();
        hypIndexToContextIndexMap_.assign(innerHyps_.size(), -1);
        for (auto const& ext : extensions_) {
            if (hypIndexToContextIndexMap_[ext.baseHypIndex] == -1) {
                hypIndexToContextIndexMap_[ext.baseHypIndex] = scoringContexts_.size();
                scoringContexts_.push_back(innerHyps_[ext.baseHypIndex].scoringContexts[scorerIdx]);
            }
        }

        std::vector<std::optional<Nn::ScoreAccessorRef>> scoreAccessors;
        {
            Core::StopWatch::Scope scoringTimer(scoringTimes_[scorerIdx]);
            scoreAccessors = labelScorer->getScoreAccessors(scoringContexts_);
        }

        // Without this check, all extensions would be pruned as unscorable below and the timestep would end
        // with an empty beam
        bool anyScored = std::any_of(scoreAccessors.begin(), scoreAccessors.end(),
                                     [](auto const& a) { return a.has_value(); });
        if (not scoringContexts_.empty() and not anyScored) {
            return false;
        }

        {
            Core::StopWatch::Scope readoutTimer(scoreReadoutTimes_[scorerIdx]);

            std::vector<std::optional<Nn::DenseScoreSpan>> denseScoreSpans(scoreAccessors.size(), std::nullopt);
            std::vector<Nn::TimeframeIndex>                scoreTimes(scoreAccessors.size(), 0);
            for (size_t accessorIdx = 0ul; accessorIdx < scoreAccessors.size(); ++accessorIdx) {
                if (scoreAccessors[accessorIdx]) {
                    denseScoreSpans[accessorIdx] = (*scoreAccessors[accessorIdx])->getDenseScores();
                    scoreTimes[accessorIdx]      = (*scoreAccessors[accessorIdx])->getTime();
                }
            }

            for (auto& ext : extensions_) {
                if (not labelScorer->scoresTransition(ext.transitionType)) {
                    continue;
                }

                auto        contextIdx    = hypIndexToContextIndexMap_[ext.baseHypIndex];
                auto const& scoreAccessor = scoreAccessors[contextIdx];
                if (not scoreAccessor) {
                    // Extension is not scorable so set the score to max in order to prune it later
                    ext.score = Core::Type<Score>::max;
                    continue;
                }

                auto const& denseScores = denseScoreSpans[contextIdx];
                ext.score += (denseScores and ext.nextToken < denseScores->size())
                                     ? (*denseScores)[ext.nextToken]
                                     : (*scoreAccessor)->getScore(ext.transitionType, ext.nextToken);
                ext.timeframe = std::max(ext.timeframe, scoreTimes[contextIdx]);
            }
        }

        Core::StopWatch::Scope pruningTimer(intermediatePruningTimes_[scorerIdx]);
        size_t                 maxBeamSize = extensions_.size();
        if (scorerIdx < labelScorers_.size() - 1) {
            maxBeamSize = maxBeamSizes_[scorerIdx];
        }
        scorePruning(extensions_, scoreThresholds_[scorerIdx], maxBeamSize, [](auto const& ext) { return ext.score; });
        if (countStatistics) {
            numHypsAfterIntermediatePruning_[scorerIdx] += extensions_.size();
            if (stepwiseStatisticsChannel_.isOpen()) {
                stepwiseStatisticsChannel_ << Core::XmlFull("num-hyps-after-intermediate-pruning", extensions_.size()) + Core::XmlAttribute("scorer", scorerIdx + 1);
            }
        }
    }
    return true;
}

void TreeRNNTTimesyncBeamSearch::buildHypsFromExtensions(std::vector<LabelHypothesis>& hyps) {
    for (auto const& extension : extensions_) {
        auto const& baseHyp = innerHyps_[extension.baseHypIndex];

        std::vector<Nn::ScoringContextRef> newScoringContexts;
        newScoringContexts.reserve(labelScorers_.size());
        for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
            newScoringContexts.push_back(labelScorers_[scorerIdx]->extendedScoringContext(
                    baseHyp.scoringContexts[scorerIdx],
                    extension.nextToken,
                    extension.transitionType));
        }

        hyps.push_back({baseHyp, extension, newScoringContexts, lengthNormScale_});
    }
}

void TreeRNNTTimesyncBeamSearch::expandWordEnds(std::vector<LabelHypothesis>& hyps) {
    wordEndExtensions_.clear();

    for (size_t hypIndex = 0ul; hypIndex < hyps.size(); ++hypIndex) {
        auto const& hyp = hyps[hypIndex];

        // All exits below share this hypothesis' scoring contexts, so each accessor is fetched only once
        wordEndScoreAccessors_.assign(labelScorers_.size(), Nn::ScoreAccessorRef());

        for (size_t i = stateExitsOffset_[hyp.currentState]; i < stateExitsOffset_[hyp.currentState + 1]; ++i) {
            auto const& exit      = stateExits_[i];
            auto const* lemmaPron = lexicon_->lemmaPronunciation(exit.pronunciation);
            auto const* lemma     = lemmaPron->lemma();

            Score       lmScore = 0;
            auto const& sts     = lemma->syntacticTokenSequence();
            if (sts.size() != 0) {
                require(sts.size() == 1);
                lmScore = languageModel_->score(hyp.lmHistory, sts.front());
            }

            auto transitionType = lemma == blankLemma_ ? Nn::TransitionType::BLANK_EXIT : Nn::TransitionType::WORD_EXIT;

            Score penalty = 0.0;
            for (size_t scorerIdx = 0ul; scorerIdx < labelScorers_.size(); ++scorerIdx) {
                if (not labelScorers_[scorerIdx]->scoresTransition(transitionType)) {
                    continue;
                }
                auto& scoreAccessor = wordEndScoreAccessors_[scorerIdx];
                if (not scoreAccessor) {
                    auto fetched = labelScorers_[scorerIdx]->getScoreAccessor(hyp.scoringContexts[scorerIdx]);
                    if (not fetched) {
                        continue;
                    }
                    scoreAccessor = *fetched;
                }
                penalty += scoreAccessor->getScore(transitionType);
            }

            wordEndExtensions_.push_back({
                    .pron           = lemmaPron,
                    .rootState      = exit.transitState,
                    .score          = hyp.score + lmScore + penalty,
                    .transitionType = transitionType,
                    .baseHypIndex   = hypIndex,
            });
        }
    }

    scorePruning(wordEndExtensions_, wordEndScoreThreshold_, wordEndExtensions_.size(), [](auto const& ext) { return ext.score; });

    wordEndHypotheses_.clear();
    for (auto const& extension : wordEndExtensions_) {
        auto const& baseHyp      = hyps[extension.baseHypIndex];
        auto        newLmHistory = baseHyp.lmHistory;
        auto const& sts          = extension.pron->lemma()->syntacticTokenSequence();
        if (sts.size() != 0) {
            newLmHistory = languageModel_->extendedHistory(newLmHistory, sts.front());
        }
        wordEndHypotheses_.push_back({baseHyp, extension, newLmHistory, blankLemma_, lengthNormScale_});
    }

    recombinationTime_.start();
    recombination(wordEndHypotheses_);
    recombinationTime_.stop();

    scorePruning(wordEndHypotheses_, Core::Type<Score>::max, maxWordEndBeamSize_, [](auto const& hyp) { return hyp.score; });

    numWordEndHyps_ += wordEndHypotheses_.size();
    hyps.insert(hyps.end(), wordEndHypotheses_.begin(), wordEndHypotheses_.end());
}

template<typename Element, typename ScoreFn>
void TreeRNNTTimesyncBeamSearch::scorePruning(std::vector<Element>& elements, Score relativeThreshold, size_t maxBeamSize, ScoreFn scoreFn) {
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

void TreeRNNTTimesyncBeamSearch::recombination(std::vector<TreeRNNTTimesyncBeamSearch::LabelHypothesis>& hypotheses) {
    if (recombinationMode_ == RecombinationModeOff) {
        return;
    }

    // Unique combination of state, currentToken, scoringContexts, LM history and, in sum mode only, outputWords,
    // since summing probabilities is only valid for the same output. Points at a hyp to avoid copying outputWords.
    struct RecombinationContext {
        LabelHypothesis const* hyp;
        bool                   ignoreOutputWords;

        bool operator==(RecombinationContext const& other) const {
            if (hyp->currentState != other.hyp->currentState or hyp->currentToken != other.hyp->currentToken) {
                return false;
            }
            if (not(hyp->lmHistory == other.hyp->lmHistory)) {
                return false;
            }
            if (not ignoreOutputWords and (hyp->outputWordsHash != other.hyp->outputWordsHash or hyp->outputWords != other.hyp->outputWords)) {
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
            size_t hash = Core::combineHashes(context.hyp->currentState, context.hyp->currentToken);
            hash        = Core::combineHashes(hash, Lm::History::Hash{}(context.hyp->lmHistory));
            for (auto const& scoringContext : context.hyp->scoringContexts) {
                hash = Core::combineHashes(hash, Nn::ScoringContextHash{}(scoringContext));
            }
            return Core::combineHashes(hash, context.ignoreOutputWords ? 0ul : context.hyp->outputWordsHash);
        }
    };

    bool ignoreOutputWords = recombinationMode_ == RecombinationModeViterbi;

    tempHypotheses_.clear();
    tempHypotheses_.reserve(hypotheses.size());  // Reserve capacity because future reallocations would break the raw pointers
    // Map each unique RecombinationContext in `hypotheses` to its representative hyp
    std::unordered_map<RecombinationContext, LabelHypothesis*, RecombinationContextHash> seenContexts;

    for (auto& hyp : hypotheses) {
        // Probe with a key pointing at `hyp` in its original location. That pointer is only dereferenced
        // for this lookup and never stored, since `hyp` may be moved-from right after.
        auto it = seenContexts.find(RecombinationContext{&hyp, ignoreOutputWords});

        if (it == seenContexts.end()) {
            // Key the map entry off of the stable copy in tempHypotheses_
            tempHypotheses_.push_back(std::move(hyp));
            LabelHypothesis* stableHyp = &tempHypotheses_.back();
            seenContexts.emplace(RecombinationContext{stableHyp, ignoreOutputWords}, stableHyp);
            continue;
        }

        // Only hyps at a root own their trace, within-word hyps share the trace of their last word end.
        // A root hyp may already carry a sibling chain from an earlier recombination, so whole chains are joined.
        bool createTraceSiblings = network_->isRoot(hyp.currentState);

        auto* existingHyp = it->second;

        // In sum mode, merge scores in probability space, in viterbi mode keep the better hyp's own score
        bool  mergeScores = recombinationMode_ == RecombinationModeSum;
        Score mergedScore = mergeScores ? scaledLogAdd(existingHyp->score, hyp.score, 1.0f, 1.0f) : 0.0f;

        if (hyp.score < existingHyp->score) {
            // New hyp is better -> keep it as representative
            if (createTraceSiblings) {
                hyp.trace->appendSiblingToChain(existingHyp->trace);
            }
            *existingHyp = std::move(hyp);  // Overwrite in-place
        }
        else if (createTraceSiblings) {
            existingHyp->trace->appendSiblingToChain(hyp.trace);
        }

        if (mergeScores) {
            existingHyp->score       = mergedScore;
            existingHyp->scaledScore = mergedScore / std::pow(existingHyp->length, lengthNormScale_);
        }
    }

    hypotheses.swap(tempHypotheses_);
}

void TreeRNNTTimesyncBeamSearch::openStepStatistics() {
    if (not stepwiseStatisticsChannel_.isOpen() or stepStatisticsOpen_) {
        return;
    }

    stepwiseStatisticsChannel_ << Core::XmlOpen("search-step-stats") + Core::XmlAttribute("step", currentSearchStep_);
    stepStatisticsOpen_ = true;
}

void TreeRNNTTimesyncBeamSearch::closeStepStatistics() {
    if (not stepStatisticsOpen_) {
        return;
    }

    stepwiseStatisticsChannel_ << Core::XmlClose("search-step-stats");
    stepStatisticsOpen_ = false;
}

void TreeRNNTTimesyncBeamSearch::logStepStatistics() {
    if (debugChannel_.isOpen()) {
        std::stringstream ss;
        for (size_t hypIdx = 0ul; hypIdx < beam_.size(); ++hypIdx) {
            ss << "Hypothesis " << hypIdx + 1ul << ":  " << beam_[hypIdx].toString() << "\n";
        }
        ss << "\n";
        debugChannel_ << ss.str();
    }

    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlFull("active-hyps", beam_.size());
        stepwiseStatisticsChannel_ << Core::XmlFull("best-hyp-score", getBestHypothesis().score);
        stepwiseStatisticsChannel_ << Core::XmlFull("worst-hyp-score", getWorstHypothesis().score);
    }

    closeStepStatistics();
}

void TreeRNNTTimesyncBeamSearch::createSuccessorLookups() {
    size_t numStates = network_->structure.stateCount();

    stateSuccessorsOffset_.assign(numStates + 1, 0);
    stateExitsOffset_.assign(numStates + 1, 0);
    stateSuccessors_.clear();
    stateExits_.clear();
    blankLemmaStates_.clear();

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
                auto const& exit = network_->exits[it.label()];
                stateExits_.push_back(exit);
                if (lexicon_->lemmaPronunciation(exit.pronunciation)->lemma() == blankLemma_) {
                    blankLemmaStates_.insert(state);
                }
            }
        }
    }
    stateSuccessorsOffset_[numStates] = stateSuccessors_.size();
    stateExitsOffset_[numStates]      = stateExits_.size();
}

void TreeRNNTTimesyncBeamSearch::finalizeHypotheses() {
    innerHyps_.clear();
    for (auto const& hyp : beam_) {
        if (network_->finalStates.contains(hyp.currentState)) {
            innerHyps_.push_back(hyp);
        }
    }

    if (innerHyps_.empty() and sentenceEndFallback_) {  // There was no valid final hypothesis in the beam
        warning("No active word-end hypothesis at segment end.");
        log() << "Use sentence-end fallback";
        // The trace of the unfinished word keeps an empty pronunciation
        innerHyps_ = beam_;
    }

    if (not innerHyps_.empty()) {
        extensions_.clear();
        for (size_t hypIndex = 0ul; hypIndex < innerHyps_.size(); ++hypIndex) {
            auto const& hyp = innerHyps_[hypIndex];
            extensions_.push_back(
                    {.nextToken      = sentenceEndLabelIndex_,
                     .nextState      = hyp.currentState,
                     .timeframe      = hyp.timeframe,
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
            for (auto const& hyp : innerHyps_) {
                scoringContexts_.push_back(hyp.scoringContexts[scorerIdx]);
            }

            auto scoreAccessors = labelScorers_[scorerIdx]->getScoreAccessors(scoringContexts_);
            for (size_t extensionIdx = 0ul; extensionIdx < extensions_.size(); ++extensionIdx) {
                auto const& scoreAccessor = scoreAccessors[extensionIdx];
                if (not scoreAccessor) {
                    continue;
                }
                auto& ext = extensions_[extensionIdx];
                ext.score += (*scoreAccessor)->getScore(ext.transitionType, sentenceEndLabelIndex_);
                ext.timeframe = std::max(ext.timeframe, (*scoreAccessor)->getTime());
            }
        }

        // The scoring contexts are not updated as no further scoring is done afterwards
        newBeam_.clear();
        for (auto const& ext : extensions_) {
            auto const& baseHyp = innerHyps_[ext.baseHypIndex];
            newBeam_.push_back({baseHyp, ext, baseHyp.scoringContexts, lengthNormScale_});
        }

        // Add the LM's sentence-end score. The LM history is not updated as this is the last LM scoring step
        tempHypotheses_.clear();
        for (size_t hypIndex = 0ul; hypIndex < newBeam_.size(); ++hypIndex) {
            auto const&               hyp = newBeam_[hypIndex];
            WordEndExtensionCandidate ext{
                    .pron           = sentenceEndLemma_ ? static_cast<Bliss::LemmaPronunciation const*>(sentenceEndLemma_->pronunciations().first) : nullptr,
                    .rootState      = hyp.currentState,
                    .score          = hyp.score + languageModel_->sentenceEndScore(hyp.lmHistory),
                    .transitionType = Nn::TransitionType::SENTENCE_END,
                    .baseHypIndex   = hypIndex,
            };
            tempHypotheses_.push_back({hyp, ext, hyp.lmHistory, blankLemma_, lengthNormScale_});
        }
    }
    else {  // No valid final hypotheses and no sentence-end fallback
        // Construct an empty hypothesis with a lattice containing only one empty pronunciation from start to end
        tempHypotheses_.clear();
        tempHypotheses_.push_back(LabelHypothesis());
        tempHypotheses_.front().trace->time          = beam_.front().trace->time;  // Retrieve the timeframe from any hyp in the old beam
        tempHypotheses_.front().trace->pronunciation = nullptr;
        tempHypotheses_.front().trace->predecessor   = Core::ref(new LatticeTrace(0, {0, 0}, {}));
    }

    beam_.swap(tempHypotheses_);

    numActiveHyps_ += beam_.size();

    // Log statistics about the final beam
    if (stepwiseStatisticsChannel_.isOpen()) {
        stepwiseStatisticsChannel_ << Core::XmlOpen("final-beam-stats");
        stepwiseStatisticsChannel_ << Core::XmlFull("active-hyps", beam_.size());
        stepwiseStatisticsChannel_ << Core::XmlFull("best-hyp-score", getBestHypothesis().score);
        stepwiseStatisticsChannel_ << Core::XmlFull("worst-hyp-score", getWorstHypothesis().score);
        stepwiseStatisticsChannel_ << Core::XmlClose("final-beam-stats");
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

void TreeRNNTTimesyncBeamSearch::maximumStableDelayPruning() {
    if (currentSearchStep_ + 1 <= maximumStableDelay_) {
        return;
    }

    auto cutoff = currentSearchStep_ + 1 - maximumStableDelay_;

    std::vector<StableDelayPruning::Hypothesis> hyps;
    hyps.reserve(beam_.size());
    for (auto& hyp : beam_) {
        bool inBlankLemma = blankLemmaStates_.contains(hyp.currentState);
        hyps.push_back({&hyp.trace, false, inBlankLemma ? blankLemma_ : nullptr, hyp.score});
    }

    StableDelayPruning pruning(cutoff, [this](Bliss::Lemma const* lemma) {
        return lemma and lemma == blankLemma_;
    });
    std::vector<bool>  keep;
    if (not pruning.apply(hyps, keep)) {
        warning() << "Most recent word in best hypothesis is before cutoff point for maximum-stable-delay-pruning so the limit will be surpassed";
    }

    tempHypotheses_.clear();
    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        if (keep[hypIndex]) {
            tempHypotheses_.push_back(beam_[hypIndex]);
        }
    }
    beam_.swap(tempHypotheses_);
}

}  // namespace Search
