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

#include "ModelCombTreeTimesyncBeamSearch.hh"

#include <algorithm>
#include <strings.h>

#include <Am/Module.hh>
#include <Core/XmlStream.hh>
#include <Lattice/LatticeAdaptor.hh>
#include <Lm/BackingOff.hh>
#include <Lm/Module.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/TracebackHelper.hh>
#include "Search/Module.hh"
#include "Search/Traceback.hh"

namespace Search {

/*
 * =======================
 * === LabelHypothesis ===
 * =======================
 */

ModelCombTreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContext(),
          currentToken(Nn::invalidLabelIndex),
          currentState(invalidTreeNodeIndex),
          pron(nullptr),
          atWordEnd(false),
          lmHistory(),
          timeframe(0),
          score(0.0),
          lastWordEndScore(0.0),
          lifetime(0),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))),
          globalWordEnds(0),
          nonGlobalWordEnds(0) {}

ModelCombTreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        ModelCombTreeTimesyncBeamSearch::LabelHypothesis const&              base,
        ModelCombTreeTimesyncBeamSearch::WithinWordExtensionCandidate const& extension,
        Nn::ScoringContextRef const&                                         newScoringContext)
        : scoringContext(newScoringContext),
          currentToken(extension.nextToken),
          currentState(extension.state),
          pron(nullptr),
          atWordEnd(false),
          lmHistory(base.lmHistory),
          timeframe(extension.timeframe),
          score(extension.score),
          lastWordEndScore(base.lastWordEndScore),
          lifetime(base.lifetime),
          trace(base.trace),
          globalWordEnds(base.globalWordEnds),
          nonGlobalWordEnds(base.nonGlobalWordEnds) {}

ModelCombTreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        ModelCombTreeTimesyncBeamSearch::LabelHypothesis const&           base,
        ModelCombTreeTimesyncBeamSearch::WordEndExtensionCandidate const& extension,
        Nn::ScoringContextRef const&                                      newScoringContext)
        : scoringContext(newScoringContext),
          currentToken(extension.nextToken),
          currentState(extension.state),
          pron(extension.pron),
          atWordEnd(true),
          lmHistory(extension.lmHistory),
          timeframe(base.timeframe),
          score(extension.score),
          lastWordEndScore(extension.lastWordEndScore),
          lifetime(extension.lifetime),
          trace(base.trace),
          globalWordEnds(base.globalWordEnds),
          nonGlobalWordEnds(base.nonGlobalWordEnds) {
    if (extension.atGlobalWordEnd) {
        globalWordEnds += 1;
    }
    if (extension.atNonGlobalWordEnd) {
        nonGlobalWordEnds += 1;
    }

    if (not extension.previousWordEndHyp) {  // Don't start a new trace for previous word-end hyps
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
}

std::string ModelCombTreeTimesyncBeamSearch::LabelHypothesis::toString() const {
    std::stringstream ss;
    ss << "Score: " << score << ", lastWordEndScore: " << lastWordEndScore << ", current state: " << currentState << ", traceback: ";

    auto traceback = trace->performTraceback();

    for (auto& item : *traceback) {
        if (item.pronunciation and item.pronunciation->lemma()) {
            ss << item.pronunciation->lemma()->symbol() << " ";
        }
    }
    return ss.str();
}

/*
 * =============
 * === Model ===
 * =============
 */

const Core::ParameterFloat ModelCombTreeTimesyncBeamSearch::Model::paramModelScale(
        "scale",
        "This model's scale for score combination at word ends.",
        1.0);

const Core::ParameterInt ModelCombTreeTimesyncBeamSearch::Model::paramMaxBeamSize(
        "max-beam-size",
        "Maximum number of within-word hypotheses in the search beam.",
        -1);

const Core::ParameterInt ModelCombTreeTimesyncBeamSearch::Model::paramMaxWordEndBeamSize(
        "max-word-end-beam-size",
        "Maximum number of word-end hypotheses in the search beam. If not set, global beam pruning will be done and word-end hypotheses will not be pruned separately.",
        -1);

const Core::ParameterInt ModelCombTreeTimesyncBeamSearch::Model::paramMaxPreviousWordEndBeamSize(
        "max-previous-word-end-beam-size",
        "Maximum number of previous word ends waiting for combination.",
        -1);

const Core::ParameterInt ModelCombTreeTimesyncBeamSearch::Model::paramMaxPreviousWordEndLifetime(
        "max-previous-word-end-lifetime",
        "Maximum number of timeframes the previous word ends can wait for combination until they are discarded.",
        -1);

const Core::ParameterFloat ModelCombTreeTimesyncBeamSearch::Model::paramScoreThreshold(
        "score-threshold",
        "Prune any within-word hypothesis with a score that is at least this much worse than the best hypothesis.",
        -1.0);

const Core::ParameterFloat ModelCombTreeTimesyncBeamSearch::Model::paramWordEndScoreThreshold(
        "word-end-score-threshold",
        "Prune any word-end hypothesis with a score that is at least this much worse than the best word-end hypothesis. This threshold is relative to the score-threshold. \
        If not set, global score pruning will be done and word-end hypotheses will not be pruned separately.",
        1.0);

const Core::ParameterFloat ModelCombTreeTimesyncBeamSearch::Model::paramPreviousWordEndScoreThreshold(
        "previous-word-end-score-threshold",
        "Prune any previous word-end hypothesis with a score that is at least this much worse than the best previous word-end hypothesis. \
         This threshold is relative to the score-threshold.",
        1.0);

const Core::ParameterBool ModelCombTreeTimesyncBeamSearch::Model::paramCollapseRepeatedLabels(
        "collapse-repeated-labels",
        "Collapse repeated emission of the same label into one output. If false, every emission is treated like a new output.",
        false);

ModelCombTreeTimesyncBeamSearch::Model::Model(Core::Configuration const& config)
        : labelScorer(),
          lexicon(),
          nonWordLemmas(),
          network(),
          acousticModel(),
          scale(paramModelScale(config)),
          collapseRepeatedLabels(paramCollapseRepeatedLabels(config)),
          maxBeamSize(paramMaxBeamSize(config)),
          maxWordEndBeamSize(paramMaxWordEndBeamSize(config)),
          maxPreviousWordEndBeamSize(paramMaxPreviousWordEndBeamSize(config)),
          scoreThreshold(paramScoreThreshold(config)),
          wordEndScoreThreshold(paramWordEndScoreThreshold(config)),
          previousWordEndScoreThreshold(paramPreviousWordEndScoreThreshold(config)),
          maxLifetime(paramMaxPreviousWordEndLifetime(config)),
          withinWordExtensions(),
          wordEndExtensions(),
          beam(),
          newBeam(),
          requests(),
          previousWordEndExtensions(),
          previousWordEndHyps(),
          requestsForPreviousExtensions(),
          withinWordHyps(),
          wordEndHyps(),
          tempHypotheses(),
          numWithinWordHyps("num-within-word-hyps"),
          numIndividualWordEndHyps("num-individual-we-hyps"),
          numGlobalWordEndHyps("num-global-we-hyps"),
          numPreviousWordEndHyps("num-previous-we-hyps") {
    wordEndScoreThreshold *= scoreThreshold;
    previousWordEndScoreThreshold *= scoreThreshold;
}

/*
 * =======================================
 * === ModelCombTreeTimesyncBeamSearch ===
 * =======================================
 */
const Core::ParameterInt ModelCombTreeTimesyncBeamSearch::paramGlobalMaxBeamSize(
        "global-max-beam-size",
        "Maximum number of within-word hypotheses in the search beam.",
        1, 1);

const Core::ParameterInt ModelCombTreeTimesyncBeamSearch::paramGlobalMaxWordEndBeamSize(
        "global-max-word-end-beam-size",
        "Maximum number of word-end hypotheses in the search beam. If not set, global beam pruning will be done and word-end hypotheses will not be pruned separately.",
        std::numeric_limits<int>::max(), 0);

const Core::ParameterInt ModelCombTreeTimesyncBeamSearch::paramGlobalMaxPreviousWordEndBeamSize(
        "global-max-previous-word-end-beam-size",
        "Maximum number of previous word ends waiting for combination.",
        std::numeric_limits<int>::max(), 0);

const Core::ParameterFloat ModelCombTreeTimesyncBeamSearch::paramGlobalScoreThreshold(
        "global-score-threshold",
        "Prune any within-word hypothesis with a score that is at least this much worse than the best hypothesis.",
        Core::Type<Score>::max, 0);

const Core::ParameterFloat ModelCombTreeTimesyncBeamSearch::paramGlobalWordEndScoreThreshold(
        "global-word-end-score-threshold",
        "Prune any word-end hypothesis with a score that is at least this much worse than the best word-end hypothesis. This threshold is relative to the score-threshold. \
        If not set, global score pruning will be done and word-end hypotheses will not be pruned separately.",
        Core::Type<Score>::max, 0);

const Core::ParameterFloat ModelCombTreeTimesyncBeamSearch::paramGlobalPreviousWordEndScoreThreshold(
        "global-previous-word-end-score-threshold",
        "Prune any previous word-end hypothesis with a score that is at least this much worse than the best previous word-end hypothesis. \
         This threshold is relative to the score-threshold.",
        Core::Type<Score>::max, 0);

const Core::ParameterInt ModelCombTreeTimesyncBeamSearch::paramGlobalMaxPreviousWordEndLifetime(
        "global-max-previous-word-end-lifetime",
        "Maximum number of timeframes the previous word ends can wait for combination until they are discarded.",
        1);

const Core::ParameterBool ModelCombTreeTimesyncBeamSearch::paramPruneWordEndHypsBeforeSplit(
        "prune-word-ends-before-split",
        "Whether word-end extensions should be pruned before splitting into global and individual hypotheses.",
        true);

const Core::ParameterBool ModelCombTreeTimesyncBeamSearch::paramKeepGlobalWordEndHyps(
        "keep-global-word-ends-as-previous",
        "Whether global, combined word-end hypotheses should additionally be kept as individual (previous) ones in their respective model, \
		so they can be combined again in a later timeframe.",
        true);

const Core::ParameterBool ModelCombTreeTimesyncBeamSearch::paramAllowNonGlobalWordEnds(
        "allow-non-global-word-ends",
        "Allow adding individual word-end hypotheses of one model to the other model's beam with non-global-word-end-penalty added.",
        false);

const Core::ParameterFloat ModelCombTreeTimesyncBeamSearch::paramNonGlobalWordEndPenalty(
        "non-global-word-end-penalty",
        "Add this penalty to the score of individual word-end hypotheses which are added to the beam of all models",
        0.0);

const Core::ParameterInt ModelCombTreeTimesyncBeamSearch::paramMaxNonGlobalWordEndBeamSize(
        "max-non-global-word-end-beam-size",
        "Maximum number of individual word-end hypotheses which are transferred between the models.",
        std::numeric_limits<int>::max(), 0);

const Core::ParameterFloat ModelCombTreeTimesyncBeamSearch::paramNonGlobalWordEndScoreThreshold(
        "non-global-word-end-score-threshold",
        "Prune any individual word-end hypothesis with a score that is at least this much worse than the best individual word-end hypothesis. \
         This threshold is relative to the score-threshold.",
        Core::Type<Score>::max, 0);

const Core::ParameterBool ModelCombTreeTimesyncBeamSearch::paramSentenceEndFallBack(
        "sentence-end-fall-back",
        "Allow for fallback solution if no active word-end hypothesis exists at the end of a segment.",
        true);

const Core::ParameterBool ModelCombTreeTimesyncBeamSearch::paramLogStepwiseStatistics(
        "log-stepwise-statistics",
        "Log statistics about the beam at every search step.",
        false);

const Core::ParameterBool ModelCombTreeTimesyncBeamSearch::paramCacheCleanupInterval(
        "cache-cleanup-interval",
        "Interval of search steps after which buffered inputs that are not needed anymore get cleaned up.",
        10);

const Core::ParameterInt ModelCombTreeTimesyncBeamSearch::paramNumModels(
        "num-models",
        "Number of models to combine.",
        1, 1);

ModelCombTreeTimesyncBeamSearch::ModelCombTreeTimesyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          numModels_(paramNumModels(config)),
          globalMaxBeamSize_(paramGlobalMaxBeamSize(config)),
          globalMaxWordEndBeamSize_(paramGlobalMaxWordEndBeamSize(config)),
          globalMaxPreviousWordEndBeamSize_(paramGlobalMaxPreviousWordEndBeamSize(config)),
          globalScoreThreshold_(paramGlobalScoreThreshold(config)),
          globalWordEndScoreThreshold_(paramGlobalWordEndScoreThreshold(config)),
          globalPreviousWordEndScoreThreshold_(paramGlobalPreviousWordEndScoreThreshold(config)),
          globalMaxPreviousWordEndLifetime_(paramGlobalMaxPreviousWordEndLifetime(config)),
          allowNonGlobalWordEnds_(paramAllowNonGlobalWordEnds(config)),
          nonGlobalWordEndPenalty_(paramNonGlobalWordEndPenalty(config)),
          nonGlobalWordEndScoreThreshold_(paramNonGlobalWordEndScoreThreshold(config)),
          maxNonGlobalWordEndBeamSize_(paramMaxNonGlobalWordEndBeamSize(config)),
          pruneWordEndHypsBeforeSplit_(paramPruneWordEndHypsBeforeSplit(config)),
          keepGlobalWordEndHyps_(paramKeepGlobalWordEndHyps(config)),
          sentenceEndFallback_(paramSentenceEndFallBack(config)),
          cacheCleanupInterval_(paramCacheCleanupInterval(config)),
          logStepwiseStatistics_(paramLogStepwiseStatistics(config)),
          debugChannel_(config, "debug"),
          recombinedHypotheses_(),
          individualWordEndHypotheses_(),
          globalWordEndExtensions_(),
          finalBeam_(),
          currentSearchStep_(0ul),
          initializationTime_(),
          featureProcessingTime_(),
          scoringTime_(),
          finishedSegment_(false) {
    if (globalScoreThreshold_ == Core::Type<Score>::max and globalWordEndScoreThreshold_ != Core::Type<Score>::max) {
        error() << "Word-end score-threshold which is relative to the score-threshold is set, but score-threshold is not set";
    }
    globalWordEndScoreThreshold_ *= globalScoreThreshold_;
    globalPreviousWordEndScoreThreshold_ *= globalScoreThreshold_;
    nonGlobalWordEndScoreThreshold_ *= globalScoreThreshold_;
}

// Mode for the global model combination
Speech::ModelCombination::Mode ModelCombTreeTimesyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLexicon | Speech::ModelCombination::useLanguageModel;
}

Am::AcousticModel::Mode ModelCombTreeTimesyncBeamSearch::requiredAcousticModel() const {
    return Am::AcousticModel::noEmissions;
}

bool ModelCombTreeTimesyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    globalLexicon_       = modelCombination.lexicon();
    globalLanguageModel_ = modelCombination.languageModel();

    models_.reserve(numModels_);

    Speech::ModelCombination::Mode modelCombinationMode = Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon | Speech::ModelCombination::useAcousticModel;

    for (size_t i = 0ul; i < numModels_; ++i) {
        // Set the ModelCombination
        Core::Configuration modelConfig = select(std::string("model-") + std::to_string(i + 1));
        Core::Ref<Speech::ModelCombination> modelComb = Core::ref(new Speech::ModelCombination(modelConfig, modelCombinationMode, requiredAcousticModel(), globalLexicon_));
        models_.push_back(Model(modelConfig));

        models_[i].lexicon       = modelComb->lexicon();
        models_[i].acousticModel = modelComb->acousticModel();
        models_[i].labelScorer   = modelComb->labelScorer();

        models_[i].nonWordLemmas = models_[i].lexicon->specialLemmas("nonword");

        // Build the search tree
        auto                                 network = Core::ref(new PersistentStateTree(modelConfig, models_[i].acousticModel, models_[i].lexicon, std::bind(&Module_::createTreeBuilder, &Search::Module::instance(), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5)));
        std::unique_ptr<AbstractTreeBuilder> builder = Search::Module::instance().createTreeBuilder(modelConfig, *models_[i].lexicon, *models_[i].acousticModel, *network);
        builder->build();
        models_[i].network = network;

        if (models_[i].lexicon->specialLemma("blank")) {
            models_[i].blankLabelIndex = models_[i].acousticModel->emissionIndex(models_[i].acousticModel->blankAllophoneStateIndex());
        }
        else {
            error() << "All models require a special blank lemma";
        }

        models_[i].sentenceEndLemma = models_[i].lexicon->specialLemma("sentence-end");
        if (not models_[i].sentenceEndLemma) {
            models_[i].sentenceEndLemma = models_[i].lexicon->specialLemma("sentence-boundary");
        }
        if (models_[i].sentenceEndLemma and models_[i].sentenceEndLemma->nPronunciations() != 0 and models_[i].sentenceEndLemma->pronunciations().first->pronunciation()->length() > 0) {
            auto const* pron = models_[i].sentenceEndLemma->pronunciations().first->pronunciation();
            require(pron->length() == 1);
            Am::Allophone           allo(models_[i].acousticModel->phonology()->allophone(*pron, 0),
                                         Am::Allophone::isInitialPhone | Am::Allophone::isFinalPhone);
            Am::AllophoneStateIndex alloStateIdx = models_[i].acousticModel->allophoneStateAlphabet()->index(&allo, 0);

            models_[i].sentenceEndLabelIndex = models_[i].acousticModel->emissionIndex(alloStateIdx);
            log() << "Use sentence-end label with index " << models_[i].sentenceEndLabelIndex;
        }
        else {
            models_[i].sentenceEndLabelIndex = Nn::invalidLabelIndex;
        }

        // If the models don't have their own pruning parameters, use the global ones
        // TODO maybe develop a better way to handle global/non-global pruning parameters in general?
        if (models_[i].maxBeamSize == -1) {
            models_[i].maxBeamSize = globalMaxBeamSize_;
        }
        if (models_[i].maxWordEndBeamSize == -1) {
            models_[i].maxWordEndBeamSize = globalMaxWordEndBeamSize_;
        }
        if (models_[i].maxPreviousWordEndBeamSize == -1) {
            models_[i].maxPreviousWordEndBeamSize = globalMaxPreviousWordEndBeamSize_;
        }
        if (models_[i].scoreThreshold == -1.0) {
            models_[i].scoreThreshold = globalScoreThreshold_;
        }
        if (models_[i].wordEndScoreThreshold == -1.0) {
            models_[i].wordEndScoreThreshold = globalWordEndScoreThreshold_;
        }
        if (models_[i].previousWordEndScoreThreshold == -1.0) {
            models_[i].previousWordEndScoreThreshold = globalPreviousWordEndScoreThreshold_;
        }
        if (models_[i].maxLifetime == -1) {
            models_[i].maxLifetime = globalMaxPreviousWordEndLifetime_;
        }
    }

    // Create look-ups for state successors and exits of each state
    createSuccessorLookups();

    reset();

    return true;
}

void ModelCombTreeTimesyncBeamSearch::reset() {
    initializationTime_.start();

    for (size_t i = 0ul; i < numModels_; ++i) {
        models_[i].labelScorer->reset();

        // Reset beam to a single empty hypothesis
        models_[i].beam.clear();
        models_[i].beam.push_back(LabelHypothesis());
        models_[i].beam.front().scoringContext = models_[i].labelScorer->getInitialScoringContext();
        models_[i].beam.front().currentState   = models_[i].network->rootState;
        models_[i].beam.front().lmHistory      = globalLanguageModel_->startHistory();

        models_[i].previousWordEndExtensions.clear();
        models_[i].previousWordEndHyps.clear();

        models_[i].withinWordHyps.clear();
        models_[i].wordEndHyps.clear();
    }

    currentSearchStep_ = 0ul;
    finishedSegment_   = false;

    initializationTime_.stop();
}

void ModelCombTreeTimesyncBeamSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    initializationTime_.start();
    for (size_t i = 0ul; i < numModels_; ++i) {
        models_[i].labelScorer->reset();
        if (segment != nullptr) {
            globalLanguageModel_->setSegment(segment);
            for (auto& hyp : models_[i].beam) {
                hyp.lmHistory = globalLanguageModel_->startHistory();
            }
        }
    }
    resetStatistics();
    initializationTime_.stop();
    finishedSegment_ = false;
}

void ModelCombTreeTimesyncBeamSearch::finishSegment() {
    featureProcessingTime_.start();
    for (size_t i = 0ul; i < numModels_; ++i) {
        models_[i].labelScorer->signalNoMoreFeatures();
    }
    featureProcessingTime_.stop();
    decodeManySteps();
    logStatistics();
    finishedSegment_ = true;
    finalizeHypotheses();
}

void ModelCombTreeTimesyncBeamSearch::putFeature(Nn::DataView const& feature) {
    featureProcessingTime_.start();
    for (size_t i = 0ul; i < numModels_; ++i) {
        models_[i].labelScorer->addInput(feature);
    }
    featureProcessingTime_.stop();
}

void ModelCombTreeTimesyncBeamSearch::putFeatures(Nn::DataView const& features, size_t nTimesteps) {
    featureProcessingTime_.start();
    for (size_t i = 0ul; i < numModels_; ++i) {
        models_[i].labelScorer->addInputs(features, nTimesteps);
    }
    featureProcessingTime_.stop();
}

Core::Ref<const Traceback> ModelCombTreeTimesyncBeamSearch::getCurrentBestTraceback() const {
    auto bestHyp = getBestHypothesis();
    clog() << Core::XmlOpen("global-word-ends") << bestHyp.globalWordEnds << Core::XmlClose("global-word-ends");
    clog() << Core::XmlOpen("non-global-word-ends") << bestHyp.nonGlobalWordEnds << Core::XmlClose("non-global-word-ends");
    return getBestHypothesis().trace->performTraceback();
}

Core::Ref<const LatticeAdaptor> ModelCombTreeTimesyncBeamSearch::getCurrentBestWordLattice() const {
    auto&        bestHypothesis = getBestHypothesis();
    LatticeTrace endTrace(bestHypothesis.trace, 0, bestHypothesis.trace->time + 1, bestHypothesis.trace->score, {});

    for (size_t hypIdx = 1ul; hypIdx < finalBeam_.size(); ++hypIdx) {
        auto& hyp          = finalBeam_[hypIdx];
        auto  siblingTrace = Core::ref(new LatticeTrace(hyp.trace, 0, hyp.trace->time, hyp.trace->score, {}));
        endTrace.appendSiblingToChain(siblingTrace);
    }

    // TODO don't hardcode models_[0].lexicon here
    // instead, maybe use globalLexicon_?
    return endTrace.buildWordLattice(models_[0].lexicon);
}

Core::Ref<const LatticeTrace> ModelCombTreeTimesyncBeamSearch::getCurrentBestLatticeTrace() const {
    return getBestHypothesis().trace;
}

Core::Ref<const LatticeTrace> ModelCombTreeTimesyncBeamSearch::getCommonPrefix() const {
    std::vector<Core::Ref<LatticeTrace>> traces(models_[0].beam.size());
    for (size_t hypIndex = 0ul; hypIndex < models_[0].beam.size(); ++hypIndex) {
        traces[hypIndex] = models_[0].beam[hypIndex].trace;
    }

    RootTraceSearcher searcher(traces);
    if (not searcher.rootTrace()) {
        warning("Common prefix of all traces is a sentinel value");
    }

    return Core::Ref<const LatticeTrace>(searcher.rootTrace());
}

bool ModelCombTreeTimesyncBeamSearch::decodeStep() {
    if (finishedSegment_) {
        return false;
    }

    if (debugChannel_.isOpen()) {
        std::stringstream ss;
        ss << "Start Search Step " << currentSearchStep_ << "\n";
        for (size_t i = 0ul; i < numModels_; ++i) {
            ss << "Model " << i << ":\n";
            for (size_t hypIdx = 0ul; hypIdx < models_[i].beam.size(); ++hypIdx) {
                ss << "Hypothesis " << hypIdx + 1ul << ":  " << models_[i].beam[hypIdx].toString() << ", AM index: " << models_[i].network->structure.state(models_[i].beam[hypIdx].currentState).stateDesc.acousticModel << "\n";
            }
            for (size_t hypIdx = 0ul; hypIdx < models_[i].previousWordEndHyps.size(); ++hypIdx) {
                ss << "Previous Word-End Hypothesis " << hypIdx + 1ul << ":  " << models_[i].previousWordEndHyps[hypIdx].toString() << ", AM index: " << models_[i].network->structure.state(models_[i].previousWordEndHyps[hypIdx].currentState).stateDesc.acousticModel << ", lifetime " << models_[i].previousWordEndHyps[hypIdx].lifetime << "\n";
            }
            ss << "\n";
        }
        debugChannel_ << ss.str();
    }

    /*
     * Collect all possible extensions for all hypotheses in the beam for all models.
     * Also create scoring requests for the label scorer.
     * Each extension candidate makes up a request.
     */
    for (size_t i = 0ul; i < numModels_; ++i) {
        models_[i].withinWordExtensions.clear();
        models_[i].requests.clear();
        models_[i].newBeam.clear();
        for (size_t hypIndex = 0ul; hypIndex < models_[i].beam.size(); ++hypIndex) {
            auto& hyp = models_[i].beam[hypIndex];

            // Iterate over the successors of this hypothesis' current state in the tree
            for (size_t j = models_[i].stateSuccessorsOffset[hyp.currentState]; j < models_[i].stateSuccessorsOffset[hyp.currentState + 1]; ++j) {
                const StateId successorState = models_[i].stateSuccessors[j];

                Nn::LabelIndex tokenIdx = models_[i].network->structure.state(successorState).stateDesc.acousticModel;

                // If we collapse repeated labels, a new word should not start with the same token as the previous word ended (except for blank itself)
                if (models_[i].collapseRepeatedLabels and
                    hyp.currentState == models_[i].network->rootState and
                    tokenIdx == hyp.currentToken and
                    tokenIdx != models_[i].blankLabelIndex) {
                    continue;
                }

                auto transitionType = inferTransitionType(hyp.currentToken, tokenIdx, models_[i].collapseRepeatedLabels, models_[i].blankLabelIndex);

                models_[i].withinWordExtensions.push_back(
                        {tokenIdx,
                         successorState,
                         hyp.score,
                         0,
                         transitionType,
                         hypIndex});
                models_[i].requests.push_back({models_[i].beam[hypIndex].scoringContext, tokenIdx, transitionType});

                // If we predict the blank lemma from the root, also keep it as a within-word extension so that we can start from the root in the next step
                // Otherwise there is the edge case that only one hypothesis is left in the beam in this state in one model
                // and if an other model does not have a blank word-end hyp, it disappears and the model's beam is empty
                if (hyp.currentState == models_[i].network->rootState and tokenIdx == models_[i].blankLabelIndex) {
                    models_[i].withinWordExtensions.push_back(
                            {tokenIdx,
                             models_[i].network->rootState,
                             hyp.score,
                             0,
                             transitionType,
                             hypIndex});
                    models_[i].requests.push_back({models_[i].beam[hypIndex].scoringContext, tokenIdx, transitionType});
                }
            }
        }

        // Create blank-extension for previous word-end extensions
        models_[i].previousWordEndExtensions.clear();
        models_[i].requestsForPreviousExtensions.clear();
        for (size_t extensionIdx = 0ul; extensionIdx < models_[i].previousWordEndHyps.size(); ++extensionIdx) {
            auto& previousHyp = models_[i].previousWordEndHyps[extensionIdx];

            auto transitionType = previousHyp.currentToken == models_[i].blankLabelIndex ? Nn::LabelScorer::TransitionType::BLANK_LOOP : Nn::LabelScorer::TransitionType::LABEL_TO_BLANK;

            models_[i].newBeam.push_back(previousHyp);

            models_[i].previousWordEndExtensions.push_back(
                    {models_[i].blankLabelIndex,
                     previousHyp.pron,
                     previousHyp.currentState,	// Stay in the root state
                     false,
                     false,
                     previousHyp.lmHistory,
                     previousHyp.score,
                     0.0,
                     0,
                     transitionType,
                     previousHyp.lifetime + 1,
                     true,
                     previousHyp.lastWordEndScore,
                     models_[i].newBeam.size() - 1});
            models_[i].requestsForPreviousExtensions.push_back({previousHyp.scoringContext, models_[i].blankLabelIndex, transitionType});
        }
    }

    if (logStepwiseStatistics_) {
        clog() << Core::XmlOpen("search-step-stats");
    }

    /*
     * Perform scoring of all the requests with the label scorer in all models.
     */
    scoringTime_.start();
    std::vector<std::optional<Nn::LabelScorer::ScoresWithTimes>> results(numModels_);
    std::vector<std::optional<Nn::LabelScorer::ScoresWithTimes>> resultsForPreviousExtensions(numModels_);
    for (size_t i = 0ul; i < numModels_; ++i) {
        auto result                     = models_[i].labelScorer->computeScoresWithTimes(models_[i].requests);
        results[i]                      = result;
        auto resultForPrevious          = models_[i].labelScorer->computeScoresWithTimes(models_[i].requestsForPreviousExtensions);
        resultsForPreviousExtensions[i] = resultForPrevious;
    }
    scoringTime_.stop();

    if (std::all_of(results.begin(), results.end(), [](const std::optional<Nn::LabelScorer::ScoresWithTimes>& result) { return not result; })) {
        // LabelScorer of all models could not compute scores -> no search step can be made.
        if (logStepwiseStatistics_) {
            clog() << Core::XmlClose("search-step-stats");
        }
        return false;
    }

    for (size_t i = 0ul; i < numModels_; ++i) {
        if (results[i]) {
            for (size_t requestIdx = 0ul; requestIdx < models_[i].withinWordExtensions.size(); ++requestIdx) {
                models_[i].withinWordExtensions[requestIdx].score += results[i]->scores[requestIdx];
                models_[i].withinWordExtensions[requestIdx].timeframe = results[i]->timeframes[requestIdx];
            }
        }

        if (resultsForPreviousExtensions[i]) {
            for (size_t requestIdx = 0ul; requestIdx < models_[i].previousWordEndExtensions.size(); ++requestIdx) {
                models_[i].previousWordEndExtensions[requestIdx].score += resultsForPreviousExtensions[i]->scores[requestIdx];
                models_[i].previousWordEndExtensions[requestIdx].timeframe = resultsForPreviousExtensions[i]->timeframes[requestIdx];
            }
        }
    }

    // Prune and recombine within-word extensions
    for (size_t i = 0ul; i < numModels_; ++i) {
        scorePruning(models_[i].withinWordExtensions, models_[i].scoreThreshold);

        models_[i].withinWordHyps.clear();
        for (auto const& extension : models_[i].withinWordExtensions) {
            auto const& baseHyp = models_[i].beam[extension.baseHypIndex];

            auto newScoringContext = models_[i].labelScorer->extendedScoringContext(
                    {baseHyp.scoringContext,
                     extension.nextToken,
                     extension.transitionType});

            models_[i].withinWordHyps.push_back({baseHyp, extension, newScoringContext});
        }

        recombination(models_[i].withinWordHyps);

        beamSizePruning(models_[i].withinWordHyps, models_[i].maxBeamSize);
    }

    if (debugChannel_.isOpen()) {
        std::stringstream ss;
        ss << "After Pruning:\n";
        for (size_t i = 0ul; i < numModels_; ++i) {
            ss << "Model " << i << "\n";
            ss << "Previous Word-End Extensions" << "\n";
            for (size_t hypIdx = 0ul; hypIdx < models_[i].previousWordEndExtensions.size(); ++hypIdx) {
                ss << "Previous Extension " << hypIdx + 1ul << ": current state: " << models_[i].previousWordEndExtensions[hypIdx].state << ", AM index: " << models_[i].network->structure.state(models_[i].previousWordEndExtensions[hypIdx].state).stateDesc.acousticModel << ", next token: " << models_[i].previousWordEndExtensions[hypIdx].nextToken << ", score: " << models_[i].previousWordEndExtensions[hypIdx].score << ", base hyp: " << models_[i].previousWordEndExtensions[hypIdx].baseHypIndex + 1 << ", pron: " << models_[i].previousWordEndExtensions[hypIdx].pron->lemma()->preferredOrthographicForm().str() << ", lifetime " << models_[i].previousWordEndExtensions[hypIdx].lifetime << "\n";
            }

            ss << "Within-word hyps" << "\n";
            for (size_t hypIdx = 0ul; hypIdx < models_[i].withinWordHyps.size(); ++hypIdx) {
                ss << "Hypothesis " << hypIdx + 1ul << ":  " << models_[i].withinWordHyps[hypIdx].toString() << ", AM index: " << models_[i].network->structure.state(models_[i].withinWordHyps[hypIdx].currentState).stateDesc.acousticModel << "\n";
            }
            ss << "\n";
        }
        debugChannel_ << ss.str();
    }

    /*
     * Expand hypotheses to word-end extensions
     */
    for (size_t i = 0ul; i < numModels_; ++i) {
        models_[i].wordEndExtensions.clear();
        for (size_t hypIndex = 0ul; hypIndex < models_[i].withinWordHyps.size(); ++hypIndex) {
            auto& hyp = models_[i].withinWordHyps[hypIndex];

            models_[i].newBeam.push_back(hyp);

            // Create one word-end hypothesis for each exit
            for (size_t j = models_[i].stateExitsOffset[hyp.currentState]; j < models_[i].stateExitsOffset[hyp.currentState + 1]; ++j) {
                const PersistentStateTree::Exit  exit      = models_[i].stateExits[j];
                const Bliss::LemmaPronunciation* lemmaPron = models_[i].lexicon->lemmaPronunciation(exit.pronunciation);
                const Bliss::Lemma*              lemma     = lemmaPron->lemma();

                Score                           penalty               = 0.0;
                Nn::LabelScorer::TransitionType wordEndtransitionType = Nn::LabelScorer::WORD_EXIT;
                if (lemma == models_[i].lexicon->specialLemma("silence")) {
                    wordEndtransitionType = Nn::LabelScorer::SILENCE_EXIT;
                }
                else if (models_[i].nonWordLemmas.contains(lemma)) {
                    wordEndtransitionType = Nn::LabelScorer::NONWORD_EXIT;
                }
                auto result = models_[i].labelScorer->computeScoreWithTime({hyp.scoringContext, Nn::invalidLabelIndex, wordEndtransitionType});
                if (result) {
                    penalty += result->score;
                }

                WordEndExtensionCandidate wordEndExtension{hyp.currentToken,
                                                           lemmaPron,
                                                           exit.transitState,  // Start from the root node (the exit's transit state) in the next step
                                                           false,
                                                           false,
                                                           hyp.lmHistory,
                                                           hyp.score + penalty,
                                                           0.0,
                                                           static_cast<TimeframeIndex>(currentSearchStep_),
                                                           wordEndtransitionType,
                                                           0,
                                                           false,
                                                           hyp.score,
                                                           models_[i].newBeam.size() - 1};

                models_[i].wordEndExtensions.push_back(wordEndExtension);
            }
        }
    }

    // Optionally prune new word-end extensions and previous word-end extensions separately
    // Recombination is not possible for new word-end extensions as the LM-history is not updated yet    TODO still possible with reduced history and considering pron?
    // And recombination for previous word-end extensions is not necessary as they have already been recombined
    for (size_t i = 0ul; i < numModels_; ++i) {
        if (pruneWordEndHypsBeforeSplit_) {
            scorePruning(models_[i].wordEndExtensions, models_[i].wordEndScoreThreshold);
            beamSizePruning(models_[i].wordEndExtensions, models_[i].maxWordEndBeamSize);

            scorePruning(models_[i].previousWordEndExtensions, models_[i].previousWordEndScoreThreshold);
            beamSizePruning(models_[i].previousWordEndExtensions, models_[i].maxPreviousWordEndBeamSize);
        }
        models_[i].wordEndExtensions.insert(models_[i].wordEndExtensions.end(), models_[i].previousWordEndExtensions.begin(), models_[i].previousWordEndExtensions.end());
    }

    if (debugChannel_.isOpen()) {
        std::stringstream ss;
        ss << "Word-End Extensions:\n";
        for (size_t i = 0ul; i < numModels_; ++i) {
            ss << "Model " << i << "\n";
            for (size_t hypIdx = 0ul; hypIdx < models_[i].wordEndExtensions.size(); ++hypIdx) {
                ss << "Extension " << hypIdx + 1ul << ": current state: " << models_[i].wordEndExtensions[hypIdx].state << ", AM index: " << models_[i].network->structure.state(models_[i].wordEndExtensions[hypIdx].state).stateDesc.acousticModel << ", next token: " << models_[i].wordEndExtensions[hypIdx].nextToken << ", score: " << models_[i].wordEndExtensions[hypIdx].score << ", lastWordEndScore: " << models_[i].wordEndExtensions[hypIdx].lastWordEndScore << ", base hyp: " << models_[i].wordEndExtensions[hypIdx].baseHypIndex + 1 << ", pron: " << models_[i].wordEndExtensions[hypIdx].pron->lemma()->preferredOrthographicForm().str() << ", LM history: " << globalLanguageModel_->formatHistory(models_[i].wordEndExtensions[hypIdx].lmHistory) << "\n";
            }
            ss << "\n";
        }
        debugChannel_ << ss.str();
    }

    // Split individual and global word-end extensions
    wordEndExtensionHandling();

    for (size_t i = 0ul; i < numModels_; ++i) {
        // Form hypotheses from global word-end extensions and previous word-end extensions, optionally prune, and recombine them separately
        if (not pruneWordEndHypsBeforeSplit_) {
            scorePruning(models_[i].wordEndExtensions, models_[i].wordEndScoreThreshold);
            scorePruning(models_[i].previousWordEndExtensions, models_[i].previousWordEndScoreThreshold);
        }

        models_[i].wordEndHyps.clear();
        for (auto& extension : models_[i].wordEndExtensions) {
            auto const& baseHyp = models_[i].newBeam[extension.baseHypIndex];

            auto newScoringContext = baseHyp.scoringContext;  // The scoring context was already updated
            if (extension.previousWordEndHyp) {
                newScoringContext = models_[i].labelScorer->extendedScoringContext(
                        {baseHyp.scoringContext,
                         extension.nextToken,
                         extension.transitionType});
            }
            models_[i].wordEndHyps.push_back({baseHyp, extension, newScoringContext});
        }
        recombination(models_[i].wordEndHyps);

        models_[i].previousWordEndHyps.clear();
        for (auto& extension : models_[i].previousWordEndExtensions) {
            auto const& baseHyp = models_[i].newBeam[extension.baseHypIndex];

            auto newScoringContext = baseHyp.scoringContext;
            if (extension.previousWordEndHyp) {
                newScoringContext = models_[i].labelScorer->extendedScoringContext(
                        {baseHyp.scoringContext,
                         extension.nextToken,
                         extension.transitionType});
            }
            models_[i].previousWordEndHyps.push_back({baseHyp, extension, newScoringContext});
        }
        recombinationPrevious(models_[i].previousWordEndHyps);

        if (not pruneWordEndHypsBeforeSplit_) {
            beamSizePruning(models_[i].wordEndHyps, models_[i].maxWordEndBeamSize);
            beamSizePruning(models_[i].previousWordEndHyps, models_[i].maxPreviousWordEndBeamSize);
        }

        // Log statistics about the beams
        models_[i].numWithinWordHyps += models_[i].withinWordHyps.size();
        models_[i].numPreviousWordEndHyps += models_[i].previousWordEndHyps.size();
        models_[i].numGlobalWordEndHyps += models_[i].wordEndHyps.size();
        models_[i].numIndividualWordEndHyps += individualWordEndHypotheses_.size();
        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("num-within-word-hyps", models_[i].withinWordHyps.size());
            clog() << Core::XmlFull("num-previous-we-hyps", models_[i].previousWordEndHyps.size());
            clog() << Core::XmlFull("num-global-we-hyps", models_[i].wordEndHyps.size());
            clog() << Core::XmlFull("num-individual-we-hyps", individualWordEndHypotheses_.size());
        }

        // Collect within-word hyps and word-end hyps in beam to continue with them in the next timestep
        // The previous word-end hyps are kept separate as they are only allowed to continue with blank
        for (size_t hypIdx = 0ul; hypIdx < individualWordEndHypotheses_.size(); ++hypIdx) {
            auto hyp         = individualWordEndHypotheses_[hypIdx];
            hyp.currentState = models_[i].network->rootState;  // Hypothesis may originate from another model, need to set the state to the model's root state
            models_[i].wordEndHyps.push_back(hyp);
        }
        models_[i].withinWordHyps.insert(models_[i].withinWordHyps.end(), models_[i].wordEndHyps.begin(), models_[i].wordEndHyps.end());
        models_[i].beam.swap(models_[i].withinWordHyps);
    }

    ++currentSearchStep_;

    /*
     * Clean up label scorer caches.
     */
    if (currentSearchStep_ % cacheCleanupInterval_ == 0) {
        for (size_t i = 0ul; i < numModels_; ++i) {
            Core::CollapsedVector<Nn::ScoringContextRef> activeContexts;
            for (auto const& hyp : models_[i].beam) {
                activeContexts.push_back(hyp.scoringContext);
            }
            models_[i].labelScorer->cleanupCaches(activeContexts);
        }
    }

    return true;
}

ModelCombTreeTimesyncBeamSearch::LabelHypothesis const& ModelCombTreeTimesyncBeamSearch::getBestHypothesis() const {
    verify(not finalBeam_.empty());

    return *std::min_element(finalBeam_.begin(), finalBeam_.end());
}

ModelCombTreeTimesyncBeamSearch::LabelHypothesis const& ModelCombTreeTimesyncBeamSearch::getWorstHypothesis() const {
    verify(not finalBeam_.empty());

    return *std::max_element(finalBeam_.begin(), finalBeam_.end());
}

void ModelCombTreeTimesyncBeamSearch::resetStatistics() {
    initializationTime_.reset();
    featureProcessingTime_.reset();
    scoringTime_.reset();
    for (size_t i = 0ul; i < numModels_; ++i) {
        models_[i].numWithinWordHyps.clear();
        models_[i].numIndividualWordEndHyps.clear();
        models_[i].numGlobalWordEndHyps.clear();
        models_[i].numPreviousWordEndHyps.clear();
    }
}

void ModelCombTreeTimesyncBeamSearch::logStatistics() const {
    clog() << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    clog() << Core::XmlOpen("initialization-time") << initializationTime_.elapsedMilliseconds() << Core::XmlClose("initialization-time");
    clog() << Core::XmlOpen("feature-processing-time") << featureProcessingTime_.elapsedMilliseconds() << Core::XmlClose("feature-processing-time");
    clog() << Core::XmlOpen("scoring-time") << scoringTime_.elapsedMilliseconds() << Core::XmlClose("scoring-time");
    clog() << Core::XmlClose("timing-statistics");
    for (size_t i = 0ul; i < numModels_; ++i) {
        models_[i].numWithinWordHyps.write(clog());
        models_[i].numIndividualWordEndHyps.write(clog());
        models_[i].numGlobalWordEndHyps.write(clog());
        models_[i].numPreviousWordEndHyps.write(clog());
    }
}

void ModelCombTreeTimesyncBeamSearch::wordEndExtensionHandling() {
    // Helper to compare extension for the same lemma pronunciation and LM-history
    // TODO find a better way for LM-history (and pron) comparison
    auto compareExtensions = [this](const WordEndExtensionCandidate& extensionCandidate1, const WordEndExtensionCandidate& extensionCandidate2) {
        return extensionCandidate1 == extensionCandidate2 and globalLanguageModel_->formatHistory(extensionCandidate1.lmHistory) == globalLanguageModel_->formatHistory(extensionCandidate2.lmHistory) and not(extensionCandidate1.previousWordEndHyp and extensionCandidate2.previousWordEndHyp);
    };

    // Helper to find a key inside a vector and return its index (or -1 if not found)
    auto findIndex = [this, &compareExtensions](const std::vector<WordEndExtensionCandidate>& extensions,
                                                const WordEndExtensionCandidate&              extensionCandidate) {
        auto it = std::find_if(extensions.begin(), extensions.end(),
                               [&extensionCandidate, &compareExtensions](const WordEndExtensionCandidate& other) {
                                   return compareExtensions(extensionCandidate, other);
                               });
        return it == extensions.end() ? static_cast<std::size_t>(-1) : static_cast<std::size_t>(std::distance(extensions.begin(), it));
    };

    // Start with the distinct word-end extensions of the first model
    globalWordEndExtensions_.clear();
    for (const auto& extension : models_[0].wordEndExtensions) {
        if (findIndex(globalWordEndExtensions_, extension) == static_cast<std::size_t>(-1)) {
            globalWordEndExtensions_.push_back(extension);
        }
    }

    // Keep only those that also appear in every other model
    for (size_t i = 1ul; i < numModels_; ++i) {
        for (auto it = globalWordEndExtensions_.begin(); it != globalWordEndExtensions_.end();) {
            bool present = std::any_of(
                    models_[i].wordEndExtensions.begin(),
                    models_[i].wordEndExtensions.end(),
                    [&](const WordEndExtensionCandidate& extension) {
                        return compareExtensions(extension, *it);
                    });

            if (!present) {
                it = globalWordEndExtensions_.erase(it);
            }
            else {
                ++it;
            }
        }
    }

    if (debugChannel_.isOpen()) {
        std::stringstream ss;
        ss << "Global Word-End Extensions:\n";
        for (size_t hypIdx = 0ul; hypIdx < globalWordEndExtensions_.size(); ++hypIdx) {
            ss << "Extension " << hypIdx + 1ul << ": pron: " << globalWordEndExtensions_[hypIdx].pron->lemma()->symbol().str() << ", history: " << globalLanguageModel_->formatHistory(globalWordEndExtensions_[hypIdx].lmHistory) << "\n";
        }
        ss << "\n";
        debugChannel_ << ss.str();
    }

    // For the global extensions, retrieve the best score per model
    std::vector<std::vector<Score>> best(numModels_, std::vector<Score>(globalWordEndExtensions_.size(), Core::Type<Score>::max));
    for (size_t i = 0ul; i < numModels_; ++i) {
        for (const auto& extension : models_[i].wordEndExtensions) {
            std::size_t idx = findIndex(globalWordEndExtensions_, extension);
            if (idx != static_cast<std::size_t>(-1)) {
                best[i][idx] = std::min(best[i][idx], extension.lastWordEndScore);
            }
        }
    }

    // Calculate the LM score for the global word-end extensions (but don't add it to the overall score for now)
    for (auto& extension : globalWordEndExtensions_) {
        const Bliss::Lemma*                 lemma = extension.pron->lemma();
        const Bliss::SyntacticTokenSequence sts   = lemma->syntacticTokenSequence();
        if (sts.size() != 0) {
            require(sts.size() == 1);
            const Bliss::SyntacticToken* st      = sts.front();
            Lm::Score                    lmScore = globalLanguageModel_->score(extension.lmHistory, st);
            extension.lmScore                    = lmScore;
        }
    }

    // Spilt each model's word-end extension into global and individual extensions
    // and rescore every global occurrence (= combine the scores of the models + add LM score)
    for (size_t i = 0ul; i < numModels_; ++i) {
        models_[i].previousWordEndExtensions.clear();

        auto& extensions = models_[i].wordEndExtensions;
        auto& previous   = models_[i].previousWordEndExtensions;
        float scale      = models_[i].scale;

        auto isGlobal = [&](const WordEndExtensionCandidate& extension) {
            return findIndex(globalWordEndExtensions_, extension) != static_cast<std::size_t>(-1);
        };

        // Re-order: [globals][non-globals]
        auto split = std::stable_partition(extensions.begin(), extensions.end(), isGlobal);
        if (not keepGlobalWordEndHyps_) {
            previous.assign(split, extensions.end());  // Move the non-global elements to previousWordEndExtensions
        }
        else {
            previous.assign(extensions.begin(), extensions.end());  // Keep global ones also als previous extensions
        }
        extensions.erase(split, extensions.end());  // Keep only the global ones in wordEndExtensions

        // Compute the score for every occurrence that remains
        // For calculating the new score, the extension's score of this model is taken as well as the best score from all other models
        for (auto& ext : extensions) {
            size_t idx = findIndex(globalWordEndExtensions_, ext);

            // Scale the real word-end score (lastWordEndScore) and re-add the score which was added after the word-end
            Score combinedScore = ext.lastWordEndScore * scale + (ext.score - ext.lastWordEndScore);

            // Now add the scaled score of the other models to form the overall score
            for (std::size_t k = 0; k < numModels_; ++k) {
                if (k == i) {  // Skip the score of this model
                    continue;
                }
                combinedScore += best[k][idx] * models_[k].scale;
            }
            ext.score   = combinedScore;
            ext.lmScore = globalWordEndExtensions_[idx].lmScore;
            ext.score += ext.lmScore;  // Add the previously calculated LM score to the total score
            ext.lastWordEndScore = 0.0;

            // Update the LM history
            const Bliss::Lemma*                 lemma = ext.pron->lemma();
            const Bliss::SyntacticTokenSequence sts   = lemma->syntacticTokenSequence();
            if (sts.size() != 0) {
                require(sts.size() == 1);
                const Bliss::SyntacticToken* st = sts.front();
                ext.lmHistory                   = globalLanguageModel_->extendedHistory(ext.lmHistory, st);
            }
            if (ext.nextToken != models_[i].blankLabelIndex) {
                ext.atGlobalWordEnd = true;
            }
        }
    }

    // Prepare the individual word-end hypotheses:
    // The non-global word-end extensions are kept separately and a penalty is added
    // They are later added to the beam of all models
    if (allowNonGlobalWordEnds_) {
        individualWordEndHypotheses_.clear();
        for (size_t i = 0ul; i < numModels_; ++i) {
            for (size_t hypIdx = 0ul; hypIdx < models_[i].previousWordEndExtensions.size(); ++hypIdx) {
                auto const prevExtension = models_[i].previousWordEndExtensions[hypIdx];

                // Make sure this is a new word-end extension from this timestep
                if (prevExtension.lifetime > 0) {
                    continue;
                }

                auto ext = WordEndExtensionCandidate(prevExtension);

                // Add the LM score and update the LM history
                const Bliss::Lemma*                 lemma = ext.pron->lemma();
                const Bliss::SyntacticTokenSequence sts   = lemma->syntacticTokenSequence();
                if (sts.size() != 0) {
                    require(sts.size() == 1);
                    const Bliss::SyntacticToken* st = sts.front();
                    ext.lmScore                     = globalLanguageModel_->score(ext.lmHistory, st);
                    ext.lmHistory                   = globalLanguageModel_->extendedHistory(ext.lmHistory, st);
                }

                if (ext.nextToken != models_[i].blankLabelIndex) {
                    ext.atNonGlobalWordEnd = true;
                }

                // Calculate the new score:
                // scale the extension's score with its model's scale and add the extension's score + penalty scaled by the other models' scale
                Score scaledExtensionScore = ext.score * models_[i].scale;
                for (size_t j = 0ul; j < numModels_; ++j) {
                    if (j != i) {
                        scaledExtensionScore += (ext.score + nonGlobalWordEndPenalty_) * models_[j].scale;
                    }
                }

                ext.score           = scaledExtensionScore + ext.lmScore;
                auto const& baseHyp = models_[i].newBeam[ext.baseHypIndex];
                individualWordEndHypotheses_.push_back({baseHyp, ext, baseHyp.scoringContext});
                // TODO this does not work with autoregressive models (if the scoringContext depends on the predicted token)
            }
        }

        // Prune and recombine them
        scorePruning(individualWordEndHypotheses_, nonGlobalWordEndScoreThreshold_);
        recombination(individualWordEndHypotheses_);
        beamSizePruning(individualWordEndHypotheses_, maxNonGlobalWordEndBeamSize_);
    }

    // Remove previous word-end hyps whose lifetime has exceeded
    for (size_t i = 0ul; i < numModels_; ++i) {
        models_[i].previousWordEndExtensions.erase(
                std::remove_if(
                        models_[i].previousWordEndExtensions.begin(),
                        models_[i].previousWordEndExtensions.end(),
                        [=](auto const& ext) { return ext.lifetime > models_[i].maxLifetime; }),
                models_[i].previousWordEndExtensions.end());
    }

    if (debugChannel_.isOpen()) {
        std::stringstream ss;
        ss << "Global Word-end Extensions:\n";
        for (size_t i = 0ul; i < numModels_; ++i) {
            ss << "Model " << i << "\n";
            for (size_t hypIdx = 0ul; hypIdx < models_[i].wordEndExtensions.size(); ++hypIdx) {
                ss << "Extension " << hypIdx + 1ul << ": current state: " << models_[i].wordEndExtensions[hypIdx].state << ", AM index: " << models_[i].network->structure.state(models_[i].wordEndExtensions[hypIdx].state).stateDesc.acousticModel << ", next token: " << models_[i].wordEndExtensions[hypIdx].nextToken << ", score: " << models_[i].wordEndExtensions[hypIdx].score << ", LM score: " << models_[i].wordEndExtensions[hypIdx].lmScore << ", base hyp: " << models_[i].wordEndExtensions[hypIdx].baseHypIndex + 1 << ", pron: " << models_[i].wordEndExtensions[hypIdx].pron->lemma()->preferredOrthographicForm().str() << ", LM history: " << globalLanguageModel_->formatHistory(models_[i].wordEndExtensions[hypIdx].lmHistory) << "\n";
            }
            ss << "\n";
        }
        ss << "Individual Word-end Extensions:\n";
        for (size_t hypIdx = 0ul; hypIdx < individualWordEndHypotheses_.size(); ++hypIdx) {
            ss << "Hypothesis " << hypIdx + 1ul << ":  " << individualWordEndHypotheses_[hypIdx].toString() << "\n";
        }
        ss << "\n";
        debugChannel_ << ss.str();
    }
}

Nn::LabelScorer::TransitionType ModelCombTreeTimesyncBeamSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel, bool collapseRepeatedLabels, Nn::LabelIndex blankLabelIndex) const {
    bool prevIsBlank = blankLabelIndex != Nn::invalidLabelIndex and prevLabel == blankLabelIndex;
    bool nextIsBlank = blankLabelIndex != Nn::invalidLabelIndex and nextLabel == blankLabelIndex;

    if (prevLabel == Nn::invalidLabelIndex) {
        if (nextIsBlank) {
            return Nn::LabelScorer::TransitionType::INITIAL_BLANK;
        }
        else {
            return Nn::LabelScorer::TransitionType::INITIAL_LABEL;
        }
    }

    if (prevIsBlank) {
        if (nextIsBlank) {
            return Nn::LabelScorer::TransitionType::BLANK_LOOP;
        }
        else {
            return Nn::LabelScorer::TransitionType::BLANK_TO_LABEL;
        }
    }
    else {
        if (nextIsBlank) {
            return Nn::LabelScorer::TransitionType::LABEL_TO_BLANK;
        }
        else if (collapseRepeatedLabels and prevLabel == nextLabel) {
            return Nn::LabelScorer::TransitionType::LABEL_LOOP;
        }
        else {
            return Nn::LabelScorer::TransitionType::LABEL_TO_LABEL;
        }
    }
}

template<typename Element>
void ModelCombTreeTimesyncBeamSearch::beamSizePruning(std::vector<Element>& hypotheses, size_t maxBeamSize) const {
    if (maxBeamSize == std::numeric_limits<int>::max() or hypotheses.size() <= maxBeamSize) {
        return;
    }

    // Sort the hypotheses by associated score value such that the first `maxBeamSize` elements are the best
    std::nth_element(hypotheses.begin(), hypotheses.begin() + maxBeamSize, hypotheses.end());
    hypotheses.resize(maxBeamSize);  // Get rid of excessive elements
}

template void ModelCombTreeTimesyncBeamSearch::beamSizePruning<ModelCombTreeTimesyncBeamSearch::LabelHypothesis>(std::vector<ModelCombTreeTimesyncBeamSearch::LabelHypothesis>&, size_t) const;
template void ModelCombTreeTimesyncBeamSearch::beamSizePruning<ModelCombTreeTimesyncBeamSearch::WordEndExtensionCandidate>(std::vector<ModelCombTreeTimesyncBeamSearch::WordEndExtensionCandidate>&, size_t) const;

template<typename Element>
void ModelCombTreeTimesyncBeamSearch::scorePruning(std::vector<Element>& hypotheses, Score scoreThreshold) const {
    if (hypotheses.empty() or scoreThreshold == Core::Type<Score>::max) {
        return;
    }

    // Compute the pruning threshold
    auto bestScore        = std::min_element(hypotheses.begin(), hypotheses.end())->score;
    auto pruningThreshold = bestScore + scoreThreshold;

    // Remove elements with score > pruningThreshold
    hypotheses.erase(
            std::remove_if(
                    hypotheses.begin(),
                    hypotheses.end(),
                    [=](auto const& ext) { return ext.score > pruningThreshold; }),
            hypotheses.end());
}

template void ModelCombTreeTimesyncBeamSearch::scorePruning<ModelCombTreeTimesyncBeamSearch::LabelHypothesis>(std::vector<ModelCombTreeTimesyncBeamSearch::LabelHypothesis>&, Score) const;
template void ModelCombTreeTimesyncBeamSearch::scorePruning<ModelCombTreeTimesyncBeamSearch::WithinWordExtensionCandidate>(std::vector<ModelCombTreeTimesyncBeamSearch::WithinWordExtensionCandidate>&, Score) const;
template void ModelCombTreeTimesyncBeamSearch::scorePruning<ModelCombTreeTimesyncBeamSearch::WordEndExtensionCandidate>(std::vector<ModelCombTreeTimesyncBeamSearch::WordEndExtensionCandidate>&, Score) const;

void ModelCombTreeTimesyncBeamSearch::recombination(std::vector<ModelCombTreeTimesyncBeamSearch::LabelHypothesis>& hypotheses) {
    // Represents a unique combination of StateId, ScoringContext and LmHistory
    struct RecombinationContext {
        StateId               state;
        Nn::ScoringContextRef scoringContext;
        Lm::History           lmHistory;

        RecombinationContext(StateId state, Nn::ScoringContextRef scoringContext, Lm::History lmHistory)
                : state(state), scoringContext(scoringContext), lmHistory(lmHistory) {}

        bool operator==(const RecombinationContext& other) const {
            return state == other.state && Nn::ScoringContextEq{}(scoringContext, other.scoringContext) && lmHistory == other.lmHistory;
        }
    };
    struct RecombinationContextHash {
        size_t operator()(const RecombinationContext& context) const {
            size_t h1 = context.state;
            size_t h2 = Nn::ScoringContextHash{}(context.scoringContext);
            size_t h3 = Lm::History::Hash{}(context.lmHistory);
            return Core::combineHashes(Core::combineHashes(h1, h2), h3);
        }
    };

    recombinedHypotheses_.clear();
    recombinedHypotheses_.reserve(hypotheses.size());
    // Map each unique combination of StateId, ScoringContext and LmHistory in newHypotheses to its hypothesis
    std::unordered_map<RecombinationContext, LabelHypothesis*, RecombinationContextHash> seenCombinations;
    for (auto const& hyp : hypotheses) {
        // Use try_emplace to check if the combination already exists and create a new entry if not at the same time
        auto [it, inserted] = seenCombinations.try_emplace({hyp.currentState, hyp.scoringContext, hyp.lmHistory}, nullptr);

        if (inserted) {
            // First time seeing this combination so move it over to `newHypotheses`
            recombinedHypotheses_.push_back(std::move(hyp));
            it->second = &recombinedHypotheses_.back();
        }
        else {
            auto* existingHyp = it->second;
            if (hyp.score < existingHyp->score) {
                // New hyp is better
                if (hyp.atWordEnd) {
                    hyp.trace->sibling = existingHyp->trace;
                }
                // Replace in `newHypotheses`
                *existingHyp = std::move(hyp);  // Overwrite in-place
            }
            else if (hyp.atWordEnd) {
                // New hyp is worse -> add to existing one as sibling if we are at a word end
                hyp.trace->sibling          = existingHyp->trace->sibling;
                existingHyp->trace->sibling = hyp.trace;
            }
        }
    }

    hypotheses.swap(recombinedHypotheses_);
}

void ModelCombTreeTimesyncBeamSearch::recombinationPrevious(std::vector<ModelCombTreeTimesyncBeamSearch::LabelHypothesis>& hypotheses) {
    // Represents a unique combination of StateId, ScoringContext, LmHistory and LemmaPronunciation
    struct RecombinationContext {
        StateId                          state;
        Nn::ScoringContextRef            scoringContext;
        Lm::History                      lmHistory;
        const Bliss::LemmaPronunciation* pron;

        RecombinationContext(StateId state, Nn::ScoringContextRef scoringContext, Lm::History lmHistory, const Bliss::LemmaPronunciation* pron)
                : state(state), scoringContext(scoringContext), lmHistory(lmHistory), pron(pron) {}

        bool operator==(const RecombinationContext& other) const {
            return state == other.state && Nn::ScoringContextEq{}(scoringContext, other.scoringContext) && lmHistory == other.lmHistory && std::string{pron->lemma()->symbol().str()} == std::string{other.pron->lemma()->symbol().str()};
        }
    };
    struct RecombinationContextHash {
        size_t operator()(const RecombinationContext& context) const {
            size_t h1 = context.state;
            size_t h2 = Nn::ScoringContextHash{}(context.scoringContext);
            size_t h3 = Lm::History::Hash{}(context.lmHistory);
            size_t h4 = context.pron->id();
            return Core::combineHashes(Core::combineHashes(Core::combineHashes(h1, h2), h3), h4);
        }
    };

    recombinedHypotheses_.clear();
    recombinedHypotheses_.reserve(hypotheses.size());
    // Map each unique combination of StateId, ScoringContext, LmHistory and LemmaPronunciation in newHypotheses to its hypothesis
    std::unordered_map<RecombinationContext, LabelHypothesis*, RecombinationContextHash> seenCombinations;
    for (auto const& hyp : hypotheses) {
        // Use try_emplace to check if the combination already exists and create a new entry if not at the same time
        auto [it, inserted] = seenCombinations.try_emplace({hyp.currentState, hyp.scoringContext, hyp.lmHistory, hyp.pron}, nullptr);

        if (inserted) {
            // First time seeing this combination so move it over to `newHypotheses`
            recombinedHypotheses_.push_back(std::move(hyp));
            it->second = &recombinedHypotheses_.back();
        }
        else {
            auto* existingHyp = it->second;
            if (hyp.score < existingHyp->score) {
                // New hyp is better
                if (hyp.atWordEnd) {
                    hyp.trace->sibling = existingHyp->trace;
                }
                // Replace in `newHypotheses`
                *existingHyp = std::move(hyp);  // Overwrite in-place
            }
            else if (hyp.atWordEnd) {
                // New hyp is worse -> add to existing one as sibling if we are at a word end
                hyp.trace->sibling          = existingHyp->trace->sibling;
                existingHyp->trace->sibling = hyp.trace;
            }
        }
    }

    hypotheses.swap(recombinedHypotheses_);
}

void ModelCombTreeTimesyncBeamSearch::createSuccessorLookups() {
    for (size_t i = 0ul; i < numModels_; ++i) {
        size_t numStates = models_[i].network->structure.stateCount();

        models_[i].stateSuccessorsOffset.assign(numStates + 1, 0);
        models_[i].stateExitsOffset.assign(numStates + 1, 0);

        for (u32 state = 1; state < numStates; ++state) {
            // The offset for the next state is the current size of the data vectors
            models_[i].stateSuccessorsOffset[state] = models_[i].stateSuccessors.size();
            models_[i].stateExitsOffset[state]      = models_[i].stateExits.size();

            // Add successor/exit data to contiguous vectors
            for (HMMStateNetwork::SuccessorIterator it = models_[i].network->structure.successors(state); it; ++it) {
                if (not it.isLabel()) {
                    models_[i].stateSuccessors.push_back(*it);
                }
                else {
                    models_[i].stateExits.push_back(models_[i].network->exits[it.label()]);
                }
            }
        }
        models_[i].stateSuccessorsOffset[numStates] = models_[i].stateSuccessors.size();
        models_[i].stateExitsOffset[numStates]      = models_[i].stateExits.size();
    }
}

void ModelCombTreeTimesyncBeamSearch::finalizeHypotheses() {
    finalBeam_.clear();
    for (size_t i = 0ul; i < numModels_; ++i) {
        models_[i].tempHypotheses.clear();
        for (size_t hypIndex = 0ul; hypIndex < models_[i].beam.size(); ++hypIndex) {  // beam does not contain the previous word-end hyps at this point
            auto& hyp = models_[i].beam[hypIndex];
            // Check if the hypotheses in the beam are at a root state and add the sentence-end LM score
            if (models_[i].network->finalStates.contains(hyp.currentState)) {
                models_[i].tempHypotheses.push_back(hyp);
            }
        }
    }

    // Fallback: Choose the best hypothesis across all models (can be within-word or previous word-end)
    if (std::all_of(models_.begin(), models_.end(), [&](Model model) { return model.tempHypotheses.empty(); }) and sentenceEndFallback_) {
        warning("No active word-end hypothesis at segment end.");
        log() << "Use sentence-end fallback";
        for (size_t i = 0ul; i < numModels_; ++i) {
            // The previous word-end hyps should also be considered now, so add them to the beam
            models_[i].beam.insert(models_[i].beam.end(), models_[i].previousWordEndHyps.begin(), models_[i].previousWordEndHyps.end());
            models_[i].tempHypotheses = models_[i].beam;
        }
    }

    if (not std::all_of(models_.begin(), models_.end(), [&](Model model) { return model.tempHypotheses.empty(); })) {
        for (size_t i = 0ul; i < numModels_; ++i) {
            if (models_[i].sentenceEndLabelIndex != Nn::invalidLabelIndex) {
                models_[i].withinWordExtensions.clear();
                for (size_t hypIndex = 0ul; hypIndex < models_[i].tempHypotheses.size(); ++hypIndex) {
                    auto& hyp = models_[i].tempHypotheses[hypIndex];
                    models_[i].withinWordExtensions.push_back(
                            {models_[i].sentenceEndLabelIndex,
                             hyp.currentState,
                             hyp.score,
                             hyp.trace->time,
                             Nn::LabelScorer::TransitionType::SENTENCE_END,
                             hypIndex});
                }
                models_[i].requests.clear();
                for (auto const& ext : models_[i].withinWordExtensions) {
                    models_[i].requests.push_back({models_[i].tempHypotheses[ext.baseHypIndex].scoringContext, ext.nextToken, ext.transitionType});
                }

                scoringTime_.start();
                auto result = models_[i].labelScorer->computeScoresWithTimes(models_[i].requests);
                scoringTime_.stop();

                if (not result) {
                    continue;
                }

                for (size_t extensionIdx = 0ul; extensionIdx < models_[i].withinWordExtensions.size(); ++extensionIdx) {
                    auto& ext = models_[i].withinWordExtensions[extensionIdx];
                    ext.score += result->scores[extensionIdx];
                    ext.timeframe = std::max(ext.timeframe, result->timeframes[extensionIdx]);
                }

                models_[i].newBeam.clear();
                for (size_t extensionIdx = 0ul; extensionIdx < models_[i].withinWordExtensions.size(); ++extensionIdx) {
                    auto&       ext     = models_[i].withinWordExtensions[extensionIdx];
                    auto const& baseHyp = models_[i].tempHypotheses[ext.baseHypIndex];
                    // The scoring context is not updated as no further scoring is done afterwards
                    models_[i].newBeam.push_back({baseHyp, ext, baseHyp.scoringContext});
                }

                models_[i].wordEndExtensions.clear();
                for (size_t hypIndex = 0ul; hypIndex < models_[i].newBeam.size(); ++hypIndex) {
                    auto& hyp = models_[i].newBeam[hypIndex];
                    // Add the LM's sentence-end score
                    // The LM history is not updated as this is the last LM scoring step
                    Lm::Score                 sentenceEndScore = globalLanguageModel_->sentenceEndScore(hyp.lmHistory);
                    WordEndExtensionCandidate wordEndExtension{hyp.currentToken,
                                                               models_[i].sentenceEndLemma->pronunciations().first,
                                                               hyp.currentState,
                                                               false,
                                                               false,
                                                               hyp.lmHistory,
                                                               hyp.score + sentenceEndScore,
                                                               0.0,
                                                               static_cast<TimeframeIndex>(currentSearchStep_),
                                                               Nn::LabelScorer::TransitionType::SENTENCE_END,
                                                               0,
                                                               false,
                                                               hyp.score,
                                                               hypIndex};
                    models_[i].wordEndExtensions.push_back(wordEndExtension);
                }

                models_[i].tempHypotheses.clear();
                for (size_t extensionIdx = 0ul; extensionIdx < models_[i].wordEndExtensions.size(); ++extensionIdx) {
                    auto&       ext     = models_[i].wordEndExtensions[extensionIdx];
                    auto const& baseHyp = models_[i].newBeam[ext.baseHypIndex];
                    models_[i].tempHypotheses.push_back({baseHyp, ext, baseHyp.scoringContext});
                }
            }
            // TODO shared word-end hyps are added one time for each model (not problematic, but unnecessary)
            finalBeam_.insert(finalBeam_.end(), models_[i].tempHypotheses.begin(), models_[i].tempHypotheses.end());
        }
    }
    else {
        for (size_t i = 0ul; i < numModels_; ++i) {
            // Construct an empty hypothesis with a lattice containing only one empty pronunciation from start to end
            finalBeam_.push_back(LabelHypothesis());
            finalBeam_.front().trace->time          = models_[i].beam.front().trace->time;  // Retrieve the timeframe from any hyp in the old beam
            finalBeam_.front().trace->pronunciation = nullptr;
            finalBeam_.front().trace->predecessor   = Core::ref(new LatticeTrace(0, {0, 0}, {}));
        }
    }
}

}  // namespace Search
