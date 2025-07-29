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

#include <Core/CollapsedVector.hh>
#include <Core/XmlStream.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Search/Traceback.hh>
#include "Search/Module.hh"

namespace Search {

/*
 * =======================
 * === LabelHypothesis ===
 * =======================
 */

TreeLabelsyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContext(),
          currentToken(Core::Type<Nn::LabelIndex>::max),
          currentState(invalidTreeNodeIndex),
          lmHistory(),
          length(0),
          score(0.0),
          scaledScore(0.0),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))),
          isActive(true) {}

TreeLabelsyncBeamSearch::LabelHypothesis::LabelHypothesis(
        TreeLabelsyncBeamSearch::LabelHypothesis const&    base,
        TreeLabelsyncBeamSearch::ExtensionCandidate const& extension,
        Nn::ScoringContextRef const&                       newScoringContext,
        float                                              lengthNormScale)
        : scoringContext(newScoringContext),
          currentToken(extension.nextToken),
          currentState(extension.state),
          lmHistory(extension.lmHistory),
          length(base.length + 1),
          score(extension.score),
          scaledScore(score / std::pow(length, lengthNormScale)),
          trace(),
          isActive(extension.transitionType != Nn::LabelScorer::TransitionType::SENTENCE_END) {
    switch (extension.transitionType) {
        case Nn::LabelScorer::TransitionType::INITIAL_LABEL:
        case Nn::LabelScorer::TransitionType::SENTENCE_END:
            // Start a new trace for the first word or for the sentence-end symbol
            trace = Core::ref(new LatticeTrace(
                    base.trace,
                    extension.pron,
                    extension.timeframe + 1,
                    {extension.score - extension.lmScore, extension.lmScore},
                    {}));
            break;

        case Nn::LabelScorer::TransitionType::LABEL_TO_LABEL:
            if (base.trace->pronunciation != nullptr) {  // A word has ended before and the first token of a new word was predicted -> start a new trace
                trace = Core::ref(new LatticeTrace(
                        base.trace,
                        extension.pron,
                        extension.timeframe + 1,
                        {base.trace->score.acoustic + (extension.score - base.score - extension.lmScore), base.trace->score.lm + extension.lmScore},
                        {}));
            }
            else {  // Word-end or within-word hypothesis and no word has ended before -> update the old trace
                trace                 = Core::ref(new LatticeTrace(*base.trace));
                trace->sibling        = {};
                trace->pronunciation  = extension.pron;
                trace->time           = extension.timeframe + 1;
                trace->score.acoustic = base.trace->score.acoustic + (extension.score - base.score - extension.lmScore);
                trace->score.lm       = base.trace->score.lm + extension.lmScore;
            }
            break;

        default:
            defect();  // Unexpected transition type which can not be produced by `inferTransitionType`
    }
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

const Core::ParameterInt TreeLabelsyncBeamSearch::paramMaxBeamSize(
        "max-beam-size",
        "Maximum number of hypotheses in the search beam.",
        1, 1);

const Core::ParameterInt TreeLabelsyncBeamSearch::paramMaxWordEndBeamSize(
        "max-word-end-beam-size",
        "Maximum number of word-end hypotheses in the search beam. If not set, global beam pruning will be done and word-end hypotheses will not be pruned separately.",
        std::numeric_limits<int>::max(), 0);

const Core::ParameterFloat TreeLabelsyncBeamSearch::paramScoreThreshold(
        "score-threshold",
        "Prune any hypotheses with a score that is at least this much worse than the best hypothesis."
        "If length normalization is enabled, the score threshold is added to the raw score before normalization."
        "If not set, no score pruning will be done.",
        Core::Type<Score>::max, 0);

const Core::ParameterFloat TreeLabelsyncBeamSearch::paramWordEndScoreThreshold(
        "word-end-score-threshold",
        "Prune any word-end hypothesis with a score that is at least this much worse than the best word-end hypothesis. If not set, global score pruning will be done \
        and word-end hypotheses will not be pruned separately. If the value is below 1.0, e.g. 0.7, then it is relative to within-word score-pruning.",
        Core::Type<Score>::max, 0);

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

const Core::ParameterBool TreeLabelsyncBeamSearch::paramLogStepwiseStatistics(
        "log-stepwise-statistics",
        "Log statistics about the beam at every search step.",
        false);

TreeLabelsyncBeamSearch::TreeLabelsyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          maxBeamSize_(paramMaxBeamSize(config)),
          maxWordEndBeamSize_(paramMaxWordEndBeamSize(config)),
          scoreThreshold_(paramScoreThreshold(config)),
          wordEndScoreThreshold_(paramWordEndScoreThreshold(config)),
          lengthNormScale_(paramLengthNormScale(config)),
          maxLabelsPerTimestep_(paramMaxLabelsPerTimestep(config)),
          sentenceEndFallback_(paramSentenceEndFallBack(config)),
          logStepwiseStatistics_(paramLogStepwiseStatistics(config)),
          debugChannel_(config, "debug"),
          labelScorer_(),
          beam_(),
          extensions_(),
          newBeam_(),
          requests_(),
          recombinedHypotheses_(),
          maxNumberOfExits_(0),
          initializationTime_(),
          featureProcessingTime_(),
          scoringTime_(),
          contextExtensionTime_(),
          numTerminatedHypsAfterScorePruning_("num-termianted-hyps-after-score-pruning"),
          numTerminatedHypsAfterBeamPruning_("num-terminated-hyps-after-beam-pruning"),
          numActiveHypsAfterScorePruning_("num-active-hyps-after-score-pruning"),
          numActiveHypsAfterBeamPruning_("num-active-hyps-after-beam-pruning"),
          numActiveWordEndHypsAfterScorePruning_("num-active-word-end-hyps-after-score-pruning"),
          numActiveWordEndHypsAfterBeamPruning_("num-active-word-end-hyps-after-beam-pruning"),
          currentSearchStep_(0ul),
          totalTimesteps_(0ul),
          finishedSegment_(false) {
    useScorePruning_ = scoreThreshold_ != Core::Type<Score>::max;
    if (wordEndScoreThreshold_ <= 1.0) {
        if (not useScorePruning_) {
            error() << "Word-end score-threshold relative to score-threshold, but score-threshold is not set";
        }
        wordEndScoreThreshold_ *= scoreThreshold_;
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
    labelScorer_   = modelCombination.labelScorer();
    acousticModel_ = modelCombination.acousticModel();
    languageModel_ = modelCombination.languageModel();

    // Build the search tree
    log() << "Start building search tree";
    network_                                     = Core::ref(new PersistentStateTree(config, acousticModel_, lexicon_, std::bind(&Module_::createTreeBuilder, &Search::Module::instance(), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5)));
    std::unique_ptr<AbstractTreeBuilder> builder = Search::Module::instance().createTreeBuilder(config, *lexicon_, *acousticModel_, *network_);
    builder->build();
    log() << "Building finished";

    auto sentenceEndLemma = lexicon_->specialLemma("sentence-end");
    if (!sentenceEndLemma) {
        sentenceEndLemma = lexicon_->specialLemma("sentence-boundary");
    }
    sentenceEndLabelIndex_ = sentenceEndLemma->id();
    log() << "Use sentence-end index " << sentenceEndLabelIndex_ << " inferred from lexicon";

    // Create look-ups for state successors and exits of each state
    createSuccessorLookups();

    // If maxWordEndBeamSize_ is not set, we need the maximum number of exits a node can have for estimating the max. size of the vectors
    int maxWordEnds = maxWordEndBeamSize_ == std::numeric_limits<int>::max() ? (maxNumberOfExits_ * maxBeamSize_) : maxWordEndBeamSize_;

    // beam_ contains all hypotheses (active or inactive) which survived pruning
    beam_.reserve(maxBeamSize_);
    // In newBeam_ all inactive and all active within-word and word-end hypotheses are collected
    newBeam_.reserve(2 * maxBeamSize_ + maxWordEnds);
    recombinedHypotheses_.reserve(2 * maxBeamSize_ + maxWordEnds);

    // Each hypothesis in the beam can yield max. one extension per phoneme in the lexicon
    extensions_.reserve(maxBeamSize_ * lexicon_->phonemeInventory()->nPhonemes());
    requests_.reserve(maxBeamSize_ * lexicon_->phonemeInventory()->nPhonemes());

    // After pruning there are maxBeamSize_ state extensions, each can yield max. maxNumberOfExits_ word-end extensions
    withinWordExtensions_.reserve(maxBeamSize_);
    wordEndExtensions_.reserve(maxBeamSize_ * maxNumberOfExits_);

    reset();
    return true;
}

void TreeLabelsyncBeamSearch::reset() {
    initializationTime_.start();

    labelScorer_->reset();

    // Reset beam to a single empty hypothesis
    beam_.clear();
    beam_.push_back(LabelHypothesis());
    beam_.front().scoringContext = labelScorer_->getInitialScoringContext();
    beam_.front().currentState   = network_->rootState;
    beam_.front().lmHistory      = languageModel_->startHistory();

    finishedSegment_   = false;
    totalTimesteps_    = 0ul;
    currentSearchStep_ = 0ul;

    initializationTime_.stop();
}

void TreeLabelsyncBeamSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    initializationTime_.start();
    labelScorer_->reset();
    resetStatistics();
    initializationTime_.stop();
    finishedSegment_   = false;
    totalTimesteps_    = 0ul;
    currentSearchStep_ = 0ul;
}

void TreeLabelsyncBeamSearch::finishSegment() {
    featureProcessingTime_.start();
    labelScorer_->signalNoMoreFeatures();
    featureProcessingTime_.stop();
    decodeManySteps();
    logStatistics();
    finishedSegment_ = true;
    finalizeLmScoring();
}

void TreeLabelsyncBeamSearch::putFeature(Nn::DataView const& feature) {
    featureProcessingTime_.start();
    labelScorer_->addInput(feature);
    ++totalTimesteps_;
    featureProcessingTime_.stop();
}

void TreeLabelsyncBeamSearch::putFeatures(Nn::DataView const& features, size_t nTimesteps) {
    featureProcessingTime_.start();
    labelScorer_->addInputs(features, nTimesteps);
    totalTimesteps_ += nTimesteps;
    featureProcessingTime_.stop();
}

Core::Ref<const Traceback> TreeLabelsyncBeamSearch::getCurrentBestTraceback() const {
    return getBestHypothesis().trace->performTraceback();
}

Core::Ref<const LatticeAdaptor> TreeLabelsyncBeamSearch::getCurrentBestWordLattice() const {
    auto& bestHypothesis = getBestHypothesis();

    LatticeTrace endTrace(bestHypothesis.trace, 0, bestHypothesis.trace->time + 1, bestHypothesis.trace->score, {});

    for (auto const& hyp : beam_) {
        if (hyp.isActive != bestHypothesis.isActive) {
            continue;
        }
        auto siblingTrace = Core::ref(new LatticeTrace(hyp.trace, 0, hyp.trace->time, hyp.trace->score, {}));
        endTrace.appendSiblingToChain(siblingTrace);
    }

    return endTrace.buildWordLattice(lexicon_);
}

bool TreeLabelsyncBeamSearch::decodeStep() {
    if (finishedSegment_) {
        return false;
    }
    if (currentSearchStep_ >= maxLabelsPerTimestep_ * std::max(totalTimesteps_, 1ul)) {
        warning() << "Terminated search due to reaching max number of labels";
        finishedSegment_ = true;
        return false;
    }

    /*
     * Collect all possible within-word extensions for all hypotheses in the beam.
     * Also create scoring requests for the label scorer.
     * Each extension candidate makes up a request.
     */
    extensions_.clear();
    requests_.clear();

    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        auto& hyp = beam_[hypIndex];

        if (not hyp.isActive) {
            continue;
        }

        // Iterate over the successors of this hypothesis' current state in the tree
        for (const auto& successorState : stateSuccessorLookup_[hyp.currentState]) {
            Nn::LabelIndex tokenIdx = network_->structure.state(successorState).stateDesc.acousticModel;

            auto transitionType = Nn::LabelScorer::TransitionType::LABEL_TO_LABEL;
            if (hyp.currentToken == Core::Type<Nn::LabelIndex>::max) {
                transitionType = Nn::LabelScorer::TransitionType::INITIAL_LABEL;
            }
            if (tokenIdx == sentenceEndLabelIndex_) {
                transitionType = Nn::LabelScorer::TransitionType::SENTENCE_END;
            }

            extensions_.push_back(
                    {tokenIdx,
                     nullptr,
                     successorState,
                     hyp.lmHistory,
                     hyp.score,
                     0.0,
                     0,
                     transitionType,
                     hypIndex});
            requests_.push_back({beam_[hypIndex].scoringContext, tokenIdx, transitionType});
        }
    }

    if (requests_.empty()) {
        // All hypotheses are terminated -> no search step can be made.
        finishedSegment_ = true;
        return false;
    }

    /*
     * Perform scoring of all the requests with the label scorer.
     */
    scoringTime_.start();
    auto result = labelScorer_->computeScoresWithTimes(requests_);
    scoringTime_.stop();

    if (not result) {
        // LabelScorer could not compute scores -> no search step can be made.
        return false;
    }

    for (size_t extensionIdx = 0ul; extensionIdx < extensions_.size(); ++extensionIdx) {
        extensions_[extensionIdx].score += result->scores[extensionIdx];
        extensions_[extensionIdx].timeframe = result->timeframes[extensionIdx];
    }

    /*
     * Prune set of possible within-word extensions by max beam size and possibly also by score.
     */

    if (logStepwiseStatistics_) {
        clog() << Core::XmlOpen("search-step-stats");
    }

    if (useScorePruning_) {
        scorePruningExtensions(extensions_, scoreThreshold_);
        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("num-extensions-after-score-pruning", extensions_.size());
        }
    }

    beamSizePruningExtensions(extensions_, maxBeamSize_);
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-extensions-after-beam-pruning", extensions_.size());
    }

    /*
     * Expand extensions to word-end hypotheses and incorporate the language model
     */
    withinWordExtensions_.clear();
    wordEndExtensions_.clear();
    for (const auto& extension : extensions_) {
        // If there is at least one state successor, keep it as within-word hypothesis
        if (not stateSuccessorLookup_[extension.state].empty()) {
            withinWordExtensions_.push_back(extension);
        }
        std::vector<PersistentStateTree::Exit> exitList = exitLookup_[extension.state];
        if (not exitList.empty()) {
            // Create one word-end hypothesis for each exit
            for (const auto& exit : exitList) {
                ExtensionCandidate               wordEndExtension(extension);
                const Bliss::LemmaPronunciation* lemmaPron = lexicon_->lemmaPronunciation(exit.pronunciation);
                const Bliss::Lemma*              lemma     = lemmaPron->lemma();
                auto                             lemmaId   = lemma->id();

                // Start from the root node (the exit's transit state) in the next step
                wordEndExtension.state = exit.transitState;
                wordEndExtension.pron  = lemmaPron;

                if (lemmaId != sentenceEndLabelIndex_) {
                    const Bliss::SyntacticTokenSequence sts = lemma->syntacticTokenSequence();
                    const Bliss::SyntacticToken*        st  = sts.front();

                    // Add the LM score and update the LM history
                    Lm::Score lmScore = languageModel_->score(wordEndExtension.lmHistory, st);
                    wordEndExtension.score += lmScore;
                    wordEndExtension.lmScore   = lmScore;
                    wordEndExtension.lmHistory = languageModel_->extendedHistory(wordEndExtension.lmHistory, st);
                }

                wordEndExtensions_.push_back(wordEndExtension);
            }
        }
    }

    /*
     * Prune set of word-end hypotheses by max beam size and possibly also by score.
     */
    if (useScorePruning_) {
        scorePruningExtensions(wordEndExtensions_, wordEndScoreThreshold_);
        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("num-word-end-extensions-after-score-pruning", wordEndExtensions_.size());
        }
    }

    beamSizePruningExtensions(wordEndExtensions_, maxWordEndBeamSize_);
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-word-end-extensions-after-beam-pruning", wordEndExtensions_.size());
    }

    /*
     * Create new beam from surviving extensions.
     */
    newBeam_.clear();
    extensions_.swap(withinWordExtensions_);
    extensions_.insert(extensions_.end(), wordEndExtensions_.begin(), wordEndExtensions_.end());

    for (auto const& hyp : beam_) {
        if (not hyp.isActive) {
            newBeam_.push_back(hyp);
        }
    }

    for (auto const& extension : extensions_) {
        auto const& baseHyp = beam_[extension.baseHypIndex];

        auto newScoringContext = labelScorer_->extendedScoringContext(
                {baseHyp.scoringContext,
                 extension.nextToken,
                 extension.transitionType});
        newBeam_.push_back({baseHyp, extension, newScoringContext, lengthNormScale_});
    }

    /*
     * For all hypotheses at the same state and with the same scoring context and LM history
     * keep only the best since they will all develop in the same way.
     */
    recombination();

    /*
     * Jointly prune terminated and active hypotheses
     */
    if (useScorePruning_) {
        scorePruning();

        size_t numActive = std::accumulate(
                newBeam_.begin(),
                newBeam_.end(),
                0ul,
                [](size_t acc, auto const& hyp) { return acc + static_cast<size_t>(hyp.isActive); });

        size_t numActiveWordEnd = std::accumulate(
                newBeam_.begin(),
                newBeam_.end(),
                0ul,
                [this](size_t acc, auto const& hyp) { return acc + static_cast<size_t>(hyp.isActive and hyp.currentState == network_->rootState); });

        numTerminatedHypsAfterScorePruning_ += newBeam_.size() - numActive;
        numActiveHypsAfterScorePruning_ += numActive;
        numActiveWordEndHypsAfterScorePruning_ += numActiveWordEnd;

        if (logStepwiseStatistics_) {
            clog() << Core::XmlFull("num-terminated-hyps-after-score-pruning", newBeam_.size() - numActive);
            clog() << Core::XmlFull("num-active-hyps-after-score-pruning", numActive);
            clog() << Core::XmlFull("num-active-word-end-hyps-after-score-pruning", numActiveWordEnd);
        }
    }

    beamSizePruning();

    size_t numActive = std::accumulate(
            newBeam_.begin(),
            newBeam_.end(),
            0ul,
            [](size_t acc, auto const& hyp) { return acc + static_cast<size_t>(hyp.isActive); });

    size_t numActiveWordEnd = std::accumulate(
            newBeam_.begin(),
            newBeam_.end(),
            0ul,
            [this](size_t acc, auto const& hyp) { return acc + static_cast<size_t>(hyp.isActive and hyp.currentState == network_->rootState); });

    numTerminatedHypsAfterBeamPruning_ += newBeam_.size() - numActive;
    numActiveHypsAfterBeamPruning_ += numActive;
    numActiveWordEndHypsAfterBeamPruning_ += numActiveWordEnd;

    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-terminated-hyps-after-beam-pruning", newBeam_.size() - numActive);
        clog() << Core::XmlFull("num-active-hyps-after-beam-pruning", numActive);
        clog() << Core::XmlFull("num-active-word-end-hyps-after-beam-pruning", numActiveWordEnd);
    }

    /*
     * Clean up label scorer caches.
     */
    Core::CollapsedVector<Nn::ScoringContextRef> activeContexts;
    for (auto const& hyp : newBeam_) {
        activeContexts.push_back(hyp.scoringContext);
    }
    labelScorer_->cleanupCaches(activeContexts);

    /*
     * Log statistics about the new beam after this step.
     */
    beam_.swap(newBeam_);

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
        auto const* bestTerminatedHyp  = getBestTerminatedHypothesis();
        auto const* worstTerminatedHyp = getWorstActiveHypothesis();
        auto const* bestActiveHyp      = getBestActiveHypothesis();
        auto const* worstActiveHyp     = getWorstActiveHypothesis();
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

    ++currentSearchStep_;
    return true;
}

TreeLabelsyncBeamSearch::LabelHypothesis const* TreeLabelsyncBeamSearch::getBestTerminatedHypothesis() const {
    LabelHypothesis const* best = nullptr;

    for (auto const& hyp : beam_) {
        if (not hyp.isActive) {
            if (best == nullptr or hyp < *best) {
                best = &hyp;
            }
        }
    }

    return best;
}

TreeLabelsyncBeamSearch::LabelHypothesis const* TreeLabelsyncBeamSearch::getWorstTerminatedHypothesis() const {
    LabelHypothesis const* worst = nullptr;

    for (auto const& hyp : beam_) {
        if (not hyp.isActive) {
            if (worst == nullptr or hyp > *worst) {
                worst = &hyp;
            }
        }
    }

    return worst;
}

TreeLabelsyncBeamSearch::LabelHypothesis const* TreeLabelsyncBeamSearch::getBestActiveHypothesis() const {
    LabelHypothesis const* best = nullptr;

    for (auto const& hyp : beam_) {
        if (hyp.isActive) {
            if (best == nullptr or hyp < *best) {
                best = &hyp;
            }
        }
    }

    return best;
}

TreeLabelsyncBeamSearch::LabelHypothesis const* TreeLabelsyncBeamSearch::getWorstActiveHypothesis() const {
    LabelHypothesis const* worst = nullptr;

    for (auto const& hyp : beam_) {
        if (hyp.isActive) {
            if (worst == nullptr or hyp > *worst) {
                worst = &hyp;
            }
        }
    }

    return worst;
}

TreeLabelsyncBeamSearch::LabelHypothesis const& TreeLabelsyncBeamSearch::getBestHypothesis() const {
    auto const* result = getBestTerminatedHypothesis();
    if (result != nullptr) {
        return *result;
    }
    result = getBestActiveHypothesis();
    verify(result != nullptr);
    return *result;
}

TreeLabelsyncBeamSearch::LabelHypothesis const& TreeLabelsyncBeamSearch::getWorstHypothesis() const {
    auto const* result = getWorstTerminatedHypothesis();
    if (result != nullptr) {
        return *result;
    }
    result = getWorstActiveHypothesis();
    verify(result != nullptr);
    return *result;
}

void TreeLabelsyncBeamSearch::resetStatistics() {
    initializationTime_.reset();
    featureProcessingTime_.reset();
    scoringTime_.reset();
    contextExtensionTime_.reset();
    numTerminatedHypsAfterScorePruning_.clear();
    numTerminatedHypsAfterBeamPruning_.clear();
    numActiveHypsAfterScorePruning_.clear();
    numActiveHypsAfterBeamPruning_.clear();
    numActiveWordEndHypsAfterScorePruning_.clear();
    numActiveWordEndHypsAfterBeamPruning_.clear();
}

void TreeLabelsyncBeamSearch::logStatistics() const {
    clog() << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    clog() << Core::XmlOpen("initialization-time") << initializationTime_.elapsedMilliseconds() << Core::XmlClose("initialization-time");
    clog() << Core::XmlOpen("feature-processing-time") << featureProcessingTime_.elapsedMilliseconds() << Core::XmlClose("feature-processing-time");
    clog() << Core::XmlOpen("scoring-time") << scoringTime_.elapsedMilliseconds() << Core::XmlClose("scoring-time");
    clog() << Core::XmlOpen("context-extension-time") << contextExtensionTime_.elapsedMilliseconds() << Core::XmlClose("context-extension-time");
    clog() << Core::XmlClose("timing-statistics");
    numTerminatedHypsAfterScorePruning_.write(clog());
    numTerminatedHypsAfterBeamPruning_.write(clog());
    numActiveHypsAfterScorePruning_.write(clog());
    numActiveHypsAfterBeamPruning_.write(clog());
    numActiveWordEndHypsAfterScorePruning_.write(clog());
    numActiveWordEndHypsAfterBeamPruning_.write(clog());
}

void TreeLabelsyncBeamSearch::beamSizePruningExtensions(std::vector<TreeLabelsyncBeamSearch::ExtensionCandidate>& extensions, size_t maxBeamSize) {
    if (extensions.size() <= maxBeamSize) {
        return;
    }

    // Reorder the hypotheses by associated score value such that the first maxBeamSize elements are the best
    std::nth_element(extensions.begin(), extensions.begin() + maxBeamSize, extensions.end());
    extensions.resize(maxBeamSize);  // Get rid of excessive elements
}

void TreeLabelsyncBeamSearch::beamSizePruning() {
    if (newBeam_.size() <= maxBeamSize_) {
        return;
    }

    // Reorder the hypotheses by associated score value such that the first `beamSizeTerminated_` elements are the best
    std::nth_element(newBeam_.begin(), newBeam_.begin() + maxBeamSize_, newBeam_.end());
    newBeam_.resize(maxBeamSize_);  // Get rid of excessive elements
}

void TreeLabelsyncBeamSearch::scorePruningExtensions(std::vector<TreeLabelsyncBeamSearch::ExtensionCandidate>& extensions, Score scoreThreshold) {
    if (extensions.empty() or scoreThreshold == Core::Type<Score>::max) {
        return;
    }

    // Compute the pruning threshold
    auto bestScore        = std::min_element(extensions.begin(), extensions.end())->score;
    auto pruningThreshold = bestScore + scoreThreshold;

    // Remove elements with score > pruningThreshold
    extensions.erase(
            std::remove_if(
                    extensions.begin(),
                    extensions.end(),
                    [&](auto const& ext) { return ext.score > pruningThreshold; }),
            extensions.end());
}

void TreeLabelsyncBeamSearch::scorePruning() {
    if (newBeam_.empty() or scoreThreshold_ == Core::Type<Score>::max) {
        return;
    }

    // Compute the pruning threshold
    auto bestHyp = *std::min_element(
            newBeam_.begin(),
            newBeam_.end());

    // Remove elements with score > pruningThreshold
    auto pruningThreshold = (bestHyp.score + scoreThreshold_) / std::pow(bestHyp.length, lengthNormScale_);
    newBeam_.erase(
            std::remove_if(
                    newBeam_.begin(),
                    newBeam_.end(),
                    [&](auto const& hyp) { return hyp.scaledScore > pruningThreshold; }),
            newBeam_.end());
}

void TreeLabelsyncBeamSearch::recombination() {
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
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };

    recombinedHypotheses_.clear();
    // Map each unique combination of StateId, ScoringContext and LmHistory in `newBeam_` to its hypothesis
    std::unordered_map<RecombinationContext, LabelHypothesis*, RecombinationContextHash> seenCombinations;
    for (auto const& hyp : newBeam_) {
        // Use try_emplace to check if the combination already exists and create a new entry if not at the same time
        auto [it, inserted] = seenCombinations.try_emplace({hyp.currentState, hyp.scoringContext, hyp.lmHistory}, nullptr);

        if (inserted) {
            // First time seeing this combination so move it over to `newHypotheses`
            recombinedHypotheses_.push_back(std::move(hyp));
            it->second = &recombinedHypotheses_.back();
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

    newBeam_.swap(recombinedHypotheses_);
}

void TreeLabelsyncBeamSearch::createSuccessorLookups() {
    stateSuccessorLookup_.resize(network_->structure.stateCount());
    exitLookup_.resize(network_->structure.stateCount());

    for (u32 state = 1; state < network_->structure.stateCount(); ++state) {
        std::vector<StateId>                   stateList;  // Collect the state successors of all nodes
        std::vector<PersistentStateTree::Exit> exitList;   // Collect the exits of all nodes
        for (HMMStateNetwork::SuccessorIterator it = network_->structure.successors(state); it; ++it) {
            if (not it.isLabel()) {
                stateList.push_back(*it);
            }
            else {
                exitList.push_back(network_->exits[it.label()]);
            }
        }
        stateSuccessorLookup_[state] = stateList;
        exitLookup_[state]           = exitList;

        // Retrieve the maximal number of exits a node in the tree can have to estimate the size of pre-allocated vectors
        if (exitList.size() > maxNumberOfExits_) {
            maxNumberOfExits_ = exitList.size();
        }
    }
}

void TreeLabelsyncBeamSearch::finalizeLmScoring() {
    newBeam_.clear();
    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        auto& hyp = beam_[hypIndex];
        // Check if the hypotheses in the beam are either terminated or at a root state and add the sentence-end LM score
        if (not hyp.isActive or hyp.currentState == network_->rootState or network_->otherRootStates.find(hyp.currentState) != network_->otherRootStates.end()) {
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

}  // namespace Search