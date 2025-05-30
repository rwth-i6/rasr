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
#include <strings.h>

#include <Core/XmlStream.hh>
#include <Lattice/LatticeAdaptor.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include "Search/Module.hh"
#include "Search/Traceback.hh"

namespace Search {

/*
 * =======================
 * === LabelHypothesis ===
 * =======================
 */

TreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis()
        : scoringContext(),
          currentToken(Core::Type<Nn::LabelIndex>::max),
          currentState(invalidTreeNodeIndex),
          lmHistory(),
          score(0.0),
          trace(Core::ref(new LatticeTrace(0, {0, 0}, {}))) {}

TreeTimesyncBeamSearch::LabelHypothesis::LabelHypothesis(
        TreeTimesyncBeamSearch::LabelHypothesis const&    base,
        TreeTimesyncBeamSearch::ExtensionCandidate const& extension,
        Nn::ScoringContextRef const&                      newScoringContext)
        : scoringContext(newScoringContext),
          currentToken(extension.nextToken),
          currentState(extension.state),
          lmHistory(extension.lmHistory),
          score(extension.score),
          trace() {
    switch (extension.transitionType) {
        case Nn::LabelScorer::INITIAL_BLANK:
        case Nn::LabelScorer::INITIAL_LABEL:
            trace = Core::ref(new LatticeTrace(
                    base.trace,
                    extension.pron,
                    extension.timeframe + 1,
                    {extension.score - extension.lmScore, extension.lmScore},
                    {}));
            break;

        case Nn::LabelScorer::LABEL_TO_LABEL:
        case Nn::LabelScorer::BLANK_TO_LABEL:
        case Nn::LabelScorer::LABEL_TO_BLANK:
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

        case Nn::LabelScorer::LABEL_LOOP:
        case Nn::LabelScorer::BLANK_LOOP:
            // Word-end or within-word hypothesis (cannot happen across words) -> update the old trace
            trace                 = Core::ref(new LatticeTrace(*base.trace));
            trace->sibling        = {};
            trace->pronunciation  = extension.pron;
            trace->time           = extension.timeframe + 1;
            trace->score.acoustic = base.trace->score.acoustic + (extension.score - base.score - extension.lmScore);
            trace->score.lm       = base.trace->score.lm + extension.lmScore;
            break;
    }
}

std::string TreeTimesyncBeamSearch::LabelHypothesis::toString() const {
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
 * =====================================
 * === TreeTimesyncBeamSearch ===
 * =====================================
 */

const Core::ParameterInt TreeTimesyncBeamSearch::paramMaxBeamSize(
        "max-beam-size",
        "Maximum number of within-word hypotheses in the search beam.",
        1, 1);

const Core::ParameterInt TreeTimesyncBeamSearch::paramMaxWordEndBeamSize(
        "max-word-end-beam-size",
        "Maximum number of word-end hypotheses in the search beam. If not set, global beam pruning will be done and word-end hypotheses will not be pruned separately.",
        std::numeric_limits<int>::max(), 0);

const Core::ParameterFloat TreeTimesyncBeamSearch::paramScoreThreshold(
        "score-threshold",
        "Prune any within-word hypothesis with a score that is at least this much worse than the best hypothesis.",
        Core::Type<Score>::max, 0);

const Core::ParameterFloat TreeTimesyncBeamSearch::paramWordEndScoreThreshold(
        "word-end-score-threshold",
        "Prune any word-end hypothesis with a score that is at least this much worse than the best word-end hypothesis. If not set, global score pruning will be done \
        and word-end hypotheses will not be pruned separately. If the value is below 1.0, e.g. 0.7, then it is relative to within-word score-pruning.",
        Core::Type<Score>::max, 0);

const Core::ParameterBool TreeTimesyncBeamSearch::paramCollapseRepeatedLabels(
        "collapse-repeated-labels",
        "Collapse repeated emission of the same label into one output. If false, every emission is treated like a new output.",
        false);

const Core::ParameterBool TreeTimesyncBeamSearch::paramForceBlankAcrossWords(
        "force-blank-between-repeated-labels-across-words",
        "Require a blank label between identical labels at word end and word begin.",
        false);

const Core::ParameterBool TreeTimesyncBeamSearch::paramSentenceEndFallBack(
        "sentence-end-fall-back",
        "Allow for fallback solution if no active word-end hypothesis exists at the end of a segment.",
        true);

const Core::ParameterBool TreeTimesyncBeamSearch::paramLogStepwiseStatistics(
        "log-stepwise-statistics",
        "Log statistics about the beam at every search step.",
        false);

TreeTimesyncBeamSearch::TreeTimesyncBeamSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          maxBeamSize_(paramMaxBeamSize(config)),
          maxWordEndBeamSize_(paramMaxWordEndBeamSize(config)),
          scoreThreshold_(paramScoreThreshold(config)),
          wordEndScoreThreshold_(paramWordEndScoreThreshold(config)),
          collapseRepeatedLabels_(paramCollapseRepeatedLabels(config)),
          forceBlankAcrossWords_(paramForceBlankAcrossWords(config)),
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
          numHypsAfterScorePruning_("num-hyps-after-score-pruning"),
          numHypsAfterBeamPruning_("num-hyps-after-beam-pruning"),
          numWordEndHypsAfterScorePruning_("num-word-end-hyps-after-score-pruning"),
          numWordEndHypsAfterBeamPruning_("num-word-end-hyps-after-beam-pruning"),
          numActiveHyps_("num-active-hyps"),
          finishedSegment_(false) {
    if (wordEndScoreThreshold_ <= 1.0) {
        if (scoreThreshold_ == Core::Type<Score>::max) {
            error() << "Word-end score-threshold relative to score-threshold, but score-threshold is not set";
        }
        wordEndScoreThreshold_ *= scoreThreshold_;
    }
}

Speech::ModelCombination::Mode TreeTimesyncBeamSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon | Speech::ModelCombination::useAcousticModel | Speech::ModelCombination::useLanguageModel;
}

Speech::ModelCombination::Mode TreeTimesyncBeamSearch::requiredAcousticModel() const {
    return Am::AcousticModel::noEmissions;
}

bool TreeTimesyncBeamSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_       = modelCombination.lexicon();
    labelScorer_   = modelCombination.labelScorer();
    acousticModel_ = modelCombination.acousticModel();
    languageModel_ = modelCombination.languageModel();

    blankLabelIndex_ = acousticModel_->emissionIndex(acousticModel_->blankAllophoneStateIndex());

    // Build the search tree
    log() << "Start building search tree";
    network_                                     = Core::ref(new PersistentStateTree(config, acousticModel_, lexicon_, std::bind(&Module_::createTreeBuilder, &Search::Module::instance(), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5)));
    std::unique_ptr<AbstractTreeBuilder> builder = Search::Module::instance().createTreeBuilder(config, *lexicon_, *acousticModel_, *network_);
    builder->build();
    log() << "Building finished";

    // Create look-ups for state successors and exits of each state
    createSuccessorLookups();

    // Pre-allocate vectors

    // If maxWordEndBeamSize_ is not set, we need the maximum number of exits a node can have for estimating the max. size of the vectors
    int maxWordEnds = maxWordEndBeamSize_ == std::numeric_limits<int>::max() ? maxNumberOfExits_ : maxWordEndBeamSize_;

    // The beam contains all within-word and word-end hypotheses which survived pruning
    beam_.reserve(maxBeamSize_ + maxWordEnds);
    newBeam_.reserve(maxBeamSize_ + maxWordEnds);
    recombinedHypotheses_.reserve(maxBeamSize_ + maxWordEnds);

    // Each hypothesis in the beam can yield max. one extension per phoneme in the lexicon
    extensions_.reserve((maxBeamSize_ + maxWordEnds) * lexicon_->phonemeInventory()->nPhonemes());
    requests_.reserve((maxBeamSize_ + maxWordEnds) * lexicon_->phonemeInventory()->nPhonemes());

    // After pruning there are maxBeamSize_ state extensions, each can yield max. maxNumberOfExits_ word-end extensions
    withinWordExtensions_.reserve(maxBeamSize_);
    wordEndExtensions_.reserve(maxBeamSize_ * maxNumberOfExits_);

    reset();
    return true;
}

void TreeTimesyncBeamSearch::reset() {
    initializationTime_.start();

    labelScorer_->reset();

    // Reset beam to a single empty hypothesis
    beam_.clear();
    beam_.push_back(LabelHypothesis());
    beam_.front().scoringContext = labelScorer_->getInitialScoringContext();
    beam_.front().currentState   = network_->rootState;
    beam_.front().lmHistory      = languageModel_->startHistory();

    finishedSegment_ = false;

    initializationTime_.stop();
}

void TreeTimesyncBeamSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    initializationTime_.start();
    labelScorer_->reset();
    if (segment != nullptr and languageModel_->setSegment(segment)) {
        for (auto& hyp : beam_) {
            hyp.lmHistory = languageModel_->startHistory();
        }
    }
    resetStatistics();
    initializationTime_.stop();
    finishedSegment_ = false;
}

void TreeTimesyncBeamSearch::finishSegment() {
    featureProcessingTime_.start();
    labelScorer_->signalNoMoreFeatures();
    featureProcessingTime_.stop();
    decodeManySteps();
    logStatistics();
    finishedSegment_ = true;
    finalizeLmScoring();
}

void TreeTimesyncBeamSearch::putFeature(Nn::DataView const& feature) {
    featureProcessingTime_.start();
    labelScorer_->addInput(feature);
    featureProcessingTime_.stop();
}

void TreeTimesyncBeamSearch::putFeatures(Nn::DataView const& features, size_t nTimesteps) {
    featureProcessingTime_.start();
    labelScorer_->addInputs(features, nTimesteps);
    featureProcessingTime_.stop();
}

Core::Ref<const Traceback> TreeTimesyncBeamSearch::getCurrentBestTraceback() const {
    return getBestHypothesis().trace->performTraceback();
}

Core::Ref<const LatticeAdaptor> TreeTimesyncBeamSearch::getCurrentBestWordLattice() const {
    auto&        bestHypothesis = getBestHypothesis();
    LatticeTrace endTrace(bestHypothesis.trace, 0, bestHypothesis.trace->time + 1, bestHypothesis.trace->score, {});

    for (size_t hypIdx = 1ul; hypIdx < beam_.size(); ++hypIdx) {
        auto& hyp          = beam_[hypIdx];
        auto  siblingTrace = Core::ref(new LatticeTrace(hyp.trace, 0, hyp.trace->time, hyp.trace->score, {}));
        endTrace.appendSiblingToChain(siblingTrace);
    }

    return endTrace.buildWordLattice(lexicon_);
}

bool TreeTimesyncBeamSearch::decodeStep() {
    if (finishedSegment_) {
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

        // Iterate over the successors of this hypothesis' current state in the tree
        for (const auto& successorState : stateSuccessorLookup_[hyp.currentState]) {
            Nn::LabelIndex tokenIdx = network_->structure.state(successorState).stateDesc.acousticModel;
            // If we want to force blank between repeated labels across words, a new word should not start with the same token as the previous word ended (except for blank itself)
            // If we don't force blank and we have a repeated label across words, we need to make sure to have label-to-Label as transition type
            if (not(forceBlankAcrossWords_ and (hyp.currentState == network_->rootState) and (tokenIdx == hyp.currentToken) and (tokenIdx != blankLabelIndex_))) {
                auto transitionType = inferTransitionType(hyp.currentToken, tokenIdx, hyp.currentState == network_->rootState);
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

    for (size_t requestIdx = 0ul; requestIdx < extensions_.size(); ++requestIdx) {
        extensions_[requestIdx].score += result->scores[requestIdx];
        extensions_[requestIdx].timeframe = result->timeframes[requestIdx];
    }

    if (logStepwiseStatistics_) {
        clog() << Core::XmlOpen("search-step-stats");
    }

    /*
     * Prune set of possible within-word extensions by max beam size and possibly also by score.
     */
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-before-score-pruning", extensions_.size());
    }
    scorePruning(extensions_, scoreThreshold_);
    numHypsAfterScorePruning_ += extensions_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-score-pruning", extensions_.size());
    }

    beamSizePruning(extensions_, maxBeamSize_);
    numHypsAfterBeamPruning_ += extensions_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-hyps-after-beam-pruning", extensions_.size());
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

                // Start from the root node (the exit's transit state) in the next step
                wordEndExtension.state = exit.transitState;
                wordEndExtension.pron  = lemmaPron;

                if (lemma != lexicon_->specialLemma("blank") and lemma != lexicon_->specialLemma("silence")) {
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
    scorePruning(wordEndExtensions_, wordEndScoreThreshold_);
    numWordEndHypsAfterScorePruning_ += wordEndExtensions_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-word-end-hyps-after-score-pruning", wordEndExtensions_.size());
    }

    beamSizePruning(wordEndExtensions_, maxWordEndBeamSize_);
    numWordEndHypsAfterBeamPruning_ += wordEndExtensions_.size();
    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("num-word-end-hyps-after-beam-pruning", wordEndExtensions_.size());
    }

    /*
     * Create new beam from surviving extensions.
     */
    newBeam_.clear();
    extensions_.swap(withinWordExtensions_);
    extensions_.insert(extensions_.end(), wordEndExtensions_.begin(), wordEndExtensions_.end());

    for (auto const& extension : extensions_) {
        auto const& baseHyp = beam_[extension.baseHypIndex];

        auto newScoringContext = labelScorer_->extendedScoringContext(
                {baseHyp.scoringContext,
                 extension.nextToken,
                 extension.transitionType});

        newBeam_.push_back({baseHyp, extension, newScoringContext});
    }

    /*
     * For all hypotheses at the same state and with the same scoring context and LM history
     * keep only the best since they will all develop in the same way.
     */
    recombination(newBeam_);
    numActiveHyps_ += newBeam_.size();

    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("active-hyps", newBeam_.size());
    }

    if (debugChannel_.isOpen()) {
        std::stringstream ss;
        for (size_t hypIdx = 0ul; hypIdx < newBeam_.size(); ++hypIdx) {
            ss << "Hypothesis " << hypIdx + 1ul << ":  " << newBeam_[hypIdx].toString() << "\n";
        }
        ss << "\n";
        debugChannel_ << ss.str();
    }

    beam_.swap(newBeam_);

    if (logStepwiseStatistics_) {
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

void TreeTimesyncBeamSearch::resetStatistics() {
    initializationTime_.reset();
    featureProcessingTime_.reset();
    scoringTime_.reset();
    contextExtensionTime_.reset();
    numHypsAfterScorePruning_.clear();
    numHypsAfterBeamPruning_.clear();
    numWordEndHypsAfterScorePruning_.clear();
    numWordEndHypsAfterBeamPruning_.clear();
    numActiveHyps_.clear();
}

void TreeTimesyncBeamSearch::logStatistics() const {
    clog() << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    clog() << Core::XmlOpen("initialization-time") << initializationTime_.elapsedMilliseconds() << Core::XmlClose("initialization-time");
    clog() << Core::XmlOpen("feature-processing-time") << featureProcessingTime_.elapsedMilliseconds() << Core::XmlClose("feature-processing-time");
    clog() << Core::XmlOpen("scoring-time") << scoringTime_.elapsedMilliseconds() << Core::XmlClose("scoring-time");
    clog() << Core::XmlOpen("context-extension-time") << contextExtensionTime_.elapsedMilliseconds() << Core::XmlClose("context-extension-time");
    clog() << Core::XmlClose("timing-statistics");
    numHypsAfterScorePruning_.write(clog());
    numHypsAfterBeamPruning_.write(clog());
    numWordEndHypsAfterScorePruning_.write(clog());
    numWordEndHypsAfterBeamPruning_.write(clog());
    numActiveHyps_.write(clog());
}

Nn::LabelScorer::TransitionType TreeTimesyncBeamSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel, bool inRoot) const {
    bool prevIsBlank = prevLabel == blankLabelIndex_;
    bool nextIsBlank = nextLabel == blankLabelIndex_;

    if (prevLabel == Core::Type<Nn::LabelIndex>::max) {
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
        else if (collapseRepeatedLabels_ and prevLabel == nextLabel and not inRoot) {
            return Nn::LabelScorer::TransitionType::LABEL_LOOP;
        }
        else {
            return Nn::LabelScorer::TransitionType::LABEL_TO_LABEL;
        }
    }
}

void TreeTimesyncBeamSearch::beamSizePruning(std::vector<TreeTimesyncBeamSearch::ExtensionCandidate>& extensions, size_t maxBeamSize) const {
    if (extensions.size() <= maxBeamSize) {
        return;
    }

    // Sort the hypotheses by associated score value such that the first `maxBeamSize` elements are the best
    std::nth_element(extensions.begin(), extensions.begin() + maxBeamSize, extensions.end());
    extensions.resize(maxBeamSize);  // Get rid of excessive elements
}

void TreeTimesyncBeamSearch::scorePruning(std::vector<TreeTimesyncBeamSearch::ExtensionCandidate>& extensions, Score scoreThreshold) const {
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
                    [=](auto const& ext) { return ext.score > pruningThreshold; }),
            extensions.end());
}

void TreeTimesyncBeamSearch::recombination(std::vector<TreeTimesyncBeamSearch::LabelHypothesis>& hypotheses) {
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

    hypotheses.swap(recombinedHypotheses_);
}

void TreeTimesyncBeamSearch::createSuccessorLookups() {
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

void TreeTimesyncBeamSearch::finalizeLmScoring() {
    newBeam_.clear();
    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        auto& hyp = beam_[hypIndex];
        // Check if the hypotheses in the beam are at a root state and add the sentence-end LM score
        if (hyp.currentState == network_->rootState or network_->otherRootStates.find(hyp.currentState) != network_->otherRootStates.end()) {
            Lm::Score sentenceEndScore = languageModel_->sentenceEndScore(hyp.lmHistory);
            hyp.score += sentenceEndScore;
            hyp.trace->score.lm += sentenceEndScore;
            newBeam_.push_back(hyp);
        }
    }

    if (newBeam_.empty()) {  // There was no word-end hypothesis in the beam
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
