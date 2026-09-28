#include "ForwardBackwardSearch.hh"

#include <Core/Assertions.hh>
#include <Core/Utility.hh>
#include <Core/XmlStream.hh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <tuple>

namespace Search {

namespace {

std::string dotEscapeLabel(std::string const& input) {
    std::string output;
    output.reserve(input.size() + 2);

    for (char c : input) {
        switch (c) {
            case '\\':
                output += "\\\\";
                break;

            case '"':
                output += "\\\"";
                break;

            case '\n':
                output += "\\l";
                break;

            case '\r':
                break;

            case '\t':
                output += "    ";
                break;

            default:
                output += c;
                break;
        }
    }
    if (output.size() < 2 || output.substr(output.size() - 2) != "\\l") {
        output += "\\l";
    }

    return output;
}

std::string dotEscapeId(std::string const& input) {
    std::string output;
    output.reserve(input.size());

    for (char c : input) {
        switch (c) {
            case '\\':
                output += "\\\\";
                break;

            case '"':
                output += "\\\"";
                break;

            case '\n':
                output += "\\n";
                break;

            case '\r':
                break;

            case '\t':
                output += "    ";
                break;

            default:
                output += c;
                break;
        }
    }

    return output;
}

std::string scoreToString(double score) {
    if (std::isinf(score)) {
        return "inf";
    }

    if (std::isnan(score)) {
        return "nan";
    }

    std::ostringstream os;
    os << std::setprecision(8) << score;
    return os.str();
}

std::string probabilityToString(double value) {
    if (std::isinf(value)) {
        return "inf";
    }

    if (std::isnan(value)) {
        return "nan";
    }

    std::ostringstream os;
    os << std::setprecision(8) << value;
    return os.str();
}

}  // namespace

const Core::ParameterInt ForwardBackwardSearch::paramBlankLabelIndex(
        "blank-label-index",
        "Index of the blank label. If unset, blank is inferred from lexicon special='blank' if available.",
        Nn::invalidLabelIndex);

const Core::ParameterInt ForwardBackwardSearch::paramSentenceEndLabelIndex(
        "sentence-end-label-index",
        "Index of sentence-end label. If unset, inferred from lexicon special='sentence-end' or 'sentence-boundary' if available.",
        Nn::invalidLabelIndex);

const Core::ParameterBool ForwardBackwardSearch::paramSkipSentenceEndLabel(
        "skip-sentence-end-label",
        "Do not include the sentence-end label in the frame-synchronous label expansion.",
        true);

const Core::ParameterBool ForwardBackwardSearch::paramCollapseRepeatedLabels(
        "collapse-repeated-labels",
        "Use LABEL_LOOP transition type for repeated non-blank labels.",
        true);

const Core::ParameterInt ForwardBackwardSearch::paramCacheCleanupInterval(
        "cache-cleanup-interval",
        "Interval of search steps after which inactive label-scorer contexts are cleaned up.",
        10,
        1);

const Core::ParameterBool ForwardBackwardSearch::paramLogStatistics(
        "log-statistics",
        "Log forward-backward statistics after finishSegment().",
        true);

const Core::ParameterInt ForwardBackwardSearch::paramMaxBeamSize(
        "max-beam-size",
        "Maximum number of states kept per layer. Unset (default) means no beam-size pruning.",
        Core::Type<s32>::max,
        1);

const Core::ParameterFloat ForwardBackwardSearch::paramScoreThreshold(
        "score-threshold",
        "Prune states whose alpha is worse than the best alpha in the layer by more than this amount. Unset (default) means no score-based pruning.",
        Core::Type<Score>::max,
        0);

const Core::ParameterInt ForwardBackwardSearch::paramNumHistogramBins(
        "num-histogram-bins",
        "Number of bins for histogram-based beam-size pruning of states (very minor effect).",
        100,
        2);

const Core::ParameterBool ForwardBackwardSearch::paramApplySentenceEndScore(
        "apply-sentence-end-score",
        "Add the LM sentence-end score to every final state, i.e. to the partition and to the final betas.",
        false);

ForwardBackwardSearch::ForwardBackwardSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          lexicon_(),
          labelScorer_(),
          labels_(),
          useBlank_(false),
          blankLabelIndex_(paramBlankLabelIndex(config)),
          useSentenceEnd_(false),
          sentenceEndLabelIndex_(paramSentenceEndLabelIndex(config)),
          skipSentenceEndLabel_(paramSkipSentenceEndLabel(config)),
          collapseRepeatedLabels_(paramCollapseRepeatedLabels(config)),
          cacheCleanupInterval_(paramCacheCleanupInterval(config)),
          logStatistics_(paramLogStatistics(config)),
          maxBeamSize_(static_cast<size_t>(paramMaxBeamSize(config))),
          scoreThreshold_(paramScoreThreshold(config)),
          scoreHistogram_(paramNumHistogramBins(config)),
          applySentenceEndScore_(paramApplySentenceEndScore(config)),
          states_(),
          layerStart_(),
          activeStates_(),
          arcsByLayer_(),
          labelGammas_(),
          partitionCost_(std::numeric_limits<double>::infinity()),
          currentSearchStep_(0ul),
          finishedSegment_(false) {
}

Speech::ModelCombination::Mode ForwardBackwardSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon | Speech::ModelCombination::useLanguageModel;
}

bool ForwardBackwardSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_      = modelCombination.lexicon();
    labelScorer_ = modelCombination.labelScorer();
    languageModel_ = modelCombination.languageModel();

    if (!lexicon_) {
        error() << "ForwardBackwardSearch requires a lexicon.";
        return false;
    }

    if (!labelScorer_) {
        error() << "ForwardBackwardSearch requires a label scorer.";
        return false;
    }

    auto blankLemma = lexicon_->specialLemma("blank");
    if (blankLemma) {
        if (blankLabelIndex_ == Nn::invalidLabelIndex) {
            blankLabelIndex_ = blankLemma->id();
            log() << "Use blank index " << blankLabelIndex_ << " inferred from lexicon.";
        }
        else if (blankLabelIndex_ != static_cast<Nn::LabelIndex>(blankLemma->id())) {
            warning() << "Blank lemma exists in lexicon with id " << blankLemma->id()
                      << " but is overwritten by config value " << blankLabelIndex_;
        }
    }

    useBlank_ = blankLabelIndex_ != Nn::invalidLabelIndex;

    Bliss::Lemma const* sentenceEndLemma = lexicon_->specialLemma("sentence-end");
    if (!sentenceEndLemma) {
        sentenceEndLemma = lexicon_->specialLemma("sentence-boundary");
    }

    if (sentenceEndLemma) {
        if (sentenceEndLabelIndex_ == Nn::invalidLabelIndex) {
            sentenceEndLabelIndex_ = sentenceEndLemma->id();
            log() << "Use sentence-end index " << sentenceEndLabelIndex_
                  << " inferred from lexicon.";
        }
        else if (sentenceEndLabelIndex_ != static_cast<Nn::LabelIndex>(sentenceEndLemma->id())) {
            warning() << "Sentence-end lemma exists in lexicon with id "
                      << sentenceEndLemma->id()
                      << " but is overwritten by config value "
                      << sentenceEndLabelIndex_;
        }
    }

    useSentenceEnd_ = sentenceEndLabelIndex_ != Nn::invalidLabelIndex;

    initializeLabelsFromLexicon();

    if (labels_.empty()) {
        error() << "ForwardBackwardSearch found no labels to expand.";
        return false;
    }

    log() << "ForwardBackwardSearch uses " << labels_.size() << " frame-synchronous labels.";

    return true;
}

void ForwardBackwardSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    labelScorer_->reset();

    states_.clear();
    layerStart_.clear();
    activeStates_.clear();
    arcsByLayer_.clear();
    labelGammas_.clear();

    partitionCost_ = std::numeric_limits<double>::infinity();

    currentSearchStep_ = 0ul;
    finishedSegment_   = false;

    states_.push_back(State{
            .alpha        = 0.0,
            .beta         = std::numeric_limits<double>::infinity(),
            .currentToken = Nn::invalidLabelIndex,
    });
    layerStart_.push_back(0u);
    activeStates_.push_back(ActiveState{
            .id             = 0u,
            .currentToken   = Nn::invalidLabelIndex,
            .scoringContext = labelScorer_->getInitialScoringContext(),
            .lmHistory      = languageModel_->startHistory(),
    });

    if (segment != nullptr) {
        languageModel_->setSegment(segment);
    }
}

void ForwardBackwardSearch::finishSegment() {
    labelScorer_->signalNoMoreFeatures();

    decodeManySteps();

    computeBackwardAndGammas();

    // The contexts of the final layer are not needed once the gammas exist.
    activeStates_.clear();

    finishedSegment_  = true;

    if (logStatistics_) {
        clog() << Core::XmlOpen("forward-backward-statistics");
        clog() << Core::XmlFull("num-layers", layerStart_.size());
        clog() << Core::XmlFull("num-states", states_.size());

        size_t numArcs = 0ul;
        for (auto const& arcs : arcsByLayer_) {
            numArcs += arcs.size();
        }

        clog() << Core::XmlFull("num-arcs", numArcs);
        clog() << Core::XmlFull("sentence-end-score-applied", applySentenceEndScore_);
        clog() << Core::XmlFull("partition-cost", partitionCost_);
        clog() << Core::XmlFull("log-likelihood", -partitionCost_);
        clog() << Core::XmlClose("forward-backward-statistics");
    }

    //dumpGraphToDot("/u/lkleppel/experiments/20260520_unsupervised_asr/output/toy_forward_backward/toy_graph.dot");
}

void ForwardBackwardSearch::putFeature(Nn::DataView const& feature) {
    labelScorer_->addInput(feature);
}

void ForwardBackwardSearch::putFeatures(Nn::DataView const& features, size_t nTimesteps) {
    labelScorer_->addInputs(features, nTimesteps);
}

Core::Ref<const Traceback> ForwardBackwardSearch::getCurrentBestTraceback() const {
    /*
     * This search is not best-path oriented. Return an empty traceback so callers
     * that expect the SearchV2 interface do not crash. Training code should use
     * labelGammas() and partitionCost().
     */
    return Core::ref(new Traceback());
}

Core::Ref<const LatticeAdaptor> ForwardBackwardSearch::getCurrentBestWordLattice() const {
    /*
     * No lattice is built here. The graph used for FB is kept internally as
     * states_ / arcsByLayer_ and is meant for posterior computation.
     */
    return Core::Ref<const LatticeAdaptor>();
}

Core::Ref<const LatticeTrace> ForwardBackwardSearch::getCurrentBestLatticeTrace() const {
    return Core::Ref<LatticeTrace>();
}

Core::Ref<const LatticeTrace> ForwardBackwardSearch::getCommonPrefix() const {
    return Core::Ref<LatticeTrace>();
}

bool ForwardBackwardSearch::decodeStep() {
    if (finishedSegment_) {
        return false;
    }

    bool builtStep = buildForwardStep();

    if (builtStep) {
        ++currentSearchStep_;

        if (currentSearchStep_ % cacheCleanupInterval_ == 0) {
            Core::CollapsedVector<Nn::ScoringContextRef> activeContexts;
            for (ActiveState const& active : activeStates_) {
                activeContexts.push_back(active.scoringContext);
            }
            labelScorer_->cleanupCaches(activeContexts);
        }
    }

    return builtStep;
}

void ForwardBackwardSearch::initializeLabelsFromLexicon() {
    // Limitation: lemmas have to be in the same order as phonemes
    // and e.g. special lemmas which do not have pronunciation (and are therefore skipped and just exists for the LM)
    // have to be at the end after the phoneme-lemmas

    labels_.clear();
    labelNames_.clear();
    labelLemmas_.clear();

    std::vector<std::tuple<Nn::LabelIndex, std::string, Bliss::Lemma const*>> labelEntries;

    auto lemmas = lexicon_->lemmas();

    for (auto lemmaIt = lemmas.first; lemmaIt != lemmas.second; ++lemmaIt) {
        Bliss::Lemma const* lemma = *lemmaIt;
        Nn::LabelIndex tokenIdx = lemma->id();
        if ((skipSentenceEndLabel_ && useSentenceEnd_ && tokenIdx == sentenceEndLabelIndex_) || lemma->nPronunciations() == 0) {    // TODO
            continue;
        }


        std::string labelName;

        if (lemma->nOrthographicForms() > 0) {
            labelName = lemma->preferredOrthographicForm().str();
        }
        else if (lemma->hasName()) {
            labelName = lemma->name().str();
        }
        else {
            labelName = Core::form("<lemma-%d>", static_cast<int>(tokenIdx));
        }

        labelEntries.emplace_back(tokenIdx, labelName, lemma);
    }

    std::sort(
            labelEntries.begin(),
            labelEntries.end(),
            [](auto const& a, auto const& b) {
                return std::get<0>(a) < std::get<0>(b);
            });

    labelEntries.erase(
            std::unique(
                    labelEntries.begin(),
                    labelEntries.end(),
                    [](auto const& a, auto const& b) {
                        return std::get<0>(a) == std::get<0>(b);
                    }),
            labelEntries.end());

    labels_.reserve(labelEntries.size());
    labelNames_.reserve(labelEntries.size());
    labelLemmas_.reserve(labelEntries.size());

    for (auto const& [label, name, lemma] : labelEntries) {
        labels_.push_back(label);
        labelNames_.push_back(name);
        labelLemmas_.push_back(lemma);
    }

    verify_eq(labels_.size(), labelNames_.size());
    verify_eq(labels_.size(), labelLemmas_.size());
}


bool ForwardBackwardSearch::buildForwardStep() {
    if (activeStates_.empty()) {
        return false;
    }

    std::vector<Nn::ScoringContextRef> scoringContexts;
    scoringContexts.reserve(activeStates_.size());

    for (ActiveState const& active : activeStates_) {
        scoringContexts.push_back(active.scoringContext);
    }

    auto scoreAccessors = labelScorer_->getScoreAccessors(scoringContexts);

    // layer is only built if all active states are scoreable
    for (auto const& scoreAccessor : scoreAccessors) {
        if (!scoreAccessor) {
            return false;
        }
    }

    // Candidate states of the next layer, recombined by StateKey. They only live
    // for this step: after pruning, the survivors are stored and the rest (with
    // their scoring contexts and LM histories) is freed.
    size_t const maxCandidates = activeStates_.size() * labels_.size();

    std::vector<StateKey> candidateKeys;
    std::vector<double>   candidateAlphas;
    std::unordered_map<StateKey, uint32_t, StateKeyHash> candidateIndex;
    candidateKeys.reserve(maxCandidates);
    candidateAlphas.reserve(maxCandidates);
    candidateIndex.reserve(maxCandidates);

    struct PendingArc {
        StateId            src;
        uint32_t           candidate;
        Nn::LabelIndex     label;
        Nn::TimeframeIndex time;
        Score              score;
        Nn::TransitionType transitionType;
    };
    std::vector<PendingArc> pendingArcs;
    pendingArcs.reserve(maxCandidates);

    for (size_t statePos = 0ul; statePos < activeStates_.size(); ++statePos) {
        ActiveState const& src = activeStates_[statePos];
        double srcAlpha = states_[src.id].alpha;

        if (std::isinf(srcAlpha)) {
            continue;
        }

        auto const& scoreAccessor = scoreAccessors[statePos];
        Nn::TimeframeIndex time = (*scoreAccessor)->getTime();

        for (size_t labelPos = 0ul; labelPos < labels_.size(); ++labelPos) {
            Nn::LabelIndex label = labels_[labelPos];
            Bliss::Lemma const* lemma = labelLemmas_[labelPos];

            Nn::TransitionType transitionType = inferTransitionType(src.currentToken, label);

            if (!labelScorer_->scoresTransition(transitionType)) {
                continue;
            }

            Score acousticScore = (*scoreAccessor)->getScore(transitionType, label);

            Score lmScore = 0.0;
            Lm::History newLmHistory = src.lmHistory;
            if (not (transitionType == Nn::TransitionType::LABEL_LOOP or transitionType == Nn::TransitionType::BLANK_LOOP)) {
                Bliss::SyntacticTokenSequence const& sts = lemma->syntacticTokenSequence();
                if (sts.size() != 0) {
                    auto const* st = sts.front();
                    lmScore = languageModel_->score(src.lmHistory, st);
                    newLmHistory = languageModel_->extendedHistory(src.lmHistory, st);
                }
            }

            Score arcScore = acousticScore + lmScore;

            if (std::isinf(static_cast<double>(arcScore))) {
                continue;
            }

            Nn::ScoringContextRef nextScoringContext = labelScorer_->extendedScoringContext(
                                                            src.scoringContext,
                                                            label,
                                                            transitionType);

            StateKey key{
                    .currentToken   = label,
                    .scoringContext = nextScoringContext,
                    .lmHistory      = newLmHistory};

            auto [it, inserted] = candidateIndex.emplace(key, static_cast<uint32_t>(candidateKeys.size()));
            if (inserted) {
                candidateKeys.push_back(std::move(key));
                candidateAlphas.push_back(std::numeric_limits<double>::infinity());
            }
            uint32_t candidate = it->second;

            candidateAlphas[candidate] = scoreSum(candidateAlphas[candidate], srcAlpha + static_cast<double>(arcScore));

            pendingArcs.push_back(PendingArc{
                    .src            = src.id,
                    .candidate      = candidate,
                    .label          = label,
                    .time           = time,
                    .score          = arcScore,
                    .transitionType = transitionType});
        }
    }

    if (pendingArcs.empty()) {
        return false;
    }

    // Prune the candidates. Same rule as before storing was deferred: keep alphas
    // at or below the threshold; if all alphas are equal, keep the first
    // maxBeamSize_ candidates.
    size_t const numCandidates = candidateAlphas.size();
    double threshold = std::numeric_limits<double>::infinity();
    size_t keepLimit = numCandidates;

    if (numCandidates > maxBeamSize_ or scoreThreshold_ != Core::Type<Score>::max) {
        auto [lowerIt, upperIt] = std::minmax_element(candidateAlphas.begin(), candidateAlphas.end());
        double lowerScore = *lowerIt;
        double upperScore = *upperIt;

        if (lowerScore == upperScore) {
            keepLimit = std::min(numCandidates, maxBeamSize_);
        }
        else {
            threshold = pruningThreshold(candidateAlphas);
        }
    }

    // Store the surviving candidates as the next layer.
    StateId const firstNewState = static_cast<StateId>(states_.size());
    std::vector<StateId> newStateId(numCandidates, invalidStateId);
    std::vector<ActiveState> nextActiveStates;
    nextActiveStates.reserve(std::min(numCandidates, maxBeamSize_));

    for (size_t candidate = 0ul; candidate < numCandidates and nextActiveStates.size() < keepLimit; ++candidate) {
        if (candidateAlphas[candidate] > threshold) {
            continue;
        }
        verify(states_.size() < static_cast<size_t>(invalidStateId));
        StateId id = static_cast<StateId>(states_.size());
        newStateId[candidate] = id;

        StateKey const& key = candidateKeys[candidate];
        states_.push_back(State{
                .alpha        = candidateAlphas[candidate],
                .beta         = std::numeric_limits<double>::infinity(),
                .currentToken = key.currentToken});
        nextActiveStates.push_back(ActiveState{
                .id             = id,
                .currentToken   = key.currentToken,
                .scoringContext = key.scoringContext,
                .lmHistory      = key.lmHistory});
    }

    // Keep only the arcs into surviving states: an arc into a pruned state would
    // get posterior 0 in the backward pass anyway.
    std::vector<Arc> arcs;
    for (PendingArc const& pending : pendingArcs) {
        StateId dst = newStateId[pending.candidate];
        if (dst == invalidStateId) {
            continue;
        }
        arcs.push_back(Arc{
                .src            = pending.src,
                .dst            = dst,
                .label          = pending.label,
                .time           = pending.time,
                .score          = pending.score,
                .transitionType = pending.transitionType});
    }
    arcs.shrink_to_fit();

    arcsByLayer_.push_back(std::move(arcs));
    layerStart_.push_back(firstNewState);
    activeStates_ = std::move(nextActiveStates);

    return true;
}

double ForwardBackwardSearch::pruningThreshold(std::vector<double> const& alphas) {
    auto [lowerIt, upperIt] = std::minmax_element(alphas.begin(), alphas.end());
    double lowerScore = *lowerIt;
    double upperScore = *upperIt;

    double absoluteThreshold = upperScore;

    // Pruning by relative score threshold
    if (scoreThreshold_ != Core::Type<Score>::max) {
        absoluteThreshold = lowerScore + static_cast<double>(scoreThreshold_);
    }

    // Pruning by max beam size
    if (alphas.size() > maxBeamSize_) {
        scoreHistogram_.clear();
        scoreHistogram_.setLimits(static_cast<Score>(lowerScore), static_cast<Score>(upperScore));

        for (double alpha : alphas) {
            scoreHistogram_ += static_cast<Score>(alpha);
        }

        absoluteThreshold = std::min(absoluteThreshold, static_cast<double>(scoreHistogram_.quantile(maxBeamSize_)));
    }

    if (absoluteThreshold >= upperScore) {
        // Nothing will be pruned
        return std::numeric_limits<double>::infinity();
    }

    return absoluteThreshold;
}

void ForwardBackwardSearch::computeBackwardAndGammas() {
    partitionCost_ = std::numeric_limits<double>::infinity();

    if (layerStart_.empty()) {
        return;
    }

    // The final layer contains all states after the last forward step, the partition is the log-sum over all final-state alphas.
    // activeStates_ still holds the final layer (one entry per final state), which is where the LM histories are.
    verify_eq(activeStates_.size(), states_.size() - layerStart_.back());
    for (ActiveState const& finalState : activeStates_) {
        // beta(final) is the cost of the remaining suffix: 0, or the LM's sentence-end cost
        double endCost = applySentenceEndScore_ ? static_cast<double>(languageModel_->sentenceEndScore(finalState.lmHistory)) : 0.0;

        partitionCost_ = scoreSum(partitionCost_, states_[finalState.id].alpha + endCost);
        states_[finalState.id].beta = endCost;
    }

    if (std::isinf(partitionCost_)) {
        warning() << "ForwardBackwardSearch partition cost is infinite. No valid path mass was found.";
        return;
    }

    // Backward recursion: For every arc src -> dst: beta(src) += arc.score + beta(dst)
    for (size_t layerIdx = arcsByLayer_.size(); layerIdx-- > 0;) {
        for (Arc const& arc : arcsByLayer_[layerIdx]) {
            double pathCost = static_cast<double>(arc.score) + states_[arc.dst].beta;
            states_[arc.src].beta = scoreSum(states_[arc.src].beta, pathCost);
        }
    }

    // Compute arc posteriors and accumulate them into frame/label gammas
    // posterior(arc) = exp(partitionCost - (alpha(src) + arcCost + beta(dst)))
    // labelGammas_[time][label] stores the posterior probability that `label` was emitted at scorer time `time`
    for (std::vector<Arc> const& layerArcs : arcsByLayer_) {
        for (Arc const& arc : layerArcs) {
            double posterior = arcPosterior(arc);

            size_t frame = static_cast<size_t>(arc.time);

            if (frame >= labelGammas_.size()) {
                size_t oldSize = labelGammas_.size();
                labelGammas_.resize(frame + 1ul);

                for (size_t i = oldSize; i < labelGammas_.size(); ++i) {
                    labelGammas_[i].assign(labels_.size(), 0.0);
                }
            }

            if (static_cast<size_t>(arc.label) >= labelGammas_[frame].size()) {
                labelGammas_[frame].resize(static_cast<size_t>(arc.label) + 1ul, 0.0);
            }

            labelGammas_[frame][arc.label] += posterior;
        }
    }
}

double ForwardBackwardSearch::arcPosterior(Arc const& arc) const {
    double arcPathCost = states_[arc.src].alpha + static_cast<double>(arc.score) + states_[arc.dst].beta;
    return std::exp(partitionCost_ - arcPathCost);
}

Nn::TransitionType ForwardBackwardSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const {
    bool prevIsBlank = useBlank_ && prevLabel == blankLabelIndex_;
    bool nextIsBlank = useBlank_ && nextLabel == blankLabelIndex_;

    if (prevLabel == Nn::invalidLabelIndex) {
        if (nextIsBlank) {
            return Nn::TransitionType::INITIAL_BLANK;
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
    else {
        if (nextIsBlank) {
            return Nn::TransitionType::LABEL_TO_BLANK;
        }
        else if (collapseRepeatedLabels_ && prevLabel == nextLabel) {
            return Nn::TransitionType::LABEL_LOOP;
        }
        else {
            return Nn::TransitionType::LABEL_TO_LABEL;
        }
    }
}

double ForwardBackwardSearch::scoreSum(double a, double b) {
    if (std::isinf(a)) {
        return b;
    }

    if (std::isinf(b)) {
        return a;
    }

    double m = std::min(a, b);
    double M = std::max(a, b);

    return m - std::log1p(std::exp(-(M - m)));
}




void ForwardBackwardSearch::dumpGraphToDot(std::string const& filename) const {
    std::ofstream out(filename.c_str());

    if (!out) {
        std::ostringstream os;
        os << "Could not open DOT output file '" << filename << "'";
        throw std::runtime_error(os.str());
    }

    out << "digraph ForwardBackwardGraph {\n";
    out << "  graph [rankdir=LR, compound=true];\n";
    out << "  node [shape=box, fontsize=10];\n";
    out << "  edge [fontsize=9];\n\n";

    out << "  labelloc=\"t\";\n";

    std::ostringstream graphLabel;
    graphLabel << "ForwardBackwardSearch graph";

    if (!std::isfinite(partitionCost_)) {
        graphLabel << "\npartitionCost=" << scoreToString(partitionCost_);
        graphLabel << "\nlogLikelihood=" << scoreToString(-partitionCost_);
    }

    out << "  label=\"" << dotEscapeLabel(graphLabel.str()) << "\";\n\n";

    // Write nodes grouped by layer (only states that survived pruning are kept)
    for (size_t layer = 0ul; layer < layerStart_.size(); ++layer) {
        StateId first = layerStart_[layer];
        StateId last  = layer + 1ul < layerStart_.size() ? layerStart_[layer + 1ul] : static_cast<StateId>(states_.size());

        out << "  subgraph cluster_layer_" << layer << " {\n";
        out << "    label=\"layer " << layer << "\";\n";
        out << "    color=lightgrey;\n";
        out << "    style=dashed;\n";
        out << "    rank=same;\n";

        for (StateId stateId = first; stateId < last; ++stateId) {
            State const& state = states_[stateId];

            std::ostringstream label;
            label << "s" << stateId
                  << "\nlayer=" << layer
                  << "\ntoken=" << state.currentToken
                  << "\nalpha=" << scoreToString(state.alpha)
                  << "\nbeta=" << scoreToString(state.beta);

            out << "    s" << stateId
                << " [label=\"" << dotEscapeLabel(label.str()) << "\"];\n";
        }

        out << "  }\n\n";
    }

    // Write arcs; posteriors are computed from alpha/beta rather than stored
    bool const havePosteriors = std::isfinite(partitionCost_);
    for (size_t layer = 0ul; layer < arcsByLayer_.size(); ++layer) {
        for (size_t arcIdx = 0ul; arcIdx < arcsByLayer_[layer].size(); ++arcIdx) {
            Arc const& arc = arcsByLayer_[layer][arcIdx];

            std::ostringstream label;
            label << "arc=" << layer << ":" << arcIdx
                  << "\ntime=" << arc.time
                  << "\nlabel=" << arc.label
                  << "\ntrans=" << toString(arc.transitionType)
                  << "\nscore=" << scoreToString(arc.score)
                  << "\ngamma=" << probabilityToString(havePosteriors ? arcPosterior(arc) : 0.0);

            out << "  s" << arc.src
                << " -> s" << arc.dst
                << " [label=\"" << dotEscapeLabel(label.str()) << "\"];\n";
        }
    }


    out << "}\n";
}

}  // namespace Search
