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

std::string scoreToString(Score score) {
    if (std::isinf(static_cast<double>(score))) {
        return "inf";
    }

    if (std::isnan(static_cast<double>(score))) {
        return "nan";
    }

    std::ostringstream os;
    os << std::setprecision(8) << static_cast<double>(score);
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
          states_(),
          layers_(),
          arcsByLayer_(),
          labelGammas_(),
          partitionCost_(std::numeric_limits<Score>::infinity()),
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
    layers_.clear();
    arcsByLayer_.clear();
    labelGammas_.clear();

    partitionCost_ = std::numeric_limits<Score>::infinity();

    currentSearchStep_ = 0ul;
    finishedSegment_   = false;

    State initialState {
            .layer           = 0u,
            .currentToken    = Nn::invalidLabelIndex,
            .scoringContext  = labelScorer_->getInitialScoringContext(),
            .lmHistory       = languageModel_->startHistory(),
            .alpha           = 0.0,
            .beta            = std::numeric_limits<Score>::infinity(),
    };
    states_.push_back(initialState);
    layers_.push_back({0ul});

    if (segment != nullptr) {
        languageModel_->setSegment(segment);
    }
}

void ForwardBackwardSearch::finishSegment() {
    labelScorer_->signalNoMoreFeatures();

    decodeManySteps();

    computeBackwardAndGammas();

    finishedSegment_  = true;

    if (logStatistics_) {
        clog() << Core::XmlOpen("forward-backward-statistics");
        clog() << Core::XmlFull("num-layers", layers_.size());
        clog() << Core::XmlFull("num-states", states_.size());

        size_t numArcs = 0ul;
        for (auto const& arcs : arcsByLayer_) {
            numArcs += arcs.size();
        }

        clog() << Core::XmlFull("num-arcs", numArcs);
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
            for (StateId stateId : layers_.back()) {
                activeContexts.push_back(states_[stateId].scoringContext);
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

    std::vector<std::pair<Nn::LabelIndex, std::string>> labelEntries;

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

        labelEntries.emplace_back(tokenIdx, labelName);
    }

    std::sort(
            labelEntries.begin(),
            labelEntries.end(),
            [](auto const& a, auto const& b) {
                return a.first < b.first;
            });

    labelEntries.erase(
            std::unique(
                    labelEntries.begin(),
                    labelEntries.end(),
                    [](auto const& a, auto const& b) {
                        return a.first == b.first;
                    }),
            labelEntries.end());

    labels_.reserve(labelEntries.size());
    labelNames_.reserve(labelEntries.size());

    for (auto const& [label, name] : labelEntries) {
        labels_.push_back(label);
        labelNames_.push_back(name);
    }

    verify_eq(labels_.size(), labelNames_.size());
}


bool ForwardBackwardSearch::buildForwardStep() {
    verify(!layers_.empty());

    auto currentLayer = layers_.back();

    if (currentLayer.empty()) {
        return false;
    }

    std::vector<Nn::ScoringContextRef> scoringContexts;
    scoringContexts.reserve(currentLayer.size());

    for (StateId stateId : currentLayer) {
        scoringContexts.push_back(states_[stateId].scoringContext);
    }

    auto scoreAccessors = labelScorer_->getScoreAccessors(scoringContexts);

    // layer is only built if all active states are scoreable
    for (auto const& scoreAccessor : scoreAccessors) {
        if (!scoreAccessor) {
            return false;
        }
    }

    std::vector<StateId> nextLayer;
    nextLayer.reserve(currentLayer.size() * labels_.size());

    StateMap nextLayerMap;
    nextLayerMap.reserve(currentLayer.size() * labels_.size());

    std::vector<Arc> arcs;
    arcs.reserve(currentLayer.size() * labels_.size());

    Speech::TimeframeIndex nextLayerIndex = static_cast<Speech::TimeframeIndex>(layers_.size());

    for (size_t statePos = 0ul; statePos < currentLayer.size(); ++statePos) {
        StateId srcStateId = currentLayer[statePos];
        Score srcAlpha = states_[srcStateId].alpha;
        Nn::LabelIndex srcCurrentToken = states_[srcStateId].currentToken;
        Nn::ScoringContextRef srcScoringContext = states_[srcStateId].scoringContext;
        Lm::History srcLmHistory = states_[srcStateId].lmHistory;

        if (std::isinf(static_cast<double>(srcAlpha))) {
            continue;
        }

        auto const& scoreAccessor = scoreAccessors[statePos];

        for (Nn::LabelIndex label : labels_) {
            Nn::TransitionType transitionType = inferTransitionType(srcCurrentToken, label);

            if (!labelScorer_->scoresTransition(transitionType)) {
                continue;
            }

            //Score arcScore = (*scoreAccessor)->getScore(transitionType, label);

            Score acousticScore = (*scoreAccessor)->getScore(transitionType, label);

            auto const* lemmaPron = lexicon_->lemmaPronunciation(label);    // TODO should work but actually not 100% correct
            // TODO labels_ muss auf jeden Fall refactored werden (Problem sentence-begin)
            // actually the ids stored in labels_ are the lemma IDs
            auto const* lemma = lemmaPron->lemma();
            const Bliss::SyntacticTokenSequence sts = lemma->syntacticTokenSequence();
            auto const* st = sts.front();

            Score lmScore = 0.0;
            Lm::History newLmHistory = srcLmHistory;
            if (not (transitionType == Nn::TransitionType::LABEL_LOOP or transitionType == Nn::TransitionType::BLANK_LOOP)) {
                lmScore = languageModel_->score(srcLmHistory, st);
                newLmHistory = languageModel_->extendedHistory(srcLmHistory, st);
            }


            Score arcScore = acousticScore + lmScore;

            if (std::isinf(static_cast<double>(arcScore))) {
                continue;
            }

            Nn::ScoringContextRef nextScoringContext = labelScorer_->extendedScoringContext(
                                                            srcScoringContext,
                                                            label,
                                                            transitionType);

            StateId dstStateId = getOrCreateState(
                    nextLayer,
                    nextLayerMap,
                    nextLayerIndex,
                    label,
                    nextScoringContext,
                    newLmHistory);

            Score pathCost = srcAlpha + arcScore;
            states_[dstStateId].alpha = scoreSum(states_[dstStateId].alpha, pathCost);

            Arc arc;
            arc.src            = srcStateId;
            arc.dst            = dstStateId;
            arc.label          = label;
            arc.transitionType = transitionType;
            arc.time           = (*scoreAccessor)->getTime();
            arc.score          = arcScore;
            arc.gamma          = 0.0;
            arcs.push_back(arc);
        }
    }

    if (arcs.empty()) {
        return false;
    }

    arcsByLayer_.push_back(arcs);
    layers_.push_back(nextLayer);

    return true;
}

void ForwardBackwardSearch::computeBackwardAndGammas() {
    partitionCost_ = std::numeric_limits<Score>::infinity();

    if (layers_.empty()) {
        return;
    }

    // The final layer contains all states after the last forward step, the partition is the log-sum over all final-state alphas
    std::vector<StateId> const& finalLayer = layers_.back();

    for (StateId stateId : finalLayer) {
        partitionCost_ = scoreSum(partitionCost_, states_[stateId].alpha);

        // beta(final) = 0 because the remaining suffix has probability 1, i.e. cost 0
        states_[stateId].beta = 0.0;
    }

    if (std::isinf(static_cast<double>(partitionCost_))) {
        warning() << "ForwardBackwardSearch partition cost is infinite. No valid path mass was found.";
        return;
    }

    // Backward recursion: For every arc src -> dst: beta(src) += arc.score + beta(dst)
    if (!arcsByLayer_.empty()) {
        for (size_t layerIdx = arcsByLayer_.size(); layerIdx-- > 0;) {
            for (Arc const& arc : arcsByLayer_[layerIdx]) {
                Score pathCost = arc.score + states_[arc.dst].beta;
                states_[arc.src].beta = scoreSum(states_[arc.src].beta, pathCost);
            }
        }
    }

    // Compute arc posteriors and accumulate them into frame/label gammas
    // posterior(arc) = exp(partitionCost - (alpha(src) + arcCost + beta(dst)))
    // labelGammas_[time][label] stores the posterior probability that `label` was emitted at scorer time `time`
    for (std::vector<Arc>& layerArcs : arcsByLayer_) {
        for (Arc& arc : layerArcs) {
            Score arcPathCost = states_[arc.src].alpha + arc.score + states_[arc.dst].beta;

            double posterior = std::exp(static_cast<double>(partitionCost_) - static_cast<double>(arcPathCost));

            arc.gamma = posterior;

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

ForwardBackwardSearch::StateId ForwardBackwardSearch::getOrCreateState(std::vector<StateId>& nextLayer, StateMap& nextLayerMap, Speech::TimeframeIndex layer, Nn::LabelIndex currentToken, Nn::ScoringContextRef scoringContext, Lm::History lmHistory) {
    StateKey key = StateKey{
                    .currentToken   = currentToken,
                    .scoringContext = scoringContext,
                    .lmHistory = lmHistory};

    auto [it, inserted] = nextLayerMap.emplace(key, invalidStateId);

    if (!inserted) {
        return it->second;
    }

    StateId newStateId = states_.size();

    State newState;
    newState.layer           = layer;
    newState.currentToken    = currentToken;
    newState.scoringContext = scoringContext;
    newState.lmHistory      = lmHistory;
    newState.alpha           = std::numeric_limits<Score>::infinity();
    newState.beta            = std::numeric_limits<Score>::infinity();

    states_.push_back(newState);
    nextLayer.push_back(newStateId);

    it->second = newStateId;

    return newStateId;
}

Score ForwardBackwardSearch::scoreSum(Score a, Score b) {
    if (std::isinf(static_cast<double>(a))) {
        return b;
    }

    if (std::isinf(static_cast<double>(b))) {
        return a;
    }

    Score m = std::min(a, b);
    Score M = std::max(a, b);

    return static_cast<Score>(static_cast<double>(m) - std::log1p(std::exp(-static_cast<double>(M - m))));
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

    // Write nodes grouped by layer
    for (size_t layer = 0ul; layer < layers_.size(); ++layer) {
        out << "  subgraph cluster_layer_" << layer << " {\n";
        out << "    label=\"layer " << layer << "\";\n";
        out << "    color=lightgrey;\n";
        out << "    style=dashed;\n";
        out << "    rank=same;\n";

        for (StateId stateId : layers_[layer]) {
            State const& state = states_[stateId];

            std::ostringstream label;
            label << "s" << stateId
                  << "\nlayer=" << state.layer
                  << "\ntoken=" << state.currentToken
                  << "\nctxHash=" << Nn::ScoringContextHash{}(state.scoringContext)
                  << "\nalpha=" << scoreToString(state.alpha)
                  << "\nbeta=" << scoreToString(state.beta);

            out << "    s" << stateId
                << " [label=\"" << dotEscapeLabel(label.str()) << "\"];\n";
        }

        out << "  }\n\n";
    }

    // Write arcs
    for (size_t layer = 0ul; layer < arcsByLayer_.size(); ++layer) {
        for (size_t arcIdx = 0ul; arcIdx < arcsByLayer_[layer].size(); ++arcIdx) {
            Arc const& arc = arcsByLayer_[layer][arcIdx];

            std::ostringstream label;
            label << "arc=" << layer << ":" << arcIdx
                  << "\ntime=" << arc.time
                  << "\nlabel=" << arc.label
                  << "\ntrans=" << toString(arc.transitionType)
                  << "\nscore=" << scoreToString(arc.score)
                  << "\ngamma=" << probabilityToString(arc.gamma);

            out << "  s" << arc.src
                << " -> s" << arc.dst
                << " [label=\"" << dotEscapeLabel(label.str()) << "\"];\n";
        }
    }


    out << "}\n";
}

}  // namespace Search