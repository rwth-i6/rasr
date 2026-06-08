#ifndef FORWARD_BACKWARD_SEARCH_HH
#define FORWARD_BACKWARD_SEARCH_HH

#include <Core/Configuration.hh>
#include <Core/Parameter.hh>
#include <Core/ReferenceCounting.hh>

#include <Bliss/Lexicon.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Nn/LabelScorer/ScoreAccessor.hh>
#include <Nn/LabelScorer/TransitionTypes.hh>
#include <Nn/LabelScorer/Types.hh>
#include <Nn/DataView.hh>

#include <Speech/ModelCombination.hh>

#include <Search/SearchV2.hh>
#include <Search/Traceback.hh>
#include <Search/Types.hh>

#include <optional>
#include <unordered_map>
#include <vector>
#include <string>

namespace Search {

/**
 * Lexicon-free, time-synchronous forward-backward search.
 *
 * This is intended for training / posterior computation, not for recognition.
 *
 * Model:
 *
 *   state_t = (currentToken, scorerContexts)
 *   arc_t(state -> nextState, label) =
 *       sum_scorers scorer.getScore(transitionType, label)
 *
 * Scores are RASR costs: lower is better.
 *
 * Forward:
 *
 *   alpha(dst) = scoreSum(alpha(dst), alpha(src) + arcCost)
 *
 * Backward:
 *
 *   beta(src) = scoreSum(beta(src), arcCost + beta(dst))
 *
 * Gamma:
 *
 *   gamma[t][label] += exp(partitionCost - (alpha(src) + arcCost + beta(dst)))
 *
 * The final probability mass is over all states in the final layer.
 * No sentence-end score is added by default.
 *
 *
 * Build a time-layered search graph in the forward direction.
 * Store all arcs.
 * Then run a backward pass over the stored graph.
 * Finally turn arc posteriors into label gammas.
 */

// TODO label loop collapse?
// TODO blank handling? (sentence end?)
// TODO LM scoring (in addition to AM score?)
class ForwardBackwardSearch : public SearchAlgorithmV2 {
public:
    static const Core::ParameterInt  paramBlankLabelIndex;
    static const Core::ParameterInt  paramSentenceEndLabelIndex;
    static const Core::ParameterBool paramSkipSentenceEndLabel;
    static const Core::ParameterBool paramCollapseRepeatedLabels;
    static const Core::ParameterInt  paramCacheCleanupInterval;
    static const Core::ParameterBool paramLogStatistics;

    explicit ForwardBackwardSearch(Core::Configuration const& config);

    Speech::ModelCombination::Mode requiredModelCombination() const override;
    bool setModelCombination(Speech::ModelCombination const& modelCombination) override;

    void enterSegment(Bliss::SpeechSegment const* segment = nullptr) override;
    void finishSegment() override;

    void putFeature(Nn::DataView const& feature) override;
    void putFeatures(Nn::DataView const& features, size_t nTimesteps) override;

    Core::Ref<const Traceback> getCurrentBestTraceback() const override;
    Core::Ref<const LatticeAdaptor> getCurrentBestWordLattice() const override;
    Core::Ref<const LatticeTrace> getCurrentBestLatticeTrace() const override;
    Core::Ref<const LatticeTrace> getCommonPrefix() const override;

    // call to buildForwardStep()
    bool decodeStep() override;

    /**
     * Cost-domain log partition:
     *
     *   partitionCost = -log sum_paths exp(-pathCost)
     */
    Score partitionCost() const {
        return partitionCost_;
    }

    /**
     * Log-probability-domain sequence log likelihood:
     *
     *   log P = -partitionCost
     */
    Score logLikelihood() const {
        return -partitionCost_;
    }

    /**
     * Label posterior gammas.
     *
     * Shape:
     *
     *   labelGammas()[t][label]
     *
     * Values are probabilities, not log-probabilities.
     */
    std::vector<std::vector<double>> const& labelGammas() const {
        return labelGammas_;
    }

    void dumpGraphToDot(std::string const& filename) const;

protected:
    // position of the State in the states_ vector
    using StateId = size_t;

    static constexpr StateId invalidStateId = static_cast<StateId>(-1);

    // represents where we are after some number of emitted labels
    // states are recombined by currentToken and scoringContexts
    // the probabilities of such equivalent states are summed
    struct State {
        Speech::TimeframeIndex layer;
        Nn::LabelIndex         currentToken;
        Nn::ScoringContextRef  scoringContext;
        Score                  alpha;
        Score                  beta;
    };

    struct Arc {
        StateId            src;
        StateId            dst;
        Nn::LabelIndex     label;
        Nn::TransitionType transitionType;
        Nn::TimeframeIndex time;
        Score              score;
        double             gamma;
    };

    // Key for state recombination
    struct StateKey {
        Nn::LabelIndex        currentToken;
        Nn::ScoringContextRef scoringContext;

        bool operator==(StateKey const& other) const {
            if (currentToken != other.currentToken) {
                return false;
            }
            if (!Nn::ScoringContextEq{}(scoringContext, other.scoringContext)) {
                return false;
            }
            return true;
        }
    };

    struct StateKeyHash {
        size_t operator()(StateKey const& key) const {
            return Core::combineHashes(key.currentToken, Nn::ScoringContextHash{}(key.scoringContext));
        }
    };

    using StateMap = std::unordered_map<StateKey, StateId, StateKeyHash>;

private:
    Bliss::LexiconRef            lexicon_;
    Core::Ref<Nn::LabelScorer>   labelScorer_;

    // Collection of all labels (IDs of the lemmas in the lexicon)
    std::vector<Nn::LabelIndex> labels_;

    bool            useBlank_;
    Nn::LabelIndex  blankLabelIndex_;

    bool            useSentenceEnd_;
    Nn::LabelIndex  sentenceEndLabelIndex_;
    bool            skipSentenceEndLabel_;

    bool            collapseRepeatedLabels_;
    size_t          cacheCleanupInterval_;
    bool            logStatistics_;

    // all states of all layers
    std::vector<State>              states_;
    // set of states at one search depth (layers_[t] = states after t search steps)
    std::vector<std::vector<StateId>> layers_;
    // arcByLayer_[t] contains all arcs from layers_[t] -> layers_[t+1]
    std::vector<std::vector<Arc>>     arcsByLayer_;

    // labelGammas_[t][label] = posterior probability that label was emitted at layer t
    std::vector<std::vector<double>> labelGammas_;

    Score partitionCost_;

    size_t currentSearchStep_;
    bool   finishedSegment_;

private:
    void initializeLabelsFromLexicon();

    // one call = one acoustic timestep/one label emission, every successful step advances the forward-backward graph by one layer
    // for each current state (from the current layer) and for each label extension, new states (+arcs to states) are created with
    // arcScore = scorer score for this transition and label
    // alpha = alpha of current state + arcScore
    // overall:
    // 1. Expand all currently reachable states by all possible labels.
    // 2. Recombine equivalent destination states using log-sum, not Viterbi max/min.
    // 3. Store arcs so that the backward pass can later walk the same graph in reverse.
    bool buildForwardStep();

    // this runs after all forward layers are built
    // - frist, it computes the total sequence probability mass by summing over all final-layer states (stored in partitionCost_)
    // - walks backwards over the graph and sums up arc scores to calculate the betas
    // - arcPathCost = alpha[src] + arc score + beta[dst]
    // - posterior of each arc = partitionCost_ - arcPathCost
    // - the label gammas are then the accumulated arc posteriors by layer and label
    void computeBackwardAndGammas();

    Nn::TransitionType inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const;

    StateId getOrCreateState(std::vector<StateId>& nextLayer, StateMap& nextLayerMap, Speech::TimeframeIndex layer, Nn::LabelIndex currentToken, Nn::ScoringContextRef scoringContext);

    /**
     * Cost-domain log-add:
     *
     *   scoreSum(a, b) = -log(exp(-a) + exp(-b))
     */
    static Score scoreSum(Score a, Score b);
};

}  // namespace Search

#endif  // FORWARD_BACKWARD_SEARCH_HH