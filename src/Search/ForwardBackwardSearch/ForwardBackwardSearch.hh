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
#include <Nn/LabelScorer/DataView.hh>

#include <Speech/ModelCombination.hh>

#include <Search/Histogram.hh>
#include <Search/SearchV2.hh>
#include <Search/Traceback.hh>
#include <Search/Types.hh>

#include <cstdint>
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
 * The final probability mass is over all states in the final layer. With
 * apply-sentence-end-score, a final state s additionally pays the LM's
 * sentence-end cost:
 *
 *   partitionCost = scoreSum_s(alpha(s) + lmSentenceEnd(s)),  beta(s) = lmSentenceEnd(s)
 *
 *
 * Build a time-layered search graph in the forward direction, then run a
 * backward pass over it and turn arc posteriors into label gammas.
 *
 * Only what the backward pass needs is kept for past layers. Each step
 * expands the current layer into candidate states, prunes the candidates, and
 * stores only the survivors (alpha, beta, token) and the arcs into them, as an arc
 * into a pruned state would get posterior 0 anyway. Scoring contexts and LM
 * histories are held for the current layer only. Alpha, beta and the partition
 * are accumulated in double precision.
 */

// TODO label loop collapse?
// TODO blank handling? (sentence end?)
// TODO LM scoring (in addition to AM score?)
class ForwardBackwardSearch : public SearchAlgorithmV2 {
public:
    static const Core::ParameterInt   paramBlankLabelIndex;
    static const Core::ParameterInt   paramSentenceEndLabelIndex;
    static const Core::ParameterBool  paramSkipSentenceEndLabel;
    static const Core::ParameterBool  paramCollapseRepeatedLabels;
    static const Core::ParameterInt   paramCacheCleanupInterval;
    static const Core::ParameterBool  paramLogStatistics;
    static const Core::ParameterInt   paramMaxBeamSize;
    static const Core::ParameterFloat paramScoreThreshold;
    static const Core::ParameterInt   paramNumHistogramBins;
    static const Core::ParameterBool  paramApplySentenceEndScore;

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
        return static_cast<Score>(partitionCost_);
    }

    /**
     * Log-probability-domain sequence log likelihood:
     *
     *   log P = -partitionCost
     */
    Score logLikelihood() const {
        return static_cast<Score>(-partitionCost_);
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

    std::vector<Nn::LabelIndex> const& labels() const {
        return labels_;
    }

    std::vector<std::string> const& labelNames() const {
        return labelNames_;
    }

    void dumpGraphToDot(std::string const& filename) const;

protected:
    // position of a stored State in the states_ vector
    using StateId = uint32_t;

    static constexpr StateId invalidStateId = static_cast<StateId>(-1);

    // A state that survived pruning
    struct State {
        double         alpha;
        double         beta;
        Nn::LabelIndex currentToken;
    };

    // The expansion data of a state in the current (last) layer. Only this
    // layer is ever expanded, so scoring contexts and LM histories of earlier
    // layers are not kept alive.
    struct ActiveState {
        StateId               id;
        Nn::LabelIndex        currentToken;
        Nn::ScoringContextRef scoringContext;
        Lm::History           lmHistory;
    };

    struct Arc {
        StateId            src;
        StateId            dst;
        Nn::LabelIndex     label;
        Nn::TimeframeIndex time;
        Score              score;
        Nn::TransitionType transitionType;
    };

    // Key for state recombination
    struct StateKey {
        Nn::LabelIndex        currentToken;
        Nn::ScoringContextRef scoringContext;
        Lm::History           lmHistory;

        bool operator==(StateKey const& other) const {
            if (currentToken != other.currentToken) {
                return false;
            }
            if (!Nn::ScoringContextEq{}(scoringContext, other.scoringContext)) {
                return false;
            }
            if (lmHistory != other.lmHistory) {
                return false;
            }
            return true;
        }
    };

    struct StateKeyHash {
        size_t operator()(StateKey const& key) const {
            return Core::combineHashes(Core::combineHashes(key.currentToken, Nn::ScoringContextHash{}(key.scoringContext)), Lm::History::Hash{}(key.lmHistory));
        }
    };

private:
    Bliss::LexiconRef                  lexicon_;
    Core::Ref<Nn::LabelScorer>         labelScorer_;
    Core::Ref<Lm::ScaledLanguageModel> languageModel_;

    // Collection of all labels (IDs of the lemmas in the lexicon)
    std::vector<Nn::LabelIndex> labels_;
    std::vector<std::string> labelNames_;
    // labelLemmas_[i] is the lemma belonging to labels_[i]; kept alongside so that
    // e.g. LM scoring can access the lemma directly instead of re-looking it up via id0
    std::vector<Bliss::Lemma const*> labelLemmas_;

    bool            useBlank_;
    Nn::LabelIndex  blankLabelIndex_;

    bool            useSentenceEnd_;
    Nn::LabelIndex  sentenceEndLabelIndex_;
    bool            skipSentenceEndLabel_;

    bool            collapseRepeatedLabels_;
    size_t          cacheCleanupInterval_;
    bool            logStatistics_;

    size_t          maxBeamSize_;
    Score           scoreThreshold_;
    Histogram       scoreHistogram_;

    bool            applySentenceEndScore_;

    // surviving states of all layers; the states of a layer are contiguous
    std::vector<State>              states_;
    // layerStart_[t] = id of the first state of layer t (states after t search steps)
    std::vector<StateId>            layerStart_;
    // expansion data of the states of the last layer
    std::vector<ActiveState>        activeStates_;
    // arcsByLayer_[t] contains the arcs from layer t into surviving states of layer t+1
    std::vector<std::vector<Arc>>   arcsByLayer_;

    // labelGammas_[t][label] = posterior probability that label was emitted at layer t
    std::vector<std::vector<double>> labelGammas_;

    double partitionCost_;

    size_t currentSearchStep_;
    bool   finishedSegment_;

private:
    void initializeLabelsFromLexicon();

    // one call = one acoustic timestep/one label emission, every successful step advances the forward-backward graph by one layer
    // 1. Expand all states of the current layer by all possible labels into candidate states
    // 2. Recombine equivalent candidates using log-sum
    // 3. Prune the candidates, then store only the survivors and the arcs into them
    bool buildForwardStep();

    // Score threshold above which a candidate alpha is pruned, keeping at most maxBeamSize_
    // (score-histogram-based, like the other beam searches' scorePruning()) and dropping
    // alphas worse than the best by more than scoreThreshold_
    // Returns +inf if nothing is pruned
    double pruningThreshold(std::vector<double> const& alphas);

    // this runs after all forward layers are built
    // - frist, it computes the total sequence probability mass by summing over all final-layer states (stored in partitionCost_)
    // - walks backwards over the graph and sums up arc scores to calculate the betas
    // - arcPathCost = alpha[src] + arc score + beta[dst]
    // - posterior of each arc = partitionCost_ - arcPathCost
    // - the label gammas are then the accumulated arc posteriors by layer and label
    void computeBackwardAndGammas();

    // posterior of an arc, valid after computeBackwardAndGammas()
    double arcPosterior(Arc const& arc) const;

    Nn::TransitionType inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const;

    /**
     * Cost-domain log-add:
     *
     *   scoreSum(a, b) = -log(exp(-a) + exp(-b))
     */
    static double scoreSum(double a, double b);
};

}  // namespace Search

#endif  // FORWARD_BACKWARD_SEARCH_HH