/** Copyright 2018 RWTH Aachen University. All rights reserved.
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
#include <Fsa/Semiring.hh>
#include <Fsa/Sssp.hh>
#include <Search/MinimumBayesRiskSearchUtil.hh>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <vector>
#include "MinimumBayesRiskSearch.hh"

namespace Search {

/**
 * Computes minimum Bayes risk hypothesis as proposed by Byrne and Goel.
 *
 */
class MinimumBayesRiskAStarSearch : public MinimumBayesRiskSearch {
public:
    // forward declarations
    class SearchNode;

    typedef Core::Ref<const MinimumBayesRiskAStarSearch::SearchNode> ConstSearchNodeRef;
    typedef Core::Ref<MinimumBayesRiskAStarSearch::SearchNode>       SearchNodeRef;

    // forward declarations
    class StackEntry;
    typedef std::map<Fsa::LabelId, SearchNodeRef> NodeMap;

    class SearchNode : public Core::ReferenceCounted {
        friend class MinimumBayesRiskAStarSearch;

    private:
        /** the sentence belonging to this node. */
        std::vector<Fsa::LabelId> hypothesis_;

        /**
         * The total amount of probability mass assigned to the hypothesis
         * represented by this stack node.
         */
        Fsa::Weight totalProbability_;

        /**
         * In case a final state is weighted, this weight has to be taken into account as well.
         */
        Fsa::Weight finalProbability_;

        /**
         * Forward probability for all partial paths belonging to this hypothesis.
         * This is needed in case the prefix-tree-search will be implemented.
         */
        std::map<Fsa::StateId, Fsa::Weight> logForwardProbabilities_;

        /**
         * Total probabilities for all paths belonging for this hypothsis.
         * This is needed in case the prefix-tree-search will be implemented.
         */
        Core::Vector<Fsa::Weight> logTotalProbabilities_;

        /**
         * Last columns of distance matrices between this hypothesis and all other hyps in stack.
         *
         * NOTE: The index of a hypothesis in the stack will always be the same here.
         */
        std::vector<std::vector<u32>> levenshteinColumns_;

        /**
         * Scores for underestimate for all competing hyps in stack.
         *
         * NOTE: The index of a hypothesis in the stack will always be the same here.
         */
        Core::Vector<Fsa::Weight> levenshteinScores_;

        /**
         * Scores for underestimate for all competing hyps in stack.
         *
         * NOTE: The index of a hypothesis in the stack will always be the same here.
         */
        Core::Vector<Fsa::Weight> levenshteinScoresOverestimate_;

        /** The current underestimate of the Bayes risk. */
        Fsa::Weight estimate_;

        /** The current overestimate of the Bayes risk. Only defined if this node is final. */
        Fsa::Weight overestimate_;

        /** length of longest path from any state of this node to a final state. */
        u32 longestDistance_;

        /** Flag whether this hypothesis is complete or not. */
        bool isFinal_;

        bool isSemifinal_;

        /** Flag whether this hypothesis is explorable or not. */
        bool isExplorable_;

        /** Index in stack vector. necessary for efficient pruning */
        u32 index_;

        /** stack nodes have to be initialized properly, so default access is forbidden. */
        SearchNode();

    public:
        /** For initial state. */
        SearchNode(u32 longestDistance, Fsa::StateId initialStateId);

        /**
         * Constructor for all other but initial states.
         *
         * @param hypothesis         the generating old hypothesis.
         * @param symbol             the symbol to add to the generating hypothesis.
         * @param totalProbability   the total probability for this node.
         * @param levensheinColumns  ...
         * @param levenshteinScores  ...
         * @param estimate           ...
         * @param oldNodePosition    index of generating node in stack.
         */
        SearchNode(
                const std::vector<Fsa::LabelId>&     hypothesis,
                const Fsa::LabelId                   symbol,
                const Fsa::Weight&                   totalProbability,
                const std::vector<std::vector<u32>>& levenshteinColumns,
                const Core::Vector<Fsa::Weight>&     levenshteinScores,
                const Fsa::Weight&                   estimate,
                u32                                  oldNodePosition);

        /** virtual dtor. */
        virtual ~SearchNode();

        /**
         * Adds state to set of states the current hypothesis ends in.
         * This is needed in case the prefix-tree-search will be implemented.
         */
        void addState(
                const Fsa::StateId state,
                u32                longestDistance,
                const Fsa::Weight  logForwardProbability,
                const Fsa::Weight  logBackwardProbability,
                const Fsa::Weight  inverseNormalizationConstant);

        /**
         * Iterating all states, the current hypothesis ends in.
         * This is needed in case the prefix-tree-search will be implemented.
         */
        std::map<Fsa::StateId, Fsa::Weight>::const_iterator begin() const;

        /**
         * Iterating all states, the current hypothesis ends in.
         * This is needed in case the prefix-tree-search will be implemented.
         */
        std::map<Fsa::StateId, Fsa::Weight>::const_iterator end() const;

        /** @return the hypothesis represented by this stack node. */
        const std::vector<Fsa::LabelId>& getHypothesis() const;

        /** Getter for <code> levenshteinColums </code>.
         * A Levenshtein column is the last column in the Levenshtein distance matrix
         * between the hypothesis this nodes represents and the hypothesis at the
         * position in the array.
         *
         * NOTE: The index of a hypothesis in the stack will always be the same here.
         */
        const std::vector<std::vector<u32>>& getLevenshteinColumns() const;

        /** Getter for <code> levenshteinScores </code>.
         * The levenshtein scores have to be accumulated by the suiting semiring.
         *
         * NOTE: The index of a hypothesis in the stack will always be the same here.
         */
        const Core::Vector<Fsa::Weight>& getLevenshteinScores() const;

        /**
         * Computes total probability for this search node.
         * This is needed in case the prefix-tree-search will be implemented.
         */
        void computeTotalProbability();

        /** Getter for <code> final </code>. */
        bool isFinal() const;

        /**
         * Setter for <code> final </code>.
         * Updating final nodes, the overestimation has to be carried out.
         */
        void setFinal();

        /** Getter for <code> explorable </code>. */
        bool isExplorable() const;

        /** Setter for <code> explorable </code>. */
        void setInexplorable();

        /**
         * For comparison issues.
         * Stack nodes are first sorted by their underestimate ascending and then by their
         * probability mass descending.
         */
        friend bool operator<(SearchNodeRef& lower, SearchNodeRef& taller);

        /**
         * dumps a short info on the writer.
         */
        void dump(Core::XmlWriter& out, Fsa::ConstAlphabetRef alphabet) const;
    };

    /** Prints the current stack to logging channel. */
    void dump(Core::XmlWriter& out, Fsa::ConstAlphabetRef alphabet) const;

    /**
     * The actual stack and stack of the A* search. Constructed and deleted in minimumBayesRisk ( ).
     */
    std::vector<SearchNodeRef>* stack_;

    /** Scaling factor during risk computation. */
    Fsa::Weight inverseBackwardPotentialsNormalizationConstant_;

    /** Needed for risk estimation. */
    Fsa::StatePotentials backwardPotentials_;

    /** The current topmost stack element. */
    SearchNodeRef minimalNode_;

    /** The current most promising stack element. */
    SearchNodeRef newMinimalNode_;

    /** ... */
    u32 newMinimalNodeCounter_;

    /**
     * Needed for test for optimality.
     */
    SearchNodeRef secondMinimalNode_;

    /**
     * The most promising node that can still be expanded.
     * In case this reference is null, then no node can be further expanded
     * and the search terminates.
     */
    SearchNodeRef minimalIncompleteNode_;

    /** @see paramExactEstimate */
    bool exactEstimate_;

    /** */
    std::vector<Fsa::StateId> distances_;

    /** */
    u32 newLength_;

    /** @see paramMaxStackSize */
    u32 maxStackSize_;

    bool shallPrune_;

    u32 initialPruningThreshold_;

    f32 thresholdFactor_;

    u32 maximumNumberHypotheses_;

    /** */
    bool dumpStack_;

    u32 numberOverestimates_;

    /** Semiring of the corresponding Fsa used for all operations on scores. */
    Fsa::ConstSemiringRef semiring_;

    std::vector<std::vector<SearchNodeRef>> stackEntries_;

    /**
     * Expands the current topmost stack node.
     */
    void expand();

    /**
     * Creates the new search nodes to bne inserted into the stack.
     * Here just the hypothesis and the probability mass is computed as well as
     * the corresponding grid nodes for all existing stack entries, i.e. the
     * lower right part of the grid matrix.
     *
     */
    std::vector<SearchNodeRef> createNewSearchNodes(ConstSearchNodeRef expansionNode);

    /**
     * Updates the old stack entries during the expansion step.
     * Here the grid nodes for the new search tree nodes are added to the existing ones.
     * This corresponds to the upper left entries in the grid matrix.
     */
    void updateStack(SearchNodeRef expansionNode, std::vector<SearchNodeRef>& newNodes);

    /**
     *
     *
     */
    void overestimateFinal(SearchNodeRef finalNode);

    /**
     * Generates the grid nodes for the new search nodes, i.e. the upper right entries in the
     * grid matrix.
     *
     */
    void generateNewGridNodes(SearchNodeRef expansionNode, std::vector<SearchNodeRef>& newNodes);

    /**
     * As all search nodes with a certain length n are maintained in the n-th entry of the
     * vector stackEntries_ pruning keeps the number of such search tree nodes up to the
     * limit of maxStackSize_. Let n be the length of the newly created stack entries.
     * These have already been inserted into the n-th entry of stackEntries_ which is now
     * sorted and just the maxStackSize_ elements are being kept. The remaining hypotheses
     * are pruned from the stackEntries_ and finally from the fringe itself.
     *
     */
    void pruneStack();

    /**
     * Checks whether the current best overestimate is better than the second best current underestimate or
     * there is no node left which could be further expanded.
     *
     * @return true if stack is optimal
     */
    bool stackIsOptimal();

    /** @return short description fo this MinimumBayesRiskSearch. */
    std::string describe() const;

public:
    /**
     * @see LogWeights
     */
    static const u32 MaxLogWeights;

    /**
     * Here the negative logrithms for i=1...MaxLogWeights are stored for computation of the natural logarithm
     * needed for the computation of the risk estimates.
     */
    static std::vector<Fsa::Weight> LogWeights;

    /**
     * Flag whether the stack shall be dumped during computation.
     */
    static const Core::ParameterBool paramShallDumpStack;

    /**
     * Flag whether the exact estimate should be used..
     */
    static const Core::ParameterBool paramExactEstimate;

    /**
     * Maximum number of entries for stack for nodes corresponding to hypotheses of a certain length.
     */
    static const Core::ParameterInt paramMaxStackSize;

    /* Flag whether summation and evaluation space shall be pruned. **/
    static const Core::ParameterBool paramShallPrune;

    /** For pruning the summation and evluation space. */
    static const Core::ParameterFloat paramInitialPruningThreshold;

    /** The factor the threshold gets multiplied with. */
    static const Core::ParameterFloat paramThresholdFactor;

    /** For initial pruning.*/
    static const Core::ParameterInt paramMaximumNumberHypotheses;

    /**
     * Initializes Component and MinimumBayesRiskSearch.
     */
    MinimumBayesRiskAStarSearch(const Core::Configuration& config);

    /**
     * The actual process of computation of hypothesis having minimum Bayes Levenshtein risk.
     * The mbrAutomaton is valid only if this method has been called.
     */
    void performSearch(Fsa::ConstAutomatonRef fsa);
};

}  //end namespace Search
