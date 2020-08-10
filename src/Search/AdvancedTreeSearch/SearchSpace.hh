/** Copyright 2020 RWTH Aachen University. All rights reserved.
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
#ifndef SEARCH_CONDITIONEDTREESEARCHSPACE_HH
#define SEARCH_CONDITIONEDTREESEARCHSPACE_HH

#include <unordered_map>
#include <unordered_set>

#include <Lm/SearchSpaceAwareLanguageModel.hh>
#include <Search/Histogram.hh>
#include <Search/Search.hh>
#include <Search/StateTree.hh>

#include "Helpers.hh"
#include "LanguageModelLookahead.hh"
#include "PersistentStateTree.hh"
#include "ScoreDependentStatistics.hh"
#include "SearchSpaceHelpers.hh"
#include "SimpleThreadPool.hh"
#include "Trace.hh"
#include "TreeStructure.hh"

struct EmissionSetCounter;
namespace AdvancedTreeSearch {
class AcousticLookAhead;
}

inline Bliss::LemmaPronunciation* epsilonLemmaPronunciation() {
    return reinterpret_cast<Bliss::LemmaPronunciation*>(1);
}

namespace Search {
class PrefixFilter;

class SearchSpaceStatistics;

class StaticSearchAutomaton : public Core::Component {
public:
    using Precursor = Core::Component;

    /// HMM length of a common phoneme
    const u32 hmmLength;
    bool      minimized;

    /// Static representation of the search network:
    PersistentStateTree network;

    // "Temporary network data": Constructed for speedup from the normal network, by the function initialize.
    // Second order means successors of the target node, always contains first and last of contiguous exits. Only used in ExpandState(Slow) for skip transitions.
    std::vector<int> secondOrderEdgeSuccessorBatches;
    std::vector<u32> quickLabelBatches;  // Contiguous successors, only stores first exit for each node.
    std::vector<s32> slowLabelBatches;   // Simple exit-index lists terminated by -1
    // -1 if there are no labels on the state
    // -2 if there are multiple labels
    // index of the label otherwise
    std::vector<int> singleLabels;
    // LM- and acoustic look-ahead ids together, for quicker access
    std::vector<std::pair<u32, u32>> lookAheadIds;
    // Sparse LM lookahead hash and acoustic lookahead id paired togeter
    std::vector<std::pair<u32, u32>> lookAheadIdAndHash;

    std::vector<int> stateDepths;
    // Inverted depth for each network-state, eg. the distance until the most distant following word-end
    std::vector<int> invertedStateDepths;
    // Only correctly filled until an inverted depth of 256, for more efficient access
    // Also adapted to the fade-in granularity
    std::vector<unsigned char> truncatedInvertedStateDepths;
    std::vector<unsigned char> truncatedStateDepths;

    // number of transitions needed untill the next word end (that is not silence)
    std::vector<unsigned> labelDistance;

    /// Optional filter which allows limiting the search space to a certain word sequence prefix
    // Checks whether the syntactical tokens of a hypothesis share a common prefix. Has all words from lexicon_,
    // is initialized in initialize() and used in filterStates->pruneStates.
    PrefixFilter* prefixFilter;

    StaticSearchAutomaton(Core::Configuration config, Core::Ref<const Am::AcousticModel> acousticModel, Bliss::LexiconRef lexicon);
    ~StaticSearchAutomaton();

    void buildNetwork();

    // Assigns a depth in the automaton_.stateDepths array to each state
    // If onlyFromRoot is true, then the depths are only computed behind the root state,
    // and the depths of not encountered states have the value -1
    void buildDepths(bool onlyFromRoot = false);

    // Clears the state depths
    void clearDepths();

    // Makes checks and works recursively.
    int fillStateDepths(Search::StateId state, int depth);
    int findStateDepth(Search::StateId state);

    int stateDepth(StateId idx) {
        verify(idx < stateDepths.size());
        return stateDepths[idx];
    }

    // fills the labelDistance array, has to be run after buildDepths (as they are used for topological sorting)
    void buildLabelDistances();

    // Creates fast look-up structures like singleOutputs_, quickOutputBatches_ and secondOrderEdgeTargetBatches_.
    void buildBatches();

private:
    Core::Ref<const Am::AcousticModel> acousticModel_;
    Bliss::LexiconRef                  lexicon_;
};


class SearchSpace : public Core::Component {
public:
    /// Statistics:
    SearchSpaceStatistics* statistics;

    void resetStatistics();

    void logStatistics(Core::XmlChannel& channel) const;

protected:
    /// ----------    Data ----------:

    // Descriptors of the current search phase:
    f64                       globalScoreOffset_;  // Offset applied to all hypothesis scores, for numerical stability reasons (corrected in lattice and traceback)
    TimeframeIndex            timeFrame_;          // Current timeframe
    Mm::FeatureScorer::Scorer scorer_;             // Feature scorer for the current timeframe

    /// Persistent models:
    Bliss::LexiconRef                            lexicon_;
    Core::Ref<const Am::AcousticModel>           acousticModel_;
    Core::Ref<const Lm::ScaledLanguageModel>     lm_;
    Core::Ref<const Lm::ScaledLanguageModel>     lookaheadLm_;
    Core::Ref<const Lm::LanguageModel>           recombinationLm_;
    Lm::SearchSpaceAwareLanguageModel const*     ssaLm_;
    AdvancedTreeSearch::LanguageModelLookahead*  lmLookahead_;
    std::vector<const Am::StateTransitionModel*> transitionModels_;

    TraceManager trace_manager_;

    PersistentStateTree const& network() const {
        return automaton_->network;
    }

    StaticSearchAutomaton const* automaton_;

    Lm::History                                                           unigramHistory_;
    AdvancedTreeSearch::LanguageModelLookahead::ContextLookaheadReference unigramLookAhead_;

    AdvancedTreeSearch::AcousticLookAhead* acousticLookAhead_;

    /// Search options
    bool     conditionPredecessorWord_;
    bool     decodeMesh_;
    bool     correctPushedBoundaryTimes_;
    bool     correctPushedAcousticScores_;
    bool     earlyBeamPruning_;
    bool     earlyWordEndPruning_;
    bool     histogramPruningIsMasterPruning_;
    bool     reducedContextWordRecombination_;
    unsigned reducedContextWordRecombinationLimit_;
    bool     onTheFlyRescoring_;
    unsigned onTheFlyRescoringMaxHistories_;
    unsigned maximumMutableSuffixLength_;
    unsigned maximumMutableSuffixPruningInterval_;

    /// Pruning thresholds
    Score acousticPruning_;       // main pruning threshold (log-scores)
    u32   acousticPruningLimit_;  // maximum number of hypotheses for Histogram pruning
    Score wordEndPruning_;
    u32   wordEndPruningLimit_;
    f64   lmStatePruning_;
    f32   wordEndPhonemePruningThreshold_;
    f32   acousticProspectFactor_;
    Score perInstanceAcousticPruningScale_;

    /// Boundary values for incremental search:
    f32 minimumBeamPruning_, maximumBeamPruning_;
    u32 minimumAcousticPruningLimit_, maximumAcousticPruningLimit_;
    f32 minimumStatesAfterPruning_, minimumWordEndsAfterPruning_, minimumWordLemmasAfterRecombination_;
    f32 maximumStatesAfterPruning_, maximumWordEndsAfterPruning_, maximumAcousticPruningSaturation_;

    // Minimum anticipated lm score expected during early word end pruning
    f64 earlyWordEndPruningAnticipatedLmScore_;

    // Distance over which the word end pruning is faded in
    u32 wordEndPruningFadeInInterval_;

    /// ---- Lookahead / Instance options -------------
    // After this number of inactive timeframes, network copies (i.e. instances) are deleted
    // It makes sense to keep instances alive for some time, to retain
    // the lm cache, context lookahead, etc.
    u32 instanceDeletionLatency_;

    // Minimum state-count in an instance at which the full look-ahead table is computed
    u32 fullLookAheadStateMinimum_;
    // Minimum dominance of an instance at which the full look-ahead table is computed
    f32 fullLookAheadDominanceMinimum_;
    // Sparse LM Look-Ahead is only activated when the instance has
    // at least this number of states. (reduce-lookahead-before-depth must be active too)
    // This takes the network-dominance into account, and is calculated at the end of every timeframe.
    u32 currentLookaheadInstanceStateThreshold_;
    // When this lookahead-id (i.e. depth) is crossed, full LM look-ahead is used
    AdvancedTreeSearch::LanguageModelLookahead::LookaheadId fullLookaheadAfterId_;

    bool sparseLookahead_, overflowLmScoreToAm_, sparseLookaheadSlowPropagation_;
    f32  unigramLookaheadBackoffFactor_;
    bool earlyBackoff_, correctBackoff_;

    bool allowSkips_;

    Score wpScale_;
    // If this is true, expensive extended statistics are collected
    bool extendStatistics_;

    bool encodeState() const {
        return encodeStateInTraceAlways_ || (encodeStateInTrace_ && recognitionContext_.latticeMode == SearchAlgorithm::RecognitionContext::No);
    }

    bool encodeStateInTrace_, encodeStateInTraceAlways_;

    /// Temporary search variables:
    mutable Score     bestScore_, bestProspect_, minWordEndScore_;
    mutable Histogram stateHistogram_;
    mutable Histogram wordEndHistogram_;

    mutable std::unordered_map<InstanceKey, Score, InstanceKey::Hash> bestInstanceProspect_;

    typedef std::vector<StateHypothesis> StateHypothesesList;

    /// The dynamic search space:
    // The state hypotheses of the two current frames.
    StateHypothesesList stateHypotheses;
    StateHypothesesList newStateHypotheses;

    WordEndHypothesisList      wordEndHypotheses;
    EarlyWordEndHypothesisList earlyWordEndHypotheses;

    typedef std::vector<Instance*> InstanceList;
    InstanceList                   activeInstances;

    typedef std::unordered_map<InstanceKey, Instance*, InstanceKey::Hash> KeyInstanceMap;
    KeyInstanceMap                                                        activeInstanceMap;

    /// Temporary search space helpers:
    StateHypothesisIndex currentTreeFirstNewStateHypothesis;

    typedef std::unordered_set<WordEndHypothesisList::iterator, WordEndHypothesis::Hash, WordEndHypothesis::Equality> WordEndHypothesisRecombinationMap;
    WordEndHypothesisRecombinationMap                                                                                 wordEndHypothesisMap;  // Map used for recombining word end hypotheses

    std::vector<StateHypothesisIndex> stateHypothesisRecombinationArray;  // Array used to recombine state hypotheses

    ScoreDependentStatistic statesOnDepth_;
    ScoreDependentStatistic statesOnInvertedDepth_;

    /// Statistics, which are reset whenever a new (partial) recognition is started.
    bool                  hadWordEnd_;
    Core::Statistics<u32> currentStatesAfterPruning, currentWordEndsAfterPruning, currentWordLemmasAfterRecombination;
    Core::Statistics<f32> currentAcousticPruningSaturation;

    /// Dynamically set context (i.e. left and right boundary conditions)
    Search::SearchAlgorithm::RecognitionContext recognitionContext_;

    /// Memory Leak workaround
    std::vector<Core::WeakRef<Trace>> altHistTraces_;

public:
    /// ----------- Public interface during search --------------:

    void setAllowHmmSkips(bool allow);

    // Startup of a segment
    void addStartupWordEndHypothesis(TimeframeIndex);

    // Called every time-frame before the expansion is started
    void setCurrentTimeFrame(TimeframeIndex timeFrame, const Mm::FeatureScorer::Scorer& scorer);

    // Start new trees based on the active word end hypotheses
    void startNewTrees();

    // Within-word HMM expansion
    void expandHmm();

    // Adds acoustic scores and applies pruning
    void pruneAndAddScores();

    // Creates early word end hypotheses from the active state hypotheses
    void findWordEnds();

    // Prunes early word end hypotheses, and expands them to normal word end hypothses
    void pruneEarlyWordEnds();

    // Applies time-, score- and transit-modification to the given trace-id, and returns the corrected trace item (as successor of the original trace item)
    inline Core::Ref<Trace> getModifiedTrace(TraceId trace, Bliss::Phoneme::Id initialPhone) const;

    // Prunes the normal word end hypotheses regarding their score (remove all hypotheses with score worse than absoluteScoreThreshold)
    void pruneWordEnds(Search::Score absoluteScoreThreshold);

    // Adds the recognized word to the traces of all word end hypotheses
    void createTraces(TimeframeIndex time);

    // Recombine word end hypotheses according to their history,
    // eventually weaving the traces to create lattices.
    void recombineWordEnds(bool shallCreateLattice = false);

    // Creates hypotheses for those pronunciations which are empty (eg. they consist of zero phonemes)
    void hypothesizeEpsilonPronunciations(Score bestScore);

    // Clear the whole dynamic search space
    void clear();

    // Rescales the scores, for better numeric stability,
    // removes the common score offset and remember it in globalScoreOffset_
    void rescale(Score offset, bool ignoreWordEnds = false);

    // Returns the info from trace manager about whether it needs cleanup
    inline bool needCleanup() const { return trace_manager_.needCleanup();}

    // Needs to be called once in a while, but not every timeframe,
    // deletes all traces that did not survive in stateHypotheses and rootStateHypotheses of activeTrees
    void cleanup();

    // Optimize the lattice, removing redundant silence occurances
    void optimizeSilenceInWordLattice(const Bliss::Lemma* silence);

    Core::Ref<Trace> getSentenceEnd(TimeframeIndex time, bool shallCreateLattice);

    Core::Ref<Trace> getSentenceEndFallBack(TimeframeIndex time, bool shallCreateLattice);

    // Returns the prefix trace which is common to all active hypotheses
    Core::Ref<Trace> getCommonPrefix() const;

    // Modifies the search space so that the given trace is the initial trace
    // The score of the given trace will be changed to zero, it will have no pronunciations
    // and no siblings. The search space will be modified so that it is correct relative to this
    // initial trace. All trace (i.e. lattice) paths not leading towards this trace are truncated.
    // Timesstamps are _not_ changed.
    void changeInitialTrace(Core::Ref<Trace> trace);

    u32 nStateHypotheses() const;
    u32 nEarlyWordEndHypotheses() const;
    u32 nWordEndHypotheses() const;
    u32 nActiveTrees() const;

    int                                         lookAheadLength() const;
    void                                        setLookAhead(const std::vector<Mm::FeatureVector>&);
    Search::SearchAlgorithm::RecognitionContext setContext(Search::SearchAlgorithm::RecognitionContext);

    ///Returns the best prospect, eg. the score of the best state hypothesis including the look-ahead score
    Score bestProspect() const;
    ///@warning: Expensive, without caching
    StateHypothesesList::const_iterator bestProspectStateHypothesis() const;
    ///Returns the best score (the look-ahead score is not included)
    ///@warning: Expensive, but with caching
    Score bestScore() const;
    ///@warning: Expensive, without caching
    StateHypothesesList::const_iterator bestScoreStateHypothesis() const;
    Score                               quantileStateScore(Score min, Score max, u32 nHyp) const;
    ///Returns the lowest word end score (without look-ahead)
    ///Always valid after findWordEnds was called
    Score minimumWordEndScore() const;
    Score quantileWordEndScore(Score min, Score max, u32 nHyp) const;

protected:
    /// ---------------- Search algorithm implementation details ------------------:

    // Apply special search space filtering
    void filterStates();
    // make sure that a prefix of the current best hyp is the prefix for all hyps
    void enforceCommonPrefix();
    // Applies state-pruning _before_ acoustic scores are added
    void pruneStatesEarly();
    void pruneStatesPerLmState();

    template<class Pruning>
    void addAcousticScoresInternal(Instance const& instance, Pruning& pruning, u32 from, u32 to);
    template<class Pruning>
    void addAcousticScores();

    // Prune states, ignoring and forgetting network-assignment
    template<class Pruning>
    void pruneStates(Pruning& pruning);

    void updateSsaLm();

    void correctPushedTransitions();

    void doStateStatisticsBeforePruning();

    // Called after pruning and after adding acoustic scores, but before finding word-ends
    void doStateStatistics();

    void doWordEndStatistics();

    /**
     * @param compute If this is true, a new look-ahead table is computed. Otherwise, only existing tables are re-used.
     * */
    void activateLmLookahead(Instance& instance, bool compute);

    template<bool sparseLookAhead, bool reduceLookAheadBasedOnDepth, class AcousticLookAhead, class Pruning>
    void applyLookaheadInInstanceInternal(Instance* _instance, AcousticLookAhead& acousticLookAhead, Pruning& pruning);

    // Must be called at a point in time where the states of the network reference "newStateHypotheses_"
    template<class Pruning, class AcousticLookAhead>
    void applyLookaheadInInstanceWithAcoustic(Instance* network, AcousticLookAhead& acousticLookAhead, Pruning& pruning);

    void applyLookaheadInInstance(Instance* network);

    inline_ void addNewStateHypothesis(const StateHypothesis& hyp) {
        newStateHypotheses.push_back(hyp);
    }

    inline_ void activateOrUpdateStateHypothesisLoop(const StateHypothesis& hyp, Score score);
    inline_ void activateOrUpdateStateHypothesisTransition(const StateHypothesis& hyp, Score score, StateId successorState);
    inline_ void activateOrUpdateStateHypothesisDirectly(const StateHypothesis& hyp);

    /// Checks whether network should be deactivated, and if so do it.
    /// Returns true iff network has been deactivated
    inline_ bool eventuallyDeactivateTree(Search::Instance*, bool increaseInactiveCounter);

    Instance* createTreeInstance(const InstanceKey& key);

    template<bool allowSkip>
    inline_ void expandState(const Search::StateHypothesis& hyp);
    template<bool expandForward, bool expandSkip>
    inline_ void expandStateSlow(const Search::StateHypothesis& hyp);

    /// ------------ Search algorithm helpers ---------------------------:

    ///@param create Whether the network should be created if it doesn't exist yet
    ///@param key The key, containing the unique properties of the network
    Instance* instanceForKey(bool create, const InstanceKey& key, Lm::History const& lookaheadHistory, Lm::History const& scoreHistory);

    template<bool earlyWordEndPruning, bool onTheFlyRescoring>
    void processOneWordEnd(Instance const& at, StateHypothesis const& hyp, s32 exit, Score exitPenalty, Score relativePruning, Score& bestWordEndPruning);

    template<bool earlyWordEndPruning, bool onTheFlyRescoring>
    void findWordEndsInternal();

    template<bool onTheFlyRescoring>
    void recombineWordEndsInternal(bool shallCreateLattice);

    u32 getLastSyntacticToken(const Core::Ref<Trace>& _trace) {
        const Trace* trace = _trace.get();
        while (trace) {
            if (trace->pronunciation &&
                trace->pronunciation != epsilonLemmaPronunciation() &&
                trace->pronunciation->lemma() &&
                trace->pronunciation->lemma()->hasSyntacticTokenSequence() &&
                trace->pronunciation->lemma()->syntacticTokenSequence().size()) {
                return trace->pronunciation->lemma()->syntacticTokenSequence()[trace->pronunciation->lemma()->syntacticTokenSequence().size() - 1]->id();
            }
            if (!trace->predecessor)
                break;
            trace = trace->predecessor.get();
        }
        return Core::Type<u32>::max;
    }

    Instance* activateOrUpdateTree(const Core::Ref<Trace>&, Lm::History, Lm::History, Lm::History, StateId, Score);

    Instance* getBackOffInstance(Instance* instance);

    inline_ const Am::StateTransitionModel* transitionModel(const StateTree::StateDesc& desc) const {
        return transitionModels_[size_t(desc.transitionModelIndex)];
    }

    void getTransitionModels() {
        transitionModels_.resize(acousticModel_->nStateTransitions(), 0);
        for (u32 t = 0; t < transitionModels_.size(); ++t) {
            transitionModels_[t] = acousticModel_->stateTransition(t);
        }
    }

    StateId rootForCoarticulation(std::pair<Bliss::Phoneme::Id, Bliss::Phoneme::Id>) const;

    void initializeLanguageModel();

    // Integrated profiling
    PerformanceCounter* applyLookaheadPerf_;
    PerformanceCounter* applyLookaheadSparsePerf_;
    PerformanceCounter* applyLookaheadSparsePrePerf_;
    PerformanceCounter* applyLookaheadStandardPerf_;
    PerformanceCounter* computeLookaheadPerf_;
    PerformanceCounter* extendedPerf_;  // This counter can be used for debugging

    // Implementation of recombination-pruning
    class RecombinationPruningBase;
    struct RecombinationDistance;
    template<bool crossPruning, class Base>
    struct ShortRecombinationDistance;
    template<bool collectDepthMinimum, class Base>
    struct Depth;
    template<bool recombinationSetPruning, class RecombinationPruning, class Base>
    struct RecombinationSet;

    void pruneSilenceSiblingTraces(Core::Ref<Trace>, const Bliss::Lemma* silence);
    void dumpWordEnds(std::ostream&, Core::Ref<const Bliss::PhonemeInventory>) const;
    void extendHistoryByLemma(WordEndHypothesis& weh, const Bliss::Lemma* lemma) const;

    std::pair<Bliss::Phoneme::Id, Bliss::Phoneme::Id> describeRootState(StateId state) const;

    struct WordEndPusher;

    // helper function to for recombineWordEnds(bool)
    template<bool onTheFlyRescoring>
    inline_ void recombineTwoHypotheses(WordEndHypothesis& a, WordEndHypothesis& b, bool shallCreateLattice);

public:
    /// ------------- External Initialization --------------------:
    SearchSpace(const Core::Configuration& configuration, Core::Ref<const Am::AcousticModel> acousticModel,
                Bliss::LexiconRef lexicon, Core::Ref<const Lm::ScaledLanguageModel> lm, Score wpScale);

    class PruningDesc : public SearchAlgorithm::Pruning {
    public:
        virtual Core::Ref<SearchAlgorithm::Pruning> clone() const {
            return Core::Ref<SearchAlgorithm::Pruning>(new PruningDesc(*this));
        }
        PruningDesc()
                : beam(Core::Type<Score>::max),
                  searchSpaceOK(true) {
        }

        Score beam;
        bool  searchSpaceOK;

        virtual bool checkSearchSpace() const {
            return searchSpaceOK;
        }

        Score beamForTime(TimeframeIndex t) const {
            if (t < timeBeamMap.size() && timeBeamMap[t] != Core::Type<Score>::max)
                return timeBeamMap[t];
            else
                return beam;
        }

        virtual Score masterBeam() const {
            return beam;
        }

        virtual Score maxMasterBeam() const {
            Score ret = beam;
            for (u32 t = 0; t < timeBeamMap.size(); ++t)
                if (timeBeamMap[t] != Core::Type<Score>::max && timeBeamMap[t] > ret)
                    ret = timeBeamMap[t];
            return ret;
        }

        bool haveTimeDependentPruning() const {
            return timeBeamMap.size();
        }

        // endTime is _inclusive_
        virtual bool merge(const Core::Ref<Pruning>& rhs, TimeframeIndex ownLength, TimeframeIndex startTime, TimeframeIndex endTime) {
            PruningDesc* rhsDesc(dynamic_cast<PruningDesc*>(rhs.get()));
            if (rhsDesc->timeBeamMap.empty() && startTime == 0 && endTime + 1 >= ownLength && timeBeamMap.empty()) {
                beam = rhsDesc->beam;
                return true;
            }
            verify(rhsDesc);
            verify(startTime <= endTime);
            if (endTime >= timeBeamMap.size())
                timeBeamMap.resize(endTime + 1, Core::Type<Score>::max);
            for (TimeframeIndex t = startTime; t <= endTime; ++t)
                timeBeamMap[t] = rhsDesc->beamForTime(t - startTime);
            return true;
        }

        /// Extend the pruning-thresholds by a score offset, a score factor, and by a specific number of timeframes
        virtual bool extend(Score scoreFactor, Score scoreOffset, TimeframeIndex timeOffset) {
            if (timeBeamMap.size()) {
                std::vector<Score> oldTimePruningMap;
                oldTimePruningMap.swap(timeBeamMap);
                timeBeamMap.resize(oldTimePruningMap.size() + timeOffset, Core::Type<Score>::max);
                for (TimeframeIndex t = 0; t < timeBeamMap.size(); ++t) {
                    int start = std::max((int)t - (int)timeOffset, 0);
                    int end   = std::min((int)t + (int)timeOffset + 1, (int)oldTimePruningMap.size());
                    for (TimeframeIndex tOld = start; tOld < end; ++tOld) {
                        if (oldTimePruningMap[tOld] != Core::Type<Score>::max &&
                            (oldTimePruningMap[tOld] > timeBeamMap[t] || timeBeamMap[t] == Core::Type<Score>::max))
                            timeBeamMap[t] = oldTimePruningMap[tOld];
                    }
                }
            }

            if (beam != Core::Type<Score>::max)
                beam = beam * scoreFactor + scoreOffset;

            for (TimeframeIndex t = 0; t < timeBeamMap.size(); ++t)
                if (timeBeamMap[t] != Core::Type<Score>::max)
                    timeBeamMap[t] = timeBeamMap[t] * scoreFactor + scoreOffset;

            return true;
        }

        virtual std::string format() const {
            std::ostringstream os;
            os << beam;
            if (timeBeamMap.size()) {
                os << " [ ";
                Score previous      = Core::Type<Score>::max;
                int   previousStart = -1;
                for (TimeframeIndex t = 0; t < timeBeamMap.size(); ++t) {
                    if (timeBeamMap[t] != previous) {
                        if (previousStart != -1 && previous != Core::Type<Score>::max)
                            os << previousStart << "-" << t - 1 << " : " << previous << " ";
                        previous      = timeBeamMap[t];
                        previousStart = t;
                    }
                }
                if (previousStart != -1 && previous != Core::Type<Score>::max)
                    os << previousStart << "-" << timeBeamMap.size() - 1 << " : " << previous << " ";
                os << "]";
            }
            return os.str();
        }

    private:
        // Core::Type<Score>::max or no value means that the standard acoustic-pruning is to be used
        std::vector<Score> timeBeamMap;
    };

    Core::Ref<PruningDesc> currentPruning_;

    SearchAlgorithm::PruningRef describePruning();
    bool                        relaxPruning(f32 factor, f32 offset);
    void                        resetPruning(SearchAlgorithm::PruningRef pruning);

    Score beamPruning() const {
        if (acousticPruning_ != Core::Type<Score>::max)
            return acousticPruning_ / lm_->scale();
        else
            return Core::Type<Score>::max;
    }

    // Sets acoustic-pruning and adapts the secondary pruning values accordingly
    void setMasterBeam(Score value);

    // Must be called after creation, after the above functions were called
    virtual void initialize();

    void initializePruning();

    virtual ~SearchSpace();

private:
    // These are implemented in Pruning.hh
    struct AcousticPruning;
    struct PerInstanceAcousticPruning;
    struct RecordMinimum;
    struct RecordMinimumPerInstance;
    struct NoPruning;
    struct BestTracePruning;

    template<class Base, bool shortRec, bool depths>
    struct FadeInPruningCollectMinimum;
    friend struct FadeInPruningCollectMinimum<RecordMinimum, true, false>;
    friend struct FadeInPruningCollectMinimum<RecordMinimum, false, false>;
    friend struct FadeInPruningCollectMinimum<RecordMinimum, true, true>;
    friend struct FadeInPruningCollectMinimum<RecordMinimum, false, true>;

    template<class Base>
    struct FadeInPruning;

    friend class AdvancedTreeSearch::AcousticLookAhead;
};
}  // namespace Search

#endif
