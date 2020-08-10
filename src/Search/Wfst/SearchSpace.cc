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
#include <Fsa/Hash.hh>
#include <OpenFst/LabelMap.hh>
#include <Search/Wfst/ComposedNetwork.hh>
#include <Search/Wfst/CompressedNetwork.hh>
#include <Search/Wfst/Lattice.hh>
#include <Search/Wfst/LatticeGenerator.hh>
#include <Search/Wfst/LatticeNetwork.hh>
#include <Search/Wfst/SearchSpace.hh>
#include <Search/Wfst/StateSequence.hh>
#include <Search/Wfst/Statistics.hh>
#include <Search/Wfst/Traceback.hh>
#include <cstring>

using OpenFst::Epsilon;

namespace Search {
namespace Wfst {

const Score                     SearchSpaceBase::InvalidScore = Core::Type<Score>::max;
const SearchSpaceBase::TraceRef SearchSpaceBase::InvalidTraceRef =
        TraceRecorder::InvalidTraceRef;

SearchSpaceBase::SearchSpaceBase()
        : statisticsCollector_(0),
          book_(0),
          stateSequences_(0),
          pruningThreshold_(Core::Type<Score>::max),
          latticePruning_(Core::Type<Score>::max),
          pruningLimit_(Core::Type<u32>::max),
          purgeInterval_(Core::Type<u32>::max),
          createLattice_(false),
          ignoreLastOutput_(false),
          initialEpsPruning_(false),
          epsilonArcPruning_(true),
          prospectivePruning_(true),
          mergeEpsPaths_(false),
          twoPassPruning_(false),
          mergeSilenceArcs_(true),
          wordEndPruning_(false),
          wordEndThreshold_(Core::Type<Score>::max),
          weightScale_(1.0),
          latticeType_(LatticeTraceRecorder::HmmLattice),
          silence_(0),
          silenceOutput_(OpenFst::InvalidLabelId) {}

SearchSpaceBase::~SearchSpaceBase() {
    delete stateSequences_;
    delete statisticsCollector_;
    delete book_;
}

void SearchSpaceBase::setTransitionModel(Core::Ref<const Am::AcousticModel> acousticModel) {
    if (acousticModel->nStateTransitions() > Core::Type<TransitionModelIndex>::max)
        Core::Application::us()->error("maximum number of transition models %d exceeded: %d",
                                       u32(Core::Type<TransitionModelIndex>::max),
                                       acousticModel->nStateTransitions());
    transitionModels_.resize(acousticModel->nStateTransitions(), 0);
    for (u32 t = 0; t < transitionModels_.size(); ++t) {
        transitionModels_[t] = acousticModel->stateTransition(t);
    }
    entryForwardScore = (*transitionModels_[Am::TransitionModel::entryM1])[Am::StateTransitionModel::forward];
    entrySkipScore    = (*transitionModels_[Am::TransitionModel::entryM1])[Am::StateTransitionModel::skip];
}

bool SearchSpaceBase::setNonWordPhones(Core::Ref<const Am::AcousticModel> am,
                                       const std::vector<std::string>&    phones) {
    require(stateSequences_);
    return wordEnds_.setNonWordPhones(am, *stateSequences_, phones);
}

void SearchSpaceBase::setUseNonWordModels(u32 nNonWordModels) {
    require(stateSequences_);
    wordEnds_.setNonWordModels(*stateSequences_, nNonWordModels);
}

void SearchSpaceBase::resetStatistics() {
    statisticsCollector_->reset();
}

void SearchSpaceBase::logStatistics(Core::XmlChannel& channel) const {
    statisticsCollector_->log(channel);
}

bool SearchSpaceBase::init(std::string& msg) {
    outputIsWordEnd_ = wordEnds_.type() == WordEndDetector::WordEndOutput;
    return true;
}

void SearchSpaceBase::setWordEndHyp(StateIndex state, bool isWordEnd) {
    if (state >= wordEndHyp_.size())
        wordEndHyp_.resize(state + 1, false);
    wordEndHyp_[state] = isWordEnd;
}

/******************************************************************************/

/**
 * SearchSpace implementation depending on
 *   1) the implementation of the Network used
 *   2) whether or not skip transitions are allowed
 */
template<class N, bool S>
class SearchSpace : public SearchSpaceBase {
private:
    typedef N                 Network;
    typedef SearchSpace<N, S> Self;
    Network*                  network_;
    static const bool         UseSkips = S;

public:
    SearchSpace(Network* network);
    virtual ~SearchSpace() {
        delete network_;
    }
    void feed(const Mm::FeatureScorer::Scorer& scorer);
    void reset();
    void setSegment(const std::string& name) {
        network_->setSegment(name);
    }
    void     getTraceback(BestPath* path);
    Lattice* createLattice(OutputType type);
    void     setStatistics(bool detailed) {
        require(!statisticsCollector_);
        if (detailed)
            statisticsCollector_ = new Statistics::DetailedCollector<Self>(this);
        else
            statisticsCollector_ = new Statistics::DefaultCollector(this);
    }
    u32 nActiveStates() const {
        return activeStates_.size();
    }
    bool init(std::string& msg);
    u32  nStates() const {
        return network_->nStates();
    }
    u32 nArcs() const {
        return network_->nArcs();
    }
    u32 nEpsilonArcs() const {
        return network_->nEpsilonArcs();
    }

private:
    typedef typename Network::ArcIterator        ArcIterator;
    typedef typename Network::EpsilonArcIterator EpsilonArcIterator;

    enum { NumIncomingHyps = 1 + UseSkips };
    /**
     * active state
     */
    struct StateHyp {
        /*! state id */
        StateIndex state;
        /*! arc start hypotheses from incoming arcs (0 = normal, 1 = skip) */
        IncomingHyp              incoming[NumIncomingHyps];
        static const IncomingHyp InitIncomingHyp;
        StateHyp(StateIndex _state)
                : state(_state) {
            incoming[0] = incoming[static_cast<size_t>(UseSkips)] = InitIncomingHyp;
        }
    };
    typedef std::vector<StateHyp> StateHypotheses;
    /**
     * compare state hypotheses by associated state id
     */
    struct SortByState {
        bool operator()(const StateHyp& a, const StateHyp& b) const {
            return a.state < b.state;
        }
    };
    StateHypotheses activeStates_;

    /**
     * arc with input epsilon label.
     * used only in expandEpsilonArcs
     */
    struct EpsilonArcHyp {
        StateIndex target;
        Label      output;
        Score      score;
        TraceRef   trace[NumIncomingHyps];
        EpsilonArcHyp(StateIndex _target, Label _output, Score _score)
                : target(_target), output(_output), score(_score) {}
        EpsilonArcHyp(StateIndex _target, Label _output, Score _score, const IncomingHyp* _in)
                : target(_target), output(_output), score(_score) {
            for (u32 i = 0; i < NumIncomingHyps; ++i)
                trace[i] = _in[i].trace;
        }
        EpsilonArcHyp(StateIndex _target, Label _output, Score _score, const TraceRef* _trace)
                : target(_target), output(_output), score(_score) {
            for (u32 i = 0; i < NumIncomingHyps; ++i)
                trace[i] = _trace[i];
        }
    };

    class StateHypIterator {
    public:
        StateHypIterator(const StateHypotheses& states)
                : iter_(states.begin()), end_(states.end()) {
            update();
        }
        bool done() const {
            return iter_ == end_;
        }
        void next() {
            ++iter_;
            update();
        }
        const StateHyp& value() const {
            return *iter_;
        }
        bool hasIncoming() const {
            return incoming_;
        }

    private:
        void update() {
            incoming_ = done() ? false : (isActiveHyp(iter_->incoming[0]) || (UseSkips && isActiveHyp(iter_->incoming[1])));
        }
        typename StateHypotheses::const_iterator iter_, end_;
        bool                                     incoming_;
    };

private:
    void      addInitialStateHypothesis();
    void      activateOrUpdateHmmState(StateHypIndex hmmState, Score score, const TraceRef& trace);
    void      addArcHyp(StateHypIndex hmmStateEnd, StateIndex source, StateIndex target,
                        const StateSequence* hmm, Label output, Score score, ArcIndex arc);
    StateHyp* getStateHyp(StateIndex stateIndex);
    bool      activateOrUpdateIncomingHyp(IncomingHyp& incoming, TraceRef trace, Label output,
                                          const StateSequence* input, TimeframeIndex traceTime,
                                          Score score, Score arcScore, bool wordEnd);
    void      expandHmmState(const ArcHyp& arcHyp, u32 hmmState, Score score, const TraceRef& trace);
    void      expandArc(const ArcHyp& arcHyp, StateHypIndex hmmStateHypBegin);
    void      expandState(const StateHyp& stateHyp, bool expandArcs, ArcHypIndex* arcHypIndex);
    void      expandStatesAndArcs();
    template<bool MergePaths>
    void      expandEpsilonArcs(Score threshold, bool anticipatedPruning = false);
    void      findEpsilonReachable(const StateHyp& stateHyp, Fsa::Stack<EpsilonArcHyp>* hyps) const;
    void      findEpsilonPaths(const EpsilonArcHyp& arc, const StateToScoreMap* visitedStates,
                               Score threshold, const bool mergePaths,
                               Fsa::Stack<EpsilonArcHyp>* statesToExplore) const;
    void      expandInterArcTransitions(Score threshold);
    void      expandTransition(const ArcHyp& arcHyp, const StateHypIndex prevArcHypEnd, const Score threshold);
    StateHyp* expandTransHyp(const HmmStateHyp& hmmStateHyp, const ArcHyp& arcHyp,
                             const StateSequence& hmm, const StateIndex hmmState,
                             const Score threshold, const bool isFinalPhone,
                             const bool isWordEnd, const bool isLastState,
                             StateHyp* nextStateHyp);
    void      pruneHmmStates(Score threshold);
    void      pruneHmmStatesInPlace(Score threshold);
    void      pruneLattice();
    void      pruneWordEnds(Score threshold);
    Score     quantileStateScore(HmmStateHypotheses::const_iterator begin,
                                 HmmStateHypotheses::const_iterator end,
                                 Score minScore, Score maxScore, u32 nHyps);
    void      updateContextStateScore(StateToScoreMap& scores, StateIndex state,
                                      Score score) const;
    void      addAcousticScores(const Mm::FeatureScorer::Scorer& scorer);
    void      purgeTraces();
    TraceRef  getSentenceEnd();
    TraceRef  getSentenceEndFallback();
    void      collectStatistics(Core::Statistics<u32>& activeStates,
                                Core::Statistics<u32>& activeArcs);

    friend class Statistics::DefaultCollector;
    friend class Statistics::DetailedCollector<Self>;
    friend class Statistics::GrammarStateStatistic<Self>;

public:
    MemoryUsage memoryUsage() {
        MemoryUsage m;
        m.bookkeeping    = book_->memoryUsage();
        m.stateSequences = stateSequences_->memoryUsage();
        m.arcs           = network_->memArcs();
        m.states         = network_->memStates();
        m.epsilonArcs    = network_->memEpsilonArcs();
        m.stateHyps      = activeStates_.capacity() * sizeof(StateHyp);
        m.arcHyps        = (activeArcs_.capacity() + newActiveArcs_.capacity()) * sizeof(ArcHyp);
        m.hmmStateHyps   = (hmmStateHypotheses_.capacity() +
                          newHmmStateHypotheses_.capacity()) *
                         sizeof(HmmStateHyp);
        return m;
    }
};

template<class N, bool S>
const typename SearchSpace<N, S>::IncomingHyp
        SearchSpace<N, S>::StateHyp::InitIncomingHyp = {InvalidTraceRef, InvalidScore};

/******************************************************************************/
template<class N, bool S>
SearchSpace<N, S>::SearchSpace(Network* network)
        : network_(network) {}

template<class N, bool S>
bool SearchSpace<N, S>::init(std::string& msg) {
    if (!SearchSpaceBase::init(msg))
        return false;
    network_->setLexicon(lexicon_);
    if (!network_->init())
        return false;
    if (createLattice_) {
        LatticeTraceRecorder* recorder = LatticeTraceRecorder::create(latticeType_, *stateSequences_);
        verify_(silence_);
        recorder->setSilence(silence_, silenceOutput_);
        recorder->setMergeSilence(mergeSilenceArcs_);
        recorder->setPruningThreshold(latticePruning_);
        book_ = recorder;
    }
    else {
        book_ = new FirstBestTraceRecorder(false);
    }
    return true;
}

template<class N, bool S>
void SearchSpace<N, S>::reset() {
    time_ = 0;
    activeArcs_.clear();
    activeStates_.clear();
    stateToHyp_.clear();
    stateTransitionModels_.clear();
    hmmStateHypotheses_.clear();
    currentBestScore_    = Core::Type<Score>::max;
    currentScale_        = 0;
    currentSentenceEnd_  = InvalidTraceRef;
    nActiveHmmStateHyps_ = 0;
    book_->clear();
    network_->reset();
    addInitialStateHypothesis();
    Score threshold = initialEpsPruning_ ? pruningThreshold_ : Core::Type<Score>::max;
    if (mergeEpsPaths_)
        expandEpsilonArcs<true>(threshold, initialEpsPruning_);
    else
        expandEpsilonArcs<false>(threshold, initialEpsPruning_);
}

template<class N, bool S>
SearchSpaceBase::Lattice* SearchSpace<N, S>::createLattice(OutputType type) {
    if (currentSentenceEnd_ == InvalidTraceRef)
        currentSentenceEnd_ = getSentenceEnd();
    Lattice* l = book_->createLattice(currentSentenceEnd_);
    l->setOutputType(type);
    return l;
}

template<class N, bool S>
typename SearchSpace<N, S>::TraceRef SearchSpace<N, S>::getSentenceEnd() {
    IncomingHyp best;
    best.trace = InvalidTraceRef;
    best.score = Core::Type<Score>::max;
    typedef typename StateHypotheses::const_iterator StateIter;
    for (StateIter s = activeStates_.begin(); s != activeStates_.end(); ++s) {
        const StateHyp& hyp = *s;
        if (network_->isFinal(hyp.state)) {
            const Score stateWeight = weightScale_ * network_->finalWeight(hyp.state);
            for (u32 i = 0; i < NumIncomingHyps; ++i) {
                if (isActiveHyp(hyp.incoming[i])) {
                    const Score finalScore = hyp.incoming[i].score + stateWeight;
                    const bool  update     = (finalScore < best.score);
                    if (update || createLattice_) {
                        TraceRef newTrace = book_->addTrace(best.trace, hyp.incoming[i].trace, Epsilon, 0, time_,
                                                            unscaledScore(finalScore), stateWeight, false);
                        if (update) {
                            best.score = finalScore;
                            best.trace = newTrace;
                        }
                    }
                }
            }
        }
    }
    return best.trace;
}

template<class N, bool S>
typename SearchSpace<N, S>::TraceRef SearchSpace<N, S>::getSentenceEndFallback() {
    IncomingHyp best;
    best.trace = InvalidTraceRef;
    best.score = Core::Type<Score>::max;
    typedef typename StateHypotheses::const_iterator StateIter;
    for (StateIter s = activeStates_.begin(); s != activeStates_.end(); ++s) {
        for (u32 i = 0; i < NumIncomingHyps; ++i) {
            const StateHyp& hyp = *s;
            if (isActiveHyp(hyp.incoming[i])) {
                if (hyp.incoming[i].score < best.score) {
                    best.score = hyp.incoming[i].score;
                    best.trace = book_->addTrace(best.trace, hyp.incoming[i].trace, Epsilon, 0,
                                                 time_, unscaledScore(best.score), 0, false);
                }
                else if (createLattice_) {
                    book_->addTrace(best.trace, hyp.incoming[i].trace, Epsilon, 0,
                                    time_, unscaledScore(best.score), 0, false);
                }
            }
        }
    }
    if (best.trace != InvalidTraceRef) {
        // insert the word end time, if the output has already been produced.
        // if no output has been produced yet, the last word will be deleted.
        if (!book_->hasWordEndTime(wordEnds_, best.trace))
            book_->updateTime(best.trace, time_);
    }
    return best.trace;
}

template<class N, bool S>
void SearchSpace<N, S>::getTraceback(BestPath* path) {
    // find best word end
    if (currentSentenceEnd_ == InvalidTraceRef) {
        currentSentenceEnd_ = getSentenceEnd();
        if (currentSentenceEnd_ == InvalidTraceRef) {
            Core::Application::us()->warning("no word end at sentence end found");
            Core::Application::us()->log("using sentence end fallback");
            currentSentenceEnd_ = getSentenceEndFallback();
        }
    }
    if (currentSentenceEnd_ != InvalidTraceRef)
        book_->createBestPath(wordEnds_, ignoreLastOutput_, currentSentenceEnd_, path);
}

/**
 * activates the initial state hypothesis in activeStates_ and
 * creates the root trace
 */
template<class N, bool S>
void SearchSpace<N, S>::addInitialStateHypothesis() {
    TraceRef   t               = book_->addTrace(TraceRecorder::InvalidTraceRef, TraceRecorder::InvalidTraceRef,
                                 Epsilon, 0, time_, 0.0, 0.0, true);
    StateIndex root            = network_->initialStateIndex();
    StateHyp*  rootHyp         = getStateHyp(root);
    rootHyp->incoming[0].trace = t;
    rootHyp->incoming[0].score = 0;
}

/**
 * adds a new HmmStateHyp in newHmmStateHypotheses_ or updates an existing hypothesis.
 * the hypothesis is identified using currentHmmStateHypBase_ and @c hmmState
 */
template<class N, bool S>
inline void SearchSpace<N, S>::activateOrUpdateHmmState(StateHypIndex hmmState, Score score, const TraceRef& trace) {
    const StateHypIndex hmmStateHypIndex = currentHmmStateHypBase_ + hmmState;
    verify_(hmmStateHypIndex >= currentHmmStateHypBase_);
    if (hmmStateHypIndex < currentHmmStateHypSize_) {
        // recombine hmm state hypothesis.
        HmmStateHyp& hmmStateHyp = newHmmStateHypotheses_[hmmStateHypIndex];
        if (score < hmmStateHyp.score) {
            // hypotheses is updated even if score >= currentThreshold_
            // in order to guarantee a correct score.
            hmmStateHyp.score = score;
            hmmStateHyp.trace = trace;
        }
    }
    else if (score < currentTreshold_) {
        // create new hmm state hyp
        const size_t currentSize = currentHmmStateHypSize_;
        currentHmmStateHypSize_  = hmmStateHypIndex + 1;
        if (currentHmmStateHypSize_ > newHmmStateHypotheses_.size()) {
            newHmmStateHypotheses_.resize(currentHmmStateHypSize_ + HmmStateSizeIncrement);
        }
        // fill up state hypotheses in between
        for (StateHypIndex i = currentSize; i < hmmStateHypIndex; ++i) {
            newHmmStateHypotheses_[i].trace = InvalidTraceRef;
        }
        // add new state hypothesis
        HmmStateHyp& hmmStateHyp = newHmmStateHypotheses_[hmmStateHypIndex];
        hmmStateHyp.trace        = trace;
        hmmStateHyp.score        = score;
    }
    if (score < currentBestScore_) {
        currentBestScore_ = score;
        currentTreshold_  = score + pruningThreshold_;
    }
}

/**
 * expand all hmm state hypotheses of an arc
 */
template<class N, bool S>
inline void SearchSpace<N, S>::expandArc(const ArcHyp& arcHyp, StateHypIndex hmmStateHypBegin) {
    const StateSequence& hmm      = *arcHyp.hmm;
    const u32            nStates  = hmm.nStates();
    StateIndex           hmmState = 0;
    for (StateHypIndex hmmStateIndex = hmmStateHypBegin; hmmStateIndex < arcHyp.end; ++hmmStateIndex, ++hmmState) {
        const HmmStateHyp& hmmStateHyp = hmmStateHypotheses_[hmmStateIndex];
        if (!isActiveHyp(hmmStateHyp)) {
            // pruned state hyp
            continue;
        }
        // StateIndex hmmState = hmmStateIndex - hmmStateHypBegin;
        verify_(hmmState < hmm.nStates());
        verify_(hmm.state(hmmState).transition_ < transitionModels_.size());
        const Am::StateTransitionModel* tdp   = transitionModels_[hmm.state(hmmState).transition_];
        Score                           score = 0;
        // loop transition
        Score baseScore      = hmmStateHyp.score;
        score                = baseScore + (*tdp)[Am::StateTransitionModel::loop];
        StateIndex nextState = hmmState;
        activateOrUpdateHmmState(hmmState, score, hmmStateHyp.trace);
        ++nextState;
        if (nextState < nStates) {
            // forward transition
            score = baseScore + (*tdp)[Am::StateTransitionModel::forward];
            activateOrUpdateHmmState(nextState, score, hmmStateHyp.trace);
            // skip transition
            if (UseSkips) {
                ++nextState;
                if (nextState < nStates) {
                    score = baseScore + (*tdp)[Am::StateTransitionModel::skip];
                    activateOrUpdateHmmState(nextState, score, hmmStateHyp.trace);
                }
            }
        }
    }
}

/**
 * create a new arc hypothesis in newActiveArcs_
 */
template<class N, bool S>
inline void SearchSpace<N, S>::addArcHyp(StateHypIndex hmmStateEnd, StateIndex source,
                                         StateIndex target, const StateSequence* hmm, Label output, Score score, ArcIndex arc) {
    if (currentArcHypSize_ + 1 > newActiveArcs_.size())
        newActiveArcs_.resize(currentArcHypSize_ + ArcSizeIncrement);
    ArcHyp& arcHyp = newActiveArcs_[currentArcHypSize_];
    arcHyp.end     = hmmStateEnd;
    arcHyp.state   = source;
    arcHyp.target  = target;
    arcHyp.hmm     = hmm;
    arcHyp.output  = output;
    arcHyp.arc     = arc;
    arcHyp.score   = score;
    ++currentArcHypSize_;
}

/**
 * expand a state hypothesis @c stateHyp, i.e. activate hmm states according to stateHyp.incoming,
 * and expand all outgoing arcs (if @c expandArcs == true).
 * @c arcHypIndex is used to identify the hypothesis of the first outgoing arc.
 * @c arcHypIndex is updated and will point to the first arc of the next state
 */
template<class N, bool S>
inline void SearchSpace<N, S>::expandState(
        const StateHyp& stateHyp, bool expandArcs, ArcHypIndex* arcHypIndex) {
    const bool incoming     = isActiveHyp(stateHyp.incoming[0]);
    const bool incomingSkip = UseSkips && isActiveHyp(stateHyp.incoming[1]);

    verify_(incoming || incomingSkip);
    StateHypIndex                         prevArcHypEnd = ((*arcHypIndex > 0) ? activeArcs_[*arcHypIndex - 1].end : 0);
    ArcIndex                              arcIndex      = 0;
    typedef typename Network::ArcIterator ArcIter;
    for (ArcIter aiter(network_, stateHyp.state); !aiter.done(); aiter.next(), ++arcIndex) {
        const typename Network::Arc& arc = aiter.value();
        verify_(arc.ilabel != Epsilon);
        const StateSequence* hmm     = &(*stateSequences_)[Network::stateSequenceIndex(arc)];
        const u32            nStates = hmm->nStates();
        // incoming forward transition
        currentHmmStateHypBase_ = currentHmmStateHypSize_;
        Score arcWeight         = Network::arcWeight(arc, weightScale_);
        if (incoming) {
            Score score = stateHyp.incoming[0].score + arcWeight;
            if (hmm->isInitial())
                score += entryForwardScore;
            activateOrUpdateHmmState(0, score, stateHyp.incoming[0].trace);
        }
        // incoming skip transition
        if (UseSkips && incomingSkip && nStates > 1) {
            Score score = stateHyp.incoming[1].score + arcWeight;
            if (hmm->isInitial())
                score += entrySkipScore;
            activateOrUpdateHmmState(1, score, stateHyp.incoming[1].trace);
        }
        if (expandArcs) {
            if (*arcHypIndex < activeArcs_.size()) {
                const ArcHyp& arcHyp = activeArcs_[*arcHypIndex];
                if (arcHyp.arc == arcIndex && arcHyp.state == stateHyp.state) {
                    // we have an arc hyp for this arc -> expand it
                    verify_(arcHyp.hmm == hmm && arcHyp.output == arc.olabel && arcHyp.target == arc.nextstate);
                    expandArc(arcHyp, prevArcHypEnd);
                    prevArcHypEnd = arcHyp.end;
                    ++(*arcHypIndex);
                }
            }
        }
        if (currentHmmStateHypSize_ != currentHmmStateHypBase_) {
            // new hmm state hypotheses created -> create an ArcHyp for this arc
            addArcHyp(currentHmmStateHypSize_, stateHyp.state, arc.nextstate, hmm, arc.olabel, arcWeight, arcIndex);
        }
    }
}

/**
 * interleaved expansion of incoming state hypotheses in activeStates_
 * and all hmm state hypotheses of arc hypotheses in activeArcs_.
 *
 * new state hypotheses stored in newHmmStateHypotheses_
 * new arc hypotheses stored in newActiveArcs_
 */
template<class N, bool S>
void SearchSpace<N, S>::expandStatesAndArcs() {
    size_t approxSize = (hmmStateHypotheses_.size() + 1) * HmmStateSizeIncreaseFactor;
    if (approxSize > newHmmStateHypotheses_.size())
        newHmmStateHypotheses_.resize(approxSize);

    approxSize = activeArcs_.size() * ArcSizeIncreaseFactor;
    if (approxSize > newActiveArcs_.size())
        newActiveArcs_.resize(approxSize);
    currentHmmStateHypSize_ = 0;
    currentArcHypSize_      = 0;
    currentBestScore_       = (prospectivePruning_ ? Core::Type<Score>::max : 0);
    // by setting currentBestScore_ to 0, currentThreshold_ won't be updated
    currentTreshold_ = Core::Type<Score>::max;

    std::sort(activeStates_.begin(), activeStates_.end(), SortByState());
    // by sorting the state hypotheses we can detect if an arc hypotheses
    // corresponds to an active state hypothesis or needs only expansion of
    // the arc itself.
    // stateHyp.state <-> arcHyp.state

    StateHypIterator stateIter(activeStates_);
    const u32        nArcHyps    = activeArcs_.size();
    ArcHypIndex      arcHypIndex = 0;
    const ArcHyp*    arcHyp      = 0;
    if (arcHypIndex < nArcHyps)
        arcHyp = &activeArcs_[arcHypIndex];
    while (!stateIter.done() || arcHypIndex < nArcHyps) {
        if (!stateIter.done() && !stateIter.hasIncoming()) {
            // we have a state hypotheses, but it has no incoming state hypotheses -> skip it
            stateIter.next();
        }
        else if (stateIter.done() || (arcHyp && arcHyp->state < stateIter.value().state)) {
            // no incoming state hyps for source of current arc
            // -> expand only hypotheses of current arc
            StateHypIndex prevArcHypEnd = ((arcHypIndex > 0) ? activeArcs_[arcHypIndex - 1].end : 0);
            do {
                verify_(arcHyp);
                currentHmmStateHypBase_ = currentHmmStateHypSize_;
                expandArc(*arcHyp, prevArcHypEnd);
                if (currentHmmStateHypBase_ != currentHmmStateHypSize_)
                    addArcHyp(currentHmmStateHypSize_, arcHyp->state, arcHyp->target,
                              arcHyp->hmm, arcHyp->output, arcHyp->score, arcHyp->arc);
                prevArcHypEnd = arcHyp->end;
                ++arcHypIndex;
                arcHyp = ((arcHypIndex < nArcHyps) ? &activeArcs_[arcHypIndex] : 0);
            } while (arcHyp && (stateIter.done() || arcHyp->state < stateIter.value().state));
        }
        else {
            bool expandStateOnly = (!arcHyp || (!stateIter.done() && arcHyp->state > stateIter.value().state));
            // expandStateOnly = true
            //   no active arcs for incoming state hypotheses
            //   -> create arc hypotheses for all outgoing arcs
            // expandStateOnly = false
            //   we have active arcs for current state
            //   create arc hyps for incoming state hyps and
            //   expand arc hyps of already active arcs
            verify_(!stateIter.done());
            expandState(stateIter.value(), !expandStateOnly, &arcHypIndex);
            stateIter.next();
            if (!expandStateOnly) {
                // arcHypIndex has been updated in expandState
                arcHyp = ((arcHypIndex < nArcHyps) ? &activeArcs_[arcHypIndex] : 0);
            }
        }
    }  // for stateHypIndex
}

template<class N, bool S>
inline void SearchSpace<N, S>::updateContextStateScore(StateToScoreMap& scores, StateIndex state, Score score) const {
    const StateIndex          context = network_->grammarState(state);
    StateToScoreMap::iterator i       = scores.find(context);
    if (i == scores.end())
        scores.insert(StateToScoreMap::value_type(context, score));
    else if (i->second > score)
        i->second = score;
}

/**
 * add acoustic scores for all hmm state hypotheses of all arcHyps in newActiveArcs_.
 * modifies state hypotheses in newHmmStateHypotheses_.
 * updates currentBestScore_, currentMaxScore_;
 */
template<class N, bool S>
void SearchSpace<N, S>::addAcousticScores(const Mm::FeatureScorer::Scorer& scorer) {
    currentBestScore_    = Core::Type<Score>::max;
    currentMaxScore_     = Core::Type<Score>::min;
    nActiveHmmStateHyps_ = 0;
    typedef typename ArcHypotheses::const_iterator ArcIter;
    const ArcIter                                  arcHypEnd     = newActiveArcs_.begin() + currentArcHypSize_;
    StateHypIndex                                  hmmStateIndex = 0;
    for (ArcIter arcHyp = newActiveArcs_.begin(); arcHyp != arcHypEnd; ++arcHyp) {
        const StateSequence& hmm      = *arcHyp->hmm;
        StateIndex           hmmState = 0;
        for (; hmmStateIndex < arcHyp->end; ++hmmStateIndex, ++hmmState) {
            HmmStateHyp& hmmStateHyp = newHmmStateHypotheses_[hmmStateIndex];
            if (isActiveHyp(hmmStateHyp)) {
                ++nActiveHmmStateHyps_;
                hmmStateHyp.score += scorer->score(hmm.state(hmmState).emission_);
                if (hmmStateHyp.score < currentBestScore_)
                    currentBestScore_ = hmmStateHyp.score;
                if (hmmStateHyp.score > currentMaxScore_)
                    currentMaxScore_ = hmmStateHyp.score;
            }
        }
    }
}

/**
 * prune hmm state hyps and arc hyps.
 * scales scores to [0, ...) by subtracting currentBestScore_
 * copies state hyps from newHmmStateHypotheses_ to hmmStateHypotheses_
 * copies arc hyps from newActiveArcs_ to activeArcs_
 */
template<class N, bool S>
void SearchSpace<N, S>::pruneHmmStates(Score threshold) {
    hmmStateHypotheses_.resize(currentHmmStateHypSize_);
    activeArcs_.resize(currentArcHypSize_);
    StateHypIndex stateHypOut   = 0;
    ArcHypIndex   arcHypOut     = 0;
    StateHypIndex hmmStateIndex = 0;
    nActiveHmmStateHyps_        = 0;
    typedef typename ArcHypotheses::const_iterator ArcIter;
    const ArcIter                                  arcHypEnd = newActiveArcs_.begin() + currentArcHypSize_;
    for (ArcIter arcHyp = newActiveArcs_.begin(); arcHyp != arcHypEnd; ++arcHyp) {
        verify_(arcHyp->end - hmmStateIndex > 0);
        bool          hasActiveState = false;
        StateHypIndex statesBegin    = stateHypOut;
        for (; hmmStateIndex < arcHyp->end; ++hmmStateIndex) {
            HmmStateHyp& state = newHmmStateHypotheses_[hmmStateIndex];
            verify_(stateHypOut < hmmStateHypotheses_.size());
            HmmStateHyp& newState = hmmStateHypotheses_[stateHypOut++];
            if (isActiveHyp(state) && state.score < threshold) {
                hasActiveState = true;
                newState.score = state.score - currentBestScore_;
                newState.trace = state.trace;
                ++nActiveHmmStateHyps_;
            }
            else {
                // disable state hypothesis
                newState.trace = InvalidTraceRef;
            }
        }
        if (!hasActiveState) {
            // complete arc hyp is pruned
            // reset hmm state hyp pointer
            stateHypOut = statesBegin;
        }
        else {
            // copy arc hyp to new position
            verify_(arcHypOut < activeArcs_.size());
            ArcHyp& a = activeArcs_[arcHypOut++];
            a         = *arcHyp;
            a.end     = stateHypOut;
        }
    }
    verify_(std::distance(hmmStateHypotheses_.begin() + stateHypOut, hmmStateHypotheses_.end()) >= 0);
    hmmStateHypotheses_.erase(hmmStateHypotheses_.begin() + stateHypOut, hmmStateHypotheses_.end());
    verify_(std::distance(activeArcs_.begin() + arcHypOut, activeArcs_.end()) >= 0);
    activeArcs_.erase(activeArcs_.begin() + arcHypOut, activeArcs_.end());
}

template<class N, bool S>
void SearchSpace<N, S>::pruneHmmStatesInPlace(Score threshold) {
    typedef typename ArcHypotheses::const_iterator ArcIter;
    typename ArcHypotheses::iterator               arcHypOut     = activeArcs_.begin();
    StateHypIndex                                  hmmStateIndex = 0, stateHypOut = 0;
    nActiveHmmStateHyps_ = 0;
    for (ArcIter arcHyp = activeArcs_.begin(); arcHyp != activeArcs_.end(); ++arcHyp) {
        bool          hasActiveState = false;
        StateHypIndex statesBegin    = stateHypOut;
        for (; hmmStateIndex < arcHyp->end; ++hmmStateIndex) {
            const HmmStateHyp& state    = hmmStateHypotheses_[hmmStateIndex];
            HmmStateHyp&       newState = hmmStateHypotheses_[stateHypOut++];
            if (isActiveHyp(state) && state.score < threshold) {
                hasActiveState = true;
                newState       = state;
                ++nActiveHmmStateHyps_;
            }
            else {
                // disable state hypothesis
                newState.trace = InvalidTraceRef;
            }
        }
        if (!hasActiveState) {
            // complete arc hyp is pruned
            // reset hmm state hyp pointer
            stateHypOut = statesBegin;
        }
        else {
            // copy arc hyp to new position
            verify_(arcHypOut != activeArcs_.end());
            *arcHypOut     = *arcHyp;
            arcHypOut->end = stateHypOut;
            ++arcHypOut;
        }
    }
    hmmStateHypotheses_.erase(hmmStateHypotheses_.begin() + stateHypOut, hmmStateHypotheses_.end());
    activeArcs_.erase(arcHypOut, activeArcs_.end());
}

/**
 * calculates score for histogram pruning
 */
template<class N, bool S>
Score SearchSpace<N, S>::quantileStateScore(HmmStateHypotheses::const_iterator begin, HmmStateHypotheses::const_iterator end,
                                            Score minScore, Score maxScore, u32 nHyps) {
    stateHistogram_.clear();
    stateHistogram_.setLimits(minScore, maxScore);
    u32 nActive = 0;
    for (; begin != end; ++begin) {
        if (isActiveHyp(*begin) && begin->score < maxScore) {
            stateHistogram_ += begin->score;
            ++nActive;
        }
    }
    if (nActive < nHyps)
        return maxScore;
    else
        return stateHistogram_.quantile(nHyps);
}

template<class N, bool S>
void SearchSpace<N, S>::pruneWordEnds(Score threshold) {
    typedef typename StateHypotheses::const_iterator ConstIter;
    typedef typename StateHypotheses::iterator       Iter;
    Score                                            bestScore = Core::Type<Score>::max;
    for (ConstIter s = activeStates_.begin(); s != activeStates_.end(); ++s) {
        if (wordEndHyp_[s->state]) {
            for (u32 i = 0; i < NumIncomingHyps; ++i) {
                if (s->incoming[i].score < bestScore)
                    bestScore = s->incoming[i].score;
            }
        }
    }
    threshold += bestScore;
    Iter o = activeStates_.begin();
    for (ConstIter s = activeStates_.begin(); s != activeStates_.end(); ++s) {
        bool active = true;
        if (wordEndHyp_[s->state]) {
            active = (s->incoming[0].score < threshold ||
                      (UseSkips && s->incoming[1].score < threshold));
        }
        if (active) {
            if (s != o)
                *o = *s;
            ++o;
        }
    }
    activeStates_.erase(o, activeStates_.end());
}

/**
 * create a new StateHyp (corresponding to @c stateIndex) in activeStates_ if does not already exist
 */
template<class N, bool S>
inline typename SearchSpace<N, S>::StateHyp*
        SearchSpace<N, S>::getStateHyp(StateIndex stateIndex) {
    StateHyp*                      stateHyp = 0;
    typename StateToHypMap::Cursor i        = stateToHyp_.find(
            StateToHypElement(stateIndex, 0));
    if (i == StateToHypMap::InvalidCursor) {
        stateToHyp_.insert(StateToHypElement(stateIndex, activeStates_.size()));
        activeStates_.push_back(StateHyp(stateIndex));
        stateHyp = &activeStates_.back();
    }
    else {
        stateHyp = &activeStates_[stateToHyp_[i].second];
    }
    return stateHyp;
}

/**
 * recombine state hypotheses.
 * @param incoming: existing (possibly empty) incoming state hypothesis
 * @return hypothesis updated
 */
template<class N, bool S>
inline bool SearchSpace<N, S>::activateOrUpdateIncomingHyp(IncomingHyp& incoming, TraceRef trace, Label output, const StateSequence* input,
                                                           TimeframeIndex traceTime, Score score, Score arcScore, bool wordEnd) {
    const bool update = score < incoming.score;
    if (update || createLattice_) {
        TraceRef newTrace = book_->addTrace(incoming.trace, trace, output, input, traceTime,
                                            unscaledScore(score), arcScore, wordEnd);
        if (update) {
            incoming.score = score;
            incoming.trace = newTrace;
        }
    }
    return update;
}

/**
 * creates incoming hypotheses (in activeStates_) for all hmm state hypotheses corresponding to
 * the last state of an allophone (i.e. an active arc).
 * creates new traces.
 * create only state hypotheses with a score < threshold
 */
template<class N, bool S>
void SearchSpace<N, S>::expandInterArcTransitions(Score threshold) {
    activeStates_.clear();
    stateToHyp_.clear();
    StateHypIndex                                  prevArcHypEnd = 0;
    typedef typename ArcHypotheses::const_iterator Iter;
    for (Iter iArc = activeArcs_.begin(); iArc != activeArcs_.end(); prevArcHypEnd = (iArc++)->end) {
        expandTransition(*iArc, prevArcHypEnd, threshold);
    }  // for iArc
}

template<class N, bool S>
void SearchSpace<N, S>::expandTransition(
        const ArcHyp& arcHyp, const StateHypIndex stateHypsBegin, const Score threshold) {
    const u32 nStateHyps = arcHyp.end - stateHypsBegin;
    verify_(nStateHyps > 0);
    verify_(arcHyp.hmm);
    const StateSequence& hmm          = *arcHyp.hmm;
    const u32            nStates      = hmm.nStates();
    const bool           isFinalPhone = hmm.isFinal();
    const bool           isWordEnd    = wordEnds_.isWordEnd(hmm, arcHyp.output);
    //                             last hmm state           state before that
    const bool hasLastState[] = {(nStateHyps == nStates), (nStates > 1 && nStateHyps >= (nStates - 1))};
    if (UseSkips) {
        if (!(hasLastState[0] || hasLastState[1]) ||                  // no last or penultimative state active
            (!hasLastState[0] && hasLastState[1] && isFinalPhone)) {  // final allophone with last hmm state inactive
            // -> no inter-arc transition to expand
            return;
        }
    }
    else {
        if (!hasLastState[0])
            return;
    }
    StateHyp* nextStateHyp = 0;
    for (u32 i = 0; i < NumIncomingHyps; ++i) {
        if (!hasLastState[i])
            continue;
        const StateIndex hmmState = nStates - (i + 1);
        verify_((stateHypsBegin + hmmState) < arcHyp.end);
        verify_((stateHypsBegin + hmmState) < hmmStateHypotheses_.size());

        const HmmStateHyp& hmmStateHyp = hmmStateHypotheses_[stateHypsBegin + hmmState];
        if (isActiveHyp(hmmStateHyp)) {  // state hyp has not been pruned
            nextStateHyp = expandTransHyp(hmmStateHyp, arcHyp, hmm, hmmState,
                                          threshold, isFinalPhone, isWordEnd, !i, nextStateHyp);
        }
    }
}

template<class N, bool S>
inline typename SearchSpace<N, S>::StateHyp*
        SearchSpace<N, S>::expandTransHyp(const HmmStateHyp& hmmStateHyp, const ArcHyp& arcHyp,
                                          const StateSequence& hmm, const StateIndex hmmState, const Score threshold,
                                          const bool isFinalPhone, const bool isWordEnd, const bool isLastState, StateHyp* nextStateHyp) {
    Score                           score     = 0;
    const Am::StateTransitionModel* tdp       = transitionModels_[hmm.state(hmmState).transition_];
    Score                           baseScore = hmmStateHyp.score;

    if (isLastState) {
        // forward transition only allowed at last state
        // if (isFinalPhone) {
        if (isWordEnd) {
            baseScore += (*tdp)[Am::StateTransitionModel::exit];
            // do not add forward TDP if we leave the word
            score = baseScore;
        }
        else {
            score = baseScore + (*tdp)[Am::StateTransitionModel::forward];
        }

        // should consider this state in expandEpsilonArcs, even if its score is too high here
        // no -- score can only get higher when expanding following epsilon transitions
        if (score < threshold) {
            if (!nextStateHyp)
                nextStateHyp = getStateHyp(arcHyp.target);
            if (activateOrUpdateIncomingHyp(nextStateHyp->incoming[0], hmmStateHyp.trace,
                                            arcHyp.output, arcHyp.hmm, time_, score, arcHyp.score, isWordEnd)) {
                if (outputIsWordEnd_)
                    stateTransitionModels_[arcHyp.target] = hmm.state(hmmState).transition_;
                if (wordEndPruning_)
                    setWordEndHyp(arcHyp.target, isWordEnd);
            }
        }
    }
    if (UseSkips) {
        // allow skip transitions only to the first state of the next allophone
        // last state of a word cannot be skipped (to be compatible with word conditioned tree search)
        const Score skipTdp = (*tdp)[Am::StateTransitionModel::skip];
        if ((!isFinalPhone || isLastState) && (skipTdp < Core::Type<Score>::max)) {
            // if we are at the last state, skip into second state of next hmm
            // if we are at the penultimate state, skip into first state of next hmm
            u32 incomingTarget = isLastState;
            if (isFinalPhone)
                score = baseScore;
            else
                score = baseScore + skipTdp;

            if (score < threshold) {
                if (!nextStateHyp)
                    nextStateHyp = getStateHyp(arcHyp.target);
                if (activateOrUpdateIncomingHyp(nextStateHyp->incoming[incomingTarget],
                                                hmmStateHyp.trace, arcHyp.output, arcHyp.hmm,
                                                time_, score, arcHyp.score, isFinalPhone)) {
                    if (outputIsWordEnd_)
                        stateTransitionModels_[arcHyp.target] = hmm.state(hmmState).transition_;
                    if (wordEndPruning_)
                        setWordEndHyp(arcHyp.target, isWordEnd);
                }
            }
        }
    }  // if hmmStateHyp.score
    return nextStateHyp;
}

template<class N, bool S>
void SearchSpace<N, S>::findEpsilonReachable(const StateHyp& stateHyp, Fsa::Stack<EpsilonArcHyp>* hyps) const {
    for (EpsilonArcIterator aiter(network_, stateHyp.state); !aiter.done(); aiter.next()) {
        const typename Network::EpsilonArc& arc = aiter.value();
        hyps->push(EpsilonArcHyp(arc.nextstate, arc.olabel,
                                 Network::arcWeight(arc, weightScale_),
                                 stateHyp.incoming));
    }
}

/**
 * expand epsilon arcs of states corresponding to an active state hypothesis.
 * processes all state hyps in activeStates_, adds/updates hypotheses in activeStates_.
 * creates only state hypotheses with a score < threshold
 */
template<class N, bool S>
template<bool MergePaths>
void SearchSpace<N, S>::expandEpsilonArcs(Score threshold, bool anticipatedPruning) {
    Fsa::Stack<EpsilonArcHyp> statesToExplore;
    const u32                 nActiveStates = activeStates_.size();
    StateToScoreMap*          visitedStates = 0;
    if (MergePaths)
        visitedStates = new StateToScoreMap();
    Score totalBestScore  = Core::Type<Score>::max;
    Score currentTreshold = (anticipatedPruning ? totalBestScore : threshold);
    for (StateIndex stateHypIndex = 0; stateHypIndex < nActiveStates; ++stateHypIndex) {
        const StateHyp&      stateHyp = activeStates_[stateHypIndex];
        TransitionModelIndex tdp      = 0;
        if (outputIsWordEnd_) {
            tdp = stateTransitionModels_[stateHyp.state];
        }
        const bool hasIncoming[2] = {isActiveHyp(stateHyp.incoming[0]),
                                     UseSkips && isActiveHyp(stateHyp.incoming[1])};

        if (!(hasIncoming[0] || hasIncoming[1])) {
            // no incoming hyps for this state have been activated in expandInterArcTransitions.
            // because scores can only get higher, we don't expand the epsilon arcs leaving
            // this state.
            continue;
        }
        findEpsilonReachable(stateHyp, &statesToExplore);
        while (!statesToExplore.empty()) {
            EpsilonArcHyp arc          = statesToExplore.pop();
            const bool    isWordEnd    = arc.output && outputIsWordEnd_;
            StateHyp*     nextStateHyp = 0;
            Score         bestScore    = Core::Type<Score>::max;
            for (u32 i = 0; i < NumIncomingHyps; ++i) {
                if (hasIncoming[i]) {
                    // stateHyp is not guaranteed to be valid,
                    // because we changed activeStates_ (with getStateHyp)
                    Score score = activeStates_[stateHypIndex].incoming[i].score + arc.score;
                    if (isWordEnd) {
                        score += (*transitionModels_[tdp])[Am::StateTransitionModel::exit];
                    }
                    if (score < bestScore)
                        bestScore = score;
                    bool visited = false;
                    if (MergePaths) {
                        StateToScoreMap::const_iterator v = visitedStates->find(arc.target);
                        visited                           = (v != visitedStates->end() && v->second < score);
                    }
                    if (!visited && score < currentTreshold) {
                        nextStateHyp = getStateHyp(arc.target);
                        verify_(activeStates_[stateHypIndex].incoming[i].trace != InvalidTraceRef);
                        bool addedHyp = activateOrUpdateIncomingHyp(nextStateHyp->incoming[i],
                                                                    arc.trace[i], arc.output,
                                                                    0, time_, score, arc.score, isWordEnd);
                        if (addedHyp) {
                            if (outputIsWordEnd_)
                                stateTransitionModels_[arc.target] = tdp;
                            if (wordEndPruning_)
                                setWordEndHyp(arc.target, isWordEnd);
                        }
                        if (MergePaths)
                            (*visitedStates)[arc.target] = score;
                    }
                }
            }
            if (anticipatedPruning && bestScore < totalBestScore) {
                totalBestScore  = bestScore;
                currentTreshold = totalBestScore + threshold;
            }

            if (bestScore < currentTreshold) {
                // follow epsilon path only if score of previous arc is low enough
                findEpsilonPaths(arc, visitedStates, currentTreshold, MergePaths, &statesToExplore);
            }
        }  // while !statesToExplore.empty()
    }      // for stateHypIndex
    if (MergePaths)
        delete visitedStates;
}

template<class N, bool S>
inline void SearchSpace<N, S>::findEpsilonPaths(const EpsilonArcHyp& arc, const StateToScoreMap* visitedStates,
                                                Score threshold, const bool mergePaths,
                                                Fsa::Stack<EpsilonArcHyp>* statesToExplore) const {
    for (EpsilonArcIterator aiter(network_, arc.target); !aiter.done(); aiter.next()) {
        const typename Network::EpsilonArc& nextArc = aiter.value();
        const Score                         score   = arc.score + Network::arcWeight(nextArc, weightScale_);
        bool                                visited = false;
        if (mergePaths) {
            StateToScoreMap::const_iterator v = visitedStates->find(nextArc.nextstate);
            visited                           = (v != visitedStates->end() && v->second < score);
        }
        if (!visited && score < threshold) {
            Label output = arc.output;
            if (nextArc.olabel != Epsilon) {
                output = nextArc.olabel;
            }
            statesToExplore->push(EpsilonArcHyp(nextArc.nextstate, output, score, arc.trace));
            if (nextArc.olabel != Epsilon && arc.output != Epsilon) {
                // more than one output label on an epsilon path
                // -> new bookkeeping entries required
                for (u32 i = 0; i < NumIncomingHyps; ++i) {
                    statesToExplore->back().trace[i] = book_->addTrace(InvalidTraceRef, arc.trace[i], arc.output, 0,
                                                                       time_, unscaledScore(score), arc.score, false);
                }
            }
        }
    }
}

/**
 * remove book keeping entries of pruned hypotheses
 */
template<class N, bool S>
void SearchSpace<N, S>::purgeTraces() {
    book_->purgeBegin();
    typedef typename HmmStateHypotheses::const_iterator StateIter;
    for (StateIter i = hmmStateHypotheses_.begin(); i != hmmStateHypotheses_.end(); ++i) {
        if (i->trace != InvalidTraceRef)
            book_->purgeNotify(i->trace);
    }
    book_->purgeEnd();
}

/**
 * prune states and arcs in the generated lattice
 */
template<class N, bool S>
void SearchSpace<N, S>::pruneLattice() {
    book_->pruneBegin();
    typedef typename StateHypotheses::const_iterator StateIter;
    for (StateIter s = activeStates_.begin(); s != activeStates_.end(); ++s)
        for (u32 i = 0; i < NumIncomingHyps; ++i)
            if (isActiveHyp(s->incoming[i]))
                book_->pruneNotify(s->incoming[i].trace);
    book_->pruneEnd();
}

template<class N, bool S>
void SearchSpace<N, S>::feed(const Mm::FeatureScorer::Scorer& scorer) {
    expandStatesAndArcs();
    addAcousticScores(scorer);
    statisticsCollector_->process(Statistics::AbstractCollector::beforePruning);
    Score threshold         = currentBestScore_ + pruningThreshold_;
    Score histogramTreshold = 0;
    if (!twoPassPruning_ && currentHmmStateHypSize_ > pruningLimit_) {
        histogramTreshold = quantileStateScore(newHmmStateHypotheses_.begin(), newHmmStateHypotheses_.begin() + currentHmmStateHypSize_,
                                               currentBestScore_, std::min(threshold, currentMaxScore_), pruningLimit_);
        threshold         = std::min(threshold, histogramTreshold);
    }
    if (pruningThreshold_ < Core::Type<f32>::max) {
        pruneHmmStates(threshold);
        currentScale_ += currentBestScore_;
        threshold -= currentBestScore_;
    }
    else {
        threshold = Core::Type<Score>::max;
        std::swap(hmmStateHypotheses_, newHmmStateHypotheses_);
        std::swap(activeArcs_, newActiveArcs_);
        activeArcs_.resize(currentArcHypSize_);
    }
    if (twoPassPruning_ && hmmStateHypotheses_.size() > pruningLimit_) {
        histogramTreshold = quantileStateScore(hmmStateHypotheses_.begin(), hmmStateHypotheses_.end(), 0, threshold, pruningLimit_);
        pruneHmmStatesInPlace(histogramTreshold);
    }
    statisticsCollector_->process(Statistics::AbstractCollector::afterPruning);
    ++time_;
    if (time_ % purgeInterval_ == 0) {
        purgeTraces();
    }
    expandInterArcTransitions(threshold);
    const Score epsArcThreshold = (epsilonArcPruning_ ? threshold : pruningThreshold_);
    if (mergeEpsPaths_)
        expandEpsilonArcs<true>(epsArcThreshold, !epsilonArcPruning_);
    else
        expandEpsilonArcs<false>(epsArcThreshold, !epsilonArcPruning_);
    if (wordEndPruning_) {
        pruneWordEnds(wordEndThreshold_);
    }
    /*! @todo add latticePruningInterval_ */
    if (createLattice_ && !(time_ % purgeInterval_)) {
        pruneLattice();
    }
    statisticsCollector_->process(Statistics::AbstractCollector::afterArcExpansion);
}

// delegate object creation to createSearchSpace template method
SearchSpaceBase* SearchSpaceBase::create(
        NetworkType networkType, bool allowSkips, const Core::Configuration& config) {
    SearchSpaceBase* result = 0;
    switch (networkType) {
        case NetworkTypeCompressed:
            result = SearchSpaceBase::createSearchSpace<CompressedNetwork, CompressedNetwork>(
                    allowSkips, config);
            break;
        case NetworkTypeStatic:
            result = SearchSpaceBase::createSearchSpace<StaticNetwork, StaticNetwork>(
                    allowSkips, config);
            break;
        case NetworkTypeLattice:
            result = SearchSpaceBase::createSearchSpace<StaticNetwork, LatticeNetwork>(
                    allowSkips, config);
            break;
        case NetworkTypeComposed:
            result = SearchSpaceBase::createSearchSpace<ComposedNetwork, ComposedNetwork>(
                    allowSkips, config);
            break;
        default:
            defect();
            break;
    }
    return result;
}

// hide the template from the client classes
template<class N, class C>
SearchSpaceBase* SearchSpaceBase::createSearchSpace(bool allowSkips, const Core::Configuration& c) {
    Core::Configuration config(c, "network");
    if (allowSkips)
        return new SearchSpace<N, true>(new C(config));
    else
        return new SearchSpace<N, false>(new C(config));
}

}  // namespace Wfst
}  // namespace Search
