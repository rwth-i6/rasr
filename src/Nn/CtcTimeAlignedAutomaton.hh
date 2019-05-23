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
#ifndef _NN_CTCTIMEALIGNEDAUTOMATON_HH
#define _NN_CTCTIMEALIGNEDAUTOMATON_HH

#include <Am/AcousticModel.hh>
#include <Core/Component.hh>
#include <Core/ReferenceCounting.hh>
#include <Core/Types.hh>
#include <Fsa/Automaton.hh>
#include <Fsa/Best.hh>
#include <Fsa/Static.hh>
#include <Math/FastMatrix.hh>
#include <Speech/Alignment.hh>
#include <unordered_map>
#include <vector>
#include "BatchStateScoreIntf.hh"

namespace Nn {

/* It's already the composed automaton of the allophone state automaton
 * and the posterior scores (with the time index as an additional state).
 * Creating an automaton representing the time-state where there are
 * arcs for all the input alphabet would be theoretically possible,
 * but not practically because the alphabet is too big.
 * We don't need to go through the whole alphabet but just through
 * the allophone state automaton states.
 */
template<typename FloatT>
struct TimeAlignedAutomaton : Fsa::Automaton {
    typedef u32                                TimeIndex;
    typedef typename Math::FastMatrix<FloatT>  PosteriorMatrix;
    typedef Core::Ref<const Am::AcousticModel> ConstAcousticModelRef;

    BatchStateScoreIntf<FloatT>*                  stateScores_;  // borrowed
    ConstAcousticModelRef                         acousticModel_;
    typedef Core::Ref<const Fsa::StaticAutomaton> ConstStaticAutomatonRef;
    ConstStaticAutomatonRef                       hypothesesAllophoneStateFsa_;
    ConstStaticAutomatonRef                       hypothesesAllophoneStateFsaTransposed_;
    TimeIndex                                     nTimeFrames_;
    Fsa::ConstAlphabetRef                         allophoneAlphabet_;

    struct State : Fsa::Automaton::State {
        bool                  initialized_ : 1;
        bool                  foundForward_ : 1;
        bool                  foundBackward_ : 1;
        bool                  dead_ : 1;  // can be set in the full search
        TimeAlignedAutomaton* parent_;
        TimeIndex             timeIdx_;
        Fsa::StateId          allophoneStateId_;
        FloatT                fwdScore_, bwdScore_;

        State(TimeAlignedAutomaton* parent,
              Fsa::StateId id, TimeIndex timeIdx, Fsa::StateId allophoneStateId)
                : Fsa::Automaton::State(id),
                  initialized_(false),
                  foundForward_(false),
                  foundBackward_(false),
                  dead_(false),
                  parent_(parent),
                  timeIdx_(timeIdx),
                  allophoneStateId_(allophoneStateId),
                  fwdScore_(Fsa::LogSemiring->zero()),
                  bwdScore_(fwdScore_) {
            Core::ReferenceCounted::acquireReference();  // we statically alloc them
        }

        State(const State& o)
                : Fsa::Automaton::State(o),
                  initialized_(o.initialized_),
                  foundForward_(o.foundForward_),
                  foundBackward_(o.foundBackward_),
                  dead_(o.dead_),
                  parent_(o.parent_),
                  timeIdx_(o.timeIdx_),
                  allophoneStateId_(o.allophoneStateId_),
                  fwdScore_(o.fwdScore_),
                  bwdScore_(o.bwdScore_) {
            require_eq(o.refCount(), 1);
            Core::ReferenceCounted::acquireReference();  // we statically alloc them
        }

        void maybeInit() {
            if (initialized_)
                return;
            initialized_ = true;
            if (dead_)
                return;  // No need to explore.

            require_lt(allophoneStateId_, parent_->hypothesesAllophoneStateFsa_->size());
            auto* allophoneState = parent_->hypothesesAllophoneStateFsa_->fastState(allophoneStateId_);
            require(allophoneState);
            if (timeIdx_ == parent_->nTimeFrames_ && allophoneState->isFinal()) {
                this->addTags(Fsa::StateTagFinal);
            }

            this->setWeight(allophoneState->weight());

            if (timeIdx_ < parent_->nTimeFrames_) {
                // If we did not reached the final time index, we have exactly
                // those outgoing arcs as the underlying allophone state.
                // All of these outgoing arcs are increasing the time index by one.
                this->arcs_.reserve(allophoneState->nArcs());
                for (size_t i = 0; i < allophoneState->nArcs(); ++i) {
                    const Arc& allophoneArc = *(*allophoneState)[i];
                    verify_ne(allophoneArc.input(), Fsa::Epsilon);
                    Fsa::StateId targetStateId = parent_->getStateId(timeIdx_ + 1, allophoneArc.target(), false);
                    // If we visited this frame in a backward search before,
                    // it can happen that we know that this arc leads nowhere.
                    if (targetStateId == Fsa::InvalidStateId)
                        continue;

                    Arc& arc = *this->newArc();
                    arc.setInput(allophoneArc.input());
                    arc.setOutput(allophoneArc.output());
                    arc.setTarget(targetStateId);
                    FloatT weight = (FloatT)allophoneArc.weight();
                    weight += parent_->getAllophoneAcousticFeatureWeight(timeIdx_, allophoneArc.input());
                    arc.setWeight((Weight)weight);
                }
            }
        }

        template<int TimeIdxDiff>
        void addDirScore(FloatT sourceScore, const Arc& allophoneArc) {
            TimeIndex t = timeIdx_;
            if (TimeIdxDiff > 0) {
                require_gt(t, 0);
                t--;
            }
            require_lt(t, parent_->nTimeFrames_);
            FloatT weight = sourceScore;
            weight += (FloatT)allophoneArc.weight();
            weight += parent_->getAllophoneAcousticFeatureWeight(t, allophoneArc.input());
            dirScore<TimeIdxDiff>() = (FloatT)Fsa::LogSemiring->collect(Fsa::Weight(dirScore<TimeIdxDiff>()), Fsa::Weight(weight));
        }

        template<int TimeIdxDiff>
        FloatT& dirScore() {
            if (TimeIdxDiff > 0)
                return fwdScore_;
            else
                return bwdScore_;
        }
    };
    mutable std::vector<State>                               states_;                 // idx = our state idx
    std::vector<std::pair<Fsa::StateId, Fsa::StateId>>       statesStartEndIdxs_;     // vector idx = time idx, value = start(incl)/end(excl) idx in states_
    std::vector<std::unordered_map<TimeIndex, Fsa::StateId>> statesByAllo_;           // vector idx = allophone state, value = map time -> our state idx
    std::vector<bool>                                        statesSearchCompleted_;  // vector idx = time idx
    bool                                                     isEmpty_;

    TimeAlignedAutomaton(BatchStateScoreIntf<FloatT>* stateScores,
                         ConstAcousticModelRef        acousticModel,
                         ConstStaticAutomatonRef      hypothesesAllophoneStateFsa)
            : stateScores_(stateScores),
              acousticModel_(acousticModel),
              hypothesesAllophoneStateFsa_(hypothesesAllophoneStateFsa),
              nTimeFrames_(stateScores_->getBatchLen()),
              allophoneAlphabet_(hypothesesAllophoneStateFsa->getInputAlphabet()),
              statesStartEndIdxs_(nTimeFrames_ + 1),
              statesByAllo_(hypothesesAllophoneStateFsa->size()),
              statesSearchCompleted_(nTimeFrames_ + 1, false),
              isEmpty_(true) {
        addProperties(Fsa::PropertyCached | Fsa::PropertyStorage);
        addProperties(Fsa::PropertyAcyclic);

        require_eq(hypothesesAllophoneStateFsa_->type(), Fsa::TypeAcceptor);
        // We expect to have the weights in -log space.
        // There are several semirings (TropicalSemiring, LogSemiring in various forms)
        // which have that.
        // So, do some more generic test and hope for the best.
        require_eq((FloatT)hypothesesAllophoneStateFsa_->semiring()->one(), 0);
        require_ge((FloatT)hypothesesAllophoneStateFsa_->semiring()->zero(), Core::Type<f32>::max);

        states_.reserve(/* some heuristic */ hypothesesAllophoneStateFsa->size() * 3);
    }

    TimeAlignedAutomaton(const TimeAlignedAutomaton&) = delete;

    ~TimeAlignedAutomaton() {
        for (State& s : states_)
            require_eq(s.refCount(), 1);  // no other ref
        states_.clear();
    }

    void clear() {
        isEmpty_ = true;
        states_.clear();
        for (TimeIndex t = 0; t <= nTimeFrames_; ++t)
            statesStartEndIdxs_[t] = std::make_pair(0, 0);
        for (TimeIndex t = 0; t <= nTimeFrames_; ++t)
            statesSearchCompleted_[t] = false;
        for (auto& stateIds : statesByAllo_)
            stateIds.clear();
    }

    void initStartState() {
        Fsa::StateId allophoneInitialStateId = hypothesesAllophoneStateFsa_->initialStateId();
        Fsa::StateId initialStateId          = getStateId(0, allophoneInitialStateId, true);
        verify_eq(initialStateId, 0);  // see this->initialStateId()
        states_[initialStateId].foundForward_ = true;
        states_[initialStateId].fwdScore_     = Fsa::LogSemiring->one();
        statesSearchCompleted_[0]             = true;
        statesStartEndIdxs_[0]                = std::make_pair(0, 1);
    }

    void initFinalStates() {
        if (!hypothesesAllophoneStateFsaTransposed_) {
            auto transposedFsa = Fsa::transpose(hypothesesAllophoneStateFsa_);
            // It should be a static automaton.
            const auto* transposedFsaStatic = dynamic_cast<const Fsa::StaticAutomaton*>(transposedFsa.get());
            require(transposedFsaStatic);
            hypothesesAllophoneStateFsaTransposed_ = Core::ref(transposedFsaStatic);
            verify_ne(hypothesesAllophoneStateFsaTransposed_->initialStateId(), Fsa::InvalidStateId);
        }
        ConstStateRef transposedInitial = hypothesesAllophoneStateFsaTransposed_->getState(hypothesesAllophoneStateFsaTransposed_->initialStateId());
        verify(transposedInitial);
        // We expect that the transpose algo introduced a new initial state with only eps arcs
        // to the original final states. Otherwise, no other eps arcs should have been introduced.
        statesStartEndIdxs_[nTimeFrames_].first = states_.size();
        for (const Arc& arc : *transposedInitial) {
            verify_eq(arc.input(), Fsa::Epsilon);
            Fsa::StateId allophoneStateId = arc.target();
            Fsa::StateId ownStateId       = getStateId(nTimeFrames_, allophoneStateId, true);  // setup final state
            verify_ne(ownStateId, Fsa::InvalidStateId);
            states_[ownStateId].foundBackward_ = true;
            states_[ownStateId].bwdScore_      = Fsa::LogSemiring->one();
        }
        statesSearchCompleted_[nTimeFrames_]     = true;
        statesStartEndIdxs_[nTimeFrames_].second = states_.size();
    }

    template<int TimeIdxDiff>
    void _search(TimeIndex timeIdx, const Fsa::StaticAutomaton& fsa) {
        bool searchAlreadyCompleted = statesSearchCompleted_[timeIdx + TimeIdxDiff];
        if (!searchAlreadyCompleted)
            statesStartEndIdxs_[timeIdx + TimeIdxDiff].first = states_.size();
        auto ownStateIdStartEnd = statesStartEndIdxs_[timeIdx];
        require_le(ownStateIdStartEnd.second, states_.size());
        for (Fsa::StateId ownStateId = ownStateIdStartEnd.first; ownStateId < ownStateIdStartEnd.second; ++ownStateId) {
            if (states_[ownStateId].dead_)
                continue;
            auto* allophoneState = fsa.fastState(states_[ownStateId].allophoneStateId_);
            verify(allophoneState);
            for (const Arc& arc : *allophoneState) {
                verify_ne(arc.input(), Fsa::Epsilon);
                Fsa::StateId targetAllophoneStateId = arc.target();
                // Maybe setup new state.
                Fsa::StateId targetOwnStateId = getStateId(timeIdx + TimeIdxDiff, targetAllophoneStateId, true);
                if (!searchAlreadyCompleted)
                    verify_ne(targetOwnStateId, Fsa::InvalidStateId);
                if (targetOwnStateId != Fsa::InvalidStateId) {
                    verify_lt(targetOwnStateId, states_.size());
                    State& targetState = states_[targetOwnStateId];
                    if (TimeIdxDiff > 0)
                        targetState.foundForward_ = true;
                    else if (TimeIdxDiff < 0)
                        targetState.foundBackward_ = true;
                    targetState.template addDirScore<TimeIdxDiff>(states_[ownStateId].template dirScore<TimeIdxDiff>(), arc);
                }
            }
        }
        if (!searchAlreadyCompleted)
            statesStartEndIdxs_[timeIdx + TimeIdxDiff].second = states_.size();
        statesSearchCompleted_[timeIdx + TimeIdxDiff] = true;
    }

    template<int TimeIdxDiff>
    FloatT getMinDirScore(TimeIndex timeIdx) {
        FloatT minScore           = Core::Type<FloatT>::max;
        auto   ownStateIdStartEnd = statesStartEndIdxs_[timeIdx];
        require_le(ownStateIdStartEnd.second, states_.size());
        for (Fsa::StateId ownStateId = ownStateIdStartEnd.first; ownStateId < ownStateIdStartEnd.second; ++ownStateId) {
            State& state = states_[ownStateId];
            if (state.dead_)
                continue;
            if (minScore > state.template dirScore<TimeIdxDiff>())
                minScore = state.template dirScore<TimeIdxDiff>();
        }
        return minScore;
    }

    template<int TimeIdxDiff>
    void _prune(TimeIndex timeIdx, FloatT threshold) {
        timeIdx += TimeIdxDiff;
        threshold += getMinDirScore<TimeIdxDiff>(timeIdx);
        auto ownStateIdStartEnd = statesStartEndIdxs_[timeIdx];
        require_le(ownStateIdStartEnd.second, states_.size());
        for (Fsa::StateId ownStateId = ownStateIdStartEnd.first; ownStateId < ownStateIdStartEnd.second; ++ownStateId) {
            State& state = states_[ownStateId];
            if (state.template dirScore<TimeIdxDiff>() > threshold)
                state.dead_ = true;
        }
    }

    void fwdPrune(TimeIndex timeIdx, FloatT threshold) {
        _prune<1>(timeIdx, threshold);
    }

    void bwdPrune(TimeIndex timeIdx, FloatT threshold) {
        _prune<-1>(timeIdx, threshold);
    }

    void forwardSearch(TimeIndex timeIdx) {
        verify_lt(timeIdx, nTimeFrames_);
        _search<1>(timeIdx, *hypothesesAllophoneStateFsa_);
    }

    void backwardSearch(TimeIndex timeIdx) {
        verify_ge(timeIdx, 1);
        verify(hypothesesAllophoneStateFsaTransposed_);
        _search<-1>(timeIdx, *hypothesesAllophoneStateFsaTransposed_);
    }

    void markDeadStates(TimeIndex timeIdx) {
        // This assumes that the time frame was both visited by the
        // forward and backward search.
        // In that case, states which have not been found by both
        // are dead states.
        auto ownStateIdStartEnd = statesStartEndIdxs_[timeIdx];
        require_le(ownStateIdStartEnd.second, states_.size());
        for (Fsa::StateId ownStateId = ownStateIdStartEnd.first; ownStateId < ownStateIdStartEnd.second; ++ownStateId) {
            State& state = states_[ownStateId];
            if (!state.foundForward_ || !state.foundBackward_)
                state.dead_ = true;
        }
    }

    bool haveAnyStates(TimeIndex timeIdx) {
        auto ownStateIdStartEnd = statesStartEndIdxs_[timeIdx];
        require_le(ownStateIdStartEnd.second, states_.size());
        for (Fsa::StateId ownStateId = ownStateIdStartEnd.first; ownStateId < ownStateIdStartEnd.second; ++ownStateId) {
            State& state = states_[ownStateId];
            if (!state.dead_)
                return true;
        }
        return false;
    }

    template<int TimeIdxDiff>
    void normalizeScores(TimeIndex timeIdx) {
        // Basically we want: x_s /= sum(x)
        // In -log-space, that is: x_s -= collect(x)
        auto* collector          = Fsa::LogSemiring->getCollector();
        auto  ownStateIdStartEnd = statesStartEndIdxs_[timeIdx];
        require_le(ownStateIdStartEnd.second, states_.size());
        for (Fsa::StateId ownStateId = ownStateIdStartEnd.first; ownStateId < ownStateIdStartEnd.second; ++ownStateId) {
            State& state = states_[ownStateId];
            if (state.dead_)
                continue;
            collector->feed(Fsa::Weight((FloatT)state.template dirScore<TimeIdxDiff>()));
        }
        FloatT score_sum = collector->get();
        delete collector;
        for (Fsa::StateId ownStateId = ownStateIdStartEnd.first; ownStateId < ownStateIdStartEnd.second; ++ownStateId) {
            State& state = states_[ownStateId];
            if (state.dead_)
                continue;
            state.template dirScore<TimeIdxDiff>() -= score_sum;
        }
    }

    void fullSearch(FloatT pruneThreshold) {
        require(isEmpty_);
        require(!statesSearchCompleted_[0]);
        require(!statesSearchCompleted_[nTimeFrames_]);
        initStartState();
        initFinalStates();

        // Search forward up to the middle frame - and the same for backward.
        // This is exclusive the middle frame - but both searches will the middle frame.
        TimeIndex middleTimeFrame = nTimeFrames_ / 2;
        for (TimeIndex timeIdx = 0; timeIdx < middleTimeFrame; ++timeIdx) {
            forwardSearch(timeIdx);
            fwdPrune(timeIdx, pruneThreshold);
        }
        for (TimeIndex timeIdx = nTimeFrames_; timeIdx > middleTimeFrame + 1; --timeIdx) {
            backwardSearch(timeIdx);
            bwdPrune(timeIdx, pruneThreshold);
        }
        backwardSearch(middleTimeFrame + 1);

        // Both searches reached the middle frame, so any state which is not
        // both found by forward + backward is a dead state.
        markDeadStates(middleTimeFrame);

        if (!haveAnyStates(middleTimeFrame))
            return;
        isEmpty_ = false;

        // Complete the search of non-dead states both forward and backward.
        // Now we can mark all the remaining dead states.
        for (TimeIndex timeIdx = middleTimeFrame; timeIdx < nTimeFrames_; ++timeIdx) {
            forwardSearch(timeIdx);
            markDeadStates(timeIdx + 1);
        }
        for (TimeIndex timeIdx = middleTimeFrame; timeIdx > 0; --timeIdx) {
            backwardSearch(timeIdx);
            markDeadStates(timeIdx - 1);
        }
    }

    void fullSearchFwdOnly(FloatT pruneThreshold) {
        require_gt(nTimeFrames_, 0);
        require(isEmpty_);
        require(!statesSearchCompleted_[0]);
        require(!statesSearchCompleted_[nTimeFrames_]);
        initStartState();
        initFinalStates();

        for (TimeIndex timeIdx = 0; timeIdx < nTimeFrames_ - 1; ++timeIdx) {
            forwardSearch(timeIdx);
            prune(timeIdx + 1, pruneThreshold);
        }
        forwardSearch(nTimeFrames_ - 1);
        markDeadStates(nTimeFrames_);
        if (!haveAnyStates(nTimeFrames_))
            return;
        isEmpty_ = false;
    }

    void fullSearchAutoIncrease(FloatT minPruneThreshold, FloatT maxPruneThreshold) {
        require_le(minPruneThreshold, maxPruneThreshold);
        FloatT pruneThreshold = minPruneThreshold;
        while (true) {
            fullSearch(pruneThreshold);
            if (!isEmpty_)
                return;
            if (pruneThreshold > maxPruneThreshold)
                break;
            pruneThreshold *= 2;
            clear();
        }
    }

    FloatT _totalScoreForArc(TimeIndex timeIdx, State& srcState, State& tgtState, FloatT arcWeight, Am::AcousticModel::EmissionIndex emissionIdx) {
        FloatT weight = 0;
        weight += arcWeight;
        weight += getEmissionAcousticFeatureWeight(timeIdx, emissionIdx);
        weight += srcState.fwdScore_;
        weight += tgtState.bwdScore_;
        return weight;
    }

    void extractAlignment(Speech::Alignment& out, FloatT minProbGT = 0, FloatT gamma = 1) {
        out.clear();
        for (TimeIndex timeIdx = 0; timeIdx < nTimeFrames_; ++timeIdx) {
            auto ownStateIdStartEnd = statesStartEndIdxs_[timeIdx];
            require_le(ownStateIdStartEnd.second, states_.size());
            for (Fsa::StateId ownStateId = ownStateIdStartEnd.first; ownStateId < ownStateIdStartEnd.second; ++ownStateId) {
                State& state = states_[ownStateId];
                if (state.dead_)
                    continue;
                auto* allophoneState = hypothesesAllophoneStateFsa_->fastState(states_[ownStateId].allophoneStateId_);
                verify(allophoneState);
                for (const Arc& arc : *allophoneState) {
                    verify_ne(arc.input(), Fsa::Epsilon);
                    Fsa::StateId targetAllophoneStateId = arc.target();
                    Fsa::StateId targetOwnStateId       = getStateId(timeIdx + 1, targetAllophoneStateId, false);
                    if (targetOwnStateId != Fsa::InvalidStateId) {
                        State&                           targetState = states_[targetOwnStateId];
                        Am::AcousticModel::EmissionIndex emissionIdx = acousticModel_->emissionIndex(arc.input());
                        out.push_back(Speech::AlignmentItem(timeIdx,
                                                            arc.input(),
                                                            _totalScoreForArc(timeIdx, state, targetState, (FloatT)arc.weight(), emissionIdx)));
                    }
                }
            }
        }
        if (out.empty())
            return;  // unlikely but would happen if FSA was empty
        out.combineItems(Fsa::LogSemiring);
        require(!out.empty());
        out.sortItems(false);  // smallest -log-score means highest scores
        out.clipWeights(0, Core::Type<Mm::Weight>::max);
        out.multiplyWeights(gamma);
        out.shiftMinToZeroWeights();  // more stable expm, is equivalent with normalizeWeights()
        out.expm();                   // to std space
        out.normalizeWeights();
        out.filterWeightsGT(minProbGT);
    }

    template<typename MatrixT>
    void extractAlignmentMatrix(MatrixT& out, u32 nClasses, bool initMatrix) {
        FloatT logZero = (FloatT)Fsa::LogSemiring->zero();
        if (initMatrix) {
            out.resize(nClasses, nTimeFrames_);
            for (FloatT& v : out)
                v = logZero;
        }
        else {
            require_eq(out.nRows(), nClasses);
            require_eq(out.nColumns(), nTimeFrames_);
        }
        for (TimeIndex timeIdx = 0; timeIdx < nTimeFrames_; ++timeIdx) {
            auto ownStateIdStartEnd = statesStartEndIdxs_[timeIdx];
            require_le(ownStateIdStartEnd.second, states_.size());
            for (Fsa::StateId ownStateId = ownStateIdStartEnd.first; ownStateId < ownStateIdStartEnd.second; ++ownStateId) {
                State& state = states_[ownStateId];
                if (state.dead_)
                    continue;
                auto* allophoneState = hypothesesAllophoneStateFsa_->fastState(states_[ownStateId].allophoneStateId_);
                verify(allophoneState);
                for (const Arc& arc : *allophoneState) {
                    verify_ne(arc.input(), Fsa::Epsilon);
                    Fsa::StateId targetAllophoneStateId = arc.target();
                    Fsa::StateId targetOwnStateId       = getStateId(timeIdx + 1, targetAllophoneStateId, false);
                    if (targetOwnStateId != Fsa::InvalidStateId) {
                        State&                           targetState = states_[targetOwnStateId];
                        Am::AcousticModel::EmissionIndex emissionIdx = acousticModel_->emissionIndex(arc.input());
                        require_lt(emissionIdx, nClasses);
                        FloatT weight = _totalScoreForArc(timeIdx, state, targetState, (FloatT)arc.weight(), emissionIdx);

                        // Check for absolut limits.
                        if (weight > logZero || std::isinf(weight) || Math::isnan(weight))
                            weight = logZero;

                        if (out.at(emissionIdx, timeIdx) >= logZero)
                            out.at(emissionIdx, timeIdx) = weight;
                        else
                            out.at(emissionIdx, timeIdx) = Fsa::LogSemiring->collect(Weight(out.at(emissionIdx, timeIdx)), Weight(weight));
                    }
                }
            }
        }
    }

    Fsa::StateId getStateId(TimeIndex timeIdx, Fsa::StateId allophoneStateId, bool autoCreateNew) {
        verify_le(timeIdx, nTimeFrames_);
        {
            // Search if it exists in cache.
            require_lt(allophoneStateId, statesByAllo_.size());
            auto& stateIdsByTime = statesByAllo_[allophoneStateId];
            auto  s              = stateIdsByTime.find(timeIdx);
            if (s != stateIdsByTime.end()) {
                Fsa::StateId stateId = s->second;
                State&       state   = states_[stateId];
                if (state.dead_)
                    return Fsa::InvalidStateId;
                return stateId;
            }
        }
        // If we already have exhausted the search on this timeframe,
        // any new state would be a dead-end.
        if (statesSearchCompleted_[timeIdx])
            return Fsa::InvalidStateId;
        // Create new one.
        require(autoCreateNew);
        Fsa::StateId stateId = states_.size();
        states_.push_back(State(this, stateId, timeIdx, allophoneStateId));
        statesByAllo_[allophoneStateId][timeIdx] = stateId;
        if (stateId == 0)
            verify_eq(timeIdx, 0);
        if (timeIdx == 0)
            verify_eq(stateId, 0);  // there is only a single start state
        return stateId;
    }

    FloatT getAllophoneAcousticFeatureWeight(TimeIndex timeIdx, Fsa::LabelId inputLabel) {
        require_ge(inputLabel, 0);  // no epsilon or other special arcs
        Am::AcousticModel::EmissionIndex emissionIdx = acousticModel_->emissionIndex(inputLabel);
        return getEmissionAcousticFeatureWeight(timeIdx, emissionIdx);
    }

    FloatT getEmissionAcousticFeatureWeight(TimeIndex timeIdx, Am::AcousticModel::EmissionIndex emissionIdx) {
        return stateScores_->getStateScore(timeIdx, emissionIdx);
    }

    virtual std::string describe() const {
        return "CTC::TimeAlignedAutomaton";
    }

    virtual Fsa::Type type() const {
        return Fsa::TypeAcceptor;
    }

    virtual ConstSemiringRef semiring() const {
        return Fsa::LogSemiring;
    }

    virtual Fsa::ConstAlphabetRef getInputAlphabet() const {
        return allophoneAlphabet_;
    }

    virtual Fsa::StateId initialStateId() const {
        if (isEmpty_)
            return Fsa::InvalidStateId;
        return Fsa::StateId(0);
    }

    virtual ConstStateRef getState(Fsa::StateId s) const {
        // Normally it is allowed to return NULL if the state-id is invalid.
        // However, we expect that this is only used for valid state-ids.
        require_lt(s, states_.size());
        State& state = states_[s];
        require(!state.dead_);
        state.maybeInit();
        require_ge(state.refCount(), 1);
        return ConstStateRef(&state);
    }

    size_t totalStateCount() const {
        return states_.size();
    }

    size_t nondeadStateCount() const {
        size_t count = 0;
        for (size_t i = 0; i < states_.size(); ++i)
            if (!states_[i].dead_)
                ++count;
        return count;
    }

    size_t allophoneStateCount() const {
        verify(hypothesesAllophoneStateFsaTransposed_);
        // This should be a static automaton.
        const auto* staticFsa = dynamic_cast<const Fsa::StaticAutomaton*>(hypothesesAllophoneStateFsaTransposed_.get());
        verify(staticFsa);
        return staticFsa->size();
    }

    size_t lastFrameStateCount() const {
        size_t count              = 0;
        auto   ownStateIdStartEnd = statesStartEndIdxs_[nTimeFrames_];
        require_le(ownStateIdStartEnd.second, states_.size());
        for (Fsa::StateId ownStateId = ownStateIdStartEnd.first; ownStateId < ownStateIdStartEnd.second; ++ownStateId) {
            const State& state = states_[ownStateId];
            if (!state.dead_)
                ++count;
        }
        return count;
    }

    size_t shortestAllophonePathLen() const {
        auto best       = Fsa::best(hypothesesAllophoneStateFsa_);
        auto bestStatic = Fsa::staticCopy(best);
        return bestStatic->size();
    }

    void dumpCount(Core::Component::Message msg) {
        size_t allophoneStateC = allophoneStateCount();
        msg << "time frames: " << nTimeFrames_;
        msg << ", allophone states: " << allophoneStateC;
        msg << ", max time*allo: " << (nTimeFrames_ * allophoneStateC);
        msg << ", shortest allo path: " << shortestAllophonePathLen();
        msg << " --- time aligned total states: " << totalStateCount();
        msg << ", non-dead states: " << nondeadStateCount();
        msg << ", states in last frame: " << lastFrameStateCount();
    }
};

}  // namespace Nn

#endif  // CTCTIMEALIGNEDAUTOMATON_HH
