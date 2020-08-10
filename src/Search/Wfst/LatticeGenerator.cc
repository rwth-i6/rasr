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
#include <OpenFst/ArcSort.hh>
#include <OpenFst/SymbolTable.hh>
#include <Search/Wfst/LatticeGenerator.hh>
#include <Search/Wfst/StateSequence.hh>
#include <Search/Wfst/Traceback.hh>
#include <Search/Wfst/WordEnd.hh>
#include <stack>
#include <fst/arcsort.h>
#include <fst/determinize.h>
#include <fst/dfs-visit.h>
#include <fst/encode.h>
#include <fst/project.h>
#include <fst/reverse.h>
#include <fst/rmepsilon.h>
#include <fst/shortest-path.h>

using namespace Search::Wfst;

namespace {
/**
 * checks for each state if it is reachable by an imbalanced silence arc,
 * i.e. an arc with silence output label but input label != silence.
 */
template<class A>
class SilenceLabelVisitor {
public:
    typedef A                     Arc;
    typedef typename Arc::StateId StateId;
    typedef typename Arc::Label   Label;

    SilenceLabelVisitor(Label ilabel, Label olabel,
                        std::vector<bool>* silence_states)
            : states_(silence_states), ilabel_(ilabel), olabel_(olabel) {}

    void InitVisit(const FstLib::Fst<Arc>& fst) {}
    bool InitState(StateId s, StateId root) {
        return true;
    }
    bool TreeArc(StateId s, const Arc& a) {
        SetArc(a);
        return true;
    }
    bool BackArc(StateId s, const Arc& a) {
        SetArc(a);
        return true;
    }
    bool ForwardOrCrossArc(StateId s, const Arc& a) {
        SetArc(a);
        return true;
    }
    void FinishState(StateId s, StateId parent, const Arc* parent_arc) {}
    void FinishVisit() {}

private:
    void SetArc(const Arc& a) {
        if (a.ilabel != ilabel_ && a.olabel == olabel_) {
            if (a.nextstate >= states_->size())
                states_->resize(a.nextstate + 1, false);
            states_->at(a.nextstate) = true;
        }
    }
    std::vector<bool>* states_;
    Label              ilabel_, olabel_;
};

}  // namespace

// =====================================================

LatticeTraceRecorder* LatticeTraceRecorder::create(LatticeType type, const StateSequenceList& s) {
    LatticeTraceRecorder* r = 0;
    switch (type) {
        case HmmLattice:
            r = new HmmLatticeTraceRecorder(s);
            break;
        case DetermisticHmmLattice:
            r = new DetermisticHmmLatticeTraceRecorder(s);
            break;
        case SimpleWordLattice:
            r = new SimpleWordLatticeRecorder(s);
            break;
        case SimpleNonDetWordLattice:
            r = new SimpleNonDetWordLatticeRecorder(s);
            break;
        default:
            break;
    }
    return r;
}

LatticeTraceRecorder::LatticeTraceRecorder(const StateSequenceList& hmms)
        : hmmsBegin_(&hmms.front()), lattice_(new Lattice), silence_(OpenFst::InvalidLabelId), silenceOutput_(OpenFst::InvalidLabelId), mergeSilence_(true), pruningThreshold_(Core::Type<Score>::max) {}

LatticeTraceRecorder::~LatticeTraceRecorder() {
    delete lattice_;
}

void LatticeTraceRecorder::setSilence(const StateSequence* hmm, OpenFst::Label output) {
    silence_       = getInputLabel(hmm);
    silenceOutput_ = output;
}

void LatticeTraceRecorder::clear() {
    lattice_->DeleteStates();
    unusedStates_.clear();
    lattice_->SetStart(newState());
}

LatticeTraceRecorder::StateId LatticeTraceRecorder::newState() {
    StateId s;
    if (!unusedStates_.empty()) {
        s = unusedStates_.back();
        unusedStates_.pop_back();
    }
    else {
        s = lattice_->AddState();
        enlarge(s + 1);
    }
    return s;
}

LatticeTraceRecorder::Label LatticeTraceRecorder::getInputLabel(
        const StateSequence* hmm) const {
    return hmm ? (hmm - hmmsBegin_) + 1 : OpenFst::Epsilon;
}

const StateSequence* LatticeTraceRecorder::getHmm(Label label) const {
    verify_(label);
    return hmmsBegin_ + (label - 1);
}

void LatticeTraceRecorder::purgeBegin() {
    active_.clear();
    active_.resize(lattice_->NumStates(), false);
}

void LatticeTraceRecorder::purgeEnd() {
    unusedStates_.clear();
    for (StateId s = 0; s < lattice_->NumStates(); ++s) {
        if (!active_[s]) {
            lattice_->DeleteArcs(s);
            unusedStates_.push_back(s);
        }
    }
}

void LatticeTraceRecorder::purgeNotify(TraceRef trace) {
    std::stack<TraceRef> stack;
    stack.push(trace);
    while (!stack.empty()) {
        TraceRef r = stack.top();
        stack.pop();
        if (!active_[r]) {
            active_[r] = true;
            for (ArcIterator a(*lattice_, r); !a.Done(); a.Next())
                stack.push(a.Value().nextstate);
        }
    }
}

void LatticeTraceRecorder::endLattice(TraceRef end) {
    lattice_->SetFinal(lattice_->Start(), Weight::One());
    lattice_->SetStart(end);
}

void LatticeTraceRecorder::reverseLattice() {
    Lattice tmp;
    FstLib::Reverse(*lattice_, &tmp);
    *lattice_ = tmp;
}

void LatticeTraceRecorder::finalize(TraceRef end) {
    endLattice(end);
    finalizeReverseLattice();
    trimLattice();
    reverseLattice();
    optimizeLattice();
}

void LatticeTraceRecorder::removeEpsilon(bool connect) {
    FstLib::RmEpsilon(lattice_, connect);
}

void LatticeTraceRecorder::shortestPath(BestPath* path) const {
    path->clear();
    Lattice best;
    FstLib::ShortestPath(*lattice_, &best);
    StateId s   = best.Start();
    Weight  sum = Weight::One();
    while (s != OpenFst::InvalidStateId) {
        verify_le(best.NumArcs(s), 1);
        ArcIterator aiter(best, s);
        if (aiter.Done()) {
            s = OpenFst::InvalidStateId;
        }
        else {
            sum = Times(sum, aiter.Value().weight);
            if (aiter.Value().olabel)
                path->append(aiter.Value().olabel, 0,
                             BestPath::ScoreVector(sum.AmScore(), sum.LmScore()));
            s = aiter.Value().nextstate;
        }
    }
    path->append(0, 0, BestPath::ScoreVector(sum.AmScore(), sum.LmScore()));
}

size_t LatticeTraceRecorder::memoryUsage() const {
    size_t s = 0, nArcs = 0;
    s += lattice_->NumStates() * sizeof(FstLib::VectorState<Arc>);
    for (StateIterator siter(*lattice_); !siter.Done(); siter.Next())
        nArcs += lattice_->NumArcs(siter.Value());
    s += nArcs * sizeof(Arc);
    s += active_.capacity() / 8;
    s += unusedStates_.capacity() * sizeof(StateId);
    return s;
}

// =====================================================

HmmLatticeTraceRecorder::HmmLatticeTraceRecorder(const StateSequenceList& hmms)
        : LatticeTraceRecorder(hmms),
          finished_(false) {}

void HmmLatticeTraceRecorder::clear() {
    LatticeTraceRecorder::clear();
    StateInfo& state           = states_[lattice_->Start()];
    state.score                = 0;
    state.time                 = 0;
    state.diff                 = 0;
    state.bestArc              = 0;
    hasEps_[lattice_->Start()] = false;
    finished_                  = false;
}

HmmLatticeTraceRecorder::StateId HmmLatticeTraceRecorder::getState(TimeframeIndex time) {
    StateId s = newState();
    verify_(s < states_.size());
    StateInfo& state = states_[s];
    state.score      = Core::Type<Score>::max;
    state.diff       = 0;
    state.time       = time;
    state.bestArc    = -1;
    hasEps_[s]       = false;
    return s;
}

void HmmLatticeTraceRecorder::addArc(StateId state, const Arc& arc, Score totalScore) {
    if (totalScore < states_[state].score) {
        states_[state].score   = totalScore;
        states_[state].bestArc = lattice_->NumArcs(state);
    }
    lattice_->AddArc(state, arc);
}

void HmmLatticeTraceRecorder::addArc(StateId state, const Arc& arc) {
    addArc(state, arc, states_[arc.nextstate].score + arc.weight.Combined().Value());
}

void HmmLatticeTraceRecorder::enlarge(StateId s) {
    if (s >= states_.size()) {
        states_.resize(s + 1);
        hasEps_.resize(s + 1);
    }
}

/**
 * Similar to FstLib::Connect, but keeps the StateInfo.
 * Generates an intermediate copy of lattice_
 */
void HmmLatticeTraceRecorder::trimLattice() {
    std::vector<bool>       access, coaccess;
    uint64                  props;
    FstLib::SccVisitor<Arc> scc_visitor(0, &access, &coaccess, &props);
    FstLib::DfsVisit(*lattice_, &scc_visitor);
    std::vector<StateId> newid(lattice_->NumStates(), 0);
    Lattice              newLattice;
    for (StateIterator siter(*lattice_); !siter.Done(); siter.Next()) {
        if (!access[siter.Value()] || !coaccess[siter.Value()]) {
            newid[siter.Value()] = OpenFst::InvalidStateId;
        }
        else {
            StateId ns           = newLattice.AddState();
            newid[siter.Value()] = ns;
            verify_(ns <= siter.Value());
            states_[ns] = states_[siter.Value()];
        }
    }
    for (StateIterator siter(*lattice_); !siter.Done(); siter.Next()) {
        StateId s = newid[siter.Value()];
        if (s != OpenFst::InvalidStateId) {
            newLattice.SetFinal(s, lattice_->Final(siter.Value()));
            newLattice.ReserveArcs(s, lattice_->NumArcs(siter.Value()));
            for (ArcIterator aiter(*lattice_, siter.Value()); !aiter.Done(); aiter.Next()) {
                Arc arc       = aiter.Value();
                arc.nextstate = newid[aiter.Value().nextstate];
                if (arc.nextstate != OpenFst::InvalidStateId)
                    newLattice.AddArc(s, arc);
            }
        }
    }
    newLattice.SetStart(newid[lattice_->Start()]);
    verify_ne(newid[lattice_->Start()], OpenFst::InvalidStateId);
    *lattice_ = newLattice;
}

TraceRecorder::TraceRef HmmLatticeTraceRecorder::addTrace(
        TraceRef sibling, TraceRef predecessor, OpenFst::Label output,
        const StateSequence* hmm, TimeframeIndex time,
        Score score, Score arcScore, bool wordEnd) {
    StateId state = (sibling == InvalidTraceRef ? getState(time) : sibling);
    verify_(states_[state].time == time);
    StateId     prev      = (predecessor == InvalidTraceRef ? lattice_->Start() : predecessor);
    const Score threshold = states_[state].score + pruningThreshold_;
    if (score > threshold) {
        // anticipated pruning of arcs
        verify_(sibling != InvalidTraceRef);
        return state;
    }
    // create reverse arc
    Label       input    = getInputLabel(hmm);
    const Score relScore = score - states_[prev].score;
    Arc         newArc(input, output, Weight(relScore - arcScore, arcScore), prev);
    // try to merge epsilon arcs as soon as possible
    if ((newArc.ilabel == OpenFst::Epsilon)) {
        if (lattice_->NumArcs(prev) == 1)
            mergeEpsilonArc(&newArc);
        if (newArc.ilabel == OpenFst::Epsilon)
            hasEps_[state] = true;
    }
    else if (hasEps_[newArc.nextstate] && lattice_->NumArcs(newArc.nextstate) == 1) {
        mergeEpsilonArc(&newArc);
    }
    bool add = true;
    // merge silence arcs
    if (mergeSilence_ && newArc.ilabel == silence_)
        add = !mergePredecessorArcs(state, newArc, score);
    if (add)
        addArc(state, newArc, score);
    return state;
}

void HmmLatticeTraceRecorder::mergeEpsilonArc(Arc* arc) const {
    ArcIterator aiter(*lattice_, arc->nextstate);
    verify_(!aiter.Done());
    const Arc& prevArc = aiter.Value();
    if (!(arc->olabel && prevArc.olabel)) {
        if (!arc->olabel)
            arc->olabel = prevArc.olabel;
        if (!arc->ilabel)
            arc->ilabel = prevArc.ilabel;
        arc->nextstate = prevArc.nextstate;
        arc->weight    = Times(prevArc.weight, arc->weight);
    }
}

bool HmmLatticeTraceRecorder::mergePredecessorArcs(StateId state, const Arc& arc, Score score) {
    bool merged = false;
    for (ArcIterator aiter(*lattice_, arc.nextstate); !aiter.Done(); aiter.Next()) {
        const Arc& prevArc = aiter.Value();
        if (prevArc.ilabel == arc.ilabel) {
            Arc newArc       = arc;
            newArc.nextstate = prevArc.nextstate;
            newArc.weight    = Times(prevArc.weight, arc.weight);
            if (!(prevArc.olabel && arc.olabel) || prevArc.olabel == arc.olabel) {
                if (!arc.olabel || arc.olabel == silenceOutput_)
                    newArc.olabel = prevArc.olabel;
                addArc(state, newArc);
                merged = true;
            }
        }
    }
    return merged;
}

void HmmLatticeTraceRecorder::updateTime(TraceRef t, TimeframeIndex time) {
    states_[t].time = time;
}

bool HmmLatticeTraceRecorder::hasWordEndTime(const WordEndDetector& wordEnds, TraceRef end) {
    return true;
}

void HmmLatticeTraceRecorder::createBestPath(const WordEndDetector& wordEnds, bool ignoreLast,
                                             TraceRef end, BestPath* path) {
    verify(!finished_);
    TraceRef trace = end;
    path->clear();
    path->append(OpenFst::Epsilon, states_[end].time,
                 BestPath::ScoreVector(0, states_[end].score));
    std::deque<Label>          outputs;
    std::deque<BestPath::Item> timeAndScore;
    bool                       ignoreOutput = ignoreLast;
    while (trace != lattice_->Start()) {
        ArcIterator aiter(*lattice_, trace);
        aiter.Seek(states_[trace].bestArc);
        verify(!aiter.Done());
        const Arc&           arc   = aiter.Value();
        const StateSequence* hmm   = (arc.ilabel ? getHmm(arc.ilabel) : 0);
        const Score          score = states_[trace].score + arc.weight.Combined().Value();
        if (hmm && wordEnds.isNonWord(hmm)) {
            path->append(OpenFst::Epsilon, states_[trace].time, score);
        }
        else if (hmm && wordEnds.isWordEnd(*hmm, arc.olabel))
            timeAndScore.push_back(BestPath::Item(OpenFst::Epsilon, states_[trace].time, score));
        if (arc.olabel != OpenFst::Epsilon) {
            if (!ignoreOutput)
                outputs.push_back(arc.olabel);
            else
                ignoreOutput = false;
        }
        if (!outputs.empty() && !timeAndScore.empty()) {
            BestPath::Item item = timeAndScore.front();
            /* TODO: The loop below is a bugfix, maybe not the best one.
             * There are multiple arcs (in case of character LM) for one word end.
             * Without the fix some of the arcs are not printed and the beginning of the sentence is truncated.
             * [kozielski]
             */
            while (outputs.size()) {
                item.word = outputs.front();
                path->append(item);
                outputs.pop_front();
            }
            timeAndScore.pop_front();
        }
        trace = arc.nextstate;
    }
    std::reverse(path->begin(), path->end());
}

LatticeTraceRecorder::Lattice* HmmLatticeTraceRecorder::createLattice(TraceRef end) {
    if (!finished_)
        finalize(end);
    Lattice* result = new Lattice(*lattice_);
    for (StateId s = 0; s < states_.size(); ++s)
        result->setWordBoundary(s, states_[s].time);
    return result;
}

void HmmLatticeTraceRecorder::reverseLattice() {
    LatticeTraceRecorder::reverseLattice();
    // an additional state is introduced by FstLib::reverse, shift state info
    size_t n = states_.size();
    states_.resize(n + 1);
    std::copy_backward(states_.begin(), states_.begin() + n, states_.end());
    states_[0].score = states_[1].score;
    states_[0].time  = states_[1].time;
}

void HmmLatticeTraceRecorder::finalizeReverseLattice() {
    removeEpsilon(false);
}

void HmmLatticeTraceRecorder::optimizeLattice() {
    if (mergeSilence_)
        reviseSilenceLabels();
}

void HmmLatticeTraceRecorder::reviseSilenceLabels() {
    std::vector<bool>        silenceStates(lattice_->NumStates(), false);
    SilenceLabelVisitor<Arc> visitor(silence_, silenceOutput_, &silenceStates);
    FstLib::DfsVisit(*lattice_, &visitor);
    for (StateIterator siter(*lattice_); !siter.Done(); siter.Next()) {
        if (silenceStates[siter.Value()]) {
            for (MutableArcIterator aiter(lattice_, siter.Value()); !aiter.Done(); aiter.Next()) {
                if (aiter.Value().ilabel == silence_ && aiter.Value().olabel == silenceOutput_) {
                    Arc arc    = aiter.Value();
                    arc.olabel = OpenFst::Epsilon;
                    aiter.SetValue(arc);
                }
            }
        }
    }
}

void HmmLatticeTraceRecorder::finalize(TraceRef end) {
    // final pruning
    pruneBegin();
    pruneNotify(end);
    pruneEnd();
    LatticeTraceRecorder::finalize(end);
    finished_ = true;
}

void HmmLatticeTraceRecorder::invalidateTimestamps() {
    for (std::vector<StateInfo>::iterator s = states_.begin(); s != states_.end(); ++s)
        s->time = 0;
}

void HmmLatticeTraceRecorder::pruneBegin() {
    curTraces_.clear();
    // resize and initialize active_
    purgeBegin();
}

void HmmLatticeTraceRecorder::pruneNotify(TraceRef trace) {
    verify_(trace < active_.size());
    if (!active_[trace]) {
        // collect only unique traces
        curTraces_.push_back(trace);
        active_[trace] = true;
    }
}

void HmmLatticeTraceRecorder::pruneEnd() {
    prune(curTraces_);
}

/**
 * calculate for each states the difference in score between the best path
 * through the state and the overall best path (w.r.t the given final states)
 */
void HmmLatticeTraceRecorder::calculatePruningScores(const std::vector<TraceRef>& finalStates) {
    std::vector<bool>  enqueued(active_.size(), false);
    std::vector<bool>& visited = active_;
    std::fill(visited.begin(), visited.end(), false);
    for (std::vector<TraceRef>::const_iterator i = finalStates.begin(); i != finalStates.end(); ++i) {
        visited[*i]  = true;
        enqueued[*i] = true;
    }
    std::deque<StateId> queue(finalStates.begin(), finalStates.end());
    while (!queue.empty()) {
        StateId s = queue.front();
        queue.pop_front();
        verify_(s < enqueued.size());
        enqueued[s] = false;
        verify_(visited[s]);
        for (ArcIterator aiter(*lattice_, s); !aiter.Done(); aiter.Next()) {
            const Arc& arc  = aiter.Value();
            StateId    ns   = arc.nextstate;
            Score      diff = arc.weight.Combined().Value() + states_[ns].score + states_[s].diff - states_[s].score;
            if (diff < 0.0)
                diff = 0;
            if (visited.size() <= ns) {
                visited.resize(ns + 1, false);
                enqueued.resize(ns + 1, false);
            }
            if (!visited[ns] || diff < states_[ns].diff) {
                // initialize or update score for ns
                states_[ns].diff = diff;
                visited[ns]      = true;
                if (!enqueued[ns]) {
                    enqueued[ns] = true;
                    queue.push_back(ns);
                }
            }
        }
    }
}

void HmmLatticeTraceRecorder::prune(const std::vector<TraceRef>& finalStates) {
    require(!finalStates.empty());
    calculatePruningScores(finalStates);
    std::deque<StateId> queue(finalStates.begin(), finalStates.end());
    std::vector<bool>&  visited = active_;
    std::fill(visited.begin(), visited.end(), false);
    while (!queue.empty()) {
        StateId s = queue.front();
        queue.pop_front();
        MutableArcIterator oiter(lattice_, s);
        for (ArcIterator aiter(*lattice_, s); !aiter.Done(); aiter.Next()) {
            const Arc& arc = aiter.Value();
            StateId    ns  = arc.nextstate;
            verify_(ns < visited.size());
            Score diff = arc.weight.Combined().Value() + states_[ns].score + states_[s].diff - states_[s].score;
            if (diff >= pruningThreshold_) {
                // prune arc
                if (oiter.Position() < states_[s].bestArc) {
                    // adjust pointer to best arc
                    --states_[s].bestArc;
                }
                continue;
            }
            if (oiter.Position() != aiter.Position())
                oiter.SetValue(aiter.Value());
            oiter.Next();
            if (!visited[ns]) {
                visited[ns] = true;
                queue.push_back(ns);
            }
        }
        size_t deletedArcs = lattice_->NumArcs(s) - oiter.Position();
        if (deletedArcs)
            lattice_->DeleteArcs(s, deletedArcs);
    }
    // non-reachable states will be removed by the next purge() call
}

size_t HmmLatticeTraceRecorder::memoryUsage() const {
    size_t s = LatticeTraceRecorder::memoryUsage();
    s += curTraces_.capacity() * sizeof(TraceRef);
    s += states_.capacity() * sizeof(StateInfo);
    s += hasEps_.capacity() / 8;
    return s;
}
// =======================================================

void DetermisticHmmLatticeTraceRecorder::finalizeReverseLattice() {
    removeEpsilon(true);
}

void DetermisticHmmLatticeTraceRecorder::trimLattice() {
    // lattice is already trim, see finalizeReverseLattice
}

void DetermisticHmmLatticeTraceRecorder::optimizeLattice() {
    HmmLatticeTraceRecorder::optimizeLattice();
    FstLib::EncodeMapper<Arc>          mapper(FstLib::kEncodeLabels, FstLib::ENCODE);
    FstLib::DeterminizeFstOptions<Arc> opts;
    opts.gc_limit = 0;
    Lattice result(FstLib::DecodeFst<Arc>(
            FstLib::DeterminizeFst<Arc>(FstLib::EncodeFst<Arc>(*lattice_, &mapper), opts),
            FstLib::EncodeMapper<Arc>(mapper, FstLib::DECODE)));
    *lattice_ = result;
    // time stamps are invalid after determinize
    states_.clear();
}

void DetermisticHmmLatticeTraceRecorder::createBestPath(
        const WordEndDetector& wordEnds, bool ignoreLast,
        TraceRef end, BestPath* path) {
    if (!finished_)
        finalize(end);
    shortestPath(path);
}

// =======================================================

void SimpleWordLatticeRecorder::trimLattice() {
    FstLib::Connect(lattice_);
}

void SimpleWordLatticeRecorder::optimizeLattice() {
    FstLib::CacheOptions epsOpts;
    epsOpts.gc       = false;
    epsOpts.gc_limit = 0;
    FstLib::DeterminizeFstOptions<Arc> detOpts;
    detOpts.gc       = true;
    detOpts.gc_limit = 0;
    Lattice result(FstLib::DeterminizeFst<Arc>(
            FstLib::RmEpsilonFst<Arc>(
                    FstLib::ProjectFst<Arc>(*lattice_, FstLib::PROJECT_OUTPUT), FstLib::RmEpsilonFstOptions(epsOpts)),
            detOpts));
    *lattice_ = result;
    states_.clear();
}

// =======================================================

void SimpleNonDetWordLatticeRecorder::optimizeLattice() {
    FstLib::CacheOptions epsOpts;
    epsOpts.gc       = false;
    epsOpts.gc_limit = 0;
    Lattice result(
            FstLib::RmEpsilonFst<Arc>(
                    FstLib::ProjectFst<Arc>(*lattice_, FstLib::PROJECT_OUTPUT), FstLib::RmEpsilonFstOptions(epsOpts)));
    *lattice_ = result;
    states_.clear();
}
