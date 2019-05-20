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
#ifndef _OPENFST_REPLACE_FST_HH
#define _OPENFST_REPLACE_FST_HH

#include <OpenFst/Types.hh>
#include <unordered_map>
#include <vector>
#include <fst/replace.h>
#include <fst/state-table.h>

namespace OpenFst {

/**
 * State in the CompactReplaceFst consisting of the id of the
 * original fst, the state id in the original fst, and
 * the replaced arc's target state.
 */
struct CompactReplaceStateTuple {
    typedef u8               PartId;
    PartId                   fstId_;
    typedef OpenFst::StateId StateId;
    StateId                  state_;
    StateId                  nextState_;

    CompactReplaceStateTuple(PartId id, StateId state, StateId nextState)
            : fstId_(id), state_(state), nextState_(nextState) {}

    CompactReplaceStateTuple()
            : fstId_(0), state_(FstLib::kNoStateId), nextState_(FstLib::kNoStateId) {}

    bool operator==(const CompactReplaceStateTuple& other) const {
        return fstId_ == other.fstId_ && state_ == other.state_ &&
               nextState_ == other.nextState_;
    }
};

/**
 * Hash functor for CompactReplaceStateTuples
 */
struct CompactReplaceStateTupleHash {
    size_t operator()(const CompactReplaceStateTuple& t) const {
        return ((t.fstId_ << 24) | t.state_) ^ (t.nextState_ << 5 | t.nextState_ >> 27);
    }
};

/**
 * State table for the CompactReplaceFst.
 * Adapted from FstLib::VectorHashStateTable.
 * States of the root fst are mapped using a vector, all other
 * state tuples are mapped using a hash map.
 */
class CompactReplaceStateTable {
public:
    typedef OpenFst::StateId         StateId;
    typedef CompactReplaceStateTuple StateTuple;

    CompactReplaceStateTable(size_t rootSize)
            : rootIds_(rootSize, FstLib::kNoStateId),
              tupleMap_(rootSize * 2) {
        tuples_.reserve(rootSize * 2);
    }

    ~CompactReplaceStateTable() {
        VLOG(2) << "state table:"
                << " # elements: " << tuples_.size() << " res= " << tuples_.capacity()
                << " # root tuples: " << (tuples_.size() - tupleMap_.size())
                << " # non-root tuples: " << tupleMap_.size()
                << " # buckets: " << tupleMap_.bucket_count();
    }

    StateId FindState(const StateTuple& tuple) {
        StateId id;
        if (!tuple.fstId_) {
            // root fst
            if ((id = rootIds_[tuple.state_]) == FstLib::kNoStateId) {
                id = tuples_.size();
                tuples_.push_back(tuple);
                rootIds_[tuple.state_] = id;
            }
        }
        else {
            TupleMap::const_iterator i = tupleMap_.find(tuple);
            if (i == tupleMap_.end()) {
                id = tuples_.size();
                tuples_.push_back(tuple);
                tupleMap_.insert(TupleMap::value_type(tuple, id));
            }
            else {
                id = i->second;
            }
        }
        return id;
    }

    const StateTuple& Tuple(StateId s) const {
        return tuples_[s];
    }

    StateId Size() const {
        return tuples_.size();
    }

private:
    std::vector<CompactReplaceStateTuple> tuples_;
    std::vector<OpenFst::StateId>         rootIds_;
    typedef std::unordered_map<CompactReplaceStateTuple, OpenFst::StateId,
                               CompactReplaceStateTupleHash>
            TupleMap;
    TupleMap tupleMap_;
};

/**
 * Implementation of the CompactReplaceFst.
 * See CompactReplaceFst.
 * Code is adapted from OpenFst ReplaceFstImpl.
 */
template<class A>
class CompactReplaceFstImpl : public FstLib::internal::CacheImpl<A> {
public:
    using FstLib::internal::FstImpl<A>::SetType;
    using FstLib::internal::FstImpl<A>::SetProperties;
    using FstLib::internal::FstImpl<A>::Properties;
    using FstLib::internal::FstImpl<A>::WriteHeader;
    using FstLib::internal::FstImpl<A>::SetInputSymbols;
    using FstLib::internal::FstImpl<A>::SetOutputSymbols;
    using FstLib::internal::FstImpl<A>::InputSymbols;
    using FstLib::internal::FstImpl<A>::OutputSymbols;

    //using FstLib::internal::CacheImpl<A>::AddArc;
    using FstLib::internal::CacheImpl<A>::HasArcs;
    using FstLib::internal::CacheImpl<A>::HasFinal;
    using FstLib::internal::CacheImpl<A>::HasStart;
    using FstLib::internal::CacheImpl<A>::SetArcs;
    using FstLib::internal::CacheImpl<A>::SetFinal;
    using FstLib::internal::CacheImpl<A>::SetStart;

    using FstLib::internal::CacheImpl<A>::Store;

    typedef typename A::Label     Label;
    typedef typename A::Weight    Weight;
    typedef typename A::StateId   StateId;
    typedef FstLib::CacheState<A> State;
    typedef A                     Arc;

    typedef CompactReplaceStateTuple       StateTuple;
    typedef FstLib::ExpandedFst<A>         PartFst;
    typedef StateTuple::PartId             PartId;
    typedef std::pair<s32, const PartFst*> PartDefinition;

    typedef CompactReplaceStateTable StateTable;

    // constructor for replace class implementation.
    // \param fst_tuples array of label/fst tuples, one for each non-terminal
    CompactReplaceFstImpl(const PartFst*                     root,
                          const std::vector<PartDefinition>& fstTuples,
                          const FstLib::CacheOptions&        opts)
            : FstLib::internal::CacheImpl<A>(opts),
              stateTable_(new StateTable(root->NumStates())) {
        SetType("compactreplace");

        if (fstTuples.size() > 0) {
            SetInputSymbols(fstTuples[0].second->InputSymbols());
            SetOutputSymbols(fstTuples[0].second->OutputSymbols());
        }

        const s32 maxPartId = (2 << (sizeof(PartId) * 8)) - 1;
        fstArray_.push_back(root);
        for (size_t i = 0; i < fstTuples.size(); ++i) {
            s32 nonterminal = fstTuples[i].first;
            verify(nonterminal < 0);
            verify((nonterminal * -1) <= maxPartId);
            PartId partId = GetPartId(nonterminal);
            if (fstArray_.size() <= partId)
                fstArray_.resize(partId + 1, 0);
            fstArray_[partId] = fstTuples[i].second;
        }

        std::vector<uint64> inprops;
        bool                allIlabelSorted = true;
        bool                allOlabelSorted = true;
        bool                allNonEmpty     = true;
        for (size_t i = 0; i < fstArray_.size(); ++i) {
            if (!fstArray_[i])
                continue;
            const PartFst* fst = fstArray_[i];
            if (fst->Start() == FstLib::kNoStateId)
                allNonEmpty = false;
            if (!fst->Properties(FstLib::kILabelSorted, false))
                allIlabelSorted = false;
            if (!fst->Properties(FstLib::kOLabelSorted, false))
                allOlabelSorted = false;
            inprops.push_back(fst->Properties(FstLib::kCopyProperties, false));
        }

        SetProperties(FstLib::ReplaceProperties(inprops, 0, false, false, false, false, false, allNonEmpty, allIlabelSorted, allOlabelSorted, false));
    }

    CompactReplaceFstImpl(const CompactReplaceFstImpl& impl)
            : FstLib::internal::CacheImpl<A>(impl),
              stateTable_(new StateTable(*impl.stateTable_)) {
        SetType("compactreplace");
        SetProperties(impl.Properties(), FstLib::kCopyProperties);
        SetInputSymbols(impl.InputSymbols());
        SetOutputSymbols(impl.OutputSymbols());
        fstArray_.resize(impl.fstArray_.size(), 0);
        for (size_t i = 0; i < impl.fstArray_.size(); ++i) {
            if (fstArray_[i])
                fstArray_[i] = impl.fstArray_[i]->Copy(true);
        }
    }

    ~CompactReplaceFstImpl() {
        VLOG(2) << "~ReplaceFstImpl: gc = "
                << (FstLib::internal::CacheImpl<A>::GetCacheGc() ? "true" : "false")
                << ", gc_size = " << FstLib::internal::CacheImpl<A>::GetCacheStore()->CacheSize()
                << ", gc_limit = " << FstLib::internal::CacheImpl<A>::GetCacheStore()->CacheLimit()
                << ", visited states: " << stateTable_->Size();
        delete stateTable_;
    }

    // Return or compute start state of replace fst
    StateId Start() {
        if (!HasStart()) {
            // root fst
            const PartFst* fst      = fstArray_[0];
            StateId        fstStart = fst->Start();
            if (fstStart == FstLib::kNoStateId) {
                // root Fst is empty
                return FstLib::kNoStateId;
            }
            StateId start = stateTable_->FindState(StateTuple(0, fstStart, FstLib::kNoStateId));
            SetStart(start);
            return start;
        }
        else {
            return FstLib::internal::CacheImpl<A>::Start();
        }
    }

    // return final weight of state (FstLib::kInfWeight means state is not final)
    Weight Final(StateId s) {
        if (!HasFinal(s)) {
            const StateTuple& tuple    = stateTable_->Tuple(s);
            StateId           fstState = tuple.state_;
            if (tuple.fstId_ == 0 && fstArray_[0]->Final(fstState) != Weight::Zero())
                SetFinal(s, fstArray_[0]->Final(fstState));
            else
                SetFinal(s, Weight::Zero());
        }
        return FstLib::internal::CacheImpl<A>::Final(s);
    }

    size_t NumArcs(StateId s) {
        if (HasArcs(s)) {
            return FstLib::internal::CacheImpl<A>::NumArcs(s);
        }
        else {
            StateTuple tuple = stateTable_->Tuple(s);
            if (tuple.state_ == FstLib::kNoStateId)
                return 0;
            const PartFst* fst      = fstArray_[tuple.fstId_];
            size_t         num_arcs = fst->NumArcs(tuple.state_);
            if (tuple.fstId_ && fst->Final(tuple.state_) != Weight::Zero())
                ++num_arcs;
            return num_arcs;
        }
    }

    bool IsNonTerminal(Label l) const {
        return l < 0;
    }

    PartId GetPartId(Label nonTerminal) const {
        return nonTerminal * -1;
    }

    size_t NumInputEpsilons(StateId s) {
        if (HasArcs(s)) {
            // If state cached, use the cached value.
            return FstLib::internal::CacheImpl<A>::NumInputEpsilons(s);
        }
        else if (!Properties(FstLib::kILabelSorted)) {
            // If always caching or if the number of input epsilons is too expensive
            // to compute without caching (i.e. not ilabel sorted),
            // then expand and cache state.
            Expand(s);
            return FstLib::internal::CacheImpl<A>::NumInputEpsilons(s);
        }
        else {
            // Otherwise, compute the number of input epsilons without caching.
            StateTuple tuple = stateTable_->Tuple(s);
            if (tuple.state_ == FstLib::kNoStateId)
                return 0;
            const PartFst* fst = fstArray_[tuple.fstId_];
            size_t         num = fst->NumInputEpsilons(tuple.state_);
            if (tuple.fstId_ && fst->Final(tuple.state_) != Weight::Zero())
                ++num;
            return num;
        }
    }

    size_t NumOutputEpsilons(StateId s) {
        if (HasArcs(s)) {
            // If state cached, use the cached value.
            return FstLib::internal::CacheImpl<A>::NumOutputEpsilons(s);
        }
        else if (!Properties(FstLib::kOLabelSorted)) {
            // If always caching or if the number of output epsilons is too expensive
            // to compute without caching (i.e. not olabel sorted),
            // then expand and cache state.
            Expand(s);
            return FstLib::internal::CacheImpl<A>::NumOutputEpsilons(s);
        }
        else {
            // Otherwise, compute the number of output epsilons without caching.
            StateTuple tuple = stateTable_->Tuple(s);
            if (tuple.state_ == FstLib::kNoStateId)
                return 0;
            const PartFst* fst = fstArray_[tuple.fstId_];
            size_t         num = fst->NumOutputEpsilons(tuple.state_);
            if (tuple.fstId_ && fst->Final(tuple.state_) != Weight::Zero())
                ++num;
            return num;
        }
    }

    // return the base arc iterator, if arcs have not been computed yet,
    // extend/recurse for new arcs.
    void InitArcIterator(StateId s, FstLib::ArcIteratorData<A>* data) const {
        if (!HasArcs(s))
            const_cast<CompactReplaceFstImpl<A>*>(this)->Expand(s);
        FstLib::internal::CacheImpl<A>::InitArcIterator(s, data);
    }

    // Extend current state (walk arcs one level deep)
    void Expand(StateId s) {
        StateTuple tuple = stateTable_->Tuple(s);

        // If local fst is empty
        if (tuple.state_ == FstLib::kNoStateId) {
            SetArcs(s);
            return;
        }
        FstLib::ArcIterator<PartFst> aiter(*(fstArray_[tuple.fstId_]), tuple.state_);
        Arc                          arc;

        // Create a final arc when needed
        if (ComputeFinalArc(tuple, &arc))
            this->PushArc(s, arc);

        // Expand all arcs leaving the state
        for (; !aiter.Done(); aiter.Next()) {
            if (ComputeArc(tuple, aiter.Value(), &arc))
                this->PushArc(s, arc);
        }
        SetArcs(s);
    }

    void Expand(StateId s, const StateTuple& tuple, const FstLib::ArcIteratorData<A>& data) {
        // If local fst is empty
        if (tuple.state_ == FstLib::kNoStateId) {
            SetArcs(s);
            return;
        }
        FstLib::ArcIterator<PartFst> aiter(data);
        Arc                          arc;

        // Create a final arc when needed
        if (ComputeFinalArc(tuple, &arc))
            AddArc(s, arc);

        // Expand all arcs leaving the state
        for (; !aiter.Done(); aiter.Next()) {
            if (ComputeArc(tuple, aiter.Value(), &arc))
                AddArc(s, arc);
        }
        SetArcs(s);
    }

    bool ComputeFinalArc(const StateTuple& tuple, A* arcp, uint32 flags = FstLib::kArcValueFlags) const {
        const PartFst* fst      = fstArray_[tuple.fstId_];
        StateId        fstState = tuple.state_;
        if (fstState == FstLib::kNoStateId)
            return false;
        if (tuple.fstId_ && fst->Final(fstState) != Weight::Zero()) {
            arcp->ilabel = 0;
            arcp->olabel = 0;
            if (flags & FstLib::kArcNextStateValue)
                arcp->nextstate = stateTable_->FindState(StateTuple(0, tuple.nextState_, FstLib::kNoStateId));
            if (flags & FstLib::kArcWeightValue)
                arcp->weight = fst->Final(fstState);
            return true;
        }
        else {
            return false;
        }
    }

    // Compute the arc in the replace fst corresponding to a given
    // in the underlying machine. Returns false if the underlying arc
    // corresponds to no arc in the replace.
    bool ComputeArc(const StateTuple& tuple, const A& arc, A* arcp, uint32 flags = FstLib::kArcValueFlags) const {
        if (flags == (flags & (FstLib::kArcILabelValue | FstLib::kArcWeightValue))) {
            *arcp = arc;
            return true;
        }
        if (IsNonTerminal(arc.olabel)) {
            PartId         fstId     = GetPartId(arc.olabel);
            const PartFst* fst       = fstArray_[fstId];
            StateId        partStart = fst->Start();
            if (partStart != FstLib::kNoStateId) {
                StateId nextstate = flags & FstLib::kArcNextStateValue ? stateTable_->FindState(StateTuple(fstId, partStart, arc.nextstate)) : FstLib::kNoStateId;
                *arcp             = A(arc.ilabel, 0, arc.weight, nextstate);
            }
            else {
                return false;
            }
        }
        else {
            StateId nextstate = flags & FstLib::kArcNextStateValue ? stateTable_->FindState(StateTuple(tuple.fstId_, arc.nextstate, tuple.nextState_)) : FstLib::kNoStateId;
            *arcp             = A(arc.ilabel, arc.olabel, arc.weight, nextstate);
        }
        return true;
    }

    // Returns the arc iterator flags supported by this Fst.
    uint32 ArcIteratorFlags() const {
        uint32 flags = FstLib::kArcValueFlags | FstLib::kArcNoCache;
        return flags;
    }

    const PartFst* GetFst(PartId partId) const {
        return fstArray_[partId];
    }

    StateTable* GetStateTable() const {
        return stateTable_;
    }

private:
    // state table
    // CompactReplaceStateTable *stateTable_;
    StateTable*                 stateTable_;
    std::vector<const PartFst*> fstArray_;
    std::vector<uint64_t>       fstSizeArray_;
    void                        operator=(const CompactReplaceFstImpl<A>&);  // disallow
};

/**
 * A simple and compact ReplaceFst.
 * Adapted from OpenFst ReplaceFst.
 * Recursive replacements are not supported. All nonterminals have to be
 * negative. At most 255 nonterminals are support. Nonterminals should be dense.
 */
template<class A>
class CompactReplaceFst : public FstLib::ImplToFst<CompactReplaceFstImpl<A>> {
public:
    friend class fst::ArcIterator<CompactReplaceFst<A>>;
    friend class fst::StateIterator<CompactReplaceFst<A>>;

    typedef A                        Arc;
    typedef typename A::Label        Label;
    typedef typename A::Weight       Weight;
    typedef typename A::StateId      StateId;
    typedef FstLib::CacheState<A>    State;
    typedef CompactReplaceFstImpl<A> Impl;

    typedef typename Impl::PartDefinition PartDefinition;
    typedef typename Impl::PartFst        PartFst;
    typedef typename Impl::Store          Store;

    using FstLib::ImplToFst<Impl>::Properties;

    CompactReplaceFst(const PartFst*                     root,
                      const std::vector<PartDefinition>& fstArray)
            : FstLib::ImplToFst<Impl>(std::make_shared<Impl>(root, fstArray, FstLib::CacheOptions())) {}

    CompactReplaceFst(const PartFst*                     root,
                      const std::vector<PartDefinition>& fstArray,
                      const FstLib::CacheOptions&        opts)
            : FstLib::ImplToFst<Impl>(std::make_shared<Impl>(root, fstArray, opts)) {}

    // See Fst<>::Copy() for doc.
    CompactReplaceFst(const CompactReplaceFst<A>& fst, bool safe = false)
            : FstLib::ImplToFst<Impl>(fst, safe) {}

    // Get a copy of this ReplaceFst. See Fst<>::Copy() for further doc.
    virtual CompactReplaceFst<A>* Copy(bool safe = false) const {
        return new CompactReplaceFst<A>(*this, safe);
    }

    virtual inline void InitStateIterator(FstLib::StateIteratorData<A>* data) const;

    virtual void InitArcIterator(StateId s, FstLib::ArcIteratorData<A>* data) const {
        GetImpl()->InitArcIterator(s, data);
    }

    virtual FstLib::MatcherBase<A>* InitMatcher(FstLib::MatchType match_type) const {
        if ((GetImpl()->ArcIteratorFlags() & FstLib::kArcNoCache) &&
            ((match_type == FstLib::MATCH_INPUT && Properties(FstLib::kILabelSorted, false)) ||
             (match_type == FstLib::MATCH_OUTPUT && Properties(FstLib::kOLabelSorted, false))))
            return new FstLib::SortedMatcher<CompactReplaceFst<A>>(*this, match_type);
        else {
            return 0;
        }
    }

private:
    // Makes visible to friends.
    Impl const* GetImpl() const {
        return FstLib::ImplToFst<Impl>::GetImpl();
    }

    Impl* GetMutableImpl() {
        return FstLib::ImplToFst<Impl>::GetMutableImpl();
    }

    void operator=(const CompactReplaceFst<A>& fst);  // disallow
};

}  // namespace OpenFst

namespace fst {
// Specialization for CompactReplaceFst.
template<class A>
class StateIterator<OpenFst::CompactReplaceFst<A>> : public FstLib::CacheStateIterator<OpenFst::CompactReplaceFst<A>> {
public:
    explicit StateIterator(const OpenFst::CompactReplaceFst<A>& fst)
            : CacheStateIterator<OpenFst::CompactReplaceFst<A>>(fst, const_cast<OpenFst::CompactReplaceFstImpl<A>*>(fst.GetImpl())) {}

private:
    DISALLOW_COPY_AND_ASSIGN(StateIterator);
};

template<class A>
class ArcIterator<OpenFst::CompactReplaceFst<A>> {
public:
    typedef typename OpenFst::CompactReplaceFst<A>::Arc     Arc;
    typedef typename OpenFst::CompactReplaceFst<A>::State   State;
    typedef typename OpenFst::CompactReplaceFst<A>::PartFst PartFst;
    typedef OpenFst::CompactReplaceStateTuple               Tuple;
    typedef typename Arc::StateId                           StateId;
    typedef OpenFst::CompactReplaceFstImpl<A>               Impl;

    ArcIterator(const OpenFst::CompactReplaceFst<A>& fst, StateId s)
            : fst_(fst), arcs_(0), hasFinal_(false), state_(s), pos_(0), flags_(0), dataFlags_(0), finalFlags_(0), nArcs_(0) {
        cacheData_.ref_count = 0;
        localData_.ref_count = 0;
        if (fst_.GetImpl()->HasArcs(state_)) {
            fst_.GetImpl()->InitArcIterator(state_, &cacheData_);
            nArcs_     = cacheData_.narcs;
            arcs_      = cacheData_.arcs;
            dataFlags_ = FstLib::kArcValueFlags;  // All the arc member values are valid.
        }
        else {
            tuple_ = fst_.GetImpl()->GetStateTable()->Tuple(state_);
            if (tuple_.state_ == FstLib::kNoStateId) {
                nArcs_ = 0;
            }
            else {
                const PartFst* fst = fst_.GetImpl()->GetFst(tuple_.fstId_);
                fst->InitArcIterator(tuple_.state_, &localData_);
                // 'arcs_' is a pointer to the arcs in the underlying machine.
                arcs_ = localData_.arcs;
                // Compute the final arc (but not its destination state)
                // if a final arc is required.
                finalFlags_ = FstLib::kArcValueFlags & ~FstLib::kArcNextStateValue;
                hasFinal_   = tuple_.fstId_ && fst_.GetImpl()->ComputeFinalArc(tuple_, &finalArc_, finalFlags_);
                nArcs_      = localData_.narcs;
                if (hasFinal_)
                    ++nArcs_;
                dataFlags_ = 0;
            }
        }
    }

    ~ArcIterator() {
        if (cacheData_.ref_count)
            --(*cacheData_.ref_count);
        if (localData_.ref_count)
            --(*localData_.ref_count);
    }

    bool Done() const {
        return pos_ >= nArcs_;
    }

    const Arc& Value() const {
        if (!dataFlags_) {
            CacheAllArcs();  // Expand and cache.
        }

        if (pos_ || !hasFinal_) {
            // The requested arc is not the 'final' arc.
            const A& arc = arcs_[pos_ - hasFinal_];
            if ((dataFlags_ & flags_) == (flags_ & FstLib::kArcValueFlags)) {
                // If the value flags for 'arc' match the recquired value flags
                // then return 'arc'.
                return arc;
            }
            else {
                // Otherwise, compute the corresponding arc on-the-fly.
                fst_.GetImpl()->ComputeArc(tuple_, arc, &arc_, flags_ & FstLib::kArcValueFlags);
                return arc_;
            }
        }
        else {
            // The requested arc is the 'final' arc.
            if ((finalFlags_ & flags_) != (flags_ & FstLib::kArcValueFlags)) {
                finalFlags_ = flags_ & FstLib::kArcValueFlags;
                fst_.GetImpl()->ComputeFinalArc(tuple_, &finalArc_, finalFlags_);
            }
            return finalArc_;
        }
    }

    void Next() {
        ++pos_;
    }

    size_t Position() const {
        return pos_;
    }

    void Reset() {
        pos_ = 0;
    }

    void Seek(size_t a) {
        pos_ = a;
    }

    uint32 Flags() const {
        return flags_;
    }

    void SetFlags(uint32 f, uint32 mask) {
        flags_ &= ~mask;
        flags_ |= (f & fst_.GetImpl()->ArcIteratorFlags());
        if (!(flags_ & FstLib::kArcNoCache) && dataFlags_ != FstLib::kArcValueFlags) {
            if (!fst_.GetImpl()->HasArcs(state_))
                dataFlags_ = 0;
        }
        if ((f & FstLib::kArcNoCache) && (!dataFlags_))
            Init();
    }

private:
    void Init() {
        if (flags_ & FstLib::kArcNoCache) {
            // caching is disabled
            arcs_      = localData_.arcs;
            dataFlags_ = FstLib::kArcWeightValue | FstLib::kArcILabelValue;
        }
        else {
            CacheAllArcs();
        }
    }

    void CacheAllArcs() const {
        fst_.InitArcIterator(state_, &cacheData_);
        arcs_      = cacheData_.arcs;
        dataFlags_ = FstLib::kArcValueFlags;
        hasFinal_  = false;
    }

    const OpenFst::CompactReplaceFst<A>& fst_;
    mutable ArcIteratorData<Arc>         cacheData_;
    mutable ArcIteratorData<Arc>         localData_;  // Arc iterator data in local fst
    mutable const A*                     arcs_;
    mutable A                            finalArc_, arc_;
    mutable bool                         hasFinal_;
    StateId                              state_;
    ssize_t                              pos_;    // Current position
    uint32                               flags_;  // Behavorial flags for the arc iterator
    mutable uint32                       dataFlags_, finalFlags_;
    ssize_t                              nArcs_;  // Number of arcs at state_
    Tuple                                tuple_;
    DISALLOW_COPY_AND_ASSIGN(ArcIterator);
};

}  // namespace fst

namespace OpenFst {
template<class A>
inline void CompactReplaceFst<A>::InitStateIterator(FstLib::StateIteratorData<A>* data) const {
    data->base = new FstLib::StateIterator<CompactReplaceFst<A>>(*this);
}

}  // namespace OpenFst

#endif  //  _OPENFST_REPLACE_FST_HH
