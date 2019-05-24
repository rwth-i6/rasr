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
#ifndef _SEARCH_WFST_EXPANDED_LM_FST_HH
#define _SEARCH_WFST_EXPANDED_LM_FST_HH

#include <Bliss/Fsa.hh>
#include <Lm/LanguageModel.hh>
#include <Lm/Module.hh>
#include <OpenFst/Types.hh>
#include <Search/Wfst/Types.hh>
#include <fst/cache.h>
#include <fst/matcher.h>

namespace Search {
namespace Wfst {

/**
 * Options for DynamicLmFst.
 */
struct DynamicLmFstOptions : public FstLib::CacheOptions {
    DynamicLmFstOptions(Core::Ref<Lm::LanguageModel> languageModel = Core::Ref<Lm::LanguageModel>(),
                        OutputType wordType = OutputLemma, f32 pronScale = 0.0,
                        OpenFst::Weight silWeight = OpenFst::Weight::One())
            : lm(languageModel), outputType(wordType), pronunciationScale(pronScale), silenceWeight(silWeight) {}
    DynamicLmFstOptions(const FstLib::CacheOptions& opts)
            : FstLib::CacheOptions(opts) {}
    Core::Ref<Lm::LanguageModel> lm;
    OutputType                   outputType;
    f32                          pronunciationScale;
    OpenFst::Weight              silenceWeight;
};

class DynamicLmFstScoreCache;

/**
 * Implementation of DynamicLmFst.
 * See DynamicLmFst.
 */
class DynamicLmFstImpl : public FstLib::internal::CacheImpl<OpenFst::Arc> {
public:
    typedef OpenFst::Arc                     Arc;
    typedef FstLib::internal::FstImpl<Arc>   FstImpl;
    typedef FstLib::internal::CacheImpl<Arc> CacheImpl;
    using FstImpl::InputSymbols;
    using FstImpl::OutputSymbols;
    using FstImpl::Properties;
    using FstImpl::SetInputSymbols;
    using FstImpl::SetOutputSymbols;
    using FstImpl::SetProperties;
    using FstImpl::SetType;
    using FstImpl::WriteHeader;

    using CacheImpl::HasArcs;
    using CacheImpl::HasFinal;
    using CacheImpl::HasStart;
    using CacheImpl::ReserveArcs;
    using CacheImpl::SetArcs;
    using CacheImpl::SetFinal;
    using CacheImpl::SetStart;
    using CacheImpl::Store;

    typedef Arc::Label              Label;
    typedef Arc::Weight             Weight;
    typedef Arc::StateId            StateId;
    typedef FstLib::CacheState<Arc> State;

    typedef std::vector<float> ScoreCache;

    DynamicLmFstImpl(const DynamicLmFstOptions& opts);

    DynamicLmFstImpl(const DynamicLmFstImpl& impl);

    ~DynamicLmFstImpl();

    StateId Start();

    Weight Final(StateId s);

    size_t NumArcs(StateId s) {
        return nArcs_;
    }

    size_t NumInputEpsilons(StateId s) {
        return 0;
    }

    size_t NumOutputEpsilons(StateId s) {
        return 0;
    }

    void Expand(StateId s);

    uint32 ArcIteratorFlags() const {
        uint32 flags = FstLib::kArcValueFlags | FstLib::kArcNoCache;
        return flags;
    }

    void CreateArc(StateId source, Label label, Arc* arc, bool cache = true);

    void SetLabelMapping(const std::vector<std::pair<Label, Label>>& map);

    Label GetLabel(Label label) const {
        return relabeling_.empty() ? label : relabeling_[label];
    }

    uint32 nLabels() const {
        return nLabels_;
    }

    const Lm::History& LmHistory(StateId s) const {
        verify_lt(s, state2History_.size());
        return state2History_[s];
    }

    void CacheScores(StateId s) const;

    const ScoreCache* GetScores(StateId s) const;

    Label SilenceLabel() const {
        return silenceLabel_;
    }

private:
    static const uint64 kProperties = FstLib::kAcceptor | FstLib::kIDeterministic | FstLib::kODeterministic |
                                      FstLib::kNoEpsilons | FstLib::kNoIEpsilons | FstLib::kNoOEpsilons |
                                      FstLib::kILabelSorted | FstLib::kOLabelSorted | FstLib::kWeighted |
                                      FstLib::kCyclic | FstLib::kInitialCyclic | FstLib::kNotTopSorted |
                                      FstLib::kAccessible | FstLib::kCoAccessible | FstLib::kNotString;
    static const u32 MaxScoreCaches = 100;

    StateId                                                             GetState(const Lm::History& history);
    const Lm::History&                                                  GetHistory(StateId state) const;
    void                                                                CacheArc(StateId s, const Arc& arc);
    bool                                                                GetCachedArc(StateId s, Label l, Arc* arc) const;
    const Bliss::SyntacticTokenSequence&                                SyntacticToken(Label wordLabel, f32* pronScore) const;
    Lm::CompiledBatchRequest*                                           CompileBatchRequest() const;
    Core::Ref<const Lm::LanguageModel>                                  lm_;
    Core::Ref<const Bliss::LemmaAlphabet>                               lemmas_;
    Core::Ref<const Bliss::LemmaPronunciationAlphabet>                  lemmaProns_;
    f32                                                                 wpScale_;
    u32                                                                 nLabels_, nArcs_, nCalculated_, nCached_;
    Label                                                               silence_, silenceLabel_;
    Weight                                                              silenceWeight_;
    typedef std::unordered_map<Lm::History, StateId, Lm::History::Hash> HistoryMap;
    typedef std::pair<StateId, Label>                                   ArcCacheKey;
    struct ArcCacheHash {
        size_t operator()(const ArcCacheKey& key) const {
            return key.first * 7853 + key.second;
        }
    };
    typedef std::unordered_map<ArcCacheKey, Arc, ArcCacheHash> ArcCache;
    std::vector<Label>                                         relabeling_;
    ArcCache                                                   cachedArcs_;
    HistoryMap                                                 history2State_;
    std::vector<Lm::History>                                   state2History_;
    Lm::CompiledBatchRequest*                                  batchRequest_;
    mutable DynamicLmFstScoreCache*                            scoreCache_;
    void                                                       operator=(const DynamicLmFstImpl&);  // disallow
};

/**
 * A Grammar Fst with on-demand computation of states and arcs based on an
 * underlying Lm::LanguageModel.
 *
 * In contrast to a conventional grammar WFST, the dynamic LM FST does not
 * contain backing-off (epsilon) transitions. Instead, a state representing
 * history h = ..,u,v has an arc for each word in the vocabulary w with
 * weight p(w|h) and target state h' = ..,u,v,w
 * (or shortened, depending on the LM order). The computation of p(w|h) may
 * involve backing-off weights, which is not represented by the
 * structure of the FST though.
 * It is important to note, that each state has an arc for every word in the
 * vocabulary, which makes it very expensive to iterate over all arcs.
 * The number of states is equal to the number of histories in the LM.
 *
 * Note that histories may be truncated. For example in a 3-gram LM,
 * the arc leaving state h = u,v with input w, weight p(w|h) may have the target
 * states h' = v,w or h' = w or even the empty history, depending on whether
 * the LM contains events with history v,w / w.
 */
class DynamicLmFst : public FstLib::ImplToFst<DynamicLmFstImpl> {
public:
    friend class FstLib::ArcIterator<DynamicLmFst>;
    friend class FstLib::StateIterator<DynamicLmFst>;

    typedef DynamicLmFstImpl::Arc   Arc;
    typedef Arc::Label              Label;
    typedef Arc::Weight             Weight;
    typedef Arc::StateId            StateId;
    typedef FstLib::CacheState<Arc> State;
    typedef DynamicLmFstImpl        Impl;
    typedef typename Impl::Store    Store;

    using FstLib::ImplToFst<Impl>::Properties;

    DynamicLmFst(const DynamicLmFstOptions& opts)
            : FstLib::ImplToFst<Impl>(std::make_shared<Impl>(opts)) {}

    DynamicLmFst(const DynamicLmFst& fst, bool safe = false)
            : FstLib::ImplToFst<Impl>(fst, safe) {}

    virtual DynamicLmFst* Copy(bool safe = false) const {
        return new DynamicLmFst(*this, safe);
    }

    virtual void InitStateIterator(FstLib::StateIteratorData<Arc>* data) const;

    virtual void InitArcIterator(StateId s, FstLib::ArcIteratorData<Arc>* data) const;

    virtual FstLib::MatcherBase<Arc>* InitMatcher(FstLib::MatchType match_type) const;

    void SetLabelMapping(const std::vector<std::pair<Label, Label>>& map) {
        GetMutableImpl()->SetLabelMapping(map);
    }

    const Lm::History LmHistory(StateId s) const {
        return GetImpl()->LmHistory(s);
    }

private:
    // Makes visible to friends.
    Impl const* GetImpl() const {
        return FstLib::ImplToFst<Impl>::GetImpl();
    }

    Impl* GetMutableImpl() {
        return FstLib::ImplToFst<Impl>::GetMutableImpl();
    }

    // disallow
    void operator=(const DynamicLmFst& fst);
};

}  // namespace Wfst
}  // namespace Search

namespace fst {

/**
 * Specialized StateIterator for DynamicLmFst.
 * Uses CacheStateIterator and therefore DynamicLmFst::Expand() which will
 * generate all arcs of a visited state.
 */
template<>
class StateIterator<Search::Wfst::DynamicLmFst> : public CacheStateIterator<Search::Wfst::DynamicLmFst> {
public:
    explicit StateIterator(const Search::Wfst::DynamicLmFst& fst)
            : CacheStateIterator<Search::Wfst::DynamicLmFst>(fst, const_cast<Search::Wfst::DynamicLmFstImpl*>(fst.GetImpl())) {}

private:
    DISALLOW_COPY_AND_ASSIGN(StateIterator);
};

/**
 * Specialized ArcIterator for DynamicLmFst.
 * Requires correct flags. By default, for all requested arcs the weights and
 * successor states are computed and cached.
 */
template<>
class ArcIterator<Search::Wfst::DynamicLmFst> : public ArcIteratorBase<OpenFst::Arc> {
public:
    typedef Search::Wfst::DynamicLmFst::Arc Arc;
    typedef Arc::StateId                    StateId;

    ArcIterator(const Search::Wfst::DynamicLmFst& fst, StateId s)
            : fst_(fst), state_(s), pos_(1), end_(fst.NumArcs(s)), flags_(FstLib::kArcValueFlags), scores_(0) {
        if ((haveCachedArcs_ = fst_.GetImpl()->HasArcs(s))) {
            fst_.GetImpl()->InitArcIterator(s, &cachedArcs_);
            end_ = cachedArcs_.narcs;
        }
        else {
            scores_ = fst_.GetImpl()->GetScores(s);
        }
    }

    ~ArcIterator() {}

    bool Done() const {
        return pos_ > end_;
    }

    const Arc& Value() const {
        const uint32 scoreOnlyMask  = (FstLib::kArcNoCache | FstLib::kArcWeightValue |
                                      FstLib::kArcOLabelValue | FstLib::kArcNextStateValue);
        const uint32 scoreOnlyFlags = (FstLib::kArcNoCache | FstLib::kArcWeightValue);
        if (haveCachedArcs_) {
            return cachedArcs_.arcs[pos_ - 1];
        }
        else if ((flags_ & scoreOnlyMask) == scoreOnlyFlags) {
            // compute score only. most probably a weight look-ahead, compute scores for all arcs
            if (!scores_) {
                fst_.GetImpl()->CacheScores(state_);
                scores_ = fst_.GetImpl()->GetScores(state_);
            }
            arc_.weight = Arc::Weight(scores_->at(pos_));
            arc_.ilabel = pos_;
        }
        else if (flags_ & (FstLib::kArcWeightValue | FstLib::kArcNextStateValue)) {
            const_cast<Search::Wfst::DynamicLmFstImpl*>(fst_.GetImpl())->CreateArc(state_, pos_, &arc_, !(flags_ & FstLib::kArcNoCache));
            verify_eq(pos_, arc_.ilabel);
        }
        else {
            arc_.ilabel = pos_;
            if (flags_ & FstLib::kArcOLabelValue)
                arc_.olabel = fst_.GetImpl()->GetLabel(pos_);
        }
        return arc_;
    }

    void Next() {
        ++pos_;
    }

    size_t Position() const {
        return pos_ - 1;
    }

    void Reset() {
        pos_ = 1;
    }

    void Seek(size_t pos) {
        pos_ = pos + 1;
    }

    uint32 Flags() const {
        return flags_;
    }

    void SetFlags(uint32 f, uint32 mask) {
        // Update the flags taking into account what flags are supported
        // by the Fst.
        flags_ &= ~mask;
        flags_ |= (f & fst_.GetImpl()->ArcIteratorFlags());
    }

private:
    virtual bool Done_() const {
        return Done();
    }
    virtual const Arc& Value_() const {
        return Value();
    }
    virtual void Next_() {
        Next();
    }
    virtual size_t Position_() const {
        return Position();
    }
    virtual void Reset_() {
        Reset();
    }
    virtual void Seek_(size_t a) {
        Seek(a);
    }
    virtual uint32 Flags_() const {
        return Flags();
    }
    virtual void SetFlags_(uint32 flags, uint32 mask) {
        SetFlags(flags, mask);
    }

    mutable Arc                                               arc_;
    const Search::Wfst::DynamicLmFst&                         fst_;
    StateId                                                   state_;
    Search::Wfst::DynamicLmFst::Label                         pos_, end_;
    uint32                                                    flags_;
    FstLib::ArcIteratorData<Arc>                              cachedArcs_;
    bool                                                      haveCachedArcs_;
    mutable const Search::Wfst::DynamicLmFstImpl::ScoreCache* scores_;

    DISALLOW_COPY_AND_ASSIGN(ArcIterator);
};

}  // namespace fst

namespace Search {
namespace Wfst {

/**
 * Specialized Matcher for DynamicLmFst.
 * Every state in a DynamicLmFst has an arc for every input symbol (or it
 * will compute such an arc).
 */
class DynamicLmFstMatcher : public FstLib::MatcherBase<DynamicLmFst::Arc> {
public:
    typedef DynamicLmFst      FST;
    typedef DynamicLmFst::Arc Arc;
    typedef Arc::StateId      StateId;
    typedef Arc::Label        Label;
    typedef Arc::Weight       Weight;

    DynamicLmFstMatcher(const DynamicLmFst& fst, FstLib::MatchType matchType)
            : fst_(fst.Copy()), arcRead_(true), aiter_(0), mtype_(matchType), state_(FstLib::kNoStateId), loop_(FstLib::kNoLabel, 0, Weight::One(), FstLib::kNoStateId) {}

    DynamicLmFstMatcher(const DynamicLmFstMatcher& m, bool safe)
            : fst_(m.fst_->Copy(safe)), arcRead_(true), aiter_(0), mtype_(m.mtype_), state_(FstLib::kNoStateId), loop_(m.loop_) {}

    virtual ~DynamicLmFstMatcher() {
        delete aiter_;
        delete fst_;
    }
    DynamicLmFstMatcher* Copy(bool safe = false) const {
        return new DynamicLmFstMatcher(*this, safe);
    }
    FstLib::MatchType Type(bool test) const {
        return mtype_;
    }
    void SetState(StateId s) {
        if (state_ != s) {
            delete aiter_;
            aiter_          = new FstLib::ArcIterator<DynamicLmFst>(*fst_, s);
            loop_.nextstate = state_ = s;
        }
    }
    bool Find(Label label) {
        isEpsilon_ = label == OpenFst::Epsilon;
        if (label <= 0 || label >= fst_->InputSymbols()->NumSymbols()) {
            return isEpsilon_;
        }
        else {
            // all labels can be matched
            aiter_->Seek(label - 1);
            arcRead_ = false;
            return true;
        }
    }
    bool Done() const {
        if (isEpsilon_)
            return false;
        else
            return arcRead_;
    }
    const Arc& Value() const {
        if (isEpsilon_)
            return loop_;
        else
            return aiter_->Value();
    }
    void Next() {
        isEpsilon_ = false;
        arcRead_   = true;
    }
    const DynamicLmFst& GetFst() const {
        return *fst_;
    }
    uint64 Properties(uint64 props) const {
        return props;
    }

private:
    virtual void SetState_(StateId s) {
        SetState(s);
    }
    virtual bool Find_(Label label) {
        return Find(label);
    }
    virtual bool Done_() const {
        return Done();
    }
    virtual const Arc& Value_() const {
        return Value();
    }
    virtual void Next_() {
        Next();
    }

    DynamicLmFst*                      fst_;
    bool                               arcRead_;
    FstLib::ArcIterator<DynamicLmFst>* aiter_;
    FstLib::MatchType                  mtype_;
    StateId                            state_;
    Arc                                loop_;
    bool                               isEpsilon_;
};

inline FstLib::MatcherBase<DynamicLmFst::Arc>* DynamicLmFst::InitMatcher(FstLib::MatchType match_type) const {
    return new DynamicLmFstMatcher(*this, match_type);
}

inline void DynamicLmFst::InitStateIterator(FstLib::StateIteratorData<Arc>* data) const {
    data->base = new FstLib::StateIterator<DynamicLmFst>(*this);
}

inline void DynamicLmFst::InitArcIterator(StateId s, FstLib::ArcIteratorData<Arc>* data) const {
    data->base = new FstLib::ArcIterator<DynamicLmFst>(*this, s);
}

}  // namespace Wfst
}  // namespace Search

#endif  // _SEARCH_WFST_EXPANDED_LM_FST_HH
