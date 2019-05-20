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
#ifndef _OPENFST_FSA_MAPPER_HH
#define _OPENFST_FSA_MAPPER_HH

#include <memory>

#include <Fsa/Automaton.hh>
#include <Fsa/Types.hh>
#include <Fsa/tBasic.hh>
#include <OpenFst/SymbolTable.hh>
#include <OpenFst/Types.hh>
#include <OpenFst/Weight.hh>
#include <fst/test-properties.h>

namespace OpenFst {

template<class F>
class FsaMapperStateIteratorBase;
template<class F>
class FsaMapperArcIteratorBase;

/**
 * implementation class for FsaMapperAutomaton
 */
template<class FsaAutomaton, class FstArc, template<class, class> class WeightConverter>
class FsaMapperAutomatonImpl : public FstLib::internal::FstImpl<FstArc> {
    typedef FstLib::internal::FstImpl<FstArc>                             Predecessor;
    typedef FsaMapperAutomatonImpl<FsaAutomaton, FstArc, WeightConverter> Self;

public:
    typedef Core::Ref<const FsaAutomaton>                          ConstAutomatonRef;
    typedef FstArc                                                 Arc;
    typedef typename FstArc::Weight                                Weight;
    typedef typename FstArc::StateId                               StateId;
    typedef WeightConverter<typename FsaAutomaton::Weight, Weight> WeightConverterType;
    using Predecessor::Properties;
    using Predecessor::SetInputSymbols;
    using Predecessor::SetOutputSymbols;
    using Predecessor::SetProperties;
    using Predecessor::SetType;

    FsaMapperAutomatonImpl(ConstAutomatonRef fsa) {
        // state ids have to be dense integers
        fsa_ = ConstAutomatonRef(new typename Ftl::NormalizeAutomaton<FsaAutomaton>(fsa));
        fsa_->initialStateId();  // initialize mapping
        SetType("fsa-mapper");
        if (fsa_->inputAlphabet())
            SetInputSymbols(convertAlphabet(fsa_->inputAlphabet(), "input"));
        if (fsa_->outputAlphabet())
            SetOutputSymbols(convertAlphabet(fsa_->outputAlphabet(), "output"));
        translateProperties();
    }
    FsaMapperAutomatonImpl(const Self& f)
            : fsa_(f.fsa_) {
        SetType(f.Type());
        SetInputSymbols(f.InputSymbols());
        SetOutputSymbols(f.OutputSymbols());
        SetProperties(f.Properties());
    }
    virtual ~FsaMapperAutomatonImpl() {}
    virtual StateId Start() const {
        return fsa_->initialStateId();
    }
    virtual Weight Final(StateId id) const {
        typename FsaAutomaton::ConstStateRef state = fsa_->getState(id);
        if (!state->isFinal())
            return Weight::Zero();
        else
            return weightConverter_(state->weight());
    }
    virtual size_t NumArcs(StateId id) const {
        typename FsaAutomaton::ConstStateRef state = fsa_->getState(id);
        return state->nArcs();
    }
    virtual size_t NumInputEpsilons(StateId id) const {
        typename FsaAutomaton::ConstStateRef state = fsa_->getState(id);
        size_t                               n     = 0;
        for (typename FsaAutomaton::State::const_iterator arc = state->begin(); arc != state->end(); ++arc)
            if (arc->input() == Fsa::Epsilon)
                ++n;
        return n;
    }
    virtual size_t NumOutputEpsilons(StateId id) const {
        typename FsaAutomaton::ConstStateRef state = fsa_->getState(id);
        size_t                               n     = 0;
        for (typename FsaAutomaton::State::const_iterator arc = state->begin(); arc != state->end(); ++arc)
            if (arc->output() == Fsa::Epsilon)
                ++n;
        return n;
    }
    typename FstArc::Weight ConvertWeight(const typename FsaAutomaton::Arc::Weight& w) const {
        return weightConverter_(w);
    }
    ConstAutomatonRef getFsa() const {
        return fsa_;
    }

private:
    void translateProperties() {
        SetProperties(Properties() & ~FstLib::kMutable);
        SetProperties(Properties() & ~FstLib::kExpanded);
        if (fsa_->type() == Fsa::TypeAcceptor) {
            SetProperties(Properties() | FstLib::kAcceptor);
            SetProperties(Properties() & ~FstLib::kNotAcceptor);
        }
        else {
            SetProperties(Properties() | FstLib::kNotAcceptor);
            SetProperties(Properties() & ~FstLib::kAcceptor);
        }
        if (fsa_->hasProperty(Fsa::PropertyAcyclic)) {
            SetProperties(Properties() | FstLib::kAcyclic);
            SetProperties(Properties() & ~FstLib::kCyclic);
        }
        if (fsa_->hasProperty(Fsa::PropertySortedByInput)) {
            SetProperties(Properties() | FstLib::kILabelSorted);
            SetProperties(Properties() & ~FstLib::kNotILabelSorted);
        }
        if (fsa_->hasProperty(Fsa::PropertySortedByOutput)) {
            SetProperties(Properties() | FstLib::kOLabelSorted);
            SetProperties(Properties() & ~FstLib::kNotOLabelSorted);
        }
    }
    ConstAutomatonRef   fsa_;
    WeightConverterType weightConverter_;
};

/**
 * OpenFst's fst-Interface for classes of the Fsa Toolkit.
 * supports delayed transducers.
 * currently without any caching.
 */
template<class FsaAutomaton, class FstArc,
         template<class, class> class WeightConverter = ImplicitWeightConverter>
class FsaMapperAutomaton : public FstLib::Fst<FstArc> {
    typedef FstLib::internal::FstImpl<FstArc>                             Predecessor;
    typedef FsaMapperAutomatonImpl<FsaAutomaton, FstArc, WeightConverter> Impl;
    typedef FsaMapperAutomaton<FsaAutomaton, FstArc, WeightConverter>     Self;
    typedef typename FstArc::Weight                                       Weight;

public:
    typedef FsaAutomaton                  FsaType;
    typedef Core::Ref<const FsaAutomaton> ConstAutomatonRef;

    FsaMapperAutomaton(ConstAutomatonRef fsa)
            : impl_(new Impl(fsa)) {}
    FsaMapperAutomaton(const Self& fst, bool reset) {
        if (reset) {
            impl_.reset(new Impl(*fst.impl_));
        }
        else {
            impl_ = fst.impl_;
        }
    }
    virtual ~FsaMapperAutomaton() {
    }
    virtual StateId Start() const {
        return impl_->Start();
    }
    virtual Weight Final(StateId id) const {
        return impl_->Final(id);
    }
    virtual size_t NumArcs(StateId id) const {
        return impl_->NumArcs(id);
    }
    virtual size_t NumInputEpsilons(StateId id) const {
        return impl_->NumInputEpsilons(id);
    }
    virtual size_t NumOutputEpsilons(StateId id) const {
        return impl_->NumOutputEpsilons(id);
    }
    virtual uint64 Properties(uint64 mask, bool test) const {
        if (test) {
            uint64 known, test = FstLib::TestProperties(*this, mask, &known);
            impl_->SetProperties(test, known);
            return test & mask;
        }
        else {
            return impl_->Properties(mask);
        }
    }
    virtual const string& Type() const {
        return impl_->Type();
    }
    virtual Self* Copy(bool reset = false) const {
        return new Self(*this, reset);
    }
    virtual const SymbolTable* InputSymbols() const {
        return impl_->InputSymbols();
    }
    virtual const SymbolTable* OutputSymbols() const {
        return impl_->OutputSymbols();
    }
    virtual void InitStateIterator(FstLib::StateIteratorData<FstArc>* data) const {
        data->base = new FsaMapperStateIteratorBase<Self>(impl_->getFsa());
    }
    virtual void InitArcIterator(StateId s, FstLib::ArcIteratorData<FstArc>* data) const {
        data->base = new FsaMapperArcIteratorBase<Self>(*this, impl_->getFsa()->getState(s));
    }

    typename FstArc::Weight ConvertWeight(const typename FsaAutomaton::Arc::Weight& w) const {
        return impl_->ConvertWeight(w);
    }

private:
    std::shared_ptr<Impl> impl_;
};

/**
 * required for StateIterators of FsaMapperAutomaton objects.
 * this works only, if the state ids are dense, which is the case
 * in the FsaMapperAutomaton because the original automaton's state
 * are mapped using Ftl::NormalizeAutomaton
 */
template<class F>
class FsaMapperStateIteratorBase : public FstLib::StateIteratorBase<typename F::Arc> {
public:
    typedef typename F::StateId StateId;
    explicit FsaMapperStateIteratorBase(typename F::ConstAutomatonRef f)
            : fsa_(f), s_(0) {}
    virtual ~FsaMapperStateIteratorBase() {}

    virtual bool Done() const {
        return !fsa_->getState(s_);
    }
    virtual StateId Value() const {
        return fsa_->getState(s_)->id();
    }
    virtual void Next() {
        ++s_;
    }
    virtual void Reset() {
        s_ = 0;
    }

private:
    typename F::ConstAutomatonRef fsa_;
    StateId                       s_;
};

/**
 * required for ArcIterators of FsaMapperAutomaton objects.
 * _all_ arcs of a state are created when the state is visited.
 */
template<class F>
class FsaMapperArcIteratorBase : public FstLib::ArcIteratorBase<typename F::Arc> {
public:
    typedef typename F::Arc                    Arc;
    typedef typename F::FsaType::State::Arc    FsaArc;
    typedef typename F::FsaType::ConstStateRef ConstStateRef;
    explicit FsaMapperArcIteratorBase(const F& parent, ConstStateRef state)
            : parent_(parent) {
        createArcs(state);
        i = 0;
    }
    virtual ~FsaMapperArcIteratorBase() {}

    virtual bool Done() const {
        return (i >= arcs_.size());
    }
    virtual const Arc& Value() const {
        return arcs_[i];
    }
    virtual void Next() {
        ++i;
    }
    virtual size_t Position() const {
        return i;
    }
    virtual void Reset() {
        i = 0;
    }
    virtual void Seek(size_t a) {
        i = a;
    }
    virtual uint32 Flags() const {
        return FstLib::kArcValueFlags;
    }
    virtual void SetFlags(uint32 flags, uint32 mask) {}

private:
    const F&         parent_;
    std::vector<Arc> arcs_;
    size_t           i;
    void             createArcs(ConstStateRef state) {
        arcs_.resize(state->nArcs());
        typename std::vector<Arc>::iterator fstArc = arcs_.begin();
        for (typename F::FsaType::State::const_iterator a = state->begin(); a != state->end(); ++a, ++fstArc) {
            fstArc->ilabel    = convertLabelFromFsa(a->input_);
            fstArc->olabel    = convertLabelFromFsa(a->output_);
            fstArc->weight    = parent_.ConvertWeight(a->weight_);
            fstArc->nextstate = a->target_;
        }
    }
};

}  // namespace OpenFst

#endif /* _OPENFST_FSA_MAPPER_HH */
