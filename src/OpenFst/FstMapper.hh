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
#ifndef _OPENFST_FSTMAPPER_HH
#define _OPENFST_FSTMAPPER_HH

#include <Fsa/Automaton.hh>
#include <Fsa/Semiring.hh>
#include <Fsa/tAutomaton.hh>
#include <OpenFst/SymbolTable.hh>
#include <OpenFst/Types.hh>
#include <OpenFst/Weight.hh>
#include <fst/fst.h>

namespace OpenFst {

template<class Semiring, class FstArc, template<class, class> class WeightConverter, class FsaAutomaton>
class FstMapperAutomatonImpl {
protected:
    typedef FstLib::Fst<FstArc>          FstAutomaton;
    typedef typename FsaAutomaton::State State;
    typedef Core::Ref<const Semiring>    ConstSemiringRef;

public:
    typedef WeightConverter<typename FstAutomaton::Weight, typename Semiring::Weight> WeightConverterType;

protected:
    const FstAutomaton*   fst_;
    ConstSemiringRef      semiring_;
    Fsa::ConstAlphabetRef inputAlphabet_, outputAlphabet_;
    WeightConverterType   weightConverter_;
    Fsa::Type             type_;

public:
    FstMapperAutomatonImpl(const FstAutomaton* fst, ConstSemiringRef semiring, WeightConverterType converter)
            : fst_(fst),
              semiring_(semiring),
              weightConverter_(converter) {
        init();
    }

    Fsa::ConstAlphabetRef inputAlphabet() const {
        return inputAlphabet_;
    }
    Fsa::ConstAlphabetRef outputAlphabet() const {
        return outputAlphabet_;
    }
    void setInputAlphabet(Fsa::ConstAlphabetRef a) {
        inputAlphabet_ = a;
    }
    void setOutputAlphabet(Fsa::ConstAlphabetRef a) {
        outputAlphabet_ = a;
    }
    ConstSemiringRef semiring() const {
        return semiring_;
    }
    void setSemiring(ConstSemiringRef s) {
        semiring_ = s;
    }
    Fsa::StateId initialStateId() const {
        return fst_->Start();
    }
    typename Semiring::Weight convertWeight(const typename FstAutomaton::Weight& w) {
        return weightConverter_(w);
    }
    const std::string describe() const {
        return "FstMapper(" + fst_->Type() + ")";
    }
    Fsa::Type type() const {
        return type_;
    }
    void setType(Fsa::Type type) {
        type_ = type;
    }
    State* createState(Fsa::StateId s) const {
        State* state = new State(s);
        if (fst_->Final(s) != FstAutomaton::Weight::Zero()) {
            state->setFinal(weightConverter_(fst_->Final(s)));
        }
        for (FstLib::ArcIterator<FstAutomaton> a(*fst_, s); !a.Done(); a.Next()) {
            const typename FstAutomaton::Arc& arc = a.Value();
            state->newArc(arc.nextstate, weightConverter_(arc.weight), convertLabelToFsa(arc.ilabel), convertLabelToFsa(arc.olabel));
        }
        return state;
    }

protected:
    void init() {
        inputAlphabet_  = convertAlphabet(fst_->InputSymbols());
        bool isAcceptor = fst_->Properties(FstLib::kAcceptor, true);
        if (!isAcceptor) {
            outputAlphabet_ = convertAlphabet(fst_->OutputSymbols());
        }
        type_ = (isAcceptor ? Fsa::TypeAcceptor : Fsa::TypeTransducer);
    }
};

template<class Semiring, class FstArc, template<class, class> class WeightConverter = ImplicitWeightConverter, class Base = Ftl::Automaton<Semiring>>
class FstMapperAutomaton : public Base {
public:
    typedef Base                            FsaAutomaton;
    typedef typename FstLib::Fst<FstArc>    FstAutomaton;
    typedef typename FsaAutomaton::State    _State;
    typedef typename FsaAutomaton::Semiring _Semiring;
    typedef Core::Ref<const _State>         _ConstStateRef;
    typedef Core::Ref<const _Semiring>      _ConstSemiringRef;

protected:
    typedef FstMapperAutomatonImpl<Semiring, FstArc, WeightConverter, Base> Impl;
    typedef typename Impl::WeightConverterType                              WeightConverterType;
    Impl*                                                                   impl_;

public:
    FstMapperAutomaton(const FstAutomaton* fst, WeightConverterType converter = WeightConverterType())
            : FsaAutomaton(),
              impl_(new Impl(fst, Fsa::ConstSemiringRef(), converter)) {}
    FstMapperAutomaton(const FstAutomaton* fst, _ConstSemiringRef semiring, WeightConverterType converter = WeightConverterType())
            : FsaAutomaton(),
              impl_(new Impl(fst, semiring, converter)) {}
    virtual ~FstMapperAutomaton() {
        delete impl_;
    }

public:
    virtual Fsa::Type type() const {
        return impl_->type();
    }
    virtual void setType(Fsa::Type type) {
        impl_->setType(type);
    }
    virtual _ConstSemiringRef semiring() const {
        return impl_->semiring();
    }
    virtual void setSemiring(_ConstSemiringRef semiring) {
        impl_->setSemiring(semiring);
    }
    virtual Fsa::StateId initialStateId() const {
        return impl_->initialStateId();
    }
    virtual Fsa::ConstAlphabetRef getInputAlphabet() const {
        return impl_->inputAlphabet();
    }
    virtual Fsa::ConstAlphabetRef getOutputAlphabet() const {
        return impl_->outputAlphabet();
    }
    void setInputAlphabet(Fsa::ConstAlphabetRef a) {
        impl_->setInputAlphabet(a);
    }
    void setOutputAlphabet(Fsa::ConstAlphabetRef a) {
        impl_->setOutputAlphabet(a);
    }

    virtual _ConstStateRef getState(Fsa::StateId s) const {
        return _ConstStateRef(impl_->createState(s));
    }
    virtual std::string describe() const {
        return impl_->describe();
    }
};

template<class FstArc, template<class, class> class WeightConverter>
class FstMapperAutomaton<Fsa::Semiring, FstArc, WeightConverter> : public Fsa::Automaton {
public:
    typedef typename Fsa::Automaton         FsaAutomaton;
    typedef typename FstLib::Fst<FstArc>    FstAutomaton;
    typedef typename FsaAutomaton::State    _State;
    typedef typename FsaAutomaton::Semiring _Semiring;
    typedef Core::Ref<const _State>         _ConstStateRef;
    typedef Core::Ref<const _Semiring>      _ConstSemiringRef;

protected:
    typedef FstMapperAutomatonImpl<Semiring, FstArc, WeightConverter, FsaAutomaton> Impl;
    typedef typename Impl::WeightConverterType                                      WeightConverterType;
    Impl*                                                                           impl_;

public:
    FstMapperAutomaton(const FstAutomaton* fst, WeightConverterType converter = WeightConverterType())
            : FsaAutomaton(),
              impl_(new Impl(fst, Fsa::ConstSemiringRef(), converter)) {}
    FstMapperAutomaton(const FstAutomaton* fst, _ConstSemiringRef semiring, WeightConverterType converter = WeightConverterType())
            : FsaAutomaton(),
              impl_(new Impl(fst, semiring, converter)) {}
    virtual ~FstMapperAutomaton() {
        delete impl_;
    }

public:
    virtual Fsa::Type type() const {
        return impl_->type();
    }
    virtual void setType(Fsa::Type type) {
        impl_->setType(type);
    }
    virtual _ConstSemiringRef semiring() const {
        return impl_->semiring();
    }
    virtual void setSemiring(_ConstSemiringRef semiring) {
        impl_->setSemiring(semiring);
    }
    virtual Fsa::StateId initialStateId() const {
        return impl_->initialStateId();
    }
    virtual Fsa::ConstAlphabetRef getInputAlphabet() const {
        return impl_->inputAlphabet();
    }
    virtual Fsa::ConstAlphabetRef getOutputAlphabet() const {
        return impl_->outputAlphabet();
    }
    virtual _ConstStateRef getState(Fsa::StateId s) const {
        return _ConstStateRef(impl_->createState(s));
    }
    virtual std::string describe() const {
        return impl_->describe();
    }
};

}  // namespace OpenFst
#endif /* _OPENFST_FSTMAPPER_HH */
