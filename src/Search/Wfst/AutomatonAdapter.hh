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
#ifndef _SEARCH_AUTOMATON_ADAPTER_HH
#define _SEARCH_AUTOMATON_ADAPTER_HH

#include <Fsa/Automaton.hh>
#include <Fsa/Basic.hh>
#include <Fsa/Static.hh>
#include <OpenFst/SymbolTable.hh>
#include <OpenFst/Types.hh>

namespace Search {
namespace Wfst {

/**
 * common interface to basic finite state automata data structures.
 * labels are always OpenFst::Label, i.e. Epsilon == 0
 */
template<class F, class A, class S, class L>
class AutomatonAdapter {
public:
    typedef F                    Automaton;
    typedef A                    Arc;
    typedef S                    StateId;
    typedef L                    Label;
    typedef typename Arc::Weight Weight;
    typedef Automaton*           AutomatonRef;

    AutomatonAdapter(const Automaton* f)
            : f_(f), nArcs_(0), nEpsArcs_(0) {}

    StateId        initialStateId() const;
    bool           isFinal(StateId s) const;
    Weight         finalWeight(StateId s) const;
    f32            finalWeightValue(StateId s) const;
    Weight         arcWeight(const Arc& arc) const;
    f32            arcWeightValue(const Arc& arc) const;
    OpenFst::Label arcInput(const Arc& arc) const;
    OpenFst::Label arcOutput(const Arc& arc) const;
    StateId        arcTarget(const Arc& arc) const;
    u32            nStates() const;
    u32            nArcs() const {
        if (!nArcs_)
            countArcs();
        return nArcs_;
    }
    u32 nEpsilonArcs() const {
        if (!nArcs_)
            countArcs();
        return nEpsArcs_;
    }

    class ArcIterator;

    ArcIterator arcs(StateId s) const;

private:
    void             countArcs() const;
    const Automaton* f_;
    mutable u32      nArcs_, nEpsArcs_;
};

/*******************************************************************/

template<>
inline Fsa::StateId AutomatonAdapter<Fsa::StaticAutomaton, Fsa::Arc, Fsa::StateId, Fsa::LabelId>::initialStateId() const {
    return f_->initialStateId();
}

template<>
inline bool AutomatonAdapter<Fsa::StaticAutomaton, Fsa::Arc, Fsa::StateId, Fsa::LabelId>::isFinal(Fsa::StateId s) const {
    return f_->fastState(s)->isFinal();
}

template<>
inline AutomatonAdapter<Fsa::StaticAutomaton, Fsa::Arc, Fsa::StateId, Fsa::LabelId>::Weight
        AutomatonAdapter<Fsa::StaticAutomaton, Fsa::Arc, Fsa::StateId, Fsa::LabelId>::finalWeight(Fsa::StateId s) const {
    return f_->fastState(s)->weight();
}

template<>
inline f32 AutomatonAdapter<Fsa::StaticAutomaton, Fsa::Arc, Fsa::StateId, Fsa::LabelId>::finalWeightValue(Fsa::StateId s) const {
    return finalWeight(s);
}

template<>
inline Fsa::Weight AutomatonAdapter<Fsa::StaticAutomaton, Fsa::Arc, Fsa::StateId, Fsa::LabelId>::arcWeight(const Fsa::Arc& arc) const {
    return arc.weight_;
}

template<>
inline f32 AutomatonAdapter<Fsa::StaticAutomaton, Fsa::Arc, Fsa::StateId, Fsa::LabelId>::arcWeightValue(const Fsa::Arc& arc) const {
    return arc.weight_;
}

template<>
inline OpenFst::Label AutomatonAdapter<Fsa::StaticAutomaton, Fsa::Arc, Fsa::StateId, Fsa::LabelId>::arcInput(const Fsa::Arc& arc) const {
    return OpenFst::convertLabelFromFsa(arc.input_);
}

template<>
inline OpenFst::Label AutomatonAdapter<Fsa::StaticAutomaton, Fsa::Arc, Fsa::StateId, Fsa::LabelId>::arcOutput(const Fsa::Arc& arc) const {
    return OpenFst::convertLabelFromFsa(arc.output_);
}

template<>
inline Fsa::StateId AutomatonAdapter<Fsa::StaticAutomaton, Fsa::Arc, Fsa::StateId, Fsa::LabelId>::arcTarget(const Fsa::Arc& arc) const {
    return arc.target_;
}

template<>
inline u32 AutomatonAdapter<Fsa::StaticAutomaton, Fsa::Arc, Fsa::StateId, Fsa::LabelId>::nStates() const {
    return f_->size();
}

template<>
inline void AutomatonAdapter<Fsa::StaticAutomaton, Fsa::Arc, Fsa::StateId, Fsa::LabelId>::countArcs() const {
    // need to count arcs in f_
    // but Fsa::count requires an Fsa::ConstAutomatonRef
    defect();
}

template<>
class AutomatonAdapter<Fsa::StaticAutomaton, Fsa::Arc, Fsa::StateId, Fsa::LabelId>::ArcIterator {
public:
    ArcIterator(const Automaton* f, StateId s) {
        const Fsa::State* state = f->fastState(s);
        a_                      = state->begin();
        end_                    = state->end();
    }
    void next() {
        ++a_;
    }
    bool done() const {
        return a_ == end_;
    }
    const Fsa::Arc& value() const {
        return *a_;
    }

private:
    Fsa::State::const_iterator a_, end_;
};

template<>
inline AutomatonAdapter<Fsa::StaticAutomaton, Fsa::Arc, Fsa::StateId, Fsa::LabelId>::ArcIterator
        AutomatonAdapter<Fsa::StaticAutomaton, Fsa::Arc, Fsa::StateId, Fsa::LabelId>::arcs(Fsa::StateId s) const {
    return ArcIterator(f_, s);
}

/*******************************************************************/

template<>
inline OpenFst::StateId AutomatonAdapter<OpenFst::VectorFst, OpenFst::Arc, OpenFst::StateId, OpenFst::Label>::initialStateId() const {
    return f_->Start();
}

template<>
inline bool AutomatonAdapter<OpenFst::VectorFst, OpenFst::Arc, OpenFst::StateId, OpenFst::Label>::isFinal(OpenFst::StateId s) const {
    return (f_->Final(s) != OpenFst::VectorFst::Weight::Zero());
}

template<>
inline AutomatonAdapter<OpenFst::VectorFst, OpenFst::Arc, OpenFst::StateId, OpenFst::Label>::Weight
        AutomatonAdapter<OpenFst::VectorFst, OpenFst::Arc, OpenFst::StateId, OpenFst::Label>::finalWeight(OpenFst::StateId s) const {
    return f_->Final(s);
}

template<>
inline f32 AutomatonAdapter<OpenFst::VectorFst, OpenFst::Arc, OpenFst::StateId, OpenFst::Label>::finalWeightValue(OpenFst::StateId s) const {
    return finalWeight(s).Value();
}

template<>
inline OpenFst::Weight AutomatonAdapter<OpenFst::VectorFst, OpenFst::Arc, OpenFst::StateId, OpenFst::Label>::arcWeight(const OpenFst::Arc& arc) const {
    return arc.weight;
}

template<>
inline f32 AutomatonAdapter<OpenFst::VectorFst, OpenFst::Arc, OpenFst::StateId, OpenFst::Label>::arcWeightValue(const OpenFst::Arc& arc) const {
    return arc.weight.Value();
}

template<>
inline OpenFst::Label AutomatonAdapter<OpenFst::VectorFst, OpenFst::Arc, OpenFst::StateId, OpenFst::Label>::arcInput(const OpenFst::Arc& arc) const {
    return arc.ilabel;
}

template<>
inline OpenFst::Label AutomatonAdapter<OpenFst::VectorFst, OpenFst::Arc, OpenFst::StateId, OpenFst::Label>::arcOutput(const OpenFst::Arc& arc) const {
    return arc.olabel;
}

template<>
inline OpenFst::StateId AutomatonAdapter<OpenFst::VectorFst, OpenFst::Arc, OpenFst::StateId, OpenFst::Label>::arcTarget(const OpenFst::Arc& arc) const {
    return arc.nextstate;
}

template<>
inline u32 AutomatonAdapter<OpenFst::VectorFst, OpenFst::Arc, OpenFst::StateId, OpenFst::Label>::nStates() const {
    return f_->NumStates();
}

template<>
inline void AutomatonAdapter<OpenFst::VectorFst, OpenFst::Arc, OpenFst::StateId, OpenFst::Label>::countArcs() const {
    nArcs_    = 0;
    nEpsArcs_ = 0;
    for (FstLib::StateIterator<OpenFst::VectorFst> siter(*f_); !siter.Done(); siter.Next()) {
        OpenFst::VectorFst::StateId s = siter.Value();
        for (FstLib::ArcIterator<OpenFst::VectorFst> aiter(*f_, s); !aiter.Done(); aiter.Next()) {
            const OpenFst::Arc& arc = aiter.Value();
            if (arc.ilabel == OpenFst::Epsilon)
                ++nEpsArcs_;
            else
                ++nArcs_;
        }
    }
}

template<>
class AutomatonAdapter<OpenFst::VectorFst, OpenFst::Arc, OpenFst::StateId, OpenFst::Label>::ArcIterator {
public:
    ArcIterator(const Automaton* f, StateId s)
            : a_(new FstLib::ArcIterator<OpenFst::VectorFst>(*f, s)) {}
    ~ArcIterator() {
        delete a_;
    }
    void next() {
        a_->Next();
    }
    bool done() const {
        return a_->Done();
    }
    const OpenFst::Arc& value() const {
        return a_->Value();
    }

private:
    FstLib::ArcIterator<OpenFst::VectorFst>* a_;
};

template<>
inline AutomatonAdapter<OpenFst::VectorFst, OpenFst::Arc, OpenFst::StateId, OpenFst::Label>::ArcIterator
        AutomatonAdapter<OpenFst::VectorFst, OpenFst::Arc, OpenFst::StateId, OpenFst::Label>::arcs(OpenFst::StateId s) const {
    return ArcIterator(f_, s);
}

/*******************************************************************/

typedef AutomatonAdapter<Fsa::StaticAutomaton, Fsa::Arc, Fsa::StateId, Fsa::LabelId>         FsaAutomatonAdapter;
typedef AutomatonAdapter<OpenFst::VectorFst, OpenFst::Arc, OpenFst::StateId, OpenFst::Label> FstAutomatonAdapter;

}  // namespace Wfst
}  // namespace Search

#endif /* _SEARCH_AUTOMATON_ADAPTER_HH */
