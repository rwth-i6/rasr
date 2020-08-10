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
#ifndef _OPENFST_OUTPUT_HH
#define _OPENFST_OUTPUT_HH

#include <Fsa/Alphabet.hh>
#include <Fsa/Basic.hh>
#include <Fsa/Dfs.hh>
#include <Fsa/Resources.hh>
#include <Fsa/Types.hh>
#include "FsaMapper.hh"
#include "SymbolTable.hh"
#include "Types.hh"

namespace OpenFst {

/**
 * @deprecated
 * initial attempt to convert an Fsa-object to an OpenFst-object.
 * use OpenFst::FsaMapper, see convertFromFsa
 */
template<class _Automaton>
class ConvertFsaDfsState : public Ftl::DfsState<_Automaton> {
private:
    typedef Ftl::DfsState<_Automaton> Precursor;

public:
    typedef typename _Automaton::State         _State;
    typedef typename _Automaton::ConstStateRef _ConstStateRef;
    typedef typename _Automaton::ConstRef      _ConstAutomatonRef;

private:
    VectorFst* fst_;

    typedef std::map<Fsa::StateId, StateId> StateIdMap;
    StateIdMap                              stateIdMap_;
    typedef std::map<Fsa::LabelId, Label>   LabelMap;
    LabelMap                                inputSymbolMap_, outputSymbolMap_;

    StateId mapState(Fsa::StateId fsaId) {
        StateIdMap::const_iterator id = stateIdMap_.find(fsaId);
        if (id == stateIdMap_.end()) {
            StateId newId = fst_->AddState();
            stateIdMap_.insert(std::make_pair(fsaId, newId));
            return newId;
        }
        else {
            return id->second;
        }
    }

    Label mapLabel(SymbolTable* fstSymbols, Fsa::ConstAlphabetRef fsaSymbols, LabelMap& map, Fsa::LabelId fsaId) {
        LabelMap::const_iterator it = map.find(fsaId);
        if (it == map.end()) {
            Label       fstId;
            std::string strLabel = fsaSymbols->symbol(fsaId);
            if (!strLabel.empty())
                fstId = fstSymbols->Find(strLabel.c_str());
            else {
                // std::cerr << "unknown label: " << fsaId << std::endl;
                fstId = fstSymbols->AddSymbol(Core::form("fsa-label-%d", fsaId));
            }
            map.insert(std::make_pair(fsaId, fstId));
            return fstId;
        }
        else {
            return it->second;
        }
    }

    Label mapInputLabel(Fsa::LabelId fsaId) {
        return mapLabel(fst_->InputSymbols(), Precursor::fsa_->getInputAlphabet(), inputSymbolMap_, fsaId);
    }

    Label mapOutputLabel(Fsa::LabelId fsaId) {
        return mapLabel(fst_->OutputSymbols(), Precursor::fsa_->getOutputAlphabet(), outputSymbolMap_, fsaId);
    }

public:
    ConvertFsaDfsState(_ConstAutomatonRef f, VectorFst* fst)
            : Precursor(f), fst_(fst) {
        inputSymbolMap_.insert(std::make_pair(Fsa::Epsilon, Epsilon));
        outputSymbolMap_.insert(std::make_pair(Fsa::Epsilon, Epsilon));
    }

    void discoverState(_ConstStateRef sp) {
        Fsa::StateId fsaId = sp->id();
        StateId      id    = mapState(fsaId);
        if (fsaId == Precursor::fsa_->initialStateId())
            fst_->SetStart(id);
        if (sp->isFinal())
            fst_->SetFinal(id, Weight(sp->weight_));
        for (typename _State::const_iterator a = sp->begin(); a != sp->end(); ++a) {
            Label output = mapOutputLabel(Precursor::fsa_->type() == Fsa::TypeTransducer ? a->output() : a->input());
            Label input  = mapInputLabel(a->input());
            fst_->AddArc(id, Arc(input, output, Weight(a->weight()), mapState(a->target())));
        }
    }
};

/**
 * convert an arbitrary Fsa Toolkit transducer to
 * a _FstAutomaton object (has to be a subclass of FstLib::Fst).
 * delayed transducers are expanded.
 */
template<class _Automaton, class _FstAutomaton>
_FstAutomaton* convertFromFsa(typename _Automaton::ConstRef f) {
    FsaMapperAutomaton<_Automaton, typename _FstAutomaton::Arc> mapper(f);
    _FstAutomaton*                                              fst = new _FstAutomaton(mapper);
    return fst;
}

VectorFst* convertFromFsa(Fsa::ConstAutomatonRef f);

/**
 * write a Fsa transducer in the OpenFst format.
 * transducer will be written as FstLib::VectorFst<StdArc>
 */
bool writeFsa(Fsa::ConstAutomatonRef f, const std::string& file);

bool writeOpenFst(const Fsa::Resources& resources, Fsa::ConstAutomatonRef f,
                  std::ostream& o, Fsa::StoredComponents what, bool progress);

/**
 * convenience method to write a FstLib::VectorFst<StdArc> object to disk using
 * FstLib::VectorFst<StdArc>::Write
 */
bool write(const VectorFst& fst, const std::string& filename);

}  // namespace OpenFst
#endif  // _OPENFST_OUTPUT_HH
