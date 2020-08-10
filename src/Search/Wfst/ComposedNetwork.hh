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
#ifndef _SEARCH_WFST_COMPOSED_NETWORK_HH
#define _SEARCH_WFST_COMPOSED_NETWORK_HH

#include <Search/Wfst/ComposeFst.hh>
#include <Search/Wfst/Network.hh>
#include <fst/compose.h>
#include <fst/replace.h>
#include <fst/state-table.h>

namespace Search {
namespace Wfst {

class AbstractGrammarFst;
class AbstractLexicalFst;

/**
 * network with dynamic composition of 2 automata
 */
class ComposedNetwork : public FstNetwork<FstLib::ComposeFst<FstLib::StdArc>> {
public:
    typedef FstLib::ComposeFst<Arc> ComposeFst;

protected:
    typedef FstNetwork<ComposeFst>     Precursor;
    static const Core::ParameterString paramNetworkLeft_;
    static const Core::ParameterString paramNetworkRight_;
    static const Core::ParameterInt    paramStateCache_;
    static const Core::ParameterInt    paramResetInterval_;
    static const Core::ParameterChoice paramGrammarType_;
    static const Core::Choice          choiceGrammarType_;

public:
    ComposedNetwork(const Core::Configuration&);
    virtual ~ComposedNetwork();
    virtual bool init();
    virtual u32  nStates() const {
        return 0;
    }
    static bool hasGrammarState() {
        return true;
    }
    StateIndex grammarState(StateIndex s) const {
        return stateTable_->rightState(s);
    }
    StateIndex lexiconState(StateIndex s) const {
        return stateTable_->leftState(s);
    }
    virtual void reset();
    void         setLexicon(Bliss::LexiconRef lexicon) {
        lexicon_ = lexicon;
    }

public:
    class ArcIterator {
    public:
        ArcIterator(const ComposedNetwork* network, StateIndex s)
                : a_(*network->f_, s) {
            if (!a_.Done() && a_.Value().ilabel == OpenFst::Epsilon)
                next();
        }
        void next() {
            do {
                a_.Next();
            } while (!a_.Done() && a_.Value().ilabel == OpenFst::Epsilon);
        }
        bool done() const {
            return a_.Done();
        }
        const Arc& value() const {
            return a_.Value();
        }
        void reset() {
            a_.Reset();
            if (!a_.Done() && a_.Value().ilabel == OpenFst::Epsilon)
                next();
        }

    private:
        FstLib::ArcIterator<ComposeFst> a_;
    };
    friend class ArcIterator;

    class EpsilonArcIterator {
    public:
        EpsilonArcIterator(const ComposedNetwork* network, StateIndex s)
                : a_(*network->f_, s) {
            if (!a_.Done() && a_.Value().ilabel != OpenFst::Epsilon)
                next();
        }
        void next() {
            do {
                a_.Next();
            } while (!a_.Done() && a_.Value().ilabel != OpenFst::Epsilon);
        }
        bool done() {
            return a_.Done();
        }
        const EpsilonArc& value() const {
            return a_.Value();
        }
        void reset() {
            a_.Reset();
            if (!a_.Done() && a_.Value().ilabel != OpenFst::Epsilon)
                next();
        }

    private:
        FstLib::ArcIterator<ComposeFst> a_;
    };
    friend class EpsilonArcIterator;

private:
    void createL();
    void createG();

    AbstractLexicalFst* l_;
    AbstractGrammarFst* r_;
    /**
     * This makes grammarState() expensive, as we can't inline but have to
     * call a virtual function instead!
     *
     * To make ComposedNetwork more efficient (at least if grammarState is used)
     * the ComposeFilter and the StateTable should be fixed.
     */
    AbstractStateTable* stateTable_;
    u32                 resetCount_, resetInterval_;
    size_t              cacheSize_;
    Bliss::LexiconRef   lexicon_;
};

}  // namespace Wfst
}  // namespace Search

#endif  // _SEARCH_WFST_COMPOSED_NETWORK_HH
