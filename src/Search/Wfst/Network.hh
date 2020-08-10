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
#ifndef _SEARCH_WFST_NETWORK_HH
#define _SEARCH_WFST_NETWORK_HH

#include <Bliss/Lexicon.hh>
#include <Core/Component.hh>
#include <Core/MemoryInfo.hh>
#include <Core/Parameter.hh>
#include <Fsa/Types.hh>
#include <OpenFst/Types.hh>
#include <Search/Types.hh>

namespace Search {
namespace Wfst {

/*! @todo: move this type out of global scope */
enum NetworkType { NetworkTypeCompressed,
                   NetworkTypeStatic,
                   NetworkTypeComposed,
                   NetworkTypeLattice };

/**
 * base class for OpenFst based networks
 */
template<class F>
class FstNetwork : public Core::Component {
protected:
    typedef F Fst;
    Fst*      f_;
    Fst*      automaton() const {
        return f_;
    }

public:
    typedef u32                 ArcIndex;
    typedef u32                 StateIndex;
    typedef typename F::Arc     Arc;
    typedef Arc                 EpsilonArc;
    typedef typename Arc::Label Label;

public:
    u32 nArcs() const {
        return 0;
    }
    u32 nEpsilonArcs() const {
        return 0;
    }
    virtual u32 nStates() const = 0;
    size_t      memStates() const {
        return 0;
    }
    size_t memArcs() const {
        return 0;
    }
    size_t memEpsilonArcs() const {
        return 0;
    }

public:
    bool isFinal(StateIndex s) const {
        return f_->Final(s) != Fst::Weight::Zero();
    }
    f32 finalWeight(StateIndex s) const {
        return f_->Final(s).Value();
    }
    StateIndex initialStateIndex() const {
        return f_->Start();
    }

    virtual bool init() {
        return true;
    }
    virtual void reset() {}
    virtual void setSegment(const std::string& name) {}
    virtual void setLexicon(Bliss::LexiconRef lexicon) {}

    static f32 arcWeight(const Arc& arc) {
        return arc.weight.Value();
    }
    static f32 arcWeight(const Arc& arc, f32 scale) {
        return scale * arc.weight.Value();
    }
    static u32 stateSequenceIndex(const Arc& arc) {
        return arc.ilabel - 1;
    }

    static Fsa::LabelId getFsaLabel(Label l) {
        return l - 1;
    }

protected:
    FstNetwork(const Core::Configuration& c)
            : Core::Component(c), f_(0), memUsageChannel_(c, "memory-info") {}
    virtual ~FstNetwork() {
        delete f_;
    }

protected:
    mutable Core::XmlChannel memUsageChannel_;
    void                     logMemoryUsage() const {
        if (memUsageChannel_.isOpen()) {
            Core::MemoryInfo meminfo;
            memUsageChannel_ << meminfo;
        }
    }
};

/**
 * network with direct access to the automaton
 */
class StaticNetwork : public FstNetwork<OpenFst::VectorFst> {
private:
    typedef FstNetwork<OpenFst::VectorFst> Precursor;
    static const Core::ParameterString     paramNetworkFile_;
    static const Core::ParameterFloat      paramScale_;

public:
    StaticNetwork(const Core::Configuration&);
    virtual ~StaticNetwork() {}
    virtual bool init();
    virtual u32  nStates() const {
        return f_->NumStates();
    }
    static bool hasGrammarState() {
        return false;
    }
    StateIndex grammarState(StateIndex) const {
        return 0;
    }

public:
    class ArcIterator {
    public:
        ArcIterator(const StaticNetwork* network, StateIndex s)
                : a_(*network->f_, s), offset_(network->f_->NumInputEpsilons(s)) {
            reset();
        }
        void next() {
            a_.Next();
        }
        bool done() const {
            return a_.Done();
        }
        const Arc& value() const {
            return a_.Value();
        }
        void reset() {
            a_.Seek(offset_);
        }

    private:
        OpenFst::ArcIterator a_;
        size_t               offset_;
    };
    friend class ArcIterator;

    class EpsilonArcIterator {
    public:
        EpsilonArcIterator(const StaticNetwork* network, StateIndex s)
                : a_(*network->f_, s) {}
        void next() {
            a_.Next();
        }
        bool done() {
            return a_.Done() || a_.Value().ilabel != OpenFst::Epsilon;
        }
        const EpsilonArc& value() const {
            return a_.Value();
        }
        void reset() {
            a_.Reset();
        }

    private:
        OpenFst::ArcIterator a_;
    };
    friend class EpsilonArcIterator;
};

}  // namespace Wfst
}  // namespace Search

#endif /* _SEARCH_WFST_NETWORK_HH */
