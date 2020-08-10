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
#ifndef _SEARCH_WFST_COMPRESSED_NETWORK_HH
#define _SEARCH_WFST_COMPRESSED_NETWORK_HH

#include <Bliss/Lexicon.hh>
#include <Core/Component.hh>
#include <Fsa/Automaton.hh>
#include <Fsa/Types.hh>
#include <OpenFst/Types.hh>
#include <Search/Types.hh>
#include <Search/Wfst/AutomatonAdapter.hh>

namespace Search {
namespace Wfst {

/**
 * Compressed version of either Fsa automaton or OpenFst automaton.
 * The data is read from a memory mapped file.
 * Limitations apply to the number of arcs per state (u16),
 * number of epsilon arcs per state (u8) and number of labels (u16).
 * Labels are stored in OpenFst format (Epsilon = 0)
 */
class CompressedNetwork : public Core::Component {
private:
    static const Core::Choice          choiceAutomatonType;
    static const Core::ParameterChoice paramAutomatonType_;
    static const Core::ParameterString paramNetworkFile_;
    static const Core::ParameterBool   paramRemoveEpsArcs_;

public:
    typedef u32            ArcIndex;
    typedef u32            StateIndex;
    typedef OpenFst::Label Label;
    typedef u16            InternalLabel;
    typedef u16            ArcCount;
    typedef u8             EpsArcCount;

    struct Arc {
        Score         weight;
        Fsa::StateId  nextstate;  // target;
        InternalLabel olabel;     // output;
        InternalLabel ilabel;     // input;
        Arc(Fsa::StateId _target,
            Label _input, Label _output, Score _weight)
                : weight(_weight), nextstate(_target), olabel(_output), ilabel(_input) {}
    };
    typedef Arc* Arcs;

    struct EpsilonArc {
        Score         weight;
        InternalLabel olabel;     // output;
        Fsa::StateId  nextstate;  // target;
        EpsilonArc(Fsa::StateId _target, Label _output, Score _weight)
                : weight(_weight), olabel(_output), nextstate(_target) {}
    };
    typedef EpsilonArc* EpsilonArcs;

    struct State {
        ArcIndex    begin;             // index of begin/end in arcs_
        ArcIndex    epsilonArcsBegin;  // index of begin/end in epsilonArcs_
        ArcCount    nArcs;
        EpsArcCount nEpsilonArcs;
        // bool final;
        Score weight;
        State(bool f = false, Score weight = 0)
                : begin(InvalidArcIndex), epsilonArcsBegin(InvalidArcIndex), nArcs(0), nEpsilonArcs(0), weight(f ? weight : NonFinalWeight) {}
    };
    typedef State* States;

private:
    static const ArcIndex InvalidArcIndex;
    static const Score    NonFinalWeight;
    States                states_;
    Arcs                  arcs_;
    EpsilonArcs           epsilonArcs_;
    StateIndex            initialStateIndex_;
    u32                   nStates_, nArcs_, nEpsilonArcs_;

public:
    CompressedNetwork(const Core::Configuration&, bool loadNetwork = true);
    virtual ~CompressedNetwork();
    bool build(Fsa::ConstAutomatonRef f, bool removeEpsArcs = false);
    bool build(const OpenFst::VectorFst* f, bool removeEpsArcs = false);
    bool write(const std::string& file) const;
    bool read(const std::string& file);
    void setLexicon(Bliss::LexiconRef lexicon) {}
    bool init();

    u32 nArcs() const {
        return nArcs_;
    }
    u32 nEpsilonArcs() const {
        return nEpsilonArcs_;
    }
    u32 nStates() const {
        return nStates_;
    }
    size_t memStates() const {
        return nStates_ * sizeof(State);
    }
    size_t memArcs() const {
        return nArcs_ * sizeof(Arc);
    }
    size_t memEpsilonArcs() const {
        return nEpsilonArcs_ * sizeof(EpsilonArc);
    }

public:
    bool isFinal(StateIndex s) const {
        return (states_[s].weight != NonFinalWeight);
    }
    Score finalWeight(StateIndex s) const {
        return states_[s].weight;
    }

    void reset() {}
    void setSegment(const std::string&) {}

    StateIndex initialStateIndex() const {
        return initialStateIndex_;
    }

    static bool hasGrammarState() {
        return false;
    }
    StateIndex grammarState(StateIndex) const {
        return 0;
    }

    class ArcIterator {
    public:
        ArcIterator(const CompressedNetwork* network, StateIndex s) {
            const States states = network->states_;
            const Arcs   arcs   = network->arcs_;
            a_                  = arcs + states[s].begin;
            end_                = a_ + static_cast<size_t>(states[s].nArcs);
        }
        void next() {
            ++a_;
        }
        bool done() const {
            return a_ == end_;
        }
        const Arc& value() const {
            return *a_;
        }
        void reset() {
            defect();
        }

    private:
        Arc *a_, *end_;
    };
    friend class ArcIterator;

    class EpsilonArcIterator {
    public:
        EpsilonArcIterator(const CompressedNetwork* network, StateIndex s) {
            const States      states = network->states_;
            const EpsilonArcs arcs   = network->epsilonArcs_;
            a_                       = arcs + states[s].epsilonArcsBegin;
            end_                     = a_ + static_cast<size_t>(states[s].nEpsilonArcs);
        }
        void next() {
            ++a_;
        }
        bool done() {
            return a_ == end_;
        }
        const EpsilonArc& value() const {
            return *a_;
        }
        void reset() {
            defect();
        }

    private:
        EpsilonArc *a_, *end_;
    };
    friend class EpsilonArcIterator;

    static f32 arcWeight(const Arc& arc) {
        return arc.weight;
    }
    static f32 arcWeight(const Arc& arc, f32 scale) {
        return scale * arc.weight;
    }
    static f32 arcWeight(const EpsilonArc& arc) {
        return arc.weight;
    }
    static f32 arcWeight(const EpsilonArc& arc, f32 scale) {
        return scale * arc.weight;
    }
    static u32 stateSequenceIndex(const Arc& arc) {
        return arc.ilabel - 1;
    }
    static Fsa::LabelId getFsaLabel(Label l) {
        return OpenFst::convertLabelToFsa(l);
    }

private:
    u32    writeData(int) const;
    bool   readData(int);
    char*  mmap_;
    size_t mmapSize_;
    bool   loadNetwork_;
    struct ImageHeader;
    friend struct ImageHeader;
    template<class A>
    class Builder;
    friend class Builder<FsaAutomatonAdapter>;
    friend class Builder<FstAutomatonAdapter>;
};

}  // namespace Wfst
}  // namespace Search

#endif  // _SEARCH_WFST_COMPRESSED_NETWORK_HH
