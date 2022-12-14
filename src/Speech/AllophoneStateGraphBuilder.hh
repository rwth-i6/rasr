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
#ifndef _SPEECH_ALLOPHONE_STATE_GRAPH_BUILDER_HH
#define _SPEECH_ALLOPHONE_STATE_GRAPH_BUILDER_HH

#include <Bliss/CorpusDescription.hh>
#include <Bliss/Lexicon.hh>
#include <Core/Component.hh>
#include <Fsa/Automaton.hh>
#include <Fsa/Static.hh>
#include <deque>

namespace Am {
class AcousticModel;
}

namespace Bliss {
class PhonemeToLemmaTransducer;
class OrthographicParser;
}  // namespace Bliss

namespace Speech {
class ModelCombination;
class Alignment;
}  // namespace Speech

namespace Speech {

typedef Fsa::Automaton::ConstRef AllophoneStateGraphRef;

/**
 *  AllophoneStateGraphBuilder
 *  - base class for FSA graph building
 *  - output flat automaton/transducer without additional transitions (e.g. loop, skip ...)
 *  - configurable inclusion of alternative paths (e.g. pronunciation variants)
 */
class AllophoneStateGraphBuilder : public Core::Component, public Core::ReferenceCounted {
    typedef Core::Component Precursor;

public:
    /**
     *   Base class for build-functor classes.
     */
    template<class... BuilderInput>
    class FunctorBase {
    protected:
        AllophoneStateGraphBuilder&       builder_;
        const std::string                 id_;
        const std::tuple<BuilderInput...> builderInput_;

    public:
        FunctorBase(AllophoneStateGraphBuilder& builder,
                    const std::string&          id,
                    BuilderInput... builderInput)
                : builder_(builder), id_(id), builderInput_(builderInput...) {}

        const std::string& id() const {
            return id_;
        }
    };

    /**
     *   Converts a call to one of the build functions into a functor.
     *
     *   Underlying function: build
     *   Input: template BuilderInput
     *   Ouptut: AllophoneStateGraphRef
     */
    template<class... BuilderInput>
    struct Functor : public FunctorBase<BuilderInput...> {
    private:
        typedef FunctorBase<BuilderInput...> Precursor;

    public:
        Functor(AllophoneStateGraphBuilder& builder,
                const std::string&          id,
                BuilderInput... builderInput)
                : FunctorBase<BuilderInput...>(builder, id, builderInput...) {}

        AllophoneStateGraphRef build() {
            return Precursor::builder_.build(Precursor::builderInput_);
        }
    };

    /**
     *   Converts a call to one of the finalizeTransducer functions into a functor.
     *
     *   Underlying function: finalizeTransducer
     *   Input: Fsa::ConstAutomatonRef
     *   Ouptut: AllophoneStateGraphRef
     */
    struct FinalizationFunctor : public FunctorBase<Fsa::ConstAutomatonRef> {
    private:
        typedef FunctorBase<Fsa::ConstAutomatonRef> Precursor;

    public:
        FinalizationFunctor(AllophoneStateGraphBuilder&   builder,
                            const std::string&            id,
                            const Fsa::ConstAutomatonRef& builderInput)
                : FunctorBase<Fsa::ConstAutomatonRef>(builder, id, builderInput) {}

        AllophoneStateGraphRef build() {
            return Precursor::builder_.finalizeTransducer(std::get<0>(Precursor::builderInput_));
        }
    };

    /**
     *   Converts a call to one of the buildTransducer functions into a functor.
     *
     *   Underlying function: buildTransducer
     *   Input: BuilderInput
     *   Ouptut: Fsa::ConstAutomatonRef
     */
    template<class... BuilderInput>
    struct TransducerFunctor : public FunctorBase<BuilderInput...> {
    private:
        typedef FunctorBase<BuilderInput...> Precursor;

    public:
        TransducerFunctor(AllophoneStateGraphBuilder& builder,
                          const std::string&          id,
                          BuilderInput... builderInput)
                : FunctorBase<BuilderInput...>(builder, id, builderInput...) {}

        Fsa::ConstAutomatonRef build() {
            return Precursor::builder_.buildTransducer(Precursor::builderInput_);
        }
    };

private:
    Bliss::LexiconRef                        lexicon_;
    Bliss::OrthographicParser*               orthographicParser_;
    Core::Ref<Fsa::StaticAutomaton>          lemmaPronunciationToLemmaTransducer_;
    Core::Ref<Fsa::StaticAutomaton>          phonemeToLemmaPronunciationTransducer_;
    Core::Ref<Fsa::StaticAutomaton>          allophoneStateToPhonemeTransducer_;
    Fsa::ConstAutomatonRef                   singlePronunciationAllophoneStateToPhonemeTransducer_;
    Core::XmlChannel                         modelChannel_;
    std::vector<const Bliss::Pronunciation*> silencesAndNoises_;

private:
    Bliss::OrthographicParser&      orthographicParser();
    Core::Ref<Fsa::StaticAutomaton> lemmaPronunciationToLemmaTransducer();
    Core::Ref<Fsa::StaticAutomaton> phonemeToLemmaPronunciationTransducer();
    Core::Ref<Fsa::StaticAutomaton> allophoneStateToPhonemeTransducer();
    Fsa::ConstAutomatonRef          singlePronunciationAllophoneStateToPhonemeTransducer();

    Fsa::ConstAutomatonRef createAlignmentGraph(const Alignment&);
    AllophoneStateGraphRef build(const Alignment& alignment, AllophoneStateGraphRef);

protected:
    Core::Ref<const Am::AcousticModel> acousticModel_;

    bool flatModelAcceptor_; // true: single path only
    u32  minDuration_; // minimum duration

protected:
    // compose-builds allophone-state transducer from lemma acceptor (no additional transitions)
    Fsa::ConstAutomatonRef buildFlatTransducer(Fsa::ConstAutomatonRef lemmaAcceptor);
    // finalize the built transducer
    Fsa::ConstAutomatonRef finishTransducer(Fsa::ConstAutomatonRef model);

public:
    AllophoneStateGraphBuilder(
            const Core::Configuration&         config,
            Core::Ref<const Bliss::Lexicon>    lexicon,
            Core::Ref<const Am::AcousticModel> acousticModel,
            bool                               flatModelAcceptor);

    virtual ~AllophoneStateGraphBuilder();

    void addSilenceOrNoise(const Bliss::Pronunciation*);
    void addSilenceOrNoise(const Bliss::Lemma* lemma);
    void setSilencesAndNoises(const std::vector<std::string>&);

    /** Builds allophone state acceptor from an orthography. */
    AllophoneStateGraphRef build(const std::string& orth);
    AllophoneStateGraphRef build(std::tuple<const std::string&> const& t) {
        return build(std::get<0>(t));
    };

    AllophoneStateGraphRef build(const std::string& orth, const std::string& leftContextOrth, const std::string& rightContextOrth);
    AllophoneStateGraphRef build(std::tuple<const std::string&, const std::string&, const std::string&> const& t) {
        return build(std::get<0>(t), std::get<1>(t), std::get<2>(t));
    }

    Functor<const std::string> createFunctor(const std::string& id, const std::string& orth) {
        return Functor<const std::string>(*this, id, orth);
    }
    Functor<const std::string, const std::string, const std::string> createFunctor(const std::string& id,
                                                                                   const std::string& orth,
                                                                                   const std::string& leftContextOrth,
                                                                                   const std::string& rightContextOrth) {
        return Functor<const std::string, const std::string, const std::string>(*this, id, orth, leftContextOrth, rightContextOrth);
    }
    Functor<const std::string> createFunctor(const Bliss::SpeechSegment& s) {
        return Functor<const std::string>(*this, s.fullName(), s.orth());
    }
    /** Builds allophone state acceptor for phoneme loops, cf. phoneme recognition. */
    enum InputLevel { lemma,
                      phone };
    AllophoneStateGraphRef build(const InputLevel& level);
    AllophoneStateGraphRef build(std::tuple<const InputLevel&> const& t) {
        return build(std::get<0>(t));
    };
    Functor<const InputLevel> createFunctor(const std::string& id, const InputLevel& level) {
        return Functor<const InputLevel>(*this, id, level);
    }
    /** Builds allophone state acceptor from a (non-coarticulated) pronunciation. */
    AllophoneStateGraphRef build(const Bliss::Pronunciation& p) {
        return build(Bliss::Coarticulated<Bliss::Pronunciation>(p));
    }
    Functor<const Bliss::Pronunciation&> createFunctor(const Bliss::Pronunciation& p) {
        Bliss::Coarticulated<Bliss::Pronunciation> cp(p);
        return Functor<const Bliss::Pronunciation&>(*this, cp.format(lexicon_->phonemeInventory()), p);
    }

    /** Builds allophone state acceptor from a coarticulated pronunciation. */
    AllophoneStateGraphRef build(const Bliss::Coarticulated<Bliss::Pronunciation>&);
    AllophoneStateGraphRef build(std::tuple<const Bliss::Coarticulated<Bliss::Pronunciation>&> t) {
        return build(std::get<0>(t));
    };
    Functor<const Bliss::Coarticulated<Bliss::Pronunciation>> createFunctor(
            const Bliss::Coarticulated<Bliss::Pronunciation>& p) {
        return Functor<const Bliss::Coarticulated<Bliss::Pronunciation>>(*this, p.format(lexicon_->phonemeInventory()), p);
    }

    AllophoneStateGraphRef    build(const Alignment&);
    Functor<const Alignment&> createFunctor(const std::string& id, const Alignment& a) {
        return Functor<const Alignment&>(*this, id, a);
    }
    /**
     *  Accelerated way of creating an alignment allophone state graph.
     *  Pronuniciation restricts the allophone state graph with which the
     *  alignment graph is composed.
     */
    AllophoneStateGraphRef build(const Alignment&, const Bliss::Coarticulated<Bliss::Pronunciation>&);

    /** Builds a allophone state to lemma pronunciation transducer from orthography. */
    Fsa::ConstAutomatonRef buildTransducer(const std::string& orth);
    Fsa::ConstAutomatonRef buildTransducer(std::string const& orth, std::string const& leftContextOrth, std::string const& rightContextOrth);

    // helpers for tuple
    Fsa::ConstAutomatonRef buildTransducer(std::tuple<const std::string&> const& t) {
        return buildTransducer(std::get<0>(t));
    };
    Fsa::ConstAutomatonRef buildTransducer(std::tuple<std::string const&, std::string const&, std::string const&> const& t) {
        return buildTransducer(std::get<0>(t), std::get<1>(t), std::get<2>(t));
    };

    TransducerFunctor<const std::string> createTransducerFunctor(const std::string& id, const std::string& orth) {
        return TransducerFunctor<const std::string>(*this, id, orth);
    }
    TransducerFunctor<const std::string, const std::string, const std::string> createTransducerFunctor(const std::string& id,
                                                                                                       const std::string& orth,
                                                                                                       const std::string& leftContextOrth,
                                                                                                       const std::string& rightContextOrth) {
        return TransducerFunctor<const std::string, const std::string, const std::string>(*this, id, orth, leftContextOrth, rightContextOrth);
    }

    /** Creates a static epsilon free acceptor from the input transducer. */
    AllophoneStateGraphRef finalizeTransducer(Fsa::ConstAutomatonRef);
    FinalizationFunctor    createFinalizationFunctor(const std::string& id, Fsa::ConstAutomatonRef t) {
        return FinalizationFunctor(*this, id, t);
    }

    /** Builds allophone state acceptor from a lemma accertor. */
    AllophoneStateGraphRef build(Fsa::ConstAutomatonRef);

    /** builds the final transducer
     *  forces concrete behavior in derived classes (mostly specific topology upon flat automaton)
     */
    virtual Fsa::ConstAutomatonRef buildTransducer(Fsa::ConstAutomatonRef) = 0;
};

// -------- Graphs of various topologies --------

class HMMTopologyGraphBuilder : public AllophoneStateGraphBuilder {
    typedef AllophoneStateGraphBuilder Precursor;

public:
    HMMTopologyGraphBuilder(
            const Core::Configuration&         config,
            Core::Ref<const Bliss::Lexicon>    lexicon,
            Core::Ref<const Am::AcousticModel> acousticModel,
            bool                               flatModelAcceptor) :
        Precursor(config, lexicon, acousticModel, flatModelAcceptor) {}

    // further apply transition model (loop, skip + weights)
    Fsa::ConstAutomatonRef buildTransducer(Fsa::ConstAutomatonRef);

private:
    Fsa::ConstAutomatonRef applyMinimumDuration(Fsa::ConstAutomatonRef);
};


class CTCTopologyGraphBuilder : public AllophoneStateGraphBuilder {
    typedef AllophoneStateGraphBuilder Precursor;

public:
    CTCTopologyGraphBuilder(
            const Core::Configuration&         config,
            Core::Ref<const Bliss::Lexicon>    lexicon,
            Core::Ref<const Am::AcousticModel> acousticModel,
            bool                               flatModelAcceptor);

    // further add label loop and blank (no weights)
    Fsa::ConstAutomatonRef buildTransducer(Fsa::ConstAutomatonRef);

protected:
    Fsa::LabelId blankId_;
    bool         labelLoop_;

    virtual void addBlank(Core::Ref<Fsa::StaticAutomaton>&, Fsa::StateId, std::deque<Fsa::StateId>&);

private:
    bool transitionChecked_;
    Fsa::StateId finalStateId_;
    Fsa::LabelId silenceId_;

    void checkTransitionModel();
};


class RNATopologyGraphBuilder : public CTCTopologyGraphBuilder {
    typedef CTCTopologyGraphBuilder Precursor;

public:
    RNATopologyGraphBuilder(
            const Core::Configuration&         config,
            Core::Ref<const Bliss::Lexicon>    lexicon,
            Core::Ref<const Am::AcousticModel> acousticModel,
            bool                               flatModelAcceptor):
        Precursor(config, lexicon, acousticModel, flatModelAcceptor) { labelLoop_ = false; }

protected:
    // no label loop
    void addBlank(Core::Ref<Fsa::StaticAutomaton>&, Fsa::StateId, std::deque<Fsa::StateId>&);
};

}  // namespace Speech

#endif  // _SPEECH_ALLOPHONE_STATE_GRAPH_BUILDER_HH
