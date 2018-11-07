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
#ifndef _SPEECH_ALLOPHONE_STATE_GRAPH_BUILDER_HH
#define _SPEECH_ALLOPHONE_STATE_GRAPH_BUILDER_HH

#include <Core/Component.hh>
#include <Bliss/Lexicon.hh>
#include <Bliss/CorpusDescription.hh>
#include <Fsa/Automaton.hh>
#include <Fsa/Static.hh>

namespace Am {
    class AcousticModel;
}

namespace Bliss {
    class PhonemeToLemmaTransducer;
    class OrthographicParser;
}

namespace Speech {
    class ModelCombination;
    class Alignment;
}

namespace Speech {

    typedef Fsa::Automaton::ConstRef AllophoneStateGraphRef;

    /**
     *  AllophoneStateGraphBuilder
     */
    class AllophoneStateGraphBuilder : public Core::Component, public Core::ReferenceCounted
    {
        typedef Core::Component Precursor;
    public:
        /**
         *   Base class for build-functor classes.
         */
        template<class ...BuilderInput>
        class FunctorBase {
        protected:
            AllophoneStateGraphBuilder &builder_;
            const std::string id_;
            const std::tuple<BuilderInput...> builderInput_;
        public:
            FunctorBase(AllophoneStateGraphBuilder &builder,
                        const std::string &id,
                        BuilderInput... builderInput) :
                builder_(builder), id_(id), builderInput_(builderInput...) {}

            const std::string &id() const { return id_; }
        };

        /**
         *   Converts a call to one of the build functions into a functor.
         *
         *   Underlying function: build
         *   Input: template BuilderInput
         *   Ouptut: AllophoneStateGraphRef
         */
        template<class ...BuilderInput>
        struct Functor : public FunctorBase<BuilderInput...> {
        private:
            typedef FunctorBase<BuilderInput...> Precursor;
        public:
            Functor(AllophoneStateGraphBuilder &builder,
                    const std::string &id,
                    BuilderInput... builderInput) :
                FunctorBase<BuilderInput...>(builder, id, builderInput...) {}

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
            FinalizationFunctor(AllophoneStateGraphBuilder &builder,
                                const std::string &id,
                                const Fsa::ConstAutomatonRef &builderInput) :
                FunctorBase<Fsa::ConstAutomatonRef>(builder, id, builderInput) {}

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
        template<class ...BuilderInput>
        struct TransducerFunctor : public FunctorBase<BuilderInput...> {
        private:
            typedef FunctorBase<BuilderInput...> Precursor;
        public:
            TransducerFunctor(AllophoneStateGraphBuilder &builder,
                              const std::string &id,
                              BuilderInput... builderInput) :
                FunctorBase<BuilderInput...>(builder, id, builderInput...) {}

            Fsa::ConstAutomatonRef build() {
                return Precursor::builder_.buildTransducer(Precursor::builderInput_);
            }
        };
    private:
        Bliss::LexiconRef lexicon_;
        Core::Ref<const Am::AcousticModel> acousticModel_;
        Bliss::OrthographicParser *orthographicParser_;
        Core::Ref<Fsa::StaticAutomaton> lemmaPronunciationToLemmaTransducer_;
        Core::Ref<Fsa::StaticAutomaton> phonemeToLemmaPronunciationTransducer_;
        Core::Ref<Fsa::StaticAutomaton> allophoneStateToPhonemeTransducer_;
        Fsa::ConstAutomatonRef singlePronunciationAllophoneStateToPhonemeTransducer_;
        Core::XmlChannel modelChannel_;
        bool flatModelAcceptor_;
        std::vector<const Bliss::Pronunciation*> silencesAndNoises_;
    private:
        Bliss::OrthographicParser &orthographicParser();
        Core::Ref<Fsa::StaticAutomaton> lemmaPronunciationToLemmaTransducer();
        Core::Ref<Fsa::StaticAutomaton> phonemeToLemmaPronunciationTransducer();
        Core::Ref<Fsa::StaticAutomaton> allophoneStateToPhonemeTransducer();
        Fsa::ConstAutomatonRef singlePronunciationAllophoneStateToPhonemeTransducer();

        Fsa::ConstAutomatonRef createAlignmentGraph(const Alignment &);
        AllophoneStateGraphRef build(const Alignment &alignment, AllophoneStateGraphRef);
    public:
        AllophoneStateGraphBuilder(
            const Core::Configuration&,
            Core::Ref<const Bliss::Lexicon> lexicon,
            Core::Ref<const Am::AcousticModel> acousticModel,
            bool flatModelAcceptor = false);

        virtual ~AllophoneStateGraphBuilder();

        void addSilenceOrNoise(const Bliss::Pronunciation*);
        void addSilenceOrNoise(const Bliss::Lemma *lemma);
        void setSilencesAndNoises(const std::vector<std::string> &);

        /** Builds allophone state acceptor from an orthography. */
        AllophoneStateGraphRef build(const std::string &orth);
        AllophoneStateGraphRef build(std::tuple<const std::string&> const& t) { return build(std::get<0>(t)); };

        AllophoneStateGraphRef build(const std::string &orth, const std::string &leftContextOrth, const std::string &rightContextOrth);
        AllophoneStateGraphRef build(std::tuple<const std::string&, const std::string&, const std::string&> const& t) { return build(std::get<0>(t), std::get<1>(t), std::get<2>(t)); }

        Functor<const std::string> createFunctor(const std::string &id, const std::string &orth) {
            return Functor<const std::string>(*this, id, orth);
        }
        Functor<const std::string, const std::string, const std::string> createFunctor(const std::string &id,
                                                                                       const std::string &orth,
                                                                                       const std::string &leftContextOrth,
                                                                                       const std::string &rightContextOrth) {
            return Functor<const std::string, const std::string, const std::string>(*this, id, orth, leftContextOrth, rightContextOrth);
        }
        Functor<const std::string> createFunctor(const Bliss::SpeechSegment &s) {
            return Functor<const std::string>(*this, s.fullName(), s.orth());
        }
        /** Builds allophone state acceptor for phoneme loops, cf. phoneme recognition. */
        enum InputLevel { lemma, phone };
        AllophoneStateGraphRef build(const InputLevel &level);
        AllophoneStateGraphRef build(std::tuple<const InputLevel&> const& t) { return build(std::get<0>(t)); };
        Functor<const InputLevel> createFunctor(const std::string &id, const InputLevel &level) {
            return Functor<const InputLevel>(*this, id, level);
        }
        /** Builds allophone state acceptor from a (non-coarticulated) pronunciation. */
        AllophoneStateGraphRef build(const Bliss::Pronunciation &p) {
            return build(Bliss::Coarticulated<Bliss::Pronunciation>(p));
        }
        Functor<const Bliss::Pronunciation&> createFunctor(const Bliss::Pronunciation &p) {
            Bliss::Coarticulated<Bliss::Pronunciation> cp(p);
            return Functor<const Bliss::Pronunciation&>(*this, cp.format(lexicon_->phonemeInventory()), p);
        }

        /** Builds allophone state acceptor from a coarticulated pronunciation. */
        AllophoneStateGraphRef build(const Bliss::Coarticulated<Bliss::Pronunciation> &);
        AllophoneStateGraphRef build(std::tuple<const Bliss::Coarticulated<Bliss::Pronunciation>&> t) { return build(std::get<0>(t)); };
        Functor<const Bliss::Coarticulated<Bliss::Pronunciation>> createFunctor(
            const Bliss::Coarticulated<Bliss::Pronunciation> &p) {
            return Functor<const Bliss::Coarticulated<Bliss::Pronunciation>>(*this, p.format(lexicon_->phonemeInventory()), p);
        }

        AllophoneStateGraphRef build(const Alignment &);
        Functor<const Alignment&> createFunctor(const std::string &id, const Alignment &a) {
            return Functor<const Alignment&>(*this, id, a);
        }
        /**
         *  Accelerated way of creating an alignment allophone state graph.
         *  Pronuniciation restricts the allophone state graph with which the
         *  alignment graph is composed.
         */
        AllophoneStateGraphRef build(const Alignment &, const Bliss::Coarticulated<Bliss::Pronunciation> &);

        /** Builds a allophone state to lemma pronunciation transducer from orthography. */
        Fsa::ConstAutomatonRef buildTransducer(const std::string &orth);
        Fsa::ConstAutomatonRef buildTransducer(std::string const& orth, std::string const& leftContextOrth, std::string const& rightContextOrth);

        // helpers for tuple
        Fsa::ConstAutomatonRef buildTransducer(std::tuple<const std::string&> const& t) { return buildTransducer(std::get<0>(t)); };
        Fsa::ConstAutomatonRef buildTransducer(std::tuple<std::string const&, std::string const&, std::string const&> const& t) { return buildTransducer(std::get<0>(t), std::get<1>(t), std::get<2>(t)); };

        TransducerFunctor<const std::string> createTransducerFunctor(const std::string &id, const std::string &orth) {
            return TransducerFunctor<const std::string>(*this, id, orth);
        }
        TransducerFunctor<const std::string, const std::string, const std::string> createTransducerFunctor(const std::string &id,
                                                                                                           const std::string &orth,
                                                                                                           const std::string &leftContextOrth,
                                                                                                           const std::string &rightContextOrth) {
            return TransducerFunctor<const std::string, const std::string, const std::string>(*this, id, orth, leftContextOrth, rightContextOrth);
        }

        /** Creates a static epsilon free acceptor from the input transducer. */
        AllophoneStateGraphRef finalizeTransducer(Fsa::ConstAutomatonRef);
        FinalizationFunctor createFinalizationFunctor(const std::string &id, Fsa::ConstAutomatonRef t) {
            return FinalizationFunctor(*this, id, t);
        }

        /** Builds allophone state acceptor from a lemma accertor. */
        AllophoneStateGraphRef build(Fsa::ConstAutomatonRef);
        /** Builds a allophone state to lemma pronunciation transducer from lemma acceptor. */
        Fsa::ConstAutomatonRef buildTransducer(Fsa::ConstAutomatonRef);
    };

} // namespace Speech

#endif // _SPEECH_ALLOPHONE_STATE_GRAPH_BUILDER_HH
