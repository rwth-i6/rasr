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
#ifndef _SEARCH_CONTEXT_TRANSDUCER_BUILDER_HH
#define _SEARCH_CONTEXT_TRANSDUCER_BUILDER_HH

#include <Am/ClassicAcousticModel.hh>
#include <Bliss/Lexicon.hh>
#include <Core/Component.hh>
#include <OpenFst/Types.hh>

namespace Search {
namespace Wfst {

class ContextTransducerBuilder : public Core::Component {
private:
    class Builder;
    class AcrossWordBuilder;
    class NonDeterministicBuilder;
    class WithinWordBuilder;
    class MonophoneBuilder;

    static const Core::ParameterBool   paramNonDeterministic;
    static const Core::ParameterBool   paramAllowNonCrossWordTransitions;
    static const Core::ParameterString paramSequenceEndSymbol;
    static const Core::ParameterBool   paramUseSentenceEndSymbol;
    static const Core::ParameterBool   paramAddWordDisambiguatorLoops;
    static const Core::ParameterBool   paramAddSuperFinalState;
    static const Core::ParameterBool   paramExploitDisambiguators;
    static const Core::ParameterBool   paramUnshiftCiPhones;
    static const Core::ParameterBool   paramAddNonWords;
    static const Core::ParameterBool   paramFinalCiLoop;
    static const Core::ParameterBool   paramMonophones;

public:
    ContextTransducerBuilder(const Core::Configuration& c, Core::Ref<const Am::AcousticModel> m, Core::Ref<const Bliss::Lexicon> l)
            : Core::Component(c),
              model_(m),
              lexicon_(l),
              phoneSymbols_(0),
              initialPhoneOffset_(-1),
              disambiguatorOffset_(-1),
              nDisambiguators_(-1),
              newWordLabelOffset_(-1),
              newDisambiguatorOffset_(-1) {}

    void setDisambiguators(u32 nDisambiguators, u32 disambiguatorOffset) {
        nDisambiguators_     = nDisambiguators;
        disambiguatorOffset_ = disambiguatorOffset;
    }
    void setWordDisambiguators(u32 nWordDisambiguators) {
        nWordDisambiguators_ = nWordDisambiguators;
    }
    void setInitialPhoneOffset(u32 offset) {
        initialPhoneOffset_ = offset;
    }
    void setPhoneSymbols(const OpenFst::SymbolTable* symbols) {
        phoneSymbols_ = symbols;
    }
    // return the word label offset in the input alphabet
    u32 getWordLabelOffset() const {
        return newWordLabelOffset_;
    }
    // return the disambiguator offset in the input alphabet
    u32 getDisambiguatorOffset() const {
        return newDisambiguatorOffset_;
    }

    OpenFst::VectorFst* build();

private:
    Core::Ref<const Am::AcousticModel> model_;
    Core::Ref<const Bliss::Lexicon>    lexicon_;
    const OpenFst::SymbolTable*        phoneSymbols_;
    s32                                initialPhoneOffset_, disambiguatorOffset_;
    s32                                nDisambiguators_;
    s32                                nWordDisambiguators_;
    s32                                newWordLabelOffset_, newDisambiguatorOffset_;
};

}  // namespace Wfst
}  // namespace Search

#endif /* _SEARCH_CONTEXT_TRANSDUCER_BUILDER_HH */
