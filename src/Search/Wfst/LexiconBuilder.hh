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
#ifndef _SEARCH_LEXICON_BUILDER_HH
#define _SEARCH_LEXICON_BUILDER_HH

#include <Bliss/Lexicon.hh>
#include <Core/Component.hh>
#include <Core/Hash.hh>
#include <OpenFst/Types.hh>

namespace Search {
namespace Wfst {

class NonWordTokens;

/**
 * Construction of the lexicon transducer.
 * For each lemma pronunciation a separate path from the initial
 * state to a final state is created.
 * The lexicon is not closed, i.e. no transitions from final states
 * to the initial state are created.
 * The closure is added using close().
 * Closure can be done using either only a epsilon transition or
 * additional transitions for all lemma pronunciation with an empty
 * syntactic token sequence (silence and noise).
 *
 * Options:
 *  add empty tokens:
 *    add lemma pronunciations with empty syntactic token
 *    sequence to the lexicon transducer.
 *  mark initial phones:
 *    add an offset to all input labels (phonemes) leaving the
 *    initial state. required for the context dependency transducer
 *    construction using across-word models.
 *  add word boundary phones:
 *    add input labels word phones at the initial and final position of a
 *    pronunciation
 *  add word disambiguators:
 *    add a transition after each pronunciation with
 *    input label = wordLabelOffset() + lemma pronunciation id.
 *    intended for adding "word" labels at the word end.
 *  disambiguate homophones:
 *    add disambiguators for homophones. not required if
 *    word disambiguators are used. disambiguators are in the range
 *      disambiguatorOffset() .. disambiguatorOffset() + nPhoneDisambiguators()
 *
 */
class LexiconBuilder : public Core::Component {
public:
    LexiconBuilder(const Core::Configuration& c, const Bliss::Lexicon& lexicon);
    ~LexiconBuilder();

    /**
     * construct the lexicon transducer.
     */
    OpenFst::VectorFst* build(bool buildClosed);

    /**
     * build the closure.
     * requires initialPhoneOffset_
     */
    void close(OpenFst::VectorFst* lexiconTransducer, bool useEmptyTokens);

    void setGrammarDisambiguators(s32 nDisambiguators) {
        nGrammarDisambiguators_ = nDisambiguators;
    }

    bool addWordDisambiguators() const;

    s32 nPhoneDisambiguators() const {
        verify(inputSymbols_);
        return inputSymbols_->AvailableKey() - disambiguatorOffset();
    }
    void setInitialPhoneOffset(int offset) {
        initialPhoneOffset_ = offset;
    }
    s32 initialPhoneOffset() const {
        return initialPhoneOffset_;
    }
    void setWordLabelOffset(int offset) {
        wordLabelOffset_ = offset;
    }
    s32 wordLabelOffset() const {
        return wordLabelOffset_;
    }
    void setDisambiguatorOffset(int offset) {
        disambiguatorOffset_ = offset;
    }
    s32 disambiguatorOffset() const {
        return disambiguatorOffset_;
    }

    /**
     * create input (phones) and output (pronunciation) symbol tables.
     */
    void createSymbolTables();

    const OpenFst::SymbolTable* inputSymbols() const {
        return inputSymbols_;
    }
    const OpenFst::SymbolTable* outputSymbols() const {
        return outputSymbols_;
    }
    static std::string phoneDisambiguatorSymbol(u32 disambiguator);

private:
    struct Options;
    typedef std::unordered_map<const Bliss::Pronunciation*, int,
                               Bliss::Pronunciation::Hash,
                               Bliss::Pronunciation::Equality>
            PronunciationHashMap;

    void addPronunciation(const Bliss::LemmaPronunciation* pron,
                          OpenFst::Label output, bool close);

    OpenFst::Label inputLabel(Bliss::Phoneme::Id phone,
                              bool initial = false, bool final = false) const;
    void           addDisambiguatorLoop();
    void           addNonWordLoop(OpenFst::StateId s, OpenFst::Weight w);
    void           addOptionalNonWordArcs(OpenFst::StateId from, OpenFst::StateId to);
    void           addWordDisambiguatorLabels();
    void           addBoundaryPhoneLabels(bool addFinal);
    void           addSentenceEnd(OpenFst::Label output, bool close);
    void           logSettings(bool buildClosed) const;
    OpenFst::Label phoneDisambiguator(u32 disambiguator);
    void           getSentenceEnd();
    void           getPronsWithoutNonWords(const std::vector<std::string>& lemmas);

    const Options*           options_;
    OpenFst::VectorFst*      result_;
    OpenFst::SymbolTable *   inputSymbols_, *outputSymbols_;
    NonWordTokens*           nonWordTokens_;
    const Bliss::Lexicon&    lexicon_;
    s32                      nGrammarDisambiguators_;
    s32                      phoneDisambiguatorOffset_;
    s32                      nPhoneDisambiguators_;
    s32                      initialPhoneOffset_;
    s32                      wordLabelOffset_;
    s32                      disambiguatorOffset_;
    s32                      sentenceEndLemma_;
    OpenFst::Label           silencePhone_;
    OpenFst::StateId         initialState_;
    std::set<OpenFst::Label> pronsWithoutNonWords_;
    PronunciationHashMap     homophones_;

public:
    static const char *initialSuffix, *finalSuffix;
    static const char* sentenceEndSymbol;
};

}  // namespace Wfst
}  // namespace Search

#endif /* _SEARCH_LEXICON_BUILDER_HH */
