/** Copyright 2026 RWTH Aachen University. All rights reserved.
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

#ifndef WORD_ASSEMBLER_HH
#define WORD_ASSEMBLER_HH

#include <string>
#include <unordered_map>
#include <vector>

#include <Bliss/Lexicon.hh>
#include <Core/Component.hh>
#include <Core/Parameter.hh>
#include <Nn/LabelScorer/Types.hh>

namespace Search {

/*
 * Assembles words from word-piece labels, whose text is the orthography of the corresponding lemma.
 * Spellings are interned, so a pending word is a small id.
 */
class WordAssembler : public Core::Component {
public:
    enum Convention {
        WordStartMarked,     // A piece starting with the marker begins a word, e.g. SentencePiece "▁"
        ContinuationMarked,  // A piece ending with the marker continues its word, e.g. BPE "@@"
    };

    static const Core::Choice          choiceConvention;
    static const Core::ParameterChoice paramConvention;
    static const Core::ParameterString paramMarker;

    typedef u32 WordId;

    // Id of the empty spelling, i.e. of no word
    static constexpr WordId noWord = 0u;

    struct Step {
        WordId finished;  // Word finished by the piece, if any
        WordId pending;   // Word pending after the piece
    };

    WordAssembler(Core::Configuration const& config);

    void setLexicon(Bliss::LexiconRef lexicon);

    // Append the piece of `label` to `pending`
    Step extend(WordId pending, Nn::LabelIndex label);

    std::string const& spelling(WordId word) const {
        return *spellings_[word];
    }

private:
    struct Piece {
        std::string text;    // Without the marker
        bool        marked;  // Whether the text carried the marker
    };

    Convention         convention_;
    std::string        marker_;
    std::vector<Piece> pieces_;

    std::unordered_map<std::string, WordId> wordIds_;
    std::vector<std::string const*>         spellings_;  // Keys of `wordIds_`
    std::unordered_map<u64, Step>           steps_;      // (pending << 32 | label) -> step

    WordId intern(std::string const& spelling);
    Step   computeStep(WordId pending, Piece const& piece);
};

}  // namespace Search

#endif  // WORD_ASSEMBLER_HH
