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

#ifndef _TOKENIZER_BPE_TOKENIZER_HH
#define _TOKENIZER_BPE_TOKENIZER_HH

#include <unordered_map>

#include <Bliss/ReturnnVocabulary.hh>

#include "AbstractTokenizer.hh"

namespace Tokenizer {

/*
 * Byte-pair encoding as applied by subword-nmt (https://github.com/rwth-i6/subword-nmt).
 *
 * Text is split at whitespace (as python's `str.split()`, e.g. blanks, tabs, newlines and non-breaking spaces) and
 * each word is segmented by applying the merge operations from `codes-file` in order of priority. All subwords except
 * the last one of a word get the `separator` suffix, e.g. "lower" -> "lo@@ wer". `detokenize` joins words with single
 * blanks.
 *
 * `vocab-file` defines the token ids as a RETURNN-style python dict literal {'<s>': 0, 'lo@@': 1, ...}, see
 * `Bliss::ReturnnVocabulary`. As in RETURNN
 * and `apply_bpe.py --vocabulary`, subwords that are not in the vocabulary are split by reversing their merges until
 * they are in the vocabulary or cannot be split further. When encoding, the remaining out-of-vocabulary tokens are
 * mapped to `unknown-token` if it is set, otherwise they result in `NotFound`.
 */
class BpeTokenizer : public AbstractTokenizer {
public:
    using Precursor = AbstractTokenizer;

    static const Core::ParameterString paramCodesFile;
    static const Core::ParameterString paramVocabFile;
    static const Core::ParameterString paramSeparator;
    static const Core::ParameterString paramUnknownToken;

    static Core::Status create(Core::Configuration const& config, TokenizerRef& tokenizer);

    Core::Status encode(std::string const& text, std::vector<TokenId>& ids) const override;
    Core::Status decode(std::vector<TokenId> const& ids, std::string& text) const override;

    Core::Status vocabularySize(u32& size) const override;
    Core::Status tokenStrToId(std::string const& token, TokenId& id) const override;
    Core::Status tokenIdToStr(TokenId id, std::string& token) const override;

    // All subwords except the last one of a word end with the separator, so only boundaries after tokens are marked
    Core::Status wordBoundaryMarks(TokenId id, WordBoundaryMarks& marks) const override;

    // Also for tokens that are not part of the vocabulary
    Core::Status tokenize(std::string const& text, std::vector<std::string>& tokens) const override;
    Core::Status detokenize(std::vector<std::string> const& tokens, std::string& text) const override;

private:
    std::string const separator_;

    // Codes version 0.2 attaches the end-of-word marker to the last character, version 0.1 appends it as separate symbol
    bool                                 endOfWordAttached_;
    std::unordered_map<std::string, u32> mergeRanks_;  // Key: "<left> <right>"

    // Merged symbol to the pair it was merged from, e.g. "er</w>" to "e" and "r</w>"
    std::unordered_map<std::string, std::pair<std::string, std::string>> mergedPairs_;

    Bliss::ReturnnVocabulary vocabulary_;
    TokenId                  unknownTokenId_;

    BpeTokenizer(Core::Configuration const& config);

    Core::Status load();
    Core::Status loadCodes(std::string const& filename);

    std::vector<std::string> segmentWord(std::string const& word) const;

    // Appends the subword or, if it is not in the vocabulary, the parts it was merged from, which are split further if
    // needed. `final` is true for the last subword of a word, which has no separator.
    void splitIntoVocabulary(std::string const& subword, bool final, std::vector<std::string>& result) const;
};

}  // namespace Tokenizer

#endif  // _TOKENIZER_BPE_TOKENIZER_HH
