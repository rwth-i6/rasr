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
 * The optional `vocab-file` defines the token ids. It contains either one token per line (further columns
 * such as counts are ignored) or a RETURNN-style python dict literal {'<s>': 0, 'lo@@': 1, ...}.
 * When encoding, out-of-vocabulary tokens are mapped to `unknown-token` if it is set.
 */
class BpeTokenizer : public AbstractTokenizer {
public:
    using Precursor = AbstractTokenizer;

    static const Core::ParameterString paramCodesFile;
    static const Core::ParameterString paramVocabFile;
    static const Core::ParameterString paramSeparator;
    static const Core::ParameterString paramUnknownToken;

    BpeTokenizer(Core::Configuration const& config);

    std::vector<std::string> tokenize(std::string const& text) const override;
    std::string              detokenize(std::vector<std::string> const& tokens) const override;

    u32         vocabularySize() const override;
    TokenId     tokenStrToId(std::string const& token) const override;
    std::string tokenIdToStr(TokenId id) const override;

    std::vector<TokenId> encode(std::string const& text) const override;

private:
    std::string const separator_;

    // Codes version 0.2 attaches the end-of-word marker to the last character, version 0.1 appends it as separate symbol
    bool                                 endOfWordAttached_;
    std::unordered_map<std::string, u32> mergeRanks_;  // Key: "<left> <right>"

    std::unordered_map<std::string, TokenId> tokenIds_;
    std::vector<std::string>                 tokens_;  // Indexed by id
    TokenId                                  unknownTokenId_;

    void loadCodes(std::string const& filename);
    void loadVocabulary(std::string const& filename);
    void addToken(std::string const& token, TokenId id);

    std::vector<std::string> segmentWord(std::string const& word) const;
};

}  // namespace Tokenizer

#endif  // _TOKENIZER_BPE_TOKENIZER_HH
