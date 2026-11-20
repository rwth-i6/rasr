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

#ifndef _TOKENIZER_ABSTRACT_TOKENIZER_HH
#define _TOKENIZER_ABSTRACT_TOKENIZER_HH

#include <memory>
#include <string>
#include <vector>

#include <Core/Component.hh>
#include <Core/Expected.hh>
#include <Core/Types.hh>

namespace Tokenizer {

/*
 * Base class for subword tokenizers (BPE, SentencePiece, ...).
 *
 * Converts text to token sequences and back. Tokens are available both as strings
 * (e.g. to look them up as lemmas in a lexicon) and as ids of the tokenizer's own vocabulary.
 *
 * Methods that can fail return a `Core::Expected`. Invalid token ids result in `OutOfRange`.
 */
class AbstractTokenizer : public virtual Core::Component {
public:
    typedef u32              TokenId;
    static constexpr TokenId invalidTokenId = Core::Type<TokenId>::max;

    /*
     * Word boundaries that a token marks before and after itself. A word boundary lies between two consecutive tokens
     * if the first one marks it after itself or the second one before itself. The start and end of a token sequence
     * are always word boundaries.
     *
     * Most tokenizers mark only one side, so false means that the token does not mark a boundary, not that there is
     * none. E.g. the BPE token "wer" in "lo@@ wer" marks a boundary after itself but not whether a word starts before
     * it, which is only shown by the previous token.
     */
    struct WordBoundaryMarks {
        bool before;
        bool after;
    };

    AbstractTokenizer(Core::Configuration const& config);
    virtual ~AbstractTokenizer() = default;

    virtual Core::Expected<std::vector<TokenId>> encode(std::string const& text) const         = 0;
    virtual Core::Expected<std::string>          decode(std::vector<TokenId> const& ids) const = 0;

    // Valid ids are in [0, vocabularySize())
    virtual u32 vocabularySize() const = 0;

    // Returns `invalidTokenId` if `token` is not part of the vocabulary
    virtual TokenId tokenStrToId(std::string const& token) const = 0;

    virtual Core::Expected<std::string>       tokenIdToStr(TokenId id) const      = 0;
    virtual Core::Expected<WordBoundaryMarks> wordBoundaryMarks(TokenId id) const = 0;

    // Default: `encode` then `tokenIdToStr` for each id
    virtual Core::Expected<std::vector<std::string>> tokenize(std::string const& text) const;

    // Default: `tokenStrToId` for each token then `decode`. Tokens that are not part of the vocabulary result in `NotFound`.
    virtual Core::Expected<std::string> detokenize(std::vector<std::string> const& tokens) const;

protected:
    // `OutOfRange` if `id` is not in [0, vocabularySize())
    Core::Status checkTokenId(TokenId id) const;
};

typedef std::shared_ptr<AbstractTokenizer> TokenizerRef;

}  // namespace Tokenizer

#endif  // _TOKENIZER_ABSTRACT_TOKENIZER_HH
