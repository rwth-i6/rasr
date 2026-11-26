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
#include <Core/Status.hh>
#include <Core/Types.hh>

namespace Tokenizer {

/*
 * Base class for subword tokenizers (BPE, SentencePiece, ...).
 *
 * Converts text to token sequences and back. Tokens are available both as strings
 * (e.g. to look them up as lemmas in a lexicon) and as ids of the tokenizer's own vocabulary.
 *
 * Methods that can fail return a status and write their result to the last argument, whose content is unspecified if
 * the status is not ok. Invalid token ids result in `OutOfRange`.
 */
class AbstractTokenizer : public virtual Core::Component {
public:
    typedef u32 TokenId;

    /*
     * Word boundaries that a token marks before and after itself. A word boundary lies between two consecutive tokens
     * if the first one marks it after itself or the second one before itself. The start and end of a token sequence
     * are always word boundaries.
     *
     * Most tokenizers mark only one side, so false means that the token does not mark a boundary, not that there is
     * none. E.g. the BPE token "wer" in "lo@@ wer" marks a boundary after itself but not whether a word starts before
     * it, which is only shown by the previous token.
     *
     * `whitespace` is set for tokens that decode to only whitespace or nothing on their own, e.g. "Ġ" of byte-level BPE
     * or a lone "▁" of SentencePiece, which `encode` puts next to user-defined symbols. Such tokens can form words of
     * their own that have no text. Special tokens are never whitespace.
     */
    struct WordBoundaryMarks {
        bool before;
        bool after;
        bool whitespace = false;
    };

    AbstractTokenizer(Core::Configuration const& config);
    virtual ~AbstractTokenizer() = default;

    virtual Core::Status encode(std::string const& text, std::vector<TokenId>& ids) const = 0;
    virtual Core::Status decode(std::vector<TokenId> const& ids, std::string& text) const = 0;

    // Valid ids are in [0, size)
    virtual Core::Status vocabularySize(u32& size) const = 0;

    // `NotFound` if `token` is not part of the vocabulary
    virtual Core::Status tokenStrToId(std::string const& token, TokenId& id) const = 0;

    virtual Core::Status tokenIdToStr(TokenId id, std::string& token) const            = 0;
    virtual Core::Status wordBoundaryMarks(TokenId id, WordBoundaryMarks& marks) const = 0;

    // Default: `encode` then `tokenIdToStr` for each id
    virtual Core::Status tokenize(std::string const& text, std::vector<std::string>& tokens) const;

    // Default: `tokenStrToId` for each token then `decode`
    virtual Core::Status detokenize(std::vector<std::string> const& tokens, std::string& text) const;

protected:
    // `OutOfRange` if `id` is not below the vocabulary size
    Core::Status checkTokenId(TokenId id) const;
};

typedef std::shared_ptr<AbstractTokenizer> TokenizerRef;

}  // namespace Tokenizer

#endif  // _TOKENIZER_ABSTRACT_TOKENIZER_HH
