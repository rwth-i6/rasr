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

#include <string>
#include <vector>

#include <Core/Component.hh>
#include <Core/ReferenceCounting.hh>
#include <Core/Types.hh>

namespace Tokenizer {

/*
 * Base class for subword tokenizers (BPE, SentencePiece, ...).
 *
 * Converts text to token sequences and back. Tokens are available both as strings
 * (e.g. to look them up as lemmas in a lexicon) and as ids of the tokenizer's own vocabulary.
 */
class AbstractTokenizer : public virtual Core::Component,
                          public Core::ReferenceCounted {
public:
    typedef u32              TokenId;
    static constexpr TokenId invalidTokenId = Core::Type<TokenId>::max;

    AbstractTokenizer(Core::Configuration const& config);
    virtual ~AbstractTokenizer() = default;

    virtual std::vector<std::string> tokenize(std::string const& text) const                  = 0;
    virtual std::string              detokenize(std::vector<std::string> const& tokens) const = 0;

    // Valid ids are in [0, vocabularySize())
    virtual u32 vocabularySize() const = 0;

    // Returns `invalidTokenId` if `token` is not part of the vocabulary
    virtual TokenId tokenStrToId(std::string const& token) const = 0;

    // Raises an error and returns an empty string for invalid ids
    virtual std::string tokenIdToStr(TokenId id) const = 0;

    // Default: Apply `tokenize` then `tokenStrToId` to each item
    virtual std::vector<TokenId> encode(std::string const& text) const;

    // Default: Apply `tokenIdToStr` to each item then `detokenize`
    virtual std::string decode(std::vector<TokenId> const& ids) const;

protected:
    // Raises an error if `id` is not in [0, vocabularySize())
    bool checkTokenId(TokenId id) const;
};

typedef Core::Ref<AbstractTokenizer> TokenizerRef;

}  // namespace Tokenizer

#endif  // _TOKENIZER_ABSTRACT_TOKENIZER_HH
