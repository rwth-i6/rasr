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

#ifndef _TOKENIZER_SENTENCE_PIECE_TOKENIZER_HH
#define _TOKENIZER_SENTENCE_PIECE_TOKENIZER_HH

#include <sentencepiece_processor.h>

#include "AbstractTokenizer.hh"

namespace Tokenizer {

/*
 * Tokenizer based on a SentencePiece model (https://github.com/google/sentencepiece).
 * When encoding, pieces not in the model are mapped to the model's own unknown piece.
 */
class SentencePieceTokenizer : public AbstractTokenizer {
public:
    using Precursor = AbstractTokenizer;

    static const Core::ParameterString paramModelFile;

    SentencePieceTokenizer(Core::Configuration const& config);

    std::vector<std::string> tokenize(std::string const& text) const override;
    std::string              detokenize(std::vector<std::string> const& tokens) const override;

    u32         vocabularySize() const override;
    TokenId     tokenStrToId(std::string const& token) const override;
    std::string tokenIdToStr(TokenId id) const override;

    // Pieces that start a word begin with "▁", so only boundaries before tokens are marked.
    // Control pieces such as "<s>" and the unknown piece are words of their own.
    WordBoundaryMarks wordBoundaryMarks(TokenId id) const override;

    std::vector<TokenId> encode(std::string const& text) const override;
    std::string          decode(std::vector<TokenId> const& ids) const override;

private:
    sentencepiece::SentencePieceProcessor processor_;

    void checkStatus(sentencepiece::util::Status const& status) const;
};

}  // namespace Tokenizer

#endif  // _TOKENIZER_SENTENCE_PIECE_TOKENIZER_HH
