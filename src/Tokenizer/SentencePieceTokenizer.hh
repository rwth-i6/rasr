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

    static Core::Status create(Core::Configuration const& config, TokenizerRef& tokenizer);

    Core::Status encode(std::string const& text, std::vector<TokenId>& ids) const override;
    Core::Status decode(std::vector<TokenId> const& ids, std::string& text) const override;

    Core::Status vocabularySize(u32& size) const override;
    Core::Status tokenStrToId(std::string const& token, TokenId& id) const override;
    Core::Status tokenIdToStr(TokenId id, std::string& token) const override;

    // SentencePiece represents blanks by "▁" at the start of pieces, or at their end for models trained with
    // `treat_whitespace_as_suffix`, so only boundaries before or after tokens are marked. Control pieces such as "<s>",
    // the unknown piece and user-defined symbols are words of their own. Pieces spanning several words, which models
    // trained without `split_by_whitespace` can contain, only mark their outer boundaries.
    Core::Status wordBoundaryMarks(TokenId id, WordBoundaryMarks& marks) const override;

    // Pieces that are not part of the model are decoded as well
    Core::Status tokenize(std::string const& text, std::vector<std::string>& tokens) const override;
    Core::Status detokenize(std::vector<std::string> const& tokens, std::string& text) const override;

private:
    sentencepiece::SentencePieceProcessor processor_;

    // Read from the model, as the SentencePiece API does not provide them
    bool              whitespaceAsSuffix_;
    std::vector<bool> standalonePieces_;  // Indexed by id, pieces that are words of their own

    SentencePieceTokenizer(Core::Configuration const& config);

    Core::Status load();
};

}  // namespace Tokenizer

#endif  // _TOKENIZER_SENTENCE_PIECE_TOKENIZER_HH
