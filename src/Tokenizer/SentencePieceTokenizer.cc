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

#include "SentencePieceTokenizer.hh"

#include <Core/StringUtilities.hh>

namespace Tokenizer {

const Core::ParameterString SentencePieceTokenizer::paramModelFile(
        "model-file",
        "SentencePiece model file",
        "");

SentencePieceTokenizer::SentencePieceTokenizer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config) {
    std::string modelFile = paramModelFile(config);
    auto        status    = processor_.Load(modelFile);
    if (not status.ok()) {
        criticalError("Failed to load SentencePiece model '%s': %s", modelFile.c_str(), status.ToString().c_str());
        return;
    }
    log("Loaded SentencePiece model with vocabulary size %u from '%s'", vocabularySize(), modelFile.c_str());
}

std::vector<std::string> SentencePieceTokenizer::tokenize(std::string const& text) const {
    std::vector<std::string> pieces;
    checkStatus(processor_.Encode(text, &pieces));
    return pieces;
}

std::string SentencePieceTokenizer::detokenize(std::vector<std::string> const& tokens) const {
    std::string text;
    checkStatus(processor_.Decode(tokens, &text));
    return text;
}

u32 SentencePieceTokenizer::vocabularySize() const {
    return processor_.GetPieceSize();
}

AbstractTokenizer::TokenId SentencePieceTokenizer::tokenStrToId(std::string const& token) const {
    // `PieceToId` returns the unknown id for unknown pieces
    int id = processor_.PieceToId(token);
    return (processor_.IsUnknown(id) and processor_.IdToPiece(id) != token) ? invalidTokenId : id;
}

std::string SentencePieceTokenizer::tokenIdToStr(TokenId id) const {
    // `IdToPiece` does not check the range
    return checkTokenId(id) ? processor_.IdToPiece(id) : std::string();
}

AbstractTokenizer::WordBoundaryMarks SentencePieceTokenizer::wordBoundaryMarks(TokenId id) const {
    if (not checkTokenId(id)) {
        return {false, false};
    }
    if (processor_.IsControl(id) or processor_.IsUnknown(id)) {
        return {true, true};
    }
    return {Core::startsWith(processor_.IdToPiece(id), "\u2581"), false};
}

std::vector<AbstractTokenizer::TokenId> SentencePieceTokenizer::encode(std::string const& text) const {
    std::vector<int> ids;
    checkStatus(processor_.Encode(text, &ids));
    return {ids.begin(), ids.end()};
}

std::string SentencePieceTokenizer::decode(std::vector<TokenId> const& ids) const {
    std::string text;
    checkStatus(processor_.Decode(std::vector<int>(ids.begin(), ids.end()), &text));
    return text;
}

void SentencePieceTokenizer::checkStatus(sentencepiece::util::Status const& status) const {
    if (not status.ok()) {
        error("SentencePiece error: %s", status.ToString().c_str());
    }
}

}  // namespace Tokenizer
