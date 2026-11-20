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

#include <string_view>

#include <Core/StringUtilities.hh>

namespace {

// SentencePiece represents blanks by this character
std::string const whitespace = "\u2581";

bool endsWith(std::string const& text, std::string const& suffix) {
    return text.size() >= suffix.size() and text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Field of a protobuf message, see https://protobuf.dev/programming-guides/encoding
struct ProtobufField {
    u32              number;
    u32              wireType;
    u64              varint;  // Value of varint fields (wire type 0)
    std::string_view bytes;   // Value of length-delimited fields (wire type 2)
};

bool readVarint(std::string_view data, size_t& position, u64& value) {
    value = 0u;
    for (u32 shift = 0u; shift < 64u and position < data.size(); shift += 7u) {
        u8 byte = static_cast<u8>(data[position++]);
        value |= static_cast<u64>(byte & 0x7fu) << shift;
        if ((byte & 0x80u) == 0u) {
            return true;
        }
    }
    return false;
}

// Calls `visit` for each field of the message, returns false for malformed messages
template<typename Visit>
bool forEachProtobufField(std::string_view message, Visit visit) {
    size_t position = 0ul;
    while (position < message.size()) {
        u64 key;
        if (not readVarint(message, position, key)) {
            return false;
        }
        ProtobufField field{static_cast<u32>(key >> 3u), static_cast<u32>(key & 7u), 0u, {}};
        u64           length = 0u;
        switch (field.wireType) {
            case 0u:
                if (not readVarint(message, position, field.varint)) {
                    return false;
                }
                break;
            case 1u: length = 8u; break;
            case 2u:
                if (not readVarint(message, position, length)) {
                    return false;
                }
                break;
            case 5u: length = 4u; break;
            default: return false;
        }
        if (length > message.size() - position) {
            return false;
        }
        if (field.wireType == 2u) {
            field.bytes = message.substr(position, length);
        }
        position += length;
        visit(field);
    }
    return true;
}

// Field numbers and piece types of sentencepiece_model.proto
u32 const modelPiecesField               = 1u;
u32 const modelTrainerSpecField          = 2u;
u32 const pieceTypeField                 = 3u;
u32 const trainerWhitespaceAsSuffixField = 24u;
u64 const normalPieceType                = 1u;
u64 const unknownPieceType               = 2u;
u64 const controlPieceType               = 3u;
u64 const userDefinedPieceType           = 4u;

// The status codes of SentencePiece are the same as those of RASR
Core::Status convertStatus(sentencepiece::util::Status const& status) {
    if (status.ok()) {
        return {};
    }
    return Core::Status(static_cast<Core::StatusCode>(status.code()), "SentencePiece error: " + status.ToString());
}

}  // namespace

namespace Tokenizer {

const Core::ParameterString SentencePieceTokenizer::paramModelFile(
        "model-file",
        "SentencePiece model file",
        "");

Core::Expected<TokenizerRef> SentencePieceTokenizer::create(Core::Configuration const& config) {
    std::shared_ptr<SentencePieceTokenizer> sentencePiece(new SentencePieceTokenizer(config));
    TRY(sentencePiece->load());
    return sentencePiece;
}

Core::Expected<std::vector<AbstractTokenizer::TokenId>> SentencePieceTokenizer::encode(std::string const& text) const {
    std::vector<int> pieceIds;
    TRY(convertStatus(processor_.Encode(text, &pieceIds)));
    return std::vector<TokenId>(pieceIds.begin(), pieceIds.end());
}

Core::Expected<std::string> SentencePieceTokenizer::decode(std::vector<TokenId> const& ids) const {
    for (auto id : ids) {
        TRY(checkTokenId(id));
    }
    std::string text;
    TRY(convertStatus(processor_.Decode(std::vector<int>(ids.begin(), ids.end()), &text)));
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

Core::Expected<std::string> SentencePieceTokenizer::tokenIdToStr(TokenId id) const {
    // `IdToPiece` does not check the range
    TRY(checkTokenId(id));
    return processor_.IdToPiece(id);
}

Core::Expected<AbstractTokenizer::WordBoundaryMarks> SentencePieceTokenizer::wordBoundaryMarks(TokenId id) const {
    TRY(checkTokenId(id));
    std::string const& piece = processor_.IdToPiece(id);
    if (standalonePieces_[id]) {
        return WordBoundaryMarks{true, true};
    }
    if (whitespaceAsSuffix_) {
        return WordBoundaryMarks{false, endsWith(piece, whitespace)};
    }
    return WordBoundaryMarks{Core::startsWith(piece, whitespace), false};
}

Core::Expected<std::vector<std::string>> SentencePieceTokenizer::tokenize(std::string const& text) const {
    std::vector<std::string> tokens;
    TRY(convertStatus(processor_.Encode(text, &tokens)));
    return tokens;
}

Core::Expected<std::string> SentencePieceTokenizer::detokenize(std::vector<std::string> const& tokens) const {
    std::string text;
    TRY(convertStatus(processor_.Decode(tokens, &text)));
    return text;
}

SentencePieceTokenizer::SentencePieceTokenizer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          whitespaceAsSuffix_(false) {}

Core::Status SentencePieceTokenizer::load() {
    std::string modelFile = paramModelFile(config);
    auto        status    = processor_.Load(modelFile);
    if (not status.ok()) {
        // SentencePiece reports malformed models as internal errors
        auto code = status.code() == sentencepiece::util::StatusCode::kNotFound ? Core::StatusCode::NotFound : Core::StatusCode::InvalidFileFormat;
        return Core::Status(code, "Failed to load SentencePiece model '" + modelFile + "': " + status.ToString());
    }

    // The piece types and the option `treat_whitespace_as_suffix` are read from the model, see sentencepiece_model.proto
    std::string model       = processor_.serialized_model_proto();
    bool        nestedValid = true;
    bool        valid       = forEachProtobufField(model, [&](ProtobufField const& field) {
        if (field.number == modelPiecesField and field.wireType == 2u) {
            u64 type = normalPieceType;
            nestedValid &= forEachProtobufField(field.bytes, [&](ProtobufField const& pieceField) {
                if (pieceField.number == pieceTypeField and pieceField.wireType == 0u) {
                    type = pieceField.varint;
                }
            });
            standalonePieces_.push_back(type == unknownPieceType or type == controlPieceType or type == userDefinedPieceType);
        }
        else if (field.number == modelTrainerSpecField and field.wireType == 2u) {
            nestedValid &= forEachProtobufField(field.bytes, [&](ProtobufField const& trainerField) {
                if (trainerField.number == trainerWhitespaceAsSuffixField and trainerField.wireType == 0u) {
                    whitespaceAsSuffix_ = trainerField.varint != 0u;
                }
            });
        }
    });
    if (not valid or not nestedValid or standalonePieces_.size() != vocabularySize()) {
        return Core::Status(Core::StatusCode::InvalidFileFormat, "Failed to read the pieces of SentencePiece model '" + modelFile + "'");
    }

    log("Loaded SentencePiece model with vocabulary size %u from '%s'", vocabularySize(), modelFile.c_str());
    return {};
}

}  // namespace Tokenizer
