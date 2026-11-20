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

#include "AbstractTokenizer.hh"

#include <Core/StringUtilities.hh>

namespace Tokenizer {

AbstractTokenizer::AbstractTokenizer(Core::Configuration const& config)
        : Core::Component(config) {}

Core::Expected<std::vector<std::string>> AbstractTokenizer::tokenize(std::string const& text) const {
    std::vector<TokenId>     ids = TRY(encode(text));
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());
    for (auto id : ids) {
        tokens.push_back(TRY(tokenIdToStr(id)));
    }
    return tokens;
}

Core::Expected<std::string> AbstractTokenizer::detokenize(std::vector<std::string> const& tokens) const {
    std::vector<TokenId> ids;
    ids.reserve(tokens.size());
    for (auto const& token : tokens) {
        ids.push_back(tokenStrToId(token));
        if (ids.back() == invalidTokenId) {
            return std::unexpected(Core::Status(Core::StatusCode::NotFound, "Token '" + token + "' is not part of the vocabulary"));
        }
    }
    return decode(ids);
}

Core::Status AbstractTokenizer::checkTokenId(TokenId id) const {
    if (id >= vocabularySize()) {
        return Core::Status(Core::StatusCode::OutOfRange, Core::form("Invalid token id %u for vocabulary size %u", id, vocabularySize()));
    }
    return {};
}

}  // namespace Tokenizer
