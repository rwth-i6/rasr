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

Core::Status AbstractTokenizer::tokenize(std::string const& text, std::vector<std::string>& tokens) const {
    std::vector<TokenId> ids;
    Core::Status         status = encode(text, ids);
    if (not status.ok()) {
        return status;
    }
    tokens.resize(ids.size());
    for (size_t i = 0ul; i < ids.size(); ++i) {
        status = tokenIdToStr(ids[i], tokens[i]);
        if (not status.ok()) {
            return status;
        }
    }
    return {};
}

Core::Status AbstractTokenizer::detokenize(std::vector<std::string> const& tokens, std::string& text) const {
    std::vector<TokenId> ids;
    ids.reserve(tokens.size());
    for (auto const& token : tokens) {
        ids.push_back(tokenStrToId(token));
        if (ids.back() == invalidTokenId) {
            return Core::Status(Core::StatusCode::NotFound, "Token '" + token + "' is not part of the vocabulary");
        }
    }
    return decode(ids, text);
}

Core::Status AbstractTokenizer::checkTokenId(TokenId id) const {
    if (id >= vocabularySize()) {
        return Core::Status(Core::StatusCode::OutOfRange, Core::form("Invalid token id %u for vocabulary size %u", id, vocabularySize()));
    }
    return {};
}

}  // namespace Tokenizer
