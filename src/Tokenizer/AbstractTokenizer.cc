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

namespace Tokenizer {

AbstractTokenizer::AbstractTokenizer(Core::Configuration const& config)
        : Core::Component(config) {}

std::vector<AbstractTokenizer::TokenId> AbstractTokenizer::encode(std::string const& text) const {
    std::vector<TokenId> ids;
    for (auto const& token : tokenize(text)) {
        TokenId id = tokenStrToId(token);
        if (id == invalidTokenId) {
            error("Token '%s' is not part of the vocabulary", token.c_str());
            continue;
        }
        ids.push_back(id);
    }
    return ids;
}

std::string AbstractTokenizer::decode(std::vector<TokenId> const& ids) const {
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());
    for (auto id : ids) {
        tokens.push_back(tokenIdToStr(id));
    }
    return detokenize(tokens);
}

bool AbstractTokenizer::checkTokenId(TokenId id) const {
    if (id >= vocabularySize()) {
        error("Invalid token id %u for vocabulary size %u", id, vocabularySize());
        return false;
    }
    return true;
}

}  // namespace Tokenizer
