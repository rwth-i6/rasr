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

#include "ReturnnVocabulary.hh"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iterator>
#include <utility>

#include <Core/CompressedStream.hh>
#include <Core/StringUtilities.hh>
#include <Core/Unicode.hh>

namespace {

typedef std::vector<std::pair<std::string, Bliss::ReturnnVocabulary::Id>> VocabularyEntries;

// Parses `digits` hex digits at `position` as unicode code point and appends it as UTF-8
bool appendCodePoint(std::string const& content, size_t& position, size_t digits, std::string& result) {
    std::string hexDigits = content.substr(position, digits);
    if (hexDigits.size() != digits or not std::all_of(hexDigits.begin(), hexDigits.end(), [](unsigned char character) { return std::isxdigit(character); })) {
        return false;
    }
    result += narrow(std::wstring(1, static_cast<wchar_t>(std::strtoul(hexDigits.c_str(), nullptr, 16))));
    position += digits;
    return true;
}

// Parses the python string literal whose opening quote is at `position` and moves `position` behind the closing quote.
// Handles the escape sequences written by python's `repr`.
bool parsePythonString(std::string const& content, size_t& position, std::string& result) {
    char const quote = content[position++];
    result.clear();
    while (position < content.size() and content[position] != quote) {
        char character = content[position++];
        if (character != '\\' or position == content.size()) {
            result += character;
            continue;
        }
        char escaped = content[position++];
        switch (escaped) {
            case 'n': result += '\n'; break;
            case 't': result += '\t'; break;
            case 'r': result += '\r'; break;
            case 'x':
            case 'u':
            case 'U': {
                size_t digits = escaped == 'x' ? 2ul : (escaped == 'u' ? 4ul : 8ul);
                if (not appendCodePoint(content, position, digits, result)) {
                    return false;
                }
                break;
            }
            default: result += escaped;  // \\, \' and \"
        }
    }
    if (position == content.size()) {
        return false;
    }
    ++position;
    return true;
}

// Parses a python dict literal {'token': id, ...}
bool parsePythonDict(std::string const& content, VocabularyEntries& entries) {
    size_t position = content.find_first_not_of(utf8::whitespace);
    if (position == std::string::npos or content[position] != '{') {
        return false;
    }
    ++position;
    while (true) {
        position = content.find_first_not_of(" \t\n\r,", position);
        if (position == std::string::npos) {
            return false;
        }
        if (content[position] == '}') {
            return true;
        }
        std::string token;
        if ((content[position] != '\'' and content[position] != '"') or not parsePythonString(content, position, token)) {
            return false;
        }
        position = content.find_first_not_of(" \t\n\r", position);
        if (position == std::string::npos or content[position] != ':') {
            return false;
        }
        char const* begin = content.c_str() + position + 1;
        char*       end   = nullptr;
        long        id    = std::strtol(begin, &end, 10);
        if (end == begin or id < 0 or id >= static_cast<long>(Bliss::ReturnnVocabulary::invalidId)) {
            return false;
        }
        entries.emplace_back(token, id);
        position = end - content.c_str();
    }
}

}  // namespace

namespace Bliss {

Core::Status ReturnnVocabulary::load(std::string const& filename, ReturnnVocabulary& vocabulary) {
    Core::CompressedInputStream in(filename.c_str());
    if (not in) {
        return Core::Status(Core::StatusCode::NotFound, "Failed to open vocabulary file '" + filename + "'");
    }
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    VocabularyEntries entries;
    if (not parsePythonDict(content, entries)) {
        return Core::Status(Core::StatusCode::InvalidFileFormat, "Malformed python dict in vocabulary file '" + filename + "'");
    }

    ReturnnVocabulary result;
    for (auto const& [token, id] : entries) {
        result.ids_[token] = id;
    }
    // There are at most as many ids as tokens, which also bounds the size of `tokens_`
    std::vector<bool> used(result.ids_.size(), false);
    result.tokens_.resize(result.ids_.size());
    for (auto const& [token, id] : result.ids_) {
        if (id >= result.ids_.size()) {
            return Core::Status(Core::StatusCode::InvalidFileFormat,
                                Core::form("Id %u of token '%s' in vocabulary file '%s' exceeds the number of tokens", id, token.c_str(), filename.c_str()));
        }
        if (not used[id] or token > result.tokens_[id]) {
            result.tokens_[id] = token;
        }
        used[id] = true;
    }
    auto size = std::find(used.begin(), used.end(), false) - used.begin();
    if (size == 0 or std::find(used.begin() + size, used.end(), true) != used.end()) {
        return Core::Status(Core::StatusCode::InvalidFileFormat, "Ids in vocabulary file '" + filename + "' do not cover 0 to N-1");
    }
    result.tokens_.resize(size);
    vocabulary = std::move(result);
    return {};
}

u32 ReturnnVocabulary::size() const {
    return tokens_.size();
}

ReturnnVocabulary::Id ReturnnVocabulary::id(std::string const& token) const {
    auto it = ids_.find(token);
    return it == ids_.end() ? invalidId : it->second;
}

std::string const& ReturnnVocabulary::token(Id id) const {
    return tokens_.at(id);
}

}  // namespace Bliss
