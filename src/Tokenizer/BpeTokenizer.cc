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

#include "BpeTokenizer.hh"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iterator>
#include <unordered_set>
#include <utility>

#include <Core/CompressedStream.hh>
#include <Core/StringUtilities.hh>
#include <Core/Unicode.hh>

namespace {

std::string const endOfWord = "</w>";

bool endsWith(std::string const& text, std::string const& suffix) {
    return text.size() >= suffix.size() and text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::string> splitUtf8Characters(std::string const& text) {
    std::vector<std::string> characters;
    for (size_t i = 0; i < text.size();) {
        size_t length = 1;
        while (i + length < text.size() and utf8::byteType(text[i + length]) == utf8::multiByteTail) {
            ++length;
        }
        characters.push_back(text.substr(i, length));
        i += length;
    }
    return characters;
}

// Whitespace characters of python's `str.split()`, which rwth-i6/subword-nmt uses to split text into words
bool isWhitespace(std::string const& character) {
    static std::unordered_set<std::string> const whitespace = {
            "\t", "\n", "\v", "\f", "\r", "\x1c", "\x1d", "\x1e", "\x1f", " ", "\u0085", "\u00a0", "\u1680",
            "\u2000", "\u2001", "\u2002", "\u2003", "\u2004", "\u2005", "\u2006", "\u2007", "\u2008", "\u2009", "\u200a",
            "\u2028", "\u2029", "\u202f", "\u205f", "\u3000"};
    return whitespace.count(character) > 0ul;
}

std::vector<std::string> splitWords(std::string const& text) {
    std::vector<std::string> words(1);
    for (auto const& character : splitUtf8Characters(text)) {
        if (not isWhitespace(character)) {
            words.back() += character;
        }
        else if (not words.back().empty()) {
            words.emplace_back();
        }
    }
    if (words.back().empty()) {
        words.pop_back();
    }
    return words;
}

typedef std::vector<std::pair<std::string, u32>> VocabularyEntries;

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

// Parses a python dict literal {'token': id, ...} as written by create-py-vocab.py of rwth-i6/subword-nmt
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
        if (end == begin or id < 0 or id >= static_cast<long>(Tokenizer::AbstractTokenizer::invalidTokenId)) {
            return false;
        }
        entries.emplace_back(token, id);
        position = end - content.c_str();
    }
}

}  // namespace

namespace Tokenizer {

const Core::ParameterString BpeTokenizer::paramCodesFile(
        "codes-file",
        "subword-nmt BPE codes file containing the merge operations",
        "");

const Core::ParameterString BpeTokenizer::paramVocabFile(
        "vocab-file",
        "RETURNN-style vocabulary file defining the token ids, i.e. a python dict literal {'token': id, ...}",
        "");

const Core::ParameterString BpeTokenizer::paramSeparator(
        "separator",
        "suffix marking non-final subwords of a word",
        "@@");

const Core::ParameterString BpeTokenizer::paramUnknownToken(
        "unknown-token",
        "token that out-of-vocabulary tokens are mapped to when encoding",
        "");

BpeTokenizer::BpeTokenizer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          separator_(paramSeparator(config)),
          endOfWordAttached_(false),
          unknownTokenId_(invalidTokenId) {
    if (separator_.empty()) {
        criticalError("BPE separator must not be empty");
    }
    loadCodes(paramCodesFile(config));
    loadVocabulary(paramVocabFile(config));
    std::string unknownToken = paramUnknownToken(config);
    if (not unknownToken.empty()) {
        unknownTokenId_ = tokenStrToId(unknownToken);
        if (unknownTokenId_ == invalidTokenId) {
            criticalError("Unknown token '%s' is not part of the vocabulary", unknownToken.c_str());
        }
    }
}

std::vector<std::string> BpeTokenizer::tokenize(std::string const& text) const {
    std::vector<std::string> tokens;
    for (auto const& word : splitWords(text)) {
        auto subwords = segmentWord(word);
        for (size_t i = 0ul; i + 1 < subwords.size(); ++i) {
            tokens.push_back(subwords[i] + separator_);
        }
        tokens.push_back(subwords.back());
    }
    return tokens;
}

std::string BpeTokenizer::detokenize(std::vector<std::string> const& tokens) const {
    std::string text;
    bool        continueWord = true;
    for (auto const& token : tokens) {
        if (not continueWord) {
            text += utf8::blank;
        }
        continueWord = endsWith(token, separator_);
        text.append(token, 0, continueWord ? token.size() - separator_.size() : token.size());
    }
    return text;
}

u32 BpeTokenizer::vocabularySize() const {
    return tokens_.size();
}

AbstractTokenizer::TokenId BpeTokenizer::tokenStrToId(std::string const& token) const {
    auto it = tokenIds_.find(token);
    return it == tokenIds_.end() ? invalidTokenId : it->second;
}

std::string BpeTokenizer::tokenIdToStr(TokenId id) const {
    return checkTokenId(id) ? tokens_[id] : std::string();
}

std::vector<AbstractTokenizer::TokenId> BpeTokenizer::encode(std::string const& text) const {
    if (unknownTokenId_ == invalidTokenId) {
        return Precursor::encode(text);
    }
    std::vector<TokenId> ids;
    for (auto const& token : tokenize(text)) {
        TokenId id = tokenStrToId(token);
        ids.push_back(id == invalidTokenId ? unknownTokenId_ : id);
    }
    return ids;
}

void BpeTokenizer::loadCodes(std::string const& filename) {
    Core::CompressedInputStream in(filename.c_str());
    if (not in) {
        criticalError("Failed to open BPE codes file '%s'", filename.c_str());
        return;
    }

    std::string line;
    u32         rank = 0u;
    while (std::getline(in, line)) {
        // Files without header are version 0.1
        if (rank == 0u and Core::startsWith(line, "#version:")) {
            std::string version = line.substr(std::string("#version:").size());
            Core::stripWhitespace(version);
            endOfWordAttached_ = version != "0.1";
            continue;
        }
        Core::stripWhitespace(line);
        if (line.empty()) {
            continue;
        }
        if (Core::split(line, " ").size() != 2ul) {
            criticalError("Invalid line in BPE codes file '%s': '%s'", filename.c_str(), line.c_str());
            return;
        }
        // On duplicates, the first occurrence takes precedence
        mergeRanks_.emplace(line, rank++);
    }
    log("Loaded %zu BPE merge operations from '%s'", mergeRanks_.size(), filename.c_str());
}

void BpeTokenizer::loadVocabulary(std::string const& filename) {
    Core::CompressedInputStream in(filename.c_str());
    if (not in) {
        criticalError("Failed to open BPE vocabulary file '%s'", filename.c_str());
        return;
    }
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    VocabularyEntries entries;
    if (not parsePythonDict(content, entries)) {
        criticalError("Malformed python dict in BPE vocabulary file '%s'", filename.c_str());
        return;
    }

    for (auto const& [token, id] : entries) {
        // Ids are expected to be dense, which also bounds the size of `tokens_`
        if (id >= entries.size()) {
            criticalError("Id %u of token '%s' in BPE vocabulary file '%s' exceeds the number of entries", id, token.c_str(), filename.c_str());
            return;
        }
        addToken(token, id);
    }
    log("Loaded BPE vocabulary of size %u from '%s'", vocabularySize(), filename.c_str());
}

void BpeTokenizer::addToken(std::string const& token, TokenId id) {
    tokenIds_.emplace(token, id);
    if (id >= tokens_.size()) {
        tokens_.resize(id + 1);
    }
    // If several tokens share an id (e.g. '<s>' and '</s>' in RETURNN vocabularies), the first one is used for decoding
    if (tokens_[id].empty()) {
        tokens_[id] = token;
    }
}

std::vector<std::string> BpeTokenizer::segmentWord(std::string const& word) const {
    auto symbols = splitUtf8Characters(word);
    if (symbols.size() == 1ul) {
        return symbols;
    }

    if (endOfWordAttached_) {
        symbols.back() += endOfWord;
    }
    else {
        symbols.push_back(endOfWord);
    }

    while (symbols.size() > 1ul) {
        // Find the adjacent pair with the highest merge priority
        u32    bestRank = Core::Type<u32>::max;
        size_t bestPos  = 0ul;
        for (size_t i = 0ul; i + 1 < symbols.size(); ++i) {
            auto it = mergeRanks_.find(symbols[i] + ' ' + symbols[i + 1]);
            if (it != mergeRanks_.end() and it->second < bestRank) {
                bestRank = it->second;
                bestPos  = i;
            }
        }
        if (bestRank == Core::Type<u32>::max) {
            break;
        }

        // Merge all non-overlapping occurrences of the pair from left to right
        std::string const        left  = symbols[bestPos];
        std::string const        right = symbols[bestPos + 1];
        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (size_t i = 0ul; i < symbols.size(); ++i) {
            if (i + 1 < symbols.size() and symbols[i] == left and symbols[i + 1] == right) {
                merged.push_back(left + right);
                ++i;
            }
            else {
                merged.push_back(symbols[i]);
            }
        }
        symbols.swap(merged);
    }

    if (symbols.back() == endOfWord) {
        symbols.pop_back();
    }
    else if (endsWith(symbols.back(), endOfWord)) {
        symbols.back().resize(symbols.back().size() - endOfWord.size());
    }
    return symbols;
}

}  // namespace Tokenizer
