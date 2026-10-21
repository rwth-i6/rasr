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
#include <sstream>
#include <utility>

#include <Core/CompressedStream.hh>
#include <Core/StringUtilities.hh>
#include <Core/Unicode.hh>

namespace {

std::string const endOfWord = "</w>";

bool endsWith(std::string const& s, std::string const& suffix) {
    return s.size() >= suffix.size() and s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::string> splitUtf8Characters(std::string const& word) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < word.size();) {
        size_t len = 1;
        while (i + len < word.size() and utf8::byteType(word[i + len]) == utf8::multiByteTail) {
            ++len;
        }
        chars.push_back(word.substr(i, len));
        i += len;
    }
    return chars;
}

typedef std::vector<std::pair<std::string, u32>> VocabularyEntries;

// Python dict literals start with "{" followed by a quote, unlike a line vocabulary whose first token is "{"
bool isPythonDict(std::string const& content) {
    size_t brace = content.find_first_not_of(utf8::whitespace);
    if (brace == std::string::npos or content[brace] != '{') {
        return false;
    }
    size_t next = content.find_first_not_of(utf8::whitespace, brace + 1);
    return next != std::string::npos and (content[next] == '\'' or content[next] == '"' or content[next] == '}');
}

// Parses `digits` hex digits at `pos` as unicode code point and appends it as UTF-8
bool appendCodePoint(std::string const& s, size_t& pos, size_t digits, std::string& result) {
    std::string hex = s.substr(pos, digits);
    if (hex.size() != digits or not std::all_of(hex.begin(), hex.end(), [](unsigned char c) { return std::isxdigit(c); })) {
        return false;
    }
    result += narrow(std::wstring(1, static_cast<wchar_t>(std::stoul(hex, nullptr, 16))));
    pos += digits;
    return true;
}

// Parses the python string literal whose opening quote is at `pos` and moves `pos` behind the closing quote.
// Handles the escape sequences written by python's `repr`.
bool parsePythonString(std::string const& s, size_t& pos, std::string& result) {
    char const quote = s[pos++];
    result.clear();
    while (pos < s.size() and s[pos] != quote) {
        char c = s[pos++];
        if (c != '\\' or pos == s.size()) {
            result += c;
            continue;
        }
        char escaped = s[pos++];
        switch (escaped) {
            case 'n': result += '\n'; break;
            case 't': result += '\t'; break;
            case 'r': result += '\r'; break;
            case 'x':
            case 'u':
            case 'U': {
                size_t digits = escaped == 'x' ? 2ul : (escaped == 'u' ? 4ul : 8ul);
                if (not appendCodePoint(s, pos, digits, result)) {
                    return false;
                }
                break;
            }
            default: result += escaped;  // \\, \' and \"
        }
    }
    if (pos == s.size()) {
        return false;
    }
    ++pos;
    return true;
}

// Parses a python dict literal {'token': id, ...} as written by create-py-vocab.py of rwth-i6/subword-nmt
bool parsePythonDict(std::string const& s, VocabularyEntries& entries) {
    size_t pos = s.find('{') + 1;
    while (true) {
        pos = s.find_first_not_of(" \t\n\r,", pos);
        if (pos == std::string::npos) {
            return false;
        }
        if (s[pos] == '}') {
            return true;
        }
        std::string token;
        if ((s[pos] != '\'' and s[pos] != '"') or not parsePythonString(s, pos, token)) {
            return false;
        }
        pos = s.find_first_not_of(" \t\n\r", pos);
        if (pos == std::string::npos or s[pos] != ':') {
            return false;
        }
        char const* begin = s.c_str() + pos + 1;
        char*       end   = nullptr;
        long        id    = std::strtol(begin, &end, 10);
        if (end == begin or id < 0 or id >= static_cast<long>(Tokenizer::AbstractTokenizer::invalidTokenId)) {
            return false;
        }
        entries.emplace_back(token, id);
        pos = end - s.c_str();
    }
}

// One token per line with the line index as id, further columns (e.g. counts) are ignored
VocabularyEntries parseLineVocabulary(std::string const& s) {
    VocabularyEntries  entries;
    std::istringstream lines(s);
    std::string        line;
    while (std::getline(lines, line)) {
        std::istringstream columns(line);
        std::string        token;
        if (columns >> token) {
            entries.emplace_back(token, entries.size());
        }
    }
    return entries;
}

}  // namespace

namespace Tokenizer {

const Core::ParameterString BpeTokenizer::paramCodesFile(
        "codes-file",
        "subword-nmt BPE codes file containing the merge operations",
        "");

const Core::ParameterString BpeTokenizer::paramVocabFile(
        "vocab-file",
        "vocabulary file defining the token ids; one token per line or a python dict literal",
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
    std::string vocabFile = paramVocabFile(config);
    if (not vocabFile.empty()) {
        loadVocabulary(vocabFile);
    }
    std::string unknownToken = paramUnknownToken(config);
    if (not unknownToken.empty()) {
        unknownTokenId_ = tokenId(unknownToken);
        if (unknownTokenId_ == invalidTokenId) {
            criticalError("Unknown token '%s' is not part of the vocabulary", unknownToken.c_str());
        }
    }
}

std::vector<std::string> BpeTokenizer::tokenize(std::string const& text) const {
    std::vector<std::string> tokens;
    std::istringstream       words(text);
    std::string              word;
    while (words >> word) {
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
    for (auto const& t : tokens) {
        if (not continueWord) {
            text += utf8::blank;
        }
        continueWord = endsWith(t, separator_);
        text.append(t, 0, continueWord ? t.size() - separator_.size() : t.size());
    }
    return text;
}

u32 BpeTokenizer::vocabularySize() const {
    return tokens_.size();
}

AbstractTokenizer::TokenId BpeTokenizer::tokenId(std::string const& token) const {
    auto it = tokenIds_.find(token);
    return it == tokenIds_.end() ? invalidTokenId : it->second;
}

std::string BpeTokenizer::token(TokenId id) const {
    return checkTokenId(id) ? tokens_[id] : std::string();
}

std::vector<AbstractTokenizer::TokenId> BpeTokenizer::encode(std::string const& text) const {
    if (unknownTokenId_ == invalidTokenId) {
        return Precursor::encode(text);
    }
    std::vector<TokenId> ids;
    for (auto const& t : tokenize(text)) {
        TokenId id = tokenId(t);
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
    if (not isPythonDict(content)) {
        entries = parseLineVocabulary(content);
    }
    else if (not parsePythonDict(content, entries)) {
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
