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

Core::Status BpeTokenizer::create(Core::Configuration const& config, TokenizerRef& tokenizer) {
    std::shared_ptr<BpeTokenizer> bpe(new BpeTokenizer(config));
    Core::Status                  status = bpe->load();
    if (status.ok()) {
        tokenizer = std::move(bpe);
    }
    return status;
}

Core::Status BpeTokenizer::encode(std::string const& text, std::vector<TokenId>& ids) const {
    std::vector<std::string> tokens;
    Core::Status             status = tokenize(text, tokens);
    if (not status.ok()) {
        return status;
    }
    ids.clear();
    for (auto const& token : tokens) {
        TokenId id = vocabulary_.id(token);
        if (id == Bliss::ReturnnVocabulary::invalidId) {
            id = unknownTokenId_;
        }
        if (id == Bliss::ReturnnVocabulary::invalidId) {
            return Core::Status(Core::StatusCode::NotFound, "Token '" + token + "' is not part of the vocabulary");
        }
        ids.push_back(id);
    }
    return {};
}

Core::Status BpeTokenizer::decode(std::vector<TokenId> const& ids, std::string& text) const {
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());
    for (auto id : ids) {
        Core::Status status = checkTokenId(id);
        if (not status.ok()) {
            return status;
        }
        tokens.push_back(vocabulary_.token(id));
    }
    return detokenize(tokens, text);
}

Core::Status BpeTokenizer::vocabularySize(u32& size) const {
    size = vocabulary_.size();
    return {};
}

Core::Status BpeTokenizer::tokenStrToId(std::string const& token, TokenId& id) const {
    id = vocabulary_.id(token);
    if (id == Bliss::ReturnnVocabulary::invalidId) {
        return Core::Status(Core::StatusCode::NotFound, "Token '" + token + "' is not part of the vocabulary");
    }
    return {};
}

Core::Status BpeTokenizer::tokenIdToStr(TokenId id, std::string& token) const {
    Core::Status status = checkTokenId(id);
    if (status.ok()) {
        token = vocabulary_.token(id);
    }
    return status;
}

Core::Status BpeTokenizer::wordBoundaryMarks(TokenId id, WordBoundaryMarks& marks) const {
    Core::Status status = checkTokenId(id);
    if (status.ok()) {
        marks = {false, not endsWith(vocabulary_.token(id), separator_)};
    }
    return status;
}

Core::Status BpeTokenizer::tokenize(std::string const& text, std::vector<std::string>& tokens) const {
    tokens.clear();
    for (auto const& word : splitWords(text)) {
        auto subwords = segmentWord(word);
        for (size_t i = 0ul; i + 1 < subwords.size(); ++i) {
            tokens.push_back(subwords[i] + separator_);
        }
        tokens.push_back(subwords.back());
    }
    return {};
}

Core::Status BpeTokenizer::detokenize(std::vector<std::string> const& tokens, std::string& text) const {
    text.clear();
    bool continueWord = true;
    for (auto const& token : tokens) {
        if (not continueWord) {
            text += utf8::blank;
        }
        continueWord = endsWith(token, separator_);
        text.append(token, 0, continueWord ? token.size() - separator_.size() : token.size());
    }
    return {};
}

BpeTokenizer::BpeTokenizer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          separator_(paramSeparator(config)),
          endOfWordAttached_(false),
          unknownTokenId_(Bliss::ReturnnVocabulary::invalidId) {}

Core::Status BpeTokenizer::load() {
    if (separator_.empty()) {
        return Core::Status(Core::StatusCode::InvalidArgument, "BPE separator must not be empty");
    }
    Core::Status status = loadCodes(paramCodesFile(config));
    if (not status.ok()) {
        return status;
    }
    std::string vocabFile = paramVocabFile(config);
    status                = Bliss::ReturnnVocabulary::load(vocabFile, vocabulary_);
    if (not status.ok()) {
        return status;
    }
    log("Loaded BPE vocabulary of size %u from '%s'", vocabulary_.size(), vocabFile.c_str());
    std::string unknownToken = paramUnknownToken(config);
    if (not unknownToken.empty()) {
        unknownTokenId_ = vocabulary_.id(unknownToken);
        if (unknownTokenId_ == Bliss::ReturnnVocabulary::invalidId) {
            return Core::Status(Core::StatusCode::InvalidArgument, "Unknown token '" + unknownToken + "' is not part of the vocabulary");
        }
    }
    return {};
}

Core::Status BpeTokenizer::loadCodes(std::string const& filename) {
    Core::CompressedInputStream in(filename.c_str());
    if (not in) {
        return Core::Status(Core::StatusCode::NotFound, "Failed to open BPE codes file '" + filename + "'");
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
        auto pair = Core::split(line, " ");
        if (pair.size() != 2ul) {
            return Core::Status(Core::StatusCode::InvalidFileFormat, "Invalid line in BPE codes file '" + filename + "': '" + line + "'");
        }
        // On duplicates, the first occurrence takes precedence
        mergeRanks_.emplace(line, rank++);
        mergedPairs_.emplace(pair[0] + pair[1], std::make_pair(pair[0], pair[1]));
    }
    log("Loaded %zu BPE merge operations from '%s'", mergeRanks_.size(), filename.c_str());
    return {};
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

    std::vector<std::string> subwords;
    for (size_t i = 0ul; i < symbols.size(); ++i) {
        splitIntoVocabulary(symbols[i], i + 1 == symbols.size(), subwords);
    }
    return subwords;
}

void BpeTokenizer::splitIntoVocabulary(std::string const& subword, bool final, std::vector<std::string>& result) const {
    if (vocabulary_.id(final ? subword : subword + separator_) != Bliss::ReturnnVocabulary::invalidId) {
        result.push_back(subword);
        return;
    }

    // The final subword was merged including the end-of-word marker, which in codes of version 0.1 is a separate
    // symbol, so then the merge of the subword itself is reversed
    auto merge = mergedPairs_.find(final ? subword + endOfWord : subword);
    if (final and merge != mergedPairs_.end() and merge->second.second == endOfWord) {
        merge = mergedPairs_.find(subword);
    }
    if (merge == mergedPairs_.end()) {
        result.push_back(subword);
        return;
    }

    auto [left, right] = merge->second;
    if (endsWith(right, endOfWord)) {
        right.resize(right.size() - endOfWord.size());
    }
    splitIntoVocabulary(left, false, result);
    splitIntoVocabulary(right, final, result);
}

}  // namespace Tokenizer
