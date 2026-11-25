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
#include "Text.hh"

#include <unordered_set>

#include <Core/Unicode.hh>

namespace Tokenizer {

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

bool isWhitespace(std::string const& text) {
    static std::unordered_set<std::string> const whitespace = {
            "\t", "\n", "\v", "\f", "\r", "\x1c", "\x1d", "\x1e", "\x1f", " ", "\u0085", " ", " ",
            " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ",
            " ", " ", " ", " ", "　"};
    for (auto const& character : splitUtf8Characters(text)) {
        if (whitespace.count(character) == 0ul) {
            return false;
        }
    }
    return true;
}

}  // namespace Tokenizer
