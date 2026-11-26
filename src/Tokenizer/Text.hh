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
#ifndef _TOKENIZER_TEXT_HH
#define _TOKENIZER_TEXT_HH

#include <string>
#include <vector>

namespace Tokenizer {

// Splits text into UTF-8 characters, each of them a lead byte with its following continuation bytes
std::vector<std::string> splitUtf8Characters(std::string const& text);

// Whether `text` consists only of whitespace characters as defined by python's `str.isspace()`, also if it is empty
bool isWhitespace(std::string const& text);

}  // namespace Tokenizer

#endif  // _TOKENIZER_TEXT_HH
