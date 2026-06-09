/** Copyright 2020 RWTH Aachen University. All rights reserved.
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
#include "Orthography.hh"

namespace Bliss {

Orthography::Span::Span(std::string const& text)
        : type_(Type::text),
          text_(text) {
    require(Orthography::isValidSpan(text_));
}

Orthography::Span::Span(std::vector<Orthography> const& alternatives)
        : type_(Type::alternatives),
          alternatives_(alternatives) {
    require(!alternatives_.empty());
}

Orthography Orthography::fromNormalized(std::string const& text) {
    return Orthography(text);
}

Orthography Orthography::fromRaw(std::string const& text) {
    std::string normalized(text);
    Core::normalizeWhitespace(normalized);
    Core::enforceTrailingBlank(normalized);
    return Orthography(normalized);
}

Orthography::Orthography(std::string const& text) {
    setNormalized(text);
}

std::string Orthography::str() const {
    std::string result;
    for (auto const& span : spans_) {
        switch (span.type()) {
            case Span::Type::text:
                result += span.text();
                break;
            case Span::Type::alternatives:
                result += span.alternatives().front().str();
                break;
        }
    }
    ensure(isValid(result));
    return result;
}

}  // namespace Bliss
