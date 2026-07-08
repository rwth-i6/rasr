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
#ifndef _BLISS_ORTHOGRAPHY_HH
#define _BLISS_ORTHOGRAPHY_HH

#include <string>
#include <variant>
#include <vector>

#include <Core/Assertions.hh>
#include <Core/StringUtilities.hh>

namespace Bliss {

class Orthography {
public:
    class Span {
    public:
        enum class Type {
            text,
            alternatives
        };

        Span(std::string const& text);
        Span(std::vector<Orthography> const& alternatives);

        Type type() const {
            return std::holds_alternative<std::string>(content_) ? Type::text : Type::alternatives;
        }

        std::string const& text() const {
            require(type() == Type::text);
            return std::get<std::string>(content_);
        }

        std::vector<Orthography> const& alternatives() const {
            require(type() == Type::alternatives);
            return std::get<std::vector<Orthography>>(content_);
        }

    private:
        std::variant<std::string, std::vector<Orthography>> content_;
    };

    using SpanList = std::vector<Span>;

    static Orthography fromNormalized(std::string const& text);
    static Orthography fromRaw(std::string const& text);

    Orthography() {}
    explicit Orthography(std::string const& text);

    std::string str() const;

    SpanList const& spans() const {
        return spans_;
    }

    void clear() {
        spans_.clear();
    }

    bool empty() const {
        return spans_.empty();
    }

    void setNormalized(std::string const& text) {
        require(isValid(text));
        spans_.clear();
        if (!text.empty()) {
            spans_.push_back(Span(text));
        }
    }

    void appendText(std::string const& text) {
        require(isValidSpan(text));
        if (!text.empty()) {
            spans_.push_back(Span(text));
        }
    }

    void appendAlternative(std::vector<Orthography> const& alternatives) {
        spans_.push_back(Span(alternatives));
    }

private:
    static bool isValid(std::string const& text) {
        return text.empty() || Core::isWhitespaceNormalized(text, Core::requireTrailingBlank);
    }

    static bool isValidSpan(std::string const& text) {
        return Core::isWhitespaceNormalized(text, Core::tolerateTrailingBlank);
    }

    SpanList spans_;
};

}  // namespace Bliss

#endif  // _BLISS_ORTHOGRAPHY_HH
