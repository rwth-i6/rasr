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

#include <Core/Assertions.hh>
#include <Core/StringUtilities.hh>

namespace Bliss {

class Orthography {
public:
    static Orthography fromNormalized(std::string const& text) {
        return Orthography(text);
    }

    static Orthography fromRaw(std::string const& text);

    Orthography() {}
    explicit Orthography(std::string const& text) {
        setNormalized(text);
    }

    std::string const& str() const {
        ensure(isValid(text_));
        return text_;
    }

    void setNormalized(std::string const& text) {
        require(isValid(text));
        text_ = text;
    }

private:
    static bool isValid(std::string const& text) {
        return text.empty() || Core::isWhitespaceNormalized(text, Core::requireTrailingBlank);
    }

    std::string text_;
};

}  // namespace Bliss

#endif  // _BLISS_ORTHOGRAPHY_HH
