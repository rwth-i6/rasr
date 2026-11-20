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

#ifndef _BLISS_RETURNN_VOCABULARY_HH
#define _BLISS_RETURNN_VOCABULARY_HH

#include <string>
#include <unordered_map>
#include <vector>

#include <Core/Status.hh>
#include <Core/Types.hh>

namespace Bliss {

/*
 * Vocabulary file as read by RETURNN: a python dict literal {'<s>': 0, 'lo@@': 1, ...} mapping tokens to ids, e.g. as
 * written by create-py-vocab.py of rwth-i6/subword-nmt. As in RETURNN, the ids have to be 0 to N-1 without gaps,
 * several tokens can share an id, of which the one that sorts last is used for the id, and the last entry of a token
 * counts.
 */
class ReturnnVocabulary {
public:
    typedef u32         Id;
    static constexpr Id invalidId = Core::Type<Id>::max;

    // `NotFound` if the file cannot be opened, `InvalidFileFormat` if it is malformed. `vocabulary` is only changed on success.
    static Core::Status load(std::string const& filename, ReturnnVocabulary& vocabulary);

    // Number of ids
    u32 size() const;

    // Returns `invalidId` if `token` is not part of the vocabulary
    Id id(std::string const& token) const;

    // Throws `std::out_of_range` for ids not below `size()`
    std::string const& token(Id id) const;

private:
    std::unordered_map<std::string, Id> ids_;
    std::vector<std::string>            tokens_;  // Indexed by id
};

}  // namespace Bliss

#endif  // _BLISS_RETURNN_VOCABULARY_HH
