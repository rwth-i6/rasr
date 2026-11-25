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

#ifndef _TOKENIZER_HUGGING_FACE_TOKENIZER_HH
#define _TOKENIZER_HUGGING_FACE_TOKENIZER_HH

#include <string>
#include <unordered_set>
#include <vector>

#include <Python/Utilities.hh>

#include "AbstractTokenizer.hh"

namespace pybind11 {
class handle;
}  // namespace pybind11

namespace Tokenizer {

/*
 * Tokenizer loaded from a HuggingFace `tokenizer.json` file (as shipped with most LLMs).
 * Runs the python `tokenizers` package (https://github.com/huggingface/tokenizers) in the embedded interpreter.
 */
class HuggingFaceTokenizer : public AbstractTokenizer {
public:
    using Precursor = AbstractTokenizer;

    static const Core::ParameterString paramTokenizerFile;
    static const Core::ParameterBool   paramAddSpecialTokensOnEncode;
    static const Core::ParameterBool   paramRemoveSpecialTokensOnDecode;
    static const Core::ParameterBool   paramEncodeSpecialTokensAsText;

    static Core::Status create(Core::Configuration const& config, TokenizerRef& tokenizer);

    ~HuggingFaceTokenizer();

    Core::Status encode(std::string const& text, std::vector<TokenId>& ids) const override;
    Core::Status decode(std::vector<TokenId> const& ids, std::string& text) const override;

    Core::Status vocabularySize(u32& size) const override;
    Core::Status tokenStrToId(std::string const& token, TokenId& id) const override;
    Core::Status tokenIdToStr(TokenId id, std::string& token) const override;

    // Determined from the decoder in tokenizer.json, special tokens are words of their own. Unsupported decoders result
    // in `Unimplemented`.
    Core::Status wordBoundaryMarks(TokenId id, WordBoundaryMarks& marks) const override;

    Core::Status tokenize(std::string const& text, std::vector<std::string>& tokens) const override;

private:
    // How tokens mark word boundaries, see https://huggingface.co/docs/tokenizers/api/decoders
    struct WordBoundaryMarking {
        enum class Type {
            Unknown,
            ByteLevel,           // Whitespace bytes are mapped to characters such as "Ġ" at the start or end of tokens
            Prefix,              // `marker` at the start of a token starts a word, e.g. "▁"
            ContinuationPrefix,  // `marker` at the start of a token continues a word, e.g. "##"
            Suffix,              // `marker` at the end of a token ends a word, e.g. "</w>"
        };
        Type        type;
        std::string marker;
    };

    bool const addSpecialTokensOnEncode_;
    bool const removeSpecialTokensOnDecode_;

    Python::ObjRef              tokenizer_;  // Instance of `tokenizers.Tokenizer`
    u32                         vocabularySize_;
    WordBoundaryMarking         wordBoundaryMarking_;
    std::unordered_set<TokenId> specialTokenIds_;

    HuggingFaceTokenizer(Core::Configuration const& config);

    Core::Status load();

    // Parses the JSON of a decoder, which may be None
    static WordBoundaryMarking parseWordBoundaryMarking(pybind11::handle decoder);
};

}  // namespace Tokenizer

#endif  // _TOKENIZER_HUGGING_FACE_TOKENIZER_HH
