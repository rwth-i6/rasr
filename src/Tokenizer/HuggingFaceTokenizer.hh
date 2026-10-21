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

#include <Python/Utilities.hh>

#include "AbstractTokenizer.hh"

namespace Tokenizer {

/*
 * Tokenizer loaded from a HuggingFace `tokenizer.json` file (as shipped with most LLMs).
 * Runs the python `tokenizers` package (https://github.com/huggingface/tokenizers) in the embedded interpreter.
 */
class HuggingFaceTokenizer : public AbstractTokenizer {
public:
    using Precursor = AbstractTokenizer;

    static const Core::ParameterString paramTokenizerFile;
    static const Core::ParameterBool   paramAddSpecialTokens;
    static const Core::ParameterBool   paramSkipSpecialTokens;

    HuggingFaceTokenizer(Core::Configuration const& config);
    ~HuggingFaceTokenizer();

    std::vector<std::string> tokenize(std::string const& text) const override;
    std::string              detokenize(std::vector<std::string> const& tokens) const override;

    u32         vocabularySize() const override;
    TokenId     tokenId(std::string const& token) const override;
    std::string token(TokenId id) const override;

    std::vector<TokenId> encode(std::string const& text) const override;
    std::string          decode(std::vector<TokenId> const& ids) const override;

private:
    bool const addSpecialTokens_;
    bool const skipSpecialTokens_;

    Python::ObjRef tokenizer_;  // Instance of `tokenizers.Tokenizer`
    u32            vocabularySize_;
};

}  // namespace Tokenizer

#endif  // _TOKENIZER_HUGGING_FACE_TOKENIZER_HH
