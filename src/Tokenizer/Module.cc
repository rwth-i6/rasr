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

#include "Module.hh"

#include "BpeTokenizer.hh"
#ifdef MODULE_SENTENCEPIECE
#include "SentencePieceTokenizer.hh"
#endif
#ifdef MODULE_PYTHON
#include "HuggingFaceTokenizer.hh"
#endif

namespace Tokenizer {

Module_::Module_()
        : tokenizerFactory_() {
    // subword-nmt style BPE
    tokenizerFactory_.registerTokenizer(
            "bpe",
            [](Core::Configuration const& config) {
                return Core::ref(new BpeTokenizer(config));
            });

#ifdef MODULE_SENTENCEPIECE
    tokenizerFactory_.registerTokenizer(
            "sentencepiece",
            [](Core::Configuration const& config) {
                return Core::ref(new SentencePieceTokenizer(config));
            });
#endif

#ifdef MODULE_PYTHON
    // Loads a tokenizer.json via the python `tokenizers` package
    tokenizerFactory_.registerTokenizer(
            "huggingface",
            [](Core::Configuration const& config) {
                return Core::ref(new HuggingFaceTokenizer(config));
            });
#endif
}

TokenizerFactory& Module_::tokenizerFactory() {
    return tokenizerFactory_;
}

}  // namespace Tokenizer
