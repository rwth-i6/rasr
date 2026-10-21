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

#ifndef TOKENIZER_TOKENIZER_FACTORY_HH
#define TOKENIZER_TOKENIZER_FACTORY_HH

#include <functional>

#include <Core/Choice.hh>
#include <Core/Configuration.hh>
#include <Core/Parameter.hh>

#include "AbstractTokenizer.hh"

namespace Tokenizer {

/*
 * Registry of tokenizer types. Allows registering tokenizers from anywhere in the codebase (e.g. src/Python).
 */
class TokenizerFactory {
private:
    // Needs to be declared before `paramTokenizerType` because `paramTokenizerType` depends on `choices_`
    Core::Choice choices_;

public:
    typedef std::function<TokenizerRef(Core::Configuration const&)> CreationFunction;

    Core::ParameterChoice paramTokenizerType;

    TokenizerFactory();

    // Registering an existing name replaces its creation function
    void registerTokenizer(const char* name, CreationFunction creationFunction);

    // Create a tokenizer of the type given by `paramTokenizerType`
    TokenizerRef createTokenizer(Core::Configuration const& config) const;

private:
    std::vector<CreationFunction> registry_;
};

}  // namespace Tokenizer

#endif  // TOKENIZER_TOKENIZER_FACTORY_HH
