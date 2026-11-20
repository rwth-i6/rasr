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

#ifndef _TOKENIZER_TOKENIZER_FACTORY_HH
#define _TOKENIZER_TOKENIZER_FACTORY_HH

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
public:
    typedef std::function<Core::Status(Core::Configuration const&, TokenizerRef&)> CreationFunction;

    // A string rather than a `Core::ParameterChoice`, which reports invalid values as configuration errors that
    // terminate the program on exit
    Core::ParameterString paramTokenizerType;

    TokenizerFactory();

    // Registering an existing name replaces its creation function
    void registerTokenizer(const char* name, CreationFunction creationFunction);

    // Create a tokenizer of the type given by `paramTokenizerType`. An unknown type results in `InvalidArgument`.
    Core::Status createTokenizer(Core::Configuration const& config, TokenizerRef& tokenizer) const;

private:
    Core::Choice                  choices_;
    std::vector<CreationFunction> registry_;
};

}  // namespace Tokenizer

#endif  // _TOKENIZER_TOKENIZER_FACTORY_HH
