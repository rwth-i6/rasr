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

#include "TokenizerFactory.hh"

#include <sstream>

#include <Core/Application.hh>

namespace Tokenizer {

TokenizerFactory::TokenizerFactory()
        : choices_(), paramTokenizerType("type", &choices_, "Choice from a set of tokenizer types.", Core::Choice::IllegalValue), registry_() {}

void TokenizerFactory::registerTokenizer(const char* name, CreationFunction creationFunction) {
    // Registering a name again replaces the previous type, e.g. when re-running python code that registers a tokenizer
    auto type = choices_[name];
    if (type != Core::Choice::IllegalValue) {
        registry_[type] = std::move(creationFunction);
        return;
    }
    choices_.addChoice(name, registry_.size());
    registry_.push_back(std::move(creationFunction));
}

TokenizerRef TokenizerFactory::createTokenizer(Core::Configuration const& config) const {
    auto type = paramTokenizerType(config);
    if (type == Core::Choice::IllegalValue) {
        std::stringstream message;
        message << "No valid tokenizer type defined in `" << config.getSelection() << "." << paramTokenizerType.name() << "`. ";
        message << "Possible values are: ";
        choices_.printIdentifiers(message);
        Core::Application::us()->criticalError("%s", message.str().c_str());
        return {};
    }
    return registry_[type](config);
}

}  // namespace Tokenizer
