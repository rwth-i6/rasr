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

#include "LlmScorer.hh"

#include <sstream>

#include <Core/Application.hh>

namespace Search {

LlmScorerFactory::LlmScorerFactory()
        : choices_(), paramLlmScorerType("type", &choices_, "Choice from a set of LLM scorer types.", Core::Choice::IllegalValue), registry_() {}

void LlmScorerFactory::registerLlmScorer(const char* name, CreationFunction creationFunction) {
    choices_.addChoice(name, registry_.size());
    registry_.push_back(std::move(creationFunction));
}

Core::Ref<LlmScorer> LlmScorerFactory::createLlmScorer(Core::Configuration const& config) const {
    auto type = paramLlmScorerType(config);
    if (type == Core::Choice::IllegalValue) {
        std::stringstream ss;
        ss << "No valid LLM scorer type defined in `" << config.getSelection() << "." << paramLlmScorerType.name() << "`. ";
        ss << "Possible values are: ";
        choices_.printIdentifiers(ss);
        ss << " (types implemented in Python have to be registered via `librasr.register_llm_scorer_type` first)";
        Core::Application::us()->criticalError("%s", ss.str().c_str());
        return {};
    }
    return registry_.at(type)(config);
}

}  // namespace Search
