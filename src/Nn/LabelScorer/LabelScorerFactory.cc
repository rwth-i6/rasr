/** Copyright 2025 RWTH Aachen University. All rights reserved.
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

#include "LabelScorerFactory.hh"

#include <Core/Application.hh>

namespace Nn {

LabelScorerFactory::LabelScorerFactory()
        : choices_(), paramLabelScorerType("type", &choices_, "Choice from a set of label scorer types.", Core::Choice::IllegalValue), registry_() {}

void LabelScorerFactory::registerLabelScorer(const char* name, CreationFunction creationFunction) {
    choices_.addChoice(name, registry_.size());
    registry_.push_back(std::move(creationFunction));
}

Core::Ref<ScaledLabelScorer> LabelScorerFactory::createLabelScorer(Core::Configuration const& config) const {
    ModelCache tempCache;
    return createLabelScorer(config, tempCache);
}

Core::Ref<ScaledLabelScorer> LabelScorerFactory::createLabelScorer(Core::Configuration const& config, ModelCache& modelCache) const {
    auto type = paramLabelScorerType(config);
    if (type == Core::Choice::IllegalValue) {
        std::stringstream ss;
        ss << "No valid label scorer type defined in `" << config.getSelection() << "." << paramLabelScorerType.name() << "`.";
        ss << "Possible values are: ";
        choices_.printIdentifiers(ss);
        Core::Application::us()->criticalError("%s", ss.str().c_str());
        return {};
    }

    auto subScorer = registry_.at(type)(config, modelCache);
    return Core::ref(new ScaledLabelScorer(config, subScorer));
}

}  // namespace Nn
