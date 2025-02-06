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
#ifndef LABEL_SCORER_FACTORY_HH
#define LABEL_SCORER_FACTORY_HH

#include <Core/Choice.hh>
#include <Core/Configuration.hh>
#include <Core/Parameter.hh>
#include <Core/ReferenceCounting.hh>
#include <functional>
#include "LabelScorer.hh"

namespace Nn {

/*
 * Factory class to register types of LabelScorers and create them.
 * Introduced so that LabelScorers can be registered from different places in the codebase
 * (e.g. inside src/Nn/LabelScorer and src/Onnx).
 */
class LabelScorerFactory : public Core::ReferenceCounted {
private:
    // Needs to be declared before `paramLabelScorerType` because `paramLabelScorerType` depends on `choices_`
    Core::Choice choices_;

public:
    Core::ParameterChoice paramLabelScorerType;

    LabelScorerFactory();

    typedef std::function<Core::Ref<LabelScorer>(Core::Configuration const&)> CreationFunction;

    /*
     * Register a new LabelScorer type by name and a factory function that can create an instance given a config object
     */
    void registerLabelScorer(const char* name, CreationFunction creationFunction);

    /*
     * Create a LabelScorer instance of the type given by `paramLabelScorerType` using the config object
     */
    Core::Ref<LabelScorer> createLabelScorer(Core::Configuration const& config) const;

private:
    typedef std::vector<CreationFunction> Registry;

    Registry registry_;
};

}  // namespace Nn

#endif  // LABEL_SCORER_FACTORY_HH
