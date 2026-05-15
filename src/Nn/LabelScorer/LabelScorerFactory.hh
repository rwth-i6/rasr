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

#include <functional>

#include <Core/Choice.hh>
#include <Core/Configuration.hh>
#include <Core/Parameter.hh>
#include <Core/ReferenceCounting.hh>

#include "EncoderFactory.hh"
#include "LabelScorer.hh"
#include "ScaledLabelScorer.hh"

namespace Nn {

/*
 * Class that contains cached model objects to re-use when creating multiple LabelScorer instances.
 * The model cache is keyed because a LabelScorer may use multiple models.
 */
class LabelScorerModelCache {
public:
    template<typename T>
    std::shared_ptr<T> get(std::string const& key) const {
        auto modelIt = models_.find(key);
        if (modelIt == models_.end()) {
            return {};
        }

        auto typeIt = types_.find(key);
        verify(typeIt != types_.end());
        verify(typeIt->second == std::type_index(typeid(T)));
        return std::static_pointer_cast<T>(modelIt->second);
    }

    template<typename T>
    void put(std::string const& key, std::shared_ptr<T> model) {
        models_.insert_or_assign(key, std::static_pointer_cast<void>(model));
        types_.insert_or_assign(key, std::type_index(typeid(T)));
    }

    bool empty(std::string const& key) const {
        return models_.find(key) == models_.end();
    }

private:
    std::unordered_map<std::string, std::shared_ptr<void>> models_;
    std::unordered_map<std::string, std::type_index>       types_;
};

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
    typedef std::function<Core::Ref<LabelScorer>(Core::Configuration const&, EncoderModelCache&, LabelScorerModelCache&)> CreationFunction;

    Core::ParameterChoice paramLabelScorerType;

    LabelScorerFactory();

    /*
     * Register a new LabelScorer type by name and a factory function that can create an instance given a config object
     */
    void registerLabelScorer(const char* name, CreationFunction creationFunction);

    /*
     * Create a ScaledLabelScorer instance of the type given by `paramLabelScorerType` using the config object.
     * Optionally supply a cache for encoder models and label scorer models. If cache entries are present,
     * they are reused by the LabelScorer constructors.
     */
    Core::Ref<ScaledLabelScorer> createLabelScorer(Core::Configuration const& config) const;
    Core::Ref<ScaledLabelScorer> createLabelScorer(Core::Configuration const& config, EncoderModelCache& encoderModelCache, LabelScorerModelCache& labelScorerModelCache) const;

private:
    typedef std::vector<CreationFunction> Registry;

    Registry registry_;
};

}  // namespace Nn

#endif  // LABEL_SCORER_FACTORY_HH
