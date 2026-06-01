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
#ifndef ENCODER_FACTORY_HH
#define ENCODER_FACTORY_HH

#include "Encoder.hh"

#include <functional>

#include <Core/Choice.hh>
#include <Core/Configuration.hh>
#include <Core/Parameter.hh>
#include <Core/ReferenceCounting.hh>

namespace Nn {

/*
 * Class that contains a cached model to re-use when creating multiple encoder instances
 */
class EncoderModelCache {
public:
    template<typename T>
    std::shared_ptr<T> get() const {
        if (not model_) {
            return {};
        }
        verify(type_ == typeid(T));
        return std::static_pointer_cast<T>(model_);
    }

    template<typename T>
    void put(std::shared_ptr<T> model) {
        model_ = std::static_pointer_cast<void>(model);
        type_  = typeid(T);
    }

    bool empty() const {
        return not model_;
    }

private:
    std::shared_ptr<void> model_;
    std::type_index       type_{typeid(void)};
};

/*
 * Factory class to register types of Encoders and create them.
 * Introduced so that Encoders can be registered from different places in the codebase
 * (e.g. inside src/Nn/LabelScorer and src/Onnx)
 */
class EncoderFactory {
private:
    // Needs to be declared before `paramEncoderType` because `paramEncoderType` depends on `choices_`.
    Core::Choice choices_;

public:
    typedef std::function<Core::Ref<Encoder>(Core::Configuration const&, EncoderModelCache&)> CreationFunction;

    Core::ParameterChoice paramEncoderType;

    EncoderFactory();

    /*
     * Register a new Encoder type by name and a factory function that can create an instance given a config object
     */
    void registerEncoder(const char* name, CreationFunction creationFunction);

    /*
     * Create an Encoder instance of type given by `paramEncoderType` using the config object. Optionally supply a
     * cache location for the model. If the cache is filled, the model is reused by the encoder constructor, otherwise
     * the encoder constructs a model itself and puts it into the cache.
     */
    Core::Ref<Encoder> createEncoder(Core::Configuration const& config) const;
    Core::Ref<Encoder> createEncoder(Core::Configuration const& config, EncoderModelCache& modelCache) const;

private:
    typedef std::vector<CreationFunction> Registry;
    Registry                              registry_;
};

}  // namespace Nn

#endif  // ENCODER_FACTORY_HH
