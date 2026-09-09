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

#ifndef MODEL_CACHE_HH
#define MODEL_CACHE_HH

#include <memory>
#include <typeindex>
#include <unordered_map>

#include <Core/Assertions.hh>

namespace Nn {

/*
 * Typed cache for model objects that can be shared across multiple Encoder and LabelScorer instances.
 */
class ModelCache {
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

    template<typename T, typename... Args>
    std::shared_ptr<T> getOrCreate(std::string const& key, Args&&... args) {
        auto model = get<T>(key);
        if (not model) {
            model = std::make_shared<T>(std::forward<Args>(args)...);
            put(key, model);
        }
        return model;
    }

    bool empty(std::string const& key) const {
        return models_.find(key) == models_.end();
    }

private:
    std::unordered_map<std::string, std::shared_ptr<void>> models_;
    std::unordered_map<std::string, std::type_index>       types_;
};

}  // namespace Nn

#endif  // MODEL_CACHE_HH
